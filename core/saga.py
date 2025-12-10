import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from core.message_bus import MessageBus, Message

logger = logging.getLogger(__name__)


class SagaStatus(Enum):
    """Saga execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"


class StepStatus(Enum):
    """Individual step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    """Definition of a single saga step"""
    name: str
    agent: str
    request_topic: str
    response_topic: str
    build_payload: Callable[[Dict[str, Any]], Dict[str, Any]]
    compensate: Optional[Callable[[MessageBus, Dict[str, Any]], Any]] = None
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 0


@dataclass
class SagaStepResult:
    """Result of a saga step execution"""
    name: str
    status: StepStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0


class Saga:
    """
    Manages a multi-step workflow with automatic compensation on failure
    """
    
    def __init__(
        self,
        saga_id: str,
        name: str,
        steps: List[SagaStep],
        message_bus: MessageBus
    ):
        self.saga_id = saga_id
        self.name = name
        self.steps = steps
        self.message_bus = message_bus
        
        self.status = SagaStatus.PENDING
        self.current_step_index = 0
        self.step_results: Dict[str, SagaStepResult] = {}
        self.all_results: Dict[str, Any] = {}
        
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        self._response_futures: Dict[str, asyncio.Future] = {}
    
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute all saga steps in sequence
        
        Returns:
            {
                "status": "success" | "failed",
                "results": {...},  # All step results
                "error": str,      # If failed
                "saga_id": str,
                "duration_ms": int
            }
        """
        self.status = SagaStatus.RUNNING
        self.started_at = datetime.now()
        
        logger.info(f"Saga '{self.name}' ({self.saga_id}) starting execution")
        
        try:
            for index, step in enumerate(self.steps):
                self.current_step_index = index
                
                logger.info(
                    f"Saga {self.saga_id}: Executing step {index + 1}/{len(self.steps)} - {step.name}"
                )
                
                step_result = await self._execute_step_with_retry(step)
                
                self.step_results[step.name] = step_result
                
                if step_result.status == StepStatus.COMPLETED:
                    self.all_results[step.name] = step_result.result
                    logger.info(f"Saga {self.saga_id}: Step '{step.name}' completed")
                else:
                    raise Exception(f"Step '{step.name}' failed: {step_result.error}")
            
            self.status = SagaStatus.COMPLETED
            self.completed_at = datetime.now()
            
            duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
            
            logger.info(
                f"Saga '{self.name}' ({self.saga_id}) completed successfully in {duration_ms}ms"
            )
            
            return {
                "status": "success",
                "results": self.all_results,
                "saga_id": self.saga_id,
                "duration_ms": duration_ms
            }
        
        except Exception as e:
            logger.error(f"Saga {self.saga_id} failed: {e}")
            self.status = SagaStatus.FAILED
            self.completed_at = datetime.now()
            
            await self._compensate()
            
            duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
            
            return {
                "status": "failed",
                "error": str(e),
                "partial_results": self.all_results,
                "saga_id": self.saga_id,
                "duration_ms": duration_ms,
                "failed_step": self.steps[self.current_step_index].name
            }
    
    
    async def _execute_step_with_retry(self, step: SagaStep) -> SagaStepResult:
        """Execute a step with retry logic"""
        result = SagaStepResult(
            name=step.name,
            status=StepStatus.RUNNING,
            started_at=datetime.now()
        )
        
        for attempt in range(step.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Saga {self.saga_id}: Retrying step '{step.name}' (attempt {attempt + 1})")
                    await asyncio.sleep(min(2 ** attempt, 10))  # Exponential backoff
                
                step_response = await self._execute_step(step)
                
                result.status = StepStatus.COMPLETED
                result.result = step_response
                result.completed_at = datetime.now()
                result.retries = attempt
                
                return result
            
            except Exception as e:
                result.error = str(e)
                
                if attempt >= step.max_retries:
                    result.status = StepStatus.FAILED
                    result.completed_at = datetime.now()
                    result.retries = attempt
                    return result
        
        return result
    
    
    async def _execute_step(self, step: SagaStep) -> Dict[str, Any]:
        """Execute a single saga step"""
        
        try:
            payload = step.build_payload(self.all_results)
        except Exception as e:
            raise Exception(f"Failed to build payload for step '{step.name}': {e}")
        
        correlation_id = f"{self.saga_id}_step_{step.name}_{uuid.uuid4().hex[:8]}"
        
        response_future = asyncio.Future()
        self._response_futures[correlation_id] = response_future
        
        async def response_handler(message: Message):
            if message.correlation_id == correlation_id:
                if not response_future.done():
                    response_future.set_result(message.payload)
        
        self.message_bus.subscribe(
            topic=step.response_topic,
            agent_id=f"saga_{self.saga_id}_{step.name}",
            callback=response_handler
        )
        
        try:
            await self.message_bus.publish(Message(
                type="request",
                sender=f"saga_{self.saga_id}",
                recipient=step.agent,
                topic=step.request_topic,
                payload=payload,
                correlation_id=correlation_id
            ))
            
            response = await asyncio.wait_for(
                response_future,
                timeout=step.timeout
            )
            
            return response
        
        except asyncio.TimeoutError:
            raise Exception(f"Step '{step.name}' timed out after {step.timeout}s")
        
        except Exception as e:
            raise Exception(f"Step '{step.name}' failed: {e}")
        
        finally:
            self.message_bus.unsubscribe(
                topic=step.response_topic,
                agent_id=f"saga_{self.saga_id}_{step.name}"
            )
            if correlation_id in self._response_futures:
                del self._response_futures[correlation_id]
    
    
    async def _compensate(self):
        """
        Rollback completed steps in reverse order
        """
        if not self.all_results:
            logger.info(f"Saga {self.saga_id}: No steps to compensate")
            return
        
        logger.warning(f"Saga {self.saga_id}: Starting compensation (rollback)")
        self.status = SagaStatus.COMPENSATING
        
        completed_steps = [
            step for step in self.steps[:self.current_step_index + 1]
            if step.name in self.all_results and step.compensate is not None
        ]
        
        for step in reversed(completed_steps):
            try:
                logger.info(f"Saga {self.saga_id}: Compensating step '{step.name}'")
                
                await step.compensate(self.message_bus, self.all_results)
                
                if step.name in self.step_results:
                    self.step_results[step.name].status = StepStatus.COMPENSATED
                
                logger.info(f"Saga {self.saga_id}: Successfully compensated '{step.name}'")
            
            except Exception as e:
                logger.error(
                    f"Saga {self.saga_id}: Compensation failed for '{step.name}': {e}",
                    exc_info=True
                )
        
        self.status = SagaStatus.COMPENSATED
        logger.info(f"Saga {self.saga_id}: Compensation completed")
    
    
    def get_status(self) -> Dict[str, Any]:
        """Get current saga status"""
        return {
            "saga_id": self.saga_id,
            "name": self.name,
            "status": self.status.value,
            "current_step": self.current_step_index + 1,
            "total_steps": len(self.steps),
            "step_results": {
                name: {
                    "status": result.status.value,
                    "retries": result.retries,
                    "error": result.error
                }
                for name, result in self.step_results.items()
            },
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class SagaCoordinator:
    """
    Coordinates multiple sagas
    """
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.active_sagas: Dict[str, Saga] = {}
        self.completed_sagas: Dict[str, Saga] = {}
    
    
    async def start_saga(
        self,
        name: str,
        steps: List[SagaStep],
        saga_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a new saga
        
        Args:
            name: Human-readable saga name
            steps: List of saga steps to execute
            saga_id: Optional custom saga ID
            
        Returns:
            Saga execution result
        """
        if not saga_id:
            saga_id = str(uuid.uuid4())
        
        saga = Saga(
            saga_id=saga_id,
            name=name,
            steps=steps,
            message_bus=self.message_bus
        )
        
        self.active_sagas[saga_id] = saga
        
        try:
            result = await saga.execute()
            
            self.completed_sagas[saga_id] = saga
            if saga_id in self.active_sagas:
                del self.active_sagas[saga_id]
            
            return result
        
        except Exception as e:
            logger.error(f"Saga coordinator error: {e}")
            
            self.completed_sagas[saga_id] = saga
            if saga_id in self.active_sagas:
                del self.active_sagas[saga_id]
            
            raise
    
    
    def get_saga_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific saga"""
        saga = self.active_sagas.get(saga_id) or self.completed_sagas.get(saga_id)
        
        if saga:
            return saga.get_status()
        
        return None
    
    
    def get_all_active_sagas(self) -> List[Dict[str, Any]]:
        """Get status of all active sagas"""
        return [saga.get_status() for saga in self.active_sagas.values()]