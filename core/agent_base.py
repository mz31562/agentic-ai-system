import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from core.message_bus import MessageBus, Message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Provides common functionality for message handling and lifecycle management.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        message_bus: MessageBus,
        capabilities: Optional[List[str]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name
            message_bus: Reference to the message bus
            capabilities: List of things this agent can do
        """
        self.agent_id = agent_id
        self.name = name
        self.message_bus = message_bus
        self.capabilities = capabilities or []
        
        self.status = "initialized"  # initialized, running, busy, stopped
        self.is_running = False
        
        self.pending_requests: Dict[str, Message] = {}  # correlation_id: message
        self.processed_count = 0
        
        logger.info(f"Agent '{self.name}' ({self.agent_id}) initialized")
    
    
    @abstractmethod
    async def setup(self):
        """
        Setup the agent - subscribe to topics, initialize resources.
        Must be implemented by derived classes.
        """
        pass
    
    
    @abstractmethod
    async def handle_message(self, message: Message):
        """
        Handle incoming messages.
        Must be implemented by derived classes.
        
        Args:
            message: The message to process
        """
        pass
    
    
    async def start(self):
        """Start the agent and begin processing messages"""
        logger.info(f"Starting agent '{self.name}'...")
        
        try:
            await self.setup()
            
            self.is_running = True
            self.status = "running"
            
            logger.info(f"Agent '{self.name}' started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start agent '{self.name}': {e}", exc_info=True)
            self.status = "error"
            raise
    
    
    async def stop(self):
        """Stop the agent gracefully"""
        logger.info(f"Stopping agent '{self.name}'...")
        
        self.is_running = False
        self.status = "stopped"
        
        if self.pending_requests:
            logger.info(f"Waiting for {len(self.pending_requests)} pending requests...")
            try:
                await asyncio.wait_for(asyncio.sleep(1), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.info(f"Shutdown interrupted or timed out for '{self.name}'")
        
        logger.info(f"Agent '{self.name}' stopped")
    
    
    async def send_message(
        self,
        topic: str,
        payload: Dict[str, Any],
        message_type: str = "notification",
        recipient: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Message:
        """
        Send a message via the message bus.
        
        Args:
            topic: Message topic
            payload: Message data
            message_type: Type of message (request, response, notification, error)
            recipient: Specific recipient agent ID (None for broadcast)
            correlation_id: ID to link related messages
            
        Returns:
            The sent message
        """
        message = Message(
            type=message_type,
            sender=self.agent_id,
            recipient=recipient,
            topic=topic,
            payload=payload,
            correlation_id=correlation_id
        )
        
        await self.message_bus.publish(message)
        
        if message_type == "request" and correlation_id:
            self.pending_requests[correlation_id] = message
        
        return message
    
    
    async def send_request(
        self,
        topic: str,
        recipient: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Send a request message and return correlation ID for tracking.
        
        Args:
            topic: Request topic
            recipient: Target agent ID
            payload: Request data
            correlation_id: Optional ID (will generate if not provided)
            
        Returns:
            Correlation ID for tracking the response
        """
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        await self.send_message(
            topic=topic,
            payload=payload,
            message_type="request",
            recipient=recipient,
            correlation_id=correlation_id
        )
        
        return correlation_id
    
    
    async def send_response(
        self,
        original_message: Message,
        payload: Dict[str, Any],
        topic: Optional[str] = None
    ):
        """
        Send a response to a received message.
        
        Args:
            original_message: The message being responded to
            payload: Response data
            topic: Optional topic (defaults to original topic + '_response')
        """
        response_topic = topic or f"{original_message.topic}_response"
        
        await self.send_message(
            topic=response_topic,
            payload=payload,
            message_type="response",
            recipient=original_message.sender,
            correlation_id=original_message.correlation_id
        )
        
        if original_message.correlation_id in self.pending_requests:
            del self.pending_requests[original_message.correlation_id]
    
    
    async def send_error(
        self,
        original_message: Message,
        error: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Send an error message in response to a failed request.
        
        Args:
            original_message: The message that caused the error
            error: Error description
            details: Additional error details
        """
        error_payload = {
            "error": error,
            "original_topic": original_message.topic,
            "original_message_id": original_message.id,
            "details": details or {}
        }
        
        await self.send_message(
            topic="error",
            payload=error_payload,
            message_type="error",
            recipient=original_message.sender,
            correlation_id=original_message.correlation_id
        )
    
    
    async def notify(
        self,
        topic: str,
        payload: Dict[str, Any]
    ):
        """
        Broadcast a notification to all subscribers.
        
        Args:
            topic: Notification topic
            payload: Notification data
        """
        await self.send_message(
            topic=topic,
            payload=payload,
            message_type="notification",
            recipient=None  # Broadcast
        )
    
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.
        
        Returns:
            Dictionary with agent status information
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status,
            "is_running": self.is_running,
            "capabilities": self.capabilities,
            "processed_count": self.processed_count,
            "pending_requests": len(self.pending_requests)
        }
    
    
    async def _process_message(self, message: Message):
        """
        Internal message processing wrapper.
        Adds error handling and status updates.
        """
        try:
            self.status = "busy"
            
            await self.handle_message(message)
            
            self.processed_count += 1
            self.status = "running"
            
        except Exception as e:
            logger.error(
                f"Agent '{self.name}' error processing message: {e}",
                exc_info=True
            )
            
            await self.send_error(
                original_message=message,
                error=str(e),
                details={"exception_type": type(e).__name__}
            )
            
            self.status = "running"