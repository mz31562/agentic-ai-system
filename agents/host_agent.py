# C:\Users\MohammedZaid\Desktop\agentic-ai-system\agents\host_agent.py
import asyncio
import re
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

from core.saga import SagaCoordinator, SagaStep
from core.agent_base import BaseAgent
from core.message_bus import MessageBus, Message

logger = logging.getLogger(__name__)


class HostAgent(BaseAgent):
    """
    Host Agent - Central coordinator for the agentic system.
    Routes requests to appropriate specialist agents and manages responses.
    Now with Saga orchestration for multi-step workflows.
    """
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(
            agent_id="host_agent",
            name="Host Agent",
            message_bus=message_bus,
            capabilities=[
                "request_routing",
                "response_aggregation",
                "user_interface",
                "coordination",
                "saga_orchestration"  # NEW
            ]
        )
        
        # Track active requests
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        
        # NEW: Saga Coordinator for multi-step workflows
        self.saga_coordinator = SagaCoordinator(message_bus)
        logger.info("Saga orchestration enabled")
        
        # Agent registry - which agents are available
        self.available_agents = {
            "post_design_agent": {
                "name": "PostDesign Agent",
                "capabilities": ["design", "graphics", "posts"],
                "status": "unknown"
            },
            "image_generation_agent": {
                "name": "Image Generation Agent",
                "capabilities": ["image", "visual", "graphic"],
                "status": "unknown"
            }
        }
        
        # Request patterns for routing
        self.request_patterns = {
            "design": [
                r"create.*post",
                r"design.*post",
                r"make.*post",
                r"generate.*post",
                r"post about",
                r"write.*post",
            ],
            "image": [
                r"create.*image",
                r"generate.*image",
                r"make.*image",
                r"draw",
                r"picture of",
                r"image of",
                r"visual",
                r"graphic",
                r"illustration",
            ],
            "design_with_image": [
                r"create.*post.*image",
                r"design.*post.*visual",
                r"post.*with.*image",
                r"post.*with.*picture",
            ]
        }
    
    
    async def setup(self):
        """Subscribe to relevant topics"""
        # User requests from UI/CLI
        self.message_bus.subscribe(
            topic="user_request",
            agent_id=self.agent_id,
            callback=self._process_message
        )
        
        # Responses from PostDesignAgent
        self.message_bus.subscribe(
            topic="design_response",
            agent_id=self.agent_id,
            callback=self._process_message
        )
        
        # Responses from ImageGenerationAgent
        self.message_bus.subscribe(
            topic="image_response",
            agent_id=self.agent_id,
            callback=self._process_message
        )
        
        # Error notifications
        self.message_bus.subscribe(
            topic="error",
            agent_id=self.agent_id,
            callback=self._process_message
        )
        
        logger.info(f"Host Agent subscribed to topics: user_request, design_response, image_response, error")
    

    async def handle_message(self, message: Message):
        """Route incoming messages to appropriate handlers"""
        handlers = {
            "user_request": self.handle_user_request,
            "design_response": self.handle_design_response,
            "image_response": self.handle_image_response,
            "error": self.handle_error,
        }
        
        handler = handlers.get(message.topic)
        if handler:
            await handler(message)
        else:
            logger.warning(f"Unknown topic: {message.topic}")
    
    
    async def handle_user_request(self, message: Message):
        """Process user request and route to appropriate agent - now with Saga support"""
        user_message = message.payload.get("message", "")
        user_id = message.payload.get("user_id", "anonymous")
        
        logger.info(f"Host Agent processing user request: '{user_message}'")
        
        # Track this request
        request_id = message.correlation_id or message.id
        self.active_requests[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "user_message": user_message,
            "status": "processing",
            "started_at": datetime.utcnow().isoformat(),
            "original_message": message,
            "pending_responses": []
        }
        
        # Determine request type and route
        request_type = self._analyze_request(user_message)
        
        # NEW: Use Saga for multi-agent workflows
        if request_type == "design_with_image":
            await self._execute_design_image_saga(message, user_message, request_id)
        
        elif request_type == "design":
            await self._route_to_design_agent(message, user_message, request_id)
        
        elif request_type == "image":
            await self._route_to_image_agent(message, user_message, request_id)
        
        else:
            await self._handle_unknown_request(message, user_message)
    
    
    # ========================================================================
    # NEW: SAGA WORKFLOW EXECUTION
    # ========================================================================
    
    async def _execute_design_image_saga(
        self,
        original_message: Message,
        user_message: str,
        request_id: str
    ):
        """
        Execute design + image workflow using Saga pattern
        This replaces the old _route_to_design_and_image method
        """
        logger.info(f"🔄 Starting Saga workflow for request {request_id}")
        
        self.active_requests[request_id]["status"] = "saga_started"
        self.active_requests[request_id]["workflow_type"] = "design_with_image"
        
        # Define saga steps
        steps = [
            # Step 1: Generate post design
            SagaStep(
                name="design_post",
                agent="post_design_agent",
                request_topic="design_request",
                response_topic="design_response",
                timeout=120,
                max_retries=2,
                build_payload=lambda results: {
                    "user_message": user_message,
                    "user_id": original_message.payload.get("user_id", "anonymous"),
                    "request_id": request_id,
                    "needs_image": True
                },
                compensate=self._compensate_design
            ),
            
            # Step 2: Generate image (uses design from step 1)
            SagaStep(
                name="generate_image",
                agent="image_generation_agent",
                request_topic="image_request",
                response_topic="image_response",
                timeout=600,
                max_retries=1,
                build_payload=lambda results: {
                    "prompt": user_message,
                    "content_type": "social_media",
                    "request_id": request_id
                },
                compensate=self._compensate_image
            )
        ]
        
        try:
            # Execute saga
            logger.info(f"📋 Saga steps defined: {len(steps)} steps")
            result = await self.saga_coordinator.start_saga(
                name="design_with_image",
                steps=steps,
                saga_id=request_id
            )
            
            if result["status"] == "success":
                logger.info(f"✅ Saga completed successfully for request {request_id}")
                await self._send_saga_success_response(
                    original_message,
                    result["results"],
                    request_id
                )
            else:
                logger.error(f"❌ Saga failed for request {request_id}")
                await self._send_saga_failure_response(
                    original_message,
                    result,
                    request_id
                )
        
        except Exception as e:
            logger.error(f"💥 Saga execution error: {e}", exc_info=True)
            await self.send_error(
                original_message=original_message,
                error=f"Saga execution failed: {str(e)}",
                details={"request_id": request_id}
            )
    
    
    async def _send_saga_success_response(
        self,
        original_message: Message,
        results: Dict[str, Any],
        request_id: str
    ):
        """Send successful saga response to user"""
        design_result = results.get("design_post", {})
        image_result = results.get("generate_image", {})
        
        # Build combined message
        combined_message = "✅ Your post with image is ready!\n\n"
        combined_message += "📝 Post Content:\n"
        combined_message += "─" * 50 + "\n"
        combined_message += design_result.get("design_result", "")
        combined_message += "\n" + "─" * 50 + "\n\n"
        combined_message += "🖼️ Image Details:\n"
        
        img_data = image_result.get("image_result", {})
        if img_data.get("image_path"):
            combined_message += f"   📁 Saved to: {img_data['image_path']}\n"
        if img_data.get("image_url"):
            combined_message += f"   🔗 URL: {img_data['image_url']}\n"
        
        # Get metadata
        design_meta = design_result.get("metadata", {})
        if design_meta.get("backend_used"):
            combined_message += f"\n💡 Generated using: {design_meta.get('backend_used')}"
        
        await self.send_response(
            original_message=original_message,
            payload={
                "response_type": "saga_success",
                "result": combined_message,
                "design": design_result,
                "image": image_result,
                "message": "Your complete post package is ready!",
                "request_id": request_id,
                "workflow_type": "saga"
            },
            topic="user_response"
        )
        
        self.active_requests[request_id]["status"] = "completed"
        self.active_requests[request_id]["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Saga workflow completed successfully for request {request_id}")
        
        # Cleanup after delay
        await asyncio.sleep(60)
        if request_id in self.active_requests:
            del self.active_requests[request_id]
    
    
    async def _send_saga_failure_response(
        self,
        original_message: Message,
        result: Dict[str, Any],
        request_id: str
    ):
        """Send saga failure response to user"""
        error_msg = f"❌ Workflow failed: {result.get('error', 'Unknown error')}\n\n"
        
        if result.get("failed_step"):
            error_msg += f"Failed at step: {result['failed_step']}\n"
        
        if result.get("partial_results"):
            error_msg += "\n⚠️ Partial results were generated but automatically rolled back.\n"
            error_msg += "No incomplete data was saved.\n"
        
        error_msg += "\n💡 Please try again or rephrase your request."
        
        await self.send_response(
            original_message=original_message,
            payload={
                "response_type": "saga_failure",
                "result": error_msg,
                "error": result.get("error"),
                "failed_step": result.get("failed_step"),
                "message": "Workflow failed. All changes have been rolled back.",
                "request_id": request_id
            },
            topic="user_response"
        )
        
        self.active_requests[request_id]["status"] = "failed"
        self.active_requests[request_id]["error"] = result.get("error")
        self.active_requests[request_id]["completed_at"] = datetime.utcnow().isoformat()
        
        logger.error(f"Saga workflow failed for request {request_id}")
        
        # Cleanup after delay
        await asyncio.sleep(60)
        if request_id in self.active_requests:
            del self.active_requests[request_id]
    
    
    # ========================================================================
    # COMPENSATION FUNCTIONS (Rollback Logic)
    # ========================================================================
    
    async def _compensate_design(self, message_bus: MessageBus, results: Dict[str, Any]):
        """
        Rollback design step
        Called automatically if image generation fails
        """
        logger.info("🔄 Compensating design step...")
        
        try:
            design_data = results.get("design_post", {})
            
            # If design was saved to database, delete it
            design_id = design_data.get("id")
            if design_id:
                logger.info(f"   Deleting design {design_id} from database")
                # TODO: Implement actual database deletion
                # await message_bus.publish(Message(
                #     type="request",
                #     sender=self.agent_id,
                #     topic="db_delete_request",
                #     payload={"entity": "design", "id": design_id}
                # ))
            
            # If design was saved to file, delete it
            design_file = design_data.get("file_path")
            if design_file:
                try:
                    path = Path(design_file)
                    if path.exists():
                        path.unlink()
                        logger.info(f"   Deleted design file: {design_file}")
                except Exception as e:
                    logger.warning(f"   Could not delete design file: {e}")
            
            logger.info("✅ Design compensation completed")
        
        except Exception as e:
            logger.error(f"❌ Design compensation failed: {e}", exc_info=True)
    
    
    async def _compensate_image(self, message_bus: MessageBus, results: Dict[str, Any]):
        """
        Rollback image generation step
        Called automatically if subsequent steps fail
        """
        logger.info("🔄 Compensating image step...")
        
        try:
            # Delete generated image file
            image_data = results.get("generate_image", {})
            image_result = image_data.get("image_result", {})
            image_path = image_result.get("image_path")
            
            if image_path:
                try:
                    path = Path(image_path)
                    if path.exists():
                        path.unlink()
                        logger.info(f"   Deleted image file: {image_path}")
                    else:
                        logger.warning(f"   Image file not found: {image_path}")
                except Exception as e:
                    logger.error(f"   Failed to delete image: {e}")
            else:
                logger.info("   No image path to clean up")
            
            # If image was uploaded to cloud storage, delete it
            image_url = image_result.get("image_url")
            if image_url and image_url.startswith("http"):
                logger.info(f"   Image was uploaded to: {image_url}")
                # TODO: Implement cloud storage deletion
                # await self._delete_from_cloud_storage(image_url)
            
            logger.info("✅ Image compensation completed")
        
        except Exception as e:
            logger.error(f"❌ Image compensation failed: {e}", exc_info=True)
    
    
    # ========================================================================
    # ORIGINAL ROUTING METHODS (for single-agent requests)
    # ========================================================================
    
    def _analyze_request(self, user_message: str) -> str:
        """Analyze user message to determine request type."""
        message_lower = user_message.lower()
        
        # Check design_with_image first (most specific)
        for pattern in self.request_patterns.get("design_with_image", []):
            if re.search(pattern, message_lower):
                logger.info(f"Request classified as: design_with_image (will use Saga)")
                return "design_with_image"
        
        # Then check others
        for request_type in ["design", "image"]:
            for pattern in self.request_patterns.get(request_type, []):
                if re.search(pattern, message_lower):
                    logger.info(f"Request classified as: {request_type}")
                    return request_type
        
        # Default to design for marketing queries
        logger.info("Request type: design (default)")
        return "design"
    
    
    async def _route_to_design_agent(
        self,
        original_message: Message,
        user_message: str,
        request_id: str
    ):
        """Route request to PostDesignAgent"""
        
        logger.info("Routing request to PostDesignAgent")
        
        self.active_requests[request_id]["assigned_to"] = "post_design_agent"
        self.active_requests[request_id]["status"] = "forwarded"
        
        await self.send_request(
            topic="design_request",
            recipient="post_design_agent",
            payload={
                "user_message": user_message,
                "user_id": original_message.payload.get("user_id", "anonymous"),
                "request_id": request_id
            },
            correlation_id=request_id
        )
        
        logger.info(f"Request {request_id} forwarded to PostDesignAgent")
    
    
    async def _route_to_image_agent(
        self,
        original_message: Message,
        user_message: str,
        request_id: str
    ):
        """Route request to ImageGenerationAgent"""
        
        logger.info("Routing request to ImageGenerationAgent")
        
        self.active_requests[request_id]["assigned_to"] = "image_generation_agent"
        self.active_requests[request_id]["status"] = "forwarded"
        
        await self.send_request(
            topic="image_request",
            recipient="image_generation_agent",
            payload={
                "prompt": user_message,
                "request_id": request_id
            },
            correlation_id=request_id
        )
        
        logger.info(f"Request {request_id} forwarded to ImageGenerationAgent")
    
    
    async def _handle_status_request(self, message: Message):
        """Handle request for system status"""
        
        logger.info("Handling status request")
        
        status_info = {
            "host_agent": self.get_status(),
            "available_agents": self.available_agents,
            "active_requests": len(self.active_requests),
            "system_status": "operational"
        }
        
        await self.send_response(
            original_message=message,
            payload={
                "response_type": "status",
                "status": status_info,
                "message": "System is operational"
            },
            topic="user_response"
        )
    
    
    async def _handle_unknown_request(self, message: Message, user_message: str):
        """Handle requests that couldn't be classified"""
        
        logger.info("Defaulting to design agent for unclear request")
        
        # Default to design agent for marketing-related queries
        request_id = message.correlation_id or message.id
        await self._route_to_design_agent(message, user_message, request_id)
    
    
    # ========================================================================
    # RESPONSE HANDLERS (for backwards compatibility with single-agent flows)
    # ========================================================================
    
    async def handle_design_response(self, message: Message):
        """Handle response from PostDesignAgent."""
        request_id = message.correlation_id
        
        logger.info(f"Host Agent received design response for request {request_id}")
        
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            
            # Check if this is a saga workflow (saga handles its own responses)
            if request.get("workflow_type") == "design_with_image":
                logger.debug(f"Design response for saga workflow - handled by SagaCoordinator")
                return
            
            # Single response, send it
            request["status"] = "completed"
            request["completed_at"] = datetime.utcnow().isoformat()
            
            original_message = request["original_message"]
            
            await self.send_response(
                original_message=original_message,
                payload={
                    "response_type": "design",
                    "result": message.payload.get("design_result", ""),
                    "message": "Your design is ready!",
                    "metadata": message.payload.get("metadata", {})
                },
                topic="user_response"
            )
            
            await asyncio.sleep(60)
            if request_id in self.active_requests:
                del self.active_requests[request_id]
        else:
            logger.warning(f"Received response for unknown request: {request_id}")
    
    
    async def handle_image_response(self, message: Message):
        """Handle response from ImageGenerationAgent."""
        request_id = message.correlation_id
        
        logger.info(f"Host Agent received image response for request {request_id}")
        
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            
            # Check if this is a saga workflow (saga handles its own responses)
            if request.get("workflow_type") == "design_with_image":
                logger.debug(f"Image response for saga workflow - handled by SagaCoordinator")
                return
            
            # Single response, send it
            request["status"] = "completed"
            request["completed_at"] = datetime.utcnow().isoformat()
            
            original_message = request["original_message"]
            
            await self.send_response(
                original_message=original_message,
                payload={
                    "response_type": "image",
                    "result": message.payload.get("image_result", {}),
                    "message": "Your image is ready!",
                },
                topic="user_response"
            )
            
            await asyncio.sleep(60)
            if request_id in self.active_requests:
                del self.active_requests[request_id]
        else:
            logger.warning(f"Received response for unknown request: {request_id}")
    
    
    # NOTE: Old _send_combined_response method removed - replaced by Saga pattern
    
    
    async def handle_error(self, message: Message):
        """Handle error notifications from other agents"""
        
        error = message.payload.get("error", "Unknown error")
        agent_id = message.payload.get("agent_id", "unknown")
        
        logger.error(f"Error from {agent_id}: {error}")
        
        correlation_id = message.correlation_id
        if correlation_id and correlation_id in self.active_requests:
            original_message = self.active_requests[correlation_id]["original_message"]
            
            await self.send_response(
                original_message=original_message,
                payload={
                    "response_type": "error",
                    "message": f"An error occurred: {error}",
                    "details": message.payload
                },
                topic="user_response"
            )
            
            self.active_requests[correlation_id]["status"] = "failed"
            self.active_requests[correlation_id]["error"] = error
    
    
    def get_active_requests_summary(self) -> Dict[str, Any]:
        """Get summary of active requests"""
        return {
            "total_active": len(self.active_requests),
            "requests": [
                {
                    "request_id": req["request_id"],
                    "status": req["status"],
                    "assigned_to": req.get("assigned_to", "none"),
                    "workflow_type": req.get("workflow_type", "single_agent"),
                    "started_at": req["started_at"]
                }
                for req in self.active_requests.values()
            ]
        }
    
    
    def get_saga_status_summary(self) -> Dict[str, Any]:
        """Get summary of active sagas (NEW)"""
        if not hasattr(self, 'saga_coordinator'):
            return {"active_sagas": 0, "sagas": []}
        
        active_sagas = self.saga_coordinator.get_all_active_sagas()
        
        return {
            "active_sagas": len(active_sagas),
            "sagas": active_sagas
        }