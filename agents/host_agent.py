import asyncio
import re
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from core.agent_base import BaseAgent
from core.message_bus import MessageBus, Message

logger = logging.getLogger(__name__)


class HostAgent(BaseAgent):
    """
    Host Agent - Central coordinator for the agentic system.
    Routes requests to appropriate specialist agents and manages responses.
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
                "coordination"
            ]
        )
        
        # Track active requests
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        
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
            ],
            "math": [
                r"calculate",
                r"solve",
                r"fibonacci",
                r"equation",
                r"math",
                r"factorial",
                r"prime",
            ],
            "status": [
                r"status",
                r"available agents",
                r"what can you do",
                r"help",
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
        
        if message.topic == "user_request":
            await self.handle_user_request(message)
        
        elif message.topic == "design_response":
            await self.handle_design_response(message)
        
        elif message.topic == "image_response":
            await self.handle_image_response(message)
        
        elif message.topic == "error":
            await self.handle_error(message)
        
        else:
            logger.warning(f"Host Agent received unknown topic: {message.topic}")
    
    
    async def handle_user_request(self, message: Message):
        """Process user request and route to appropriate agent."""
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
        
        if request_type == "design_with_image":
            await self._route_to_design_and_image(message, user_message, request_id)
        
        elif request_type == "design":
            await self._route_to_design_agent(message, user_message, request_id)
        
        elif request_type == "image":
            await self._route_to_image_agent(message, user_message, request_id)
        
        elif request_type == "status":
            await self._handle_status_request(message)
        
        elif request_type == "math":
            # Math is handled by PostDesign Agent
            await self._route_to_design_agent(message, user_message, request_id)
        
        else:
            await self._handle_unknown_request(message, user_message)
    
    
    def _analyze_request(self, user_message: str) -> str:
        """Analyze user message to determine request type."""
        message_lower = user_message.lower()
        
        for request_type, patterns in self.request_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    logger.info(f"Request classified as: {request_type}")
                    return request_type
        
        logger.info("Request type: unknown")
        return "unknown"
    
    
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
    
    
    async def _route_to_design_and_image(
        self,
        original_message: Message,
        user_message: str,
        request_id: str
    ):
        """Route request to both PostDesign and Image agents for coordinated response"""
        
        logger.info("Routing request to both PostDesign and Image agents")
        
        self.active_requests[request_id]["assigned_to"] = "post_design_agent,image_generation_agent"
        self.active_requests[request_id]["status"] = "forwarded_multi"
        self.active_requests[request_id]["pending_responses"] = ["design", "image"]
        
        # Send to both agents
        await self.send_request(
            topic="design_request",
            recipient="post_design_agent",
            payload={
                "user_message": user_message,
                "user_id": original_message.payload.get("user_id", "anonymous"),
                "request_id": request_id,
                "needs_image": True
            },
            correlation_id=request_id
        )
        
        await self.send_request(
            topic="image_request",
            recipient="image_generation_agent",
            payload={
                "prompt": user_message,
                "request_id": request_id
            },
            correlation_id=request_id
        )
        
        logger.info(f"Request {request_id} forwarded to both agents")
    
    
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
        
        logger.info("Handling unknown request type")
        
        response_text = (
            f"I'm not sure how to help with: '{user_message}'\n\n"
            "I can help you with:\n"
            "- Creating posts and designs\n"
            "- Generating images and graphics\n"
            "- Math calculations and visualizations\n"
            "- Checking system status\n\n"
            "Try rephrasing your request!"
        )
        
        await self.send_response(
            original_message=message,
            payload={
                "response_type": "error",
                "message": response_text
            },
            topic="user_response"
        )
    
    
    async def handle_design_response(self, message: Message):
        """Handle response from PostDesignAgent."""
        request_id = message.correlation_id
        
        logger.info(f"Host Agent received design response for request {request_id}")
        
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            
            # Check if waiting for multiple responses
            if "pending_responses" in request and request["pending_responses"]:
                request["design_result"] = message.payload
                request["pending_responses"].remove("design")
                
                # If still waiting for image, don't respond yet
                if request["pending_responses"]:
                    logger.info(f"Waiting for remaining responses: {request['pending_responses']}")
                    return
                
                # Both responses received, combine them
                await self._send_combined_response(request_id)
            else:
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
            
            # Check if waiting for multiple responses
            if "pending_responses" in request and request["pending_responses"]:
                request["image_result"] = message.payload
                request["pending_responses"].remove("image")
                
                # If still waiting for design, don't respond yet
                if request["pending_responses"]:
                    logger.info(f"Waiting for remaining responses: {request['pending_responses']}")
                    return
                
                # Both responses received, combine them
                await self._send_combined_response(request_id)
            else:
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
    
    
    async def _send_combined_response(self, request_id: str):
        """Send combined response from multiple agents"""
        request = self.active_requests[request_id]
        request["status"] = "completed"
        request["completed_at"] = datetime.utcnow().isoformat()
        
        original_message = request["original_message"]
        
        design_result = request.get("design_result", {})
        image_result = request.get("image_result", {})
        
        combined_message = "Your post with image is ready!\n\n"
        combined_message += "📝 Post Content:\n"
        combined_message += design_result.get("design_result", "")
        combined_message += "\n\n🖼️ Image Details:\n"
        
        img_data = image_result.get("image_result", {})
        if img_data.get("image_path"):
            combined_message += f"Saved to: {img_data['image_path']}\n"
        if img_data.get("image_url"):
            combined_message += f"URL: {img_data['image_url']}\n"
        
        await self.send_response(
            original_message=original_message,
            payload={
                "response_type": "combined",
                "result": combined_message,
                "design": design_result,
                "image": image_result,
                "message": "Your complete post package is ready!"
            },
            topic="user_response"
        )
        
        await asyncio.sleep(60)
        if request_id in self.active_requests:
            del self.active_requests[request_id]
    
    
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
                    "started_at": req["started_at"]
                }
                for req in self.active_requests.values()
            ]
        }