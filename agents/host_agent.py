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
    Uses Saga orchestration for multi-step workflows.

    Changes from original:
    - _analyze_request replaced with cleaner keyword approach (no fragile regex)
    - _analyze_request_with_llm added for optional LLM-based classification
    - active_requests cleanup moved to a dedicated _cleanup_request() helper
      instead of raw asyncio.sleep(60) inside response handlers
    - llm_manager optionally accepted for LLM-based routing
    """

    def __init__(self, message_bus: MessageBus, llm_manager=None):
        super().__init__(
            agent_id="host_agent",
            name="Host Agent",
            message_bus=message_bus,
            capabilities=[
                "request_routing",
                "response_aggregation",
                "user_interface",
                "coordination",
                "saga_orchestration"
            ]
        )

        self.llm_manager = llm_manager  # optional — used for LLM-based intent classification
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.saga_coordinator = SagaCoordinator(message_bus)
        logger.info("Saga orchestration enabled")

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


    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def setup(self):
        """Subscribe to relevant topics"""
        for topic in ["user_request", "design_response", "image_response", "error"]:
            self.message_bus.subscribe(
                topic=topic,
                agent_id=self.agent_id,
                callback=self._process_message
            )
        logger.info("Host Agent subscribed to: user_request, design_response, image_response, error")


    # -------------------------------------------------------------------------
    # Message routing
    # -------------------------------------------------------------------------

    async def handle_message(self, message: Message):
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


    # -------------------------------------------------------------------------
    # Intent classification
    # -------------------------------------------------------------------------

    def _analyze_request(self, user_message: str) -> str:
        """
        Classify user intent using keyword matching.
        Replaces the original fragile regex approach with a cleaner
        priority-based keyword check.

        Returns one of: "design_with_image", "image", "design"
        """
        msg = user_message.lower().strip()

        has_post_intent = any(kw in msg for kw in [
            "post", "caption", "copy", "write", "content",
            "email", "tweet", "article", "blog", "pitch", "ad"
        ])
        has_image_intent = any(kw in msg for kw in [
            "image", "picture", "photo", "visual", "graphic",
            "illustration", "draw", "render", "generate image"
        ])

        if has_post_intent and has_image_intent:
            logger.info("Request classified as: design_with_image")
            return "design_with_image"

        if has_image_intent and not has_post_intent:
            logger.info("Request classified as: image")
            return "image"

        logger.info("Request classified as: design (default)")
        return "design"


    async def _analyze_request_with_llm(self, user_message: str) -> str:
        """
        LLM-based intent classifier — higher accuracy than keyword matching.
        Automatically used if llm_manager was passed into __init__.
        Falls back to keyword matching on any failure.

        Returns one of: "design", "image", "design_with_image"
        """
        if not self.llm_manager:
            return self._analyze_request(user_message)

        prompt = f"""Classify this user request into exactly one category.

User request: "{user_message}"

Categories:
- "design"            → user wants text content only (post, caption, email, article, ad copy, etc.)
- "image"             → user wants an image only (photo, illustration, graphic, etc.)
- "design_with_image" → user wants BOTH text content AND an image together

Rules:
- Reply with ONLY the category name, nothing else
- When in doubt, default to "design"
- "create a post" alone = "design"
- "create a post with an image" = "design_with_image"
- "generate an image of a cat" = "image"

Category:"""

        try:
            result = await self.llm_manager.complete(
                prompt=prompt,
                task_type="reasoning",
                complexity="simple",
                max_tokens=10,
                temperature=0.0
            )
            category = result["content"].strip().lower().strip('"').strip("'")
            valid = {"design", "image", "design_with_image"}
            if category in valid:
                logger.info(f"LLM classified request as: {category}")
                return category
            logger.warning(f"LLM returned unexpected category '{category}', falling back to keywords")
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to keywords")

        return self._analyze_request(user_message)


    # -------------------------------------------------------------------------
    # Request handling
    # -------------------------------------------------------------------------

    async def handle_user_request(self, message: Message):
        """Process user request and route to appropriate agent"""
        user_message = message.payload.get("message", "")
        user_id = message.payload.get("user_id", "anonymous")

        # Input validation
        if not user_message or not user_message.strip():
            await self.send_response(
                original_message=message,
                payload={"response_type": "error", "message": "Please enter a message."},
                topic="user_response"
            )
            return

        logger.info(f"Host Agent processing: '{user_message}'")

        request_id = message.correlation_id or message.id
        self.active_requests[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "user_message": user_message,
            "status": "processing",
            "started_at": datetime.utcnow().isoformat(),
            "original_message": message,
        }

        # Use LLM classification if available, otherwise keyword matching
        request_type = await self._analyze_request_with_llm(user_message)

        if request_type == "design_with_image":
            await self._execute_design_image_saga(message, user_message, request_id)
        elif request_type == "image":
            await self._route_to_image_agent(message, user_message, request_id)
        else:
            await self._route_to_design_agent(message, user_message, request_id)


    # -------------------------------------------------------------------------
    # Routing helpers
    # -------------------------------------------------------------------------

    async def _route_to_design_agent(self, original_message: Message, user_message: str, request_id: str):
        """Route request to PostDesignAgent"""
        logger.info("Routing to PostDesignAgent")
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


    async def _route_to_image_agent(self, original_message: Message, user_message: str, request_id: str):
        """Route request to ImageGenerationAgent"""
        logger.info("Routing to ImageGenerationAgent")
        self.active_requests[request_id]["assigned_to"] = "image_generation_agent"
        self.active_requests[request_id]["status"] = "forwarded"
        await self.send_request(
            topic="image_request",
            recipient="image_generation_agent",
            payload={"prompt": user_message, "request_id": request_id},
            correlation_id=request_id
        )


    # -------------------------------------------------------------------------
    # Saga workflow
    # -------------------------------------------------------------------------

    async def _execute_design_image_saga(self, original_message: Message, user_message: str, request_id: str):
        """Execute design + image workflow using Saga pattern"""
        logger.info(f"Starting Saga workflow for request {request_id}")
        self.active_requests[request_id]["status"] = "saga_started"
        self.active_requests[request_id]["workflow_type"] = "design_with_image"

        steps = [
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
            result = await self.saga_coordinator.start_saga(
                name="design_with_image",
                steps=steps,
                saga_id=request_id
            )
            if result["status"] == "success":
                await self._send_saga_success_response(original_message, result["results"], request_id)
            else:
                await self._send_saga_failure_response(original_message, result, request_id)
        except Exception as e:
            logger.error(f"Saga execution error: {e}", exc_info=True)
            await self.send_error(
                original_message=original_message,
                error=f"Saga execution failed: {str(e)}",
                details={"request_id": request_id}
            )


    async def _send_saga_success_response(self, original_message: Message, results: Dict[str, Any], request_id: str):
        design_result = results.get("design_post", {})
        image_result = results.get("generate_image", {})
        img_data = image_result.get("image_result", {})

        combined = "Your post with image is ready.\n\n"
        combined += "Post Content:\n" + "─" * 50 + "\n"
        combined += design_result.get("design_result", "") + "\n" + "─" * 50 + "\n\n"
        combined += "Image Details:\n"
        if img_data.get("image_path"):
            combined += f"   Saved to: {img_data['image_path']}\n"
        if img_data.get("image_url"):
            combined += f"   URL: {img_data['image_url']}\n"

        design_meta = design_result.get("metadata", {})
        if design_meta.get("backend_used"):
            combined += f"\nGenerated using: {design_meta.get('backend_used')}"

        await self.send_response(
            original_message=original_message,
            payload={
                "response_type": "saga_success",
                "result": combined,
                "design": design_result,
                "image": image_result,
                "message": "Your complete post package is ready",
                "request_id": request_id,
                "workflow_type": "saga"
            },
            topic="user_response"
        )
        self._cleanup_request(request_id, status="completed")


    async def _send_saga_failure_response(self, original_message: Message, result: Dict[str, Any], request_id: str):
        error_msg = f"Workflow failed: {result.get('error', 'Unknown error')}\n\n"
        if result.get("failed_step"):
            error_msg += f"Failed at step: {result['failed_step']}\n"
        if result.get("partial_results"):
            error_msg += "\nPartial results were generated but automatically rolled back.\n"
        error_msg += "\nPlease try again or rephrase your request."

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
        self._cleanup_request(request_id, status="failed", error=result.get("error"))


    # -------------------------------------------------------------------------
    # Saga compensation (rollback)
    # -------------------------------------------------------------------------

    async def _compensate_design(self, message_bus: MessageBus, results: Dict[str, Any]):
        """Rollback design step"""
        logger.info("Compensating design step...")
        try:
            design_data = results.get("design_post", {})
            design_file = design_data.get("file_path")
            if design_file:
                path = Path(design_file)
                if path.exists():
                    path.unlink()
                    logger.info(f"Deleted design file: {design_file}")
            logger.info("Design compensation completed")
        except Exception as e:
            logger.error(f"Design compensation failed: {e}", exc_info=True)


    async def _compensate_image(self, message_bus: MessageBus, results: Dict[str, Any]):
        """Rollback image generation step"""
        logger.info("Compensating image step...")
        try:
            image_data = results.get("generate_image", {})
            image_result = image_data.get("image_result", {})
            image_path = image_result.get("image_path")
            if image_path:
                path = Path(image_path)
                if path.exists():
                    path.unlink()
                    logger.info(f"Deleted image file: {image_path}")
                else:
                    logger.warning(f"Image file not found: {image_path}")
            logger.info("Image compensation completed")
        except Exception as e:
            logger.error(f"Image compensation failed: {e}", exc_info=True)


    # -------------------------------------------------------------------------
    # Response handlers
    # -------------------------------------------------------------------------

    async def handle_design_response(self, message: Message):
        request_id = message.correlation_id
        logger.info(f"Received design response for request {request_id}")

        if request_id not in self.active_requests:
            logger.warning(f"Received response for unknown request: {request_id}")
            return

        request = self.active_requests[request_id]

        # Saga responses are handled by the SagaCoordinator, not here
        if request.get("workflow_type") == "design_with_image":
            logger.debug("Design response for saga workflow — handled by SagaCoordinator")
            return

        await self.send_response(
            original_message=request["original_message"],
            payload={
                "response_type": "design",
                "result": message.payload.get("design_result", ""),
                "message": "Your design is ready",
                "metadata": message.payload.get("metadata", {})
            },
            topic="user_response"
        )
        self._cleanup_request(request_id, status="completed")


    async def handle_image_response(self, message: Message):
        request_id = message.correlation_id
        logger.info(f"Received image response for request {request_id}")

        if request_id not in self.active_requests:
            logger.warning(f"Received response for unknown request: {request_id}")
            return

        request = self.active_requests[request_id]

        if request.get("workflow_type") == "design_with_image":
            logger.debug("Image response for saga workflow — handled by SagaCoordinator")
            return

        await self.send_response(
            original_message=request["original_message"],
            payload={
                "response_type": "image",
                "result": message.payload.get("image_result", {}),
                "message": "Your image is ready",
            },
            topic="user_response"
        )
        self._cleanup_request(request_id, status="completed")


    async def handle_error(self, message: Message):
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
            self._cleanup_request(correlation_id, status="failed", error=error)


    # -------------------------------------------------------------------------
    # Request lifecycle management
    # -------------------------------------------------------------------------

    def _cleanup_request(self, request_id: str, status: str = "completed", error: str = None):
        """
        Mark request as done and schedule deferred removal from active_requests.
        Replaces the old pattern of calling asyncio.sleep(60) inside response
        handlers, which blocked the handler coroutine unnecessarily.
        """
        if request_id not in self.active_requests:
            return

        self.active_requests[request_id]["status"] = status
        self.active_requests[request_id]["completed_at"] = datetime.utcnow().isoformat()
        if error:
            self.active_requests[request_id]["error"] = error

        # Schedule removal without blocking the current coroutine
        asyncio.create_task(self._deferred_remove(request_id, delay=60))


    async def _deferred_remove(self, request_id: str, delay: int = 60):
        """Remove a completed request from active_requests after a delay."""
        await asyncio.sleep(delay)
        if request_id in self.active_requests:
            del self.active_requests[request_id]
            logger.debug(f"Cleaned up request {request_id}")


    # -------------------------------------------------------------------------
    # Status / introspection
    # -------------------------------------------------------------------------

    async def _handle_status_request(self, message: Message):
        await self.send_response(
            original_message=message,
            payload={
                "response_type": "status",
                "status": {
                    "host_agent": self.get_status(),
                    "available_agents": self.available_agents,
                    "active_requests": len(self.active_requests),
                    "system_status": "operational"
                },
                "message": "System is operational"
            },
            topic="user_response"
        )


    def get_active_requests_summary(self) -> Dict[str, Any]:
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
        if not hasattr(self, 'saga_coordinator'):
            return {"active_sagas": 0, "sagas": []}
        active_sagas = self.saga_coordinator.get_all_active_sagas()
        return {"active_sagas": len(active_sagas), "sagas": active_sagas}