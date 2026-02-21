"""
api/main.py — FastAPI application

Exposes the agentic system as a REST API so the React frontend
and MCP server can talk to it without going through the CLI.

Endpoints:
  POST /api/generate/post          → PostDesignAgent
  POST /api/generate/image         → ImageGenerationAgent
  POST /api/generate/post-with-image → Full Saga workflow
  GET  /api/status                 → System status
  GET  /api/health                 → Health check
  WS   /api/ws/{client_id}         → WebSocket for streaming responses
"""

import sys
import os
import asyncio
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

# Windows event loop fix — must be before any asyncio usage
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from core.logging_config import setup_logging
setup_logging(debug=os.getenv("DEBUG_MODE", "false").lower() == "true")

from core.message_bus import MessageBus, Message
from core.llm_factory import create_llm_manager
from agents.host_agent import HostAgent
from agents.post_design_agent import PostDesignAgent
from agents.image_generation_agent import ImageGenerationAgent

logger = logging.getLogger(__name__)

# ── Global system state ────────────────────────────────────────────────────
message_bus: Optional[MessageBus] = None
host_agent: Optional[HostAgent] = None
post_design_agent: Optional[PostDesignAgent] = None
image_generation_agent: Optional[ImageGenerationAgent] = None
llm_manager = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, client_id: str, ws: WebSocket):
        await ws.accept()
        self.active[client_id] = ws
        logger.info(f"WebSocket connected: {client_id}")

    def disconnect(self, client_id: str):
        self.active.pop(client_id, None)
        logger.info(f"WebSocket disconnected: {client_id}")

    async def send(self, client_id: str, data: dict):
        ws = self.active.get(client_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.warning(f"WebSocket send failed for {client_id}: {e}")
                self.disconnect(client_id)

ws_manager = ConnectionManager()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents on startup, shut them down cleanly on exit"""
    global message_bus, host_agent, post_design_agent, image_generation_agent, llm_manager

    logger.info("API starting up...")

    message_bus = MessageBus()
    llm_manager = create_llm_manager()

    image_config = {
        "backend": os.getenv("IMAGE_BACKEND", "comfyui"),
        "model": os.getenv("IMAGE_MODEL", "sdxl"),
        "api_key": os.getenv("IMAGE_API_KEY"),
        "output_dir": os.getenv("IMAGE_OUTPUT_DIR", "generated_images"),
        "comfyui_url": os.getenv("COMFYUI_URL", "http://127.0.0.1:8188"),
    }

    host_agent = HostAgent(message_bus, llm_manager=llm_manager)
    post_design_agent = PostDesignAgent(message_bus, llm_manager)
    image_generation_agent = ImageGenerationAgent(message_bus, image_config=image_config)

    # Subscribe API response handler to forward agent responses to callers
    message_bus.subscribe(
        topic="user_response",
        agent_id="api_server",
        callback=_handle_agent_response
    )

    await host_agent.start()
    await post_design_agent.start()
    await image_generation_agent.start()

    logger.info("API ready")
    yield

    # Shutdown
    logger.info("API shutting down...")
    agents = [host_agent, post_design_agent, image_generation_agent]
    await asyncio.gather(*[a.stop() for a in agents if a], return_exceptions=True)
    if message_bus:
        await message_bus.shutdown()


# ── Pending request registry ───────────────────────────────────────────────
# Maps correlation_id → asyncio.Future so HTTP endpoints can await agent responses
_pending: dict[str, asyncio.Future] = {}


async def _handle_agent_response(message: Message):
    """Bridge agent responses back to waiting HTTP requests or WebSocket clients"""
    correlation_id = message.correlation_id
    payload = message.payload

    # If there's a waiting HTTP future, resolve it
    if correlation_id and correlation_id in _pending:
        future = _pending[correlation_id]
        if not future.done():
            future.set_result(payload)
        return

    # Otherwise try to push via WebSocket (for streaming use cases)
    user_id = payload.get("user_id", "")
    if user_id:
        await ws_manager.send(user_id, {
            "type": "response",
            "data": payload
        })


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic AI Marketing System",
    description="Multi-agent AI system for marketing content and image generation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────
class GeneratePostRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="What to generate")
    user_id: str = Field(default="api_user")

class GenerateImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    size: str = Field(default="1024x1024", pattern=r"^\d+x\d+$")
    style: str = Field(default="vivid")
    content_type: str = Field(default="social_media")
    user_id: str = Field(default="api_user")

class GeneratePostWithImageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(default="api_user")

class APIResponse(BaseModel):
    success: bool
    request_id: str
    data: dict


# ── Helpers ────────────────────────────────────────────────────────────────
async def _send_and_wait(topic: str, payload: dict, timeout: int = 120) -> dict:
    """
    Publish a message to the agent system and wait for the response.
    Used by all HTTP endpoints to bridge sync request/response with async agents.
    """
    request_id = str(uuid.uuid4())
    payload["request_id"] = request_id

    future: asyncio.Future = asyncio.get_event_loop().create_future()
    _pending[request_id] = future

    try:
        message = Message(
            type="request",
            sender="api_server",
            topic=topic,
            payload=payload,
            correlation_id=request_id
        )
        await message_bus.publish(message)

        result = await asyncio.wait_for(future, timeout=timeout)
        return result

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Agent did not respond within {timeout}s"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _pending.pop(request_id, None)


def _check_ready():
    """Raise 503 if agents aren't initialized yet"""
    if not host_agent or not host_agent.is_running:
        raise HTTPException(status_code=503, detail="System not ready")


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/status")
async def status():
    _check_ready()
    stats = llm_manager.get_stats() if llm_manager else {}
    return {
        "agents": {
            "host": host_agent.get_status() if host_agent else None,
            "post_design": post_design_agent.get_status() if post_design_agent else None,
            "image": image_generation_agent.get_status() if image_generation_agent else None,
        },
        "llm": stats,
        "active_requests": host_agent.get_active_requests_summary() if host_agent else {},
        "sagas": host_agent.get_saga_status_summary() if host_agent else {},
    }


@app.post("/api/generate/post", response_model=APIResponse)
async def generate_post(req: GeneratePostRequest):
    """Generate marketing post / copy via PostDesignAgent"""
    _check_ready()
    result = await _send_and_wait(
        topic="user_request",
        payload={"message": req.message, "user_id": req.user_id},
        timeout=120
    )
    return APIResponse(
        success=True,
        request_id=result.get("request_id", ""),
        data=result
    )


@app.post("/api/generate/image", response_model=APIResponse)
async def generate_image(req: GenerateImageRequest):
    """Generate image via ImageGenerationAgent"""
    _check_ready()
    result = await _send_and_wait(
        topic="image_request",
        payload={
            "prompt": req.prompt,
            "size": req.size,
            "style": req.style,
            "content_type": req.content_type,
            "user_id": req.user_id,
        },
        timeout=600   # image gen can be slow
    )
    return APIResponse(
        success=True,
        request_id=result.get("request_id", ""),
        data=result
    )


@app.post("/api/generate/post-with-image", response_model=APIResponse)
async def generate_post_with_image(req: GeneratePostWithImageRequest):
    """Generate post + image via full Saga workflow"""
    _check_ready()
    result = await _send_and_wait(
        topic="user_request",
        payload={
            # Force design_with_image routing by including image keyword
            "message": req.message + " with image",
            "user_id": req.user_id,
        },
        timeout=720   # saga = post + image
    )
    return APIResponse(
        success=True,
        request_id=result.get("request_id", ""),
        data=result
    )


# ── WebSocket endpoint ─────────────────────────────────────────────────────
@app.websocket("/api/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket for real-time streaming responses.
    Client sends: { "type": "request", "message": "...", "mode": "post|image|both" }
    Server sends: { "type": "response", "data": {...} } or { "type": "error", "detail": "..." }
    """
    await ws_manager.connect(client_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type != "request":
                await ws_manager.send(client_id, {"type": "error", "detail": "Unknown message type"})
                continue

            user_message = data.get("message", "").strip()
            if not user_message:
                await ws_manager.send(client_id, {"type": "error", "detail": "Message cannot be empty"})
                continue

            # Publish to agent system — response comes back via _handle_agent_response -> ws_manager
            request_id = str(uuid.uuid4())
            await message_bus.publish(Message(
                type="request",
                sender="api_server",
                topic="user_request",
                payload={
                    "message": user_message,
                    "user_id": client_id,
                    "request_id": request_id
                },
                correlation_id=request_id
            ))

            await ws_manager.send(client_id, {
                "type": "ack",
                "request_id": request_id,
                "message": "Processing..."
            })

    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        ws_manager.disconnect(client_id)