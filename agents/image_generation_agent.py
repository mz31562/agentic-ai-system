import asyncio
import os
import json
import uuid
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from pathlib import Path
import time
import re

from core.agent_base import BaseAgent
from core.message_bus import MessageBus, Message

logger = logging.getLogger(__name__)


class ImageGenerationAgent(BaseAgent):
    """
    Image Generation Agent with multi-backend support.
    - ComfyUI (local SDXL), DALL-E, Replicate, Segmind
    - Timeout enforced on every attempt via asyncio.wait_for
    - Circuit breaker + exponential backoff retry
    - Text-in-image detection and prompt optimization
    - Sampler/scheduler validation against ComfyUI's actual available list
    """

    def __init__(self, message_bus: MessageBus, image_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="image_generation_agent",
            name="Image Generation Agent",
            message_bus=message_bus,
            capabilities=[
                "text_to_image",
                "style_transfer",
                "image_editing",
                "social_media_graphics",
                "robust_generation",
                "auto_retry",
                "fallback_support",
                "prompt_enhancement",
                "comfyui_stability",
                "text_in_image_optimization"
            ]
        )

        default_config = {
            "backend": os.getenv("IMAGE_BACKEND", "comfyui"),
            "model": os.getenv("IMAGE_MODEL", "sdxl"),
            "api_key": os.getenv("IMAGE_API_KEY"),
            "output_dir": os.getenv("IMAGE_OUTPUT_DIR", "generated_images"),
            "comfyui_url": os.getenv("COMFYUI_URL", "http://127.0.0.1:8188"),
            "default_size": os.getenv("DEFAULT_IMAGE_SIZE", "1024x1024"),
            "default_style": os.getenv("DEFAULT_IMAGE_STYLE", "vivid"),
            "max_retries": int(os.getenv("IMAGE_MAX_RETRIES", "3")),
            "timeout_seconds": int(os.getenv("IMAGE_TIMEOUT_SECONDS", "600")),
            "enable_fallback": os.getenv("IMAGE_ENABLE_FALLBACK", "true").lower() == "true",
            "check_interval_seconds": 2,
            "enable_text_optimization": os.getenv("ENABLE_TEXT_OPTIMIZATION", "true").lower() == "true",
            "text_detection_keywords": ["text that says", "text:", "words:", "letters:", "sign that says"]
        }

        if image_config:
            default_config.update(image_config)

        self.image_config = default_config
        self.backend = self.image_config["backend"]
        self.model = self.image_config["model"]
        self.output_dir = Path(self.image_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.circuit_open = False
        self.generation_count = 0
        self.last_successful_generation = None

        self.available_samplers = None
        self.available_schedulers = None

        self.client = None
        self._initialize_backend()


    # -------------------------------------------------------------------------
    # Backend initialization
    # -------------------------------------------------------------------------

    def _initialize_backend(self):
        """Initialize the image generation backend"""
        logger.info(f"Initializing image backend: {self.backend}")
        try:
            if self.backend == "comfyui":
                self._init_comfyui()
            elif self.backend == "dalle":
                self._init_dalle()
            elif self.backend == "replicate":
                self._init_replicate()
            elif self.backend == "segmind":
                self._init_segmind()
            elif self.backend == "mock":
                logger.info("Using mock mode (no actual image generation)")
                self.client = None
            else:
                logger.warning(f"Unknown backend '{self.backend}', falling back to mock mode")
                self.backend = "mock"
                self.client = None
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            logger.info("Falling back to mock mode")
            self.backend = "mock"
            self.client = None


    def _init_comfyui(self):
        """Initialize ComfyUI client and fetch available samplers"""
        import requests
        comfyui_url = self.image_config["comfyui_url"]
        self.client = {"url": comfyui_url, "requests": requests}
        response = requests.get(f"{comfyui_url}/system_stats", timeout=5)
        if response.status_code == 200:
            logger.info(f"ComfyUI initialized at {comfyui_url}")
            self.consecutive_failures = 0
            self._fetch_available_samplers()
        else:
            raise Exception(f"ComfyUI returned status {response.status_code}")


    def _fetch_available_samplers(self):
        """Fetch list of available samplers/schedulers from ComfyUI"""
        try:
            import requests
            response = requests.get(f"{self.client['url']}/object_info", timeout=5)
            if response.status_code == 200:
                object_info = response.json()
                if "KSampler" in object_info:
                    sampler_info = object_info["KSampler"]["input"]["required"]
                    if "sampler_name" in sampler_info:
                        self.available_samplers = sampler_info["sampler_name"][0]
                        logger.info(f"Available samplers: {', '.join(self.available_samplers[:5])}...")
                    if "scheduler" in sampler_info:
                        self.available_schedulers = sampler_info["scheduler"][0]
                        logger.info(f"Available schedulers: {', '.join(self.available_schedulers[:5])}...")
        except Exception as e:
            logger.warning(f"Failed to fetch available samplers: {e}")


    def _init_dalle(self):
        """Initialize OpenAI DALL-E client"""
        from openai import OpenAI
        api_key = self.image_config["api_key"] or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        self.client = OpenAI(api_key=api_key)
        logger.info("DALL-E initialized")


    def _init_replicate(self):
        """Initialize Replicate client"""
        import replicate
        api_key = self.image_config["api_key"] or os.getenv("REPLICATE_API_KEY")
        if not api_key:
            raise ValueError("REPLICATE_API_KEY not provided")
        os.environ["REPLICATE_API_TOKEN"] = api_key
        self.client = replicate
        logger.info("Replicate initialized")


    def _init_segmind(self):
        """Initialize Segmind client"""
        import requests
        api_key = self.image_config["api_key"] or os.getenv("SEGMIND_API_KEY")
        if not api_key:
            raise ValueError("SEGMIND_API_KEY not provided")
        self.client = {"api_key": api_key, "requests": requests}
        logger.info("Segmind initialized")


    # -------------------------------------------------------------------------
    # Sampler / scheduler validation
    # -------------------------------------------------------------------------

    def _get_safe_sampler(self, preferred: str, is_text: bool = False) -> str:
        if not self.available_samplers:
            return "euler"
        if preferred in self.available_samplers:
            return preferred
        fallbacks = ["euler", "dpmpp_2m", "dpmpp_sde", "ddim", "dpmpp_2m_sde"]
        for s in fallbacks:
            if s in self.available_samplers:
                logger.info(f"Sampler '{preferred}' not available, using '{s}'")
                return s
        return self.available_samplers[0]


    def _get_safe_scheduler(self, preferred: str) -> str:
        if not self.available_schedulers:
            return "normal"
        if preferred in self.available_schedulers:
            return preferred
        for s in ["normal", "karras", "simple", "exponential"]:
            if s in self.available_schedulers:
                logger.info(f"Scheduler '{preferred}' not available, using '{s}'")
                return s
        return self.available_schedulers[0]


    # -------------------------------------------------------------------------
    # Text-in-image detection & prompt enhancement
    # -------------------------------------------------------------------------

    def _detect_text_request(self, prompt: str):
        """Detect if prompt contains a text-in-image request. Returns (has_text, extracted_text)."""
        if not self.image_config["enable_text_optimization"]:
            return False, None
        prompt_lower = prompt.lower()
        for keyword in self.image_config["text_detection_keywords"]:
            if keyword in prompt_lower:
                for pattern in [
                    r'["\']([^"\']+)["\']',
                    r'says\s+([A-Z0-9\s]+?)(?:,|\.|$)',
                    r'text:\s*([A-Z0-9\s]+?)(?:,|\.|$)',
                ]:
                    match = re.search(pattern, prompt, re.IGNORECASE)
                    if match:
                        text_content = match.group(1).strip()
                        logger.info(f"Text detected: '{text_content}'")
                        return True, text_content
                return True, None
        return False, None


    def _enhance_prompt_for_text(self, user_prompt: str, text_content: str = None):
        """Enhance prompt specifically for text-in-image generation."""
        text_quality_keywords = [
            "clear legible text", "sharp typography",
            "readable font", "professional text",
            "crisp letters", "well-defined text"
        ]
        text_negative = [
            "blurry text", "illegible text", "distorted letters",
            "misspelled words", "garbled text", "random characters",
            "duplicated text", "multiple text versions",
            "ugly", "low quality", "deformed"
        ]
        enhanced = user_prompt.strip()
        enhanced += ", " + ", ".join(text_quality_keywords[:4])
        enhanced += ", masterpiece, best quality, highly detailed, 8k uhd"
        return enhanced, ", ".join(text_negative)


    def _enhance_prompt_for_quality(self, user_prompt: str, content_type: str = "social_media", has_text: bool = False):
        """Enhance prompt with quality and style modifiers."""
        if has_text:
            return self._enhance_prompt_for_text(user_prompt)

        style_presets = {
            "marketing": ["commercial photography style", "studio lighting", "vibrant colors"],
            "social_media": ["trendy aesthetic", "instagram worthy", "modern style"],
            "product": ["product photography", "white background", "detailed texture"],
            "lifestyle": ["lifestyle photography", "natural lighting", "authentic"],
        }
        quality_keywords = ["masterpiece", "best quality", "highly detailed", "professional photography"]
        negative_prompt = "ugly, blurry, low quality, distorted, deformed, watermark"

        style_keywords = style_presets.get(content_type, style_presets["social_media"])
        enhanced = user_prompt.strip()
        enhanced += ", " + ", ".join(style_keywords[:2])
        enhanced += ", " + ", ".join(quality_keywords[:4])
        return enhanced, negative_prompt


    # -------------------------------------------------------------------------
    # Agent lifecycle
    # -------------------------------------------------------------------------

    async def setup(self):
        """Subscribe to image generation requests"""
        self.message_bus.subscribe(
            topic="image_request",
            agent_id=self.agent_id,
            callback=self._process_message
        )
        logger.info(f"Image Generation Agent ready | backend={self.backend} | model={self.model}")


    async def handle_message(self, message: Message):
        if message.topic == "image_request":
            await self.handle_image_request(message)
        else:
            logger.warning(f"Unknown topic: {message.topic}")


    # -------------------------------------------------------------------------
    # Core request handler — timeout bug fixed here
    # -------------------------------------------------------------------------

    async def handle_image_request(self, message: Message):
        """
        Handle image generation with retry logic.
        FIX: asyncio.wait_for now wraps _generate_image_internal on every attempt,
        so IMAGE_TIMEOUT_SECONDS is actually enforced (previously it was bypassed).
        """
        prompt = message.payload.get("prompt", "")
        style = message.payload.get("style", self.image_config["default_style"])
        size = message.payload.get("size", self.image_config["default_size"])
        content_type = message.payload.get("content_type", "social_media")
        request_id = message.payload.get("request_id", "")

        # Input validation
        if not prompt or not prompt.strip():
            await self.send_error(
                original_message=message,
                error="Prompt cannot be empty",
                details={"request_id": request_id}
            )
            return

        if len(prompt) > 2000:
            prompt = prompt[:2000]
            logger.warning(f"Prompt truncated to 2000 chars for request {request_id}")

        logger.info(f"Processing image request: '{prompt[:50]}...' | backend={self.backend}")

        if self.circuit_open:
            logger.warning("Circuit breaker OPEN")
            if self.image_config["enable_fallback"]:
                result = self._generate_mock_image(prompt, style, size)
                await self._send_success_response(message, result, request_id)
            else:
                await self.send_error(
                    original_message=message,
                    error="Image generation temporarily unavailable (circuit breaker open)",
                    details={"request_id": request_id}
                )
            return

        max_retries = self.image_config["max_retries"]
        timeout = self.image_config["timeout_seconds"]
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    wait_time = min(2 ** attempt, 10)
                    logger.info(f"Retry {attempt}/{max_retries}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)

                # FIX: timeout is enforced here on every attempt
                result = await asyncio.wait_for(
                    self._generate_image_internal(prompt, style, size, content_type),
                    timeout=timeout
                )

                # Success path
                self.consecutive_failures = 0
                self.circuit_open = False
                self.last_successful_generation = time.time()
                self.generation_count += 1
                await self._send_success_response(message, result, request_id)
                logger.info(f"Request {request_id} completed (total: {self.generation_count})")
                return

            except asyncio.TimeoutError:
                last_error = f"Timed out after {timeout}s"
                logger.error(f"Attempt {attempt + 1} timed out")
                self.consecutive_failures += 1

            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                self.consecutive_failures += 1

        # All attempts exhausted
        self._record_failure()
        if self.image_config["enable_fallback"]:
            logger.warning("All retries failed, using mock fallback")
            result = self._generate_mock_image(prompt, style, size)
            await self._send_success_response(message, result, request_id)
        else:
            await self.send_error(
                original_message=message,
                error=f"Image generation failed after {max_retries + 1} attempts: {last_error}",
                details={"request_id": request_id, "attempts": max_retries + 1}
            )


    def _record_failure(self):
        """Record failure and open circuit breaker if threshold reached"""
        self.consecutive_failures += 1
        logger.warning(f"Consecutive failures: {self.consecutive_failures}/{self.max_consecutive_failures}")
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.circuit_open = True
            logger.error("CIRCUIT BREAKER OPENED")


    async def _send_success_response(self, message: Message, result: Dict[str, Any], request_id: str):
        await self.send_response(
            original_message=message,
            payload={
                "image_result": result,
                "request_id": request_id,
                "status": "completed",
                "backend_used": self.backend
            },
            topic="image_response"
        )


    # -------------------------------------------------------------------------
    # Generation dispatch
    # -------------------------------------------------------------------------

    async def _generate_image_internal(self, prompt: str, style: str, size: str, content_type: str = "social_media") -> Dict[str, Any]:
        if self.backend == "comfyui":
            return await self._call_comfyui(prompt, size, content_type)
        elif self.backend == "dalle":
            return await self._call_dalle(prompt, style, size)
        elif self.backend == "replicate":
            return await self._call_replicate(prompt, size)
        elif self.backend == "segmind":
            return await self._call_segmind(prompt, size)
        else:
            return self._generate_mock_image(prompt, style, size)


    # -------------------------------------------------------------------------
    # ComfyUI backend
    # -------------------------------------------------------------------------

    async def _call_comfyui(self, prompt: str, size: str, content_type: str = "social_media") -> Dict[str, Any]:
        import requests

        width, height = map(int, size.split('x'))
        has_text, text_content = self._detect_text_request(prompt)

        if has_text:
            logger.info(f"Text-in-image mode | target text: '{text_content}'")

        enhanced_prompt, negative_prompt = self._enhance_prompt_for_quality(
            prompt, content_type, has_text=has_text
        )

        workflow = self._build_comfyui_workflow(enhanced_prompt, width, height, negative_prompt, has_text=has_text)
        url = f"{self.client['url']}/prompt"

        try:
            response = await asyncio.to_thread(
                self.client["requests"].post,
                url,
                json={"prompt": workflow},
                timeout=10
            )
            if response.status_code != 200:
                raise Exception(f"ComfyUI error (status {response.status_code}): {response.text[:200]}")
            prompt_id = response.json()['prompt_id']
            logger.info(f"ComfyUI job submitted | prompt_id={prompt_id}")
        except requests.exceptions.Timeout:
            raise Exception("ComfyUI submission timed out (10s)")
        except requests.exceptions.ConnectionError as e:
            raise Exception(f"Cannot connect to ComfyUI: {e}")

        image_path = await self._wait_for_comfyui_result(prompt_id, prompt)

        return {
            "image_url": None,
            "image_path": str(image_path),
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "size": size,
            "backend": "comfyui_local",
            "model": self.model,
            "has_text": has_text,
            "text_content": text_content,
            "settings": {
                "steps": workflow["3"]["inputs"]["steps"],
                "cfg": workflow["3"]["inputs"]["cfg"],
                "sampler": workflow["3"]["inputs"]["sampler_name"],
                "scheduler": workflow["3"]["inputs"]["scheduler"]
            }
        }


    def _build_comfyui_workflow(self, prompt: str, width: int, height: int, negative_prompt: str = None, has_text: bool = False) -> Dict:
        if "sdxl" in self.model.lower():
            checkpoint = "sd_xl_base_1.0.safetensors"
            if has_text:
                steps, cfg, sampler_pref, scheduler_pref = 30, 8.5, "euler", "normal"
            else:
                steps, cfg, sampler_pref, scheduler_pref = 25, 7.0, "dpmpp_2m", "karras"
        else:
            checkpoint = "v1-5-pruned-emaonly.safetensors"
            if has_text:
                steps, cfg, sampler_pref, scheduler_pref = 25, 8.0, "euler", "normal"
            else:
                steps, cfg, sampler_pref, scheduler_pref = 20, 7.0, "dpmpp_2m", "karras"

        sampler = self._get_safe_sampler(sampler_pref, is_text=has_text)
        scheduler = self._get_safe_scheduler(scheduler_pref)

        if not negative_prompt:
            if has_text:
                negative_prompt = "blurry text, illegible text, distorted letters, ugly, blurry, low quality"
            else:
                negative_prompt = "ugly, blurry, low quality, distorted, deformed, watermark, text"

        logger.info(f"Workflow: {steps} steps, CFG {cfg}, {sampler} sampler, {scheduler} scheduler")

        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": int(time.time() * 1000) % 2147483647,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "agentic_ai_pro", "images": ["8", 0]}}
        }


    async def _wait_for_comfyui_result(self, prompt_id: str, prompt: str) -> Path:
        import requests
        url = f"{self.client['url']}/history/{prompt_id}"
        check_interval = self.image_config["check_interval_seconds"]
        timeout = self.image_config["timeout_seconds"]
        start_time = time.time()
        last_log_time = start_time

        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            if elapsed - (last_log_time - start_time) >= 10:
                logger.info(f"Waiting for ComfyUI... ({elapsed:.1f}s elapsed)")
                last_log_time = time.time()

            try:
                response = await asyncio.to_thread(
                    self.client["requests"].get, url, timeout=5
                )
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        if 'error' in history[prompt_id]:
                            raise Exception(f"ComfyUI error: {history[prompt_id]['error']}")
                        for node_id, output in history[prompt_id].get('outputs', {}).items():
                            if 'images' in output:
                                return await self._download_from_comfyui(output['images'][0], prompt)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Poll failed: {e}")
            except Exception as e:
                raise

            await asyncio.sleep(check_interval)

        raise asyncio.TimeoutError(f"ComfyUI timed out after {timeout}s")


    async def _download_from_comfyui(self, image_info: Dict, prompt: str) -> Path:
        import requests
        params = {
            'filename': image_info['filename'],
            'subfolder': image_info.get('subfolder', ''),
            'type': 'output'
        }
        for attempt in range(3):
            try:
                img_response = await asyncio.to_thread(
                    self.client["requests"].get,
                    f"{self.client['url']}/view",
                    params=params,
                    timeout=30
                )
                if img_response.status_code != 200:
                    raise Exception(f"Download failed: {img_response.status_code}")

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
                image_path = self.output_dir / f"{safe_prompt}_{timestamp}.png"

                with open(image_path, 'wb') as f:
                    f.write(img_response.content)

                logger.info(f"Image saved: {image_path} ({image_path.stat().st_size / 1024:.1f} KB)")
                return image_path

            except Exception as e:
                if attempt < 2:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(1)
                else:
                    raise


    # -------------------------------------------------------------------------
    # Cloud backends
    # -------------------------------------------------------------------------

    async def _call_dalle(self, prompt: str, style: str, size: str) -> Dict[str, Any]:
        response = await asyncio.to_thread(
            self.client.images.generate,
            model=self.model or "dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            style=style,
            n=1
        )
        image_url = response.data[0].url
        image_path = await self._download_image(image_url, prompt)
        return {"image_url": image_url, "image_path": str(image_path), "prompt": prompt, "size": size, "backend": "dalle", "model": self.model}


    async def _call_replicate(self, prompt: str, size: str) -> Dict[str, Any]:
        model_name = self.model or "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
        output = await asyncio.to_thread(self.client.run, model_name, input={"prompt": prompt})
        image_url = output[0] if isinstance(output, list) else output
        image_path = await self._download_image(image_url, prompt)
        return {"image_url": image_url, "image_path": str(image_path), "prompt": prompt, "size": size, "backend": "replicate", "model": self.model}


    async def _call_segmind(self, prompt: str, size: str) -> Dict[str, Any]:
        import requests
        width, height = map(int, size.split('x'))
        data = {
            "prompt": prompt,
            "negative_prompt": "ugly, blurry, low quality",
            "samples": 1,
            "scheduler": "UniPC",
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "seed": 12345,
            "img_width": width,
            "img_height": height
        }
        response = await asyncio.to_thread(
            self.client["requests"].post,
            "https://api.segmind.com/v1/sd1.5-txt2img",
            json=data,
            headers={"x-api-key": self.client["api_key"]},
            timeout=60
        )
        image_path = self.output_dir / f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with open(image_path, 'wb') as f:
            f.write(response.content)
        return {"image_url": None, "image_path": str(image_path), "prompt": prompt, "size": size, "backend": "segmind", "model": self.model}


    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _generate_mock_image(self, prompt: str, style: str, size: str) -> Dict[str, Any]:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = self.output_dir / f"mock_img_{timestamp}.txt"
        with open(image_path, 'w') as f:
            f.write(f"MOCK IMAGE\n{'=' * 50}\n")
            f.write(f"Prompt: {prompt}\nStyle: {style}\nSize: {size}\n")
            f.write(f"Backend: {self.backend}\nGenerated: {datetime.now().isoformat()}\n")
            f.write(f"{'=' * 50}\n[Configure IMAGE_BACKEND in .env for real generation]\n")
        return {
            "image_url": None,
            "image_path": str(image_path),
            "prompt": prompt,
            "size": size,
            "style": style,
            "mock": True,
            "message": "Mock image placeholder (fallback mode)"
        }


    async def _download_image(self, url: str, prompt: str) -> Path:
        import aiohttp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        image_path = self.output_dir / f"{safe_prompt}_{timestamp}.png"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    with open(image_path, 'wb') as f:
                        f.write(await response.read())
        logger.info(f"Image saved: {image_path}")
        return image_path


    async def stop(self):
        logger.info("Stopping Image Generation Agent...")
        await super().stop()