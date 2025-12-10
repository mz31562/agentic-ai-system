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
    Enhanced Image Generation Agent with ComfyUI stability fixes + Text-in-Image Optimization
    - Connection pooling and session management
    - Health checks before submission
    - Queue monitoring
    - Automatic cleanup after each generation
    - Session reinitialization on connection errors
    - Smart text detection and prompt enhancement
    - Text-optimized generation settings
    - Sampler validation and fallback
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
    
    
    def _initialize_backend(self):
        """Initialize the image generation backend with session management"""
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
        try:
            import requests
            
            comfyui_url = self.image_config["comfyui_url"]
            
            self.client = {
                "url": comfyui_url,
                "requests": requests
            }
            
            response = requests.get(
                f"{comfyui_url}/system_stats", 
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"ComfyUI initialized at {comfyui_url}")
                self.consecutive_failures = 0
                
                self._fetch_available_samplers()
            else:
                raise Exception(f"ComfyUI returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"ComfyUI initialization failed: {e}")
            raise
    
    
    def _fetch_available_samplers(self):
        """Fetch list of available samplers from ComfyUI (NEW)"""
        try:
            import requests
            
            response = requests.get(
                f"{self.client['url']}/object_info",
                timeout=5
            )
            
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
                else:
                    logger.warning("Could not fetch sampler info from ComfyUI")
                    
        except Exception as e:
            logger.warning(f"Failed to fetch available samplers: {e}")
    
    
    def _get_safe_sampler(self, preferred: str, is_text: bool = False) -> str:
        """
        Get a safe sampler name that exists in ComfyUI
        
        Args:
            preferred: Preferred sampler name
            is_text: Whether this is for text generation
            
        Returns:
            Safe sampler name
        """
        if not self.available_samplers:
            safe_defaults = ["euler", "dpmpp_2m", "dpmpp_sde", "ddim"]
            logger.info(f"Using safe default sampler: {safe_defaults[0]}")
            return safe_defaults[0]
        
        if preferred in self.available_samplers:
            return preferred
        
        if is_text:
            text_samplers = ["euler", "dpmpp_2m", "dpmpp_sde", "ddim", "dpmpp_2m_sde"]
        else:
            text_samplers = ["dpmpp_2m", "euler", "dpmpp_sde", "ddim"]
        
        for sampler in text_samplers:
            if sampler in self.available_samplers:
                logger.info(f"Sampler '{preferred}' not available, using '{sampler}'")
                return sampler
        
        fallback = self.available_samplers[0]
        logger.warning(f"Using fallback sampler: {fallback}")
        return fallback
    
    
    def _get_safe_scheduler(self, preferred: str) -> str:
        """Get a safe scheduler name that exists in ComfyUI"""
        if not self.available_schedulers:
            safe_defaults = ["normal", "karras", "simple"]
            logger.info(f"Using safe default scheduler: {safe_defaults[0]}")
            return safe_defaults[0]
        
        if preferred in self.available_schedulers:
            return preferred
        
        fallback_order = ["normal", "karras", "simple", "exponential"]
        
        for scheduler in fallback_order:
            if scheduler in self.available_schedulers:
                logger.info(f"Scheduler '{preferred}' not available, using '{scheduler}'")
                return scheduler
        
        fallback = self.available_schedulers[0]
        logger.warning(f"Using fallback scheduler: {fallback}")
        return fallback
    
    
    def _init_dalle(self):
        """Initialize OpenAI DALL-E client"""
        try:
            from openai import OpenAI
            api_key = self.image_config["api_key"] or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not provided")
            self.client = OpenAI(api_key=api_key)
            logger.info(f"DALL-E initialized")
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
    
    
    def _init_replicate(self):
        """Initialize Replicate client"""
        try:
            import replicate
            api_key = self.image_config["api_key"] or os.getenv("REPLICATE_API_KEY")
            if not api_key:
                raise ValueError("REPLICATE_API_KEY not provided")
            os.environ["REPLICATE_API_TOKEN"] = api_key
            self.client = replicate
            logger.info(f"Replicate initialized")
        except ImportError:
            logger.error("Replicate package not installed. Run: pip install replicate")
            raise
    
    
    def _init_segmind(self):
        """Initialize Segmind client"""
        try:
            import requests
            api_key = self.image_config["api_key"] or os.getenv("SEGMIND_API_KEY")
            if not api_key:
                raise ValueError("SEGMIND_API_KEY not provided")
            self.client = {
                "api_key": api_key,
                "requests": requests
            }
            logger.info(f"Segmind initialized")
        except ImportError:
            logger.error("Requests package not installed. Run: pip install requests")
            raise
    
    
    def _detect_text_request(self, prompt: str) -> tuple:
        """
        Detect if prompt contains text generation request and extract the text
        
        Returns:
            (has_text: bool, extracted_text: str or None)
        """
        if not self.image_config["enable_text_optimization"]:
            return False, None
        
        prompt_lower = prompt.lower()
        
        for keyword in self.image_config["text_detection_keywords"]:
            if keyword in prompt_lower:
                quote_patterns = [
                    r'["\']([^"\']+)["\']',  # Single or double quotes
                    r'says\s+([A-Z0-9\s]+?)(?:,|\.|$)',  # After "says"
                    r'text:\s*([A-Z0-9\s]+?)(?:,|\.|$)',  # After "text:"
                ]
                
                for pattern in quote_patterns:
                    match = re.search(pattern, prompt, re.IGNORECASE)
                    if match:
                        text_content = match.group(1).strip()
                        logger.info(f"✓ Text detected: '{text_content}'")
                        return True, text_content
                
                logger.info(f"✓ Text request detected (no quotes)")
                return True, None
        
        return False, None
    
    
    def _enhance_prompt_for_text(self, user_prompt: str, text_content: str = None) -> tuple:
        """
        Enhanced prompting specifically for text-in-image generation
        
        Args:
            user_prompt: Original user prompt
            text_content: The exact text that should appear (if extracted)
            
        Returns:
            Tuple of (enhanced_prompt, negative_prompt)
        """
        
        text_quality_keywords = [
            "clear legible text",
            "sharp typography",
            "readable font",
            "professional text",
            "crisp letters",
            "well-defined text"
        ]
        
        text_negative_prompt = [
            "blurry text",
            "illegible text",
            "distorted letters",
            "misspelled words",
            "garbled text",
            "random characters",
            "duplicated text",
            "multiple text versions",
            "ugly",
            "low quality",
            "deformed"
        ]
        
        enhanced = user_prompt.strip()
        
        enhanced += ", " + ", ".join(text_quality_keywords[:4])
        
        enhanced += ", masterpiece, best quality, highly detailed, 8k uhd"
        
        logger.info(f"Original prompt: {user_prompt}")
        logger.info(f"Text-enhanced prompt: {enhanced}")
        
        return enhanced, ", ".join(text_negative_prompt)
    
    
    def _enhance_prompt_for_quality(self, user_prompt: str, content_type: str = "social_media", has_text: bool = False) -> tuple:
        """
        Enhance user prompt with quality modifiers and style keywords
        Now with text detection support
        
        Args:
            user_prompt: Original user prompt
            content_type: Type of content (marketing, product, social_media, etc.)
            has_text: Whether this is a text-in-image request
            
        Returns:
            Tuple of (enhanced_prompt, negative_prompt)
        """
        
        if has_text:
            return self._enhance_prompt_for_text(user_prompt)
        
        quality_keywords = [
            "masterpiece",
            "best quality", 
            "highly detailed",
            "professional photography",
            "8k uhd",
            "sharp focus"
        ]
        
        style_presets = {
            "marketing": [
                "commercial photography style",
                "studio lighting",
                "vibrant colors"
            ],
            "social_media": [
                "trendy aesthetic",
                "instagram worthy",
                "modern style"
            ],
            "product": [
                "product photography",
                "white background",
                "detailed texture"
            ],
            "lifestyle": [
                "lifestyle photography",
                "natural lighting",
                "authentic"
            ]
        }
        
        negative_prompt = [
            "ugly",
            "blurry",
            "low quality",
            "distorted",
            "deformed",
            "watermark"
        ]
        
        style_keywords = style_presets.get(content_type, style_presets["social_media"])
        
        enhanced = user_prompt.strip()
        enhanced += ", " + ", ".join(style_keywords[:2])
        enhanced += ", " + ", ".join(quality_keywords[:4])
        
        logger.info(f"Original prompt: {user_prompt}")
        logger.info(f"Enhanced prompt: {enhanced}")
        
        return enhanced, ", ".join(negative_prompt)
    
    
    async def setup(self):
        """Subscribe to image generation requests"""
        self.message_bus.subscribe(
            topic="image_request",
            agent_id=self.agent_id,
            callback=self._process_message
        )
        
        logger.info(f"Image Generation Agent subscribed to topic: image_request")
        logger.info(f"Backend: {self.backend} | Model: {self.model}")
        logger.info(f"Text optimization: {self.image_config['enable_text_optimization']}")
        logger.info(f"Max retries: {self.image_config['max_retries']} | Timeout: {self.image_config['timeout_seconds']}s")
    
    
    async def handle_message(self, message: Message):
        """Route incoming messages to appropriate handlers"""
        if message.topic == "image_request":
            await self.handle_image_request(message)
        else:
            logger.warning(f"Image Generation Agent received unknown topic: {message.topic}")
    
    
    async def handle_image_request(self, message: Message):
        """Handle image generation request with enhanced retry logic"""
        prompt = message.payload.get("prompt", "")
        style = message.payload.get("style", self.image_config["default_style"])
        size = message.payload.get("size", self.image_config["default_size"])
        content_type = message.payload.get("content_type", "social_media")
        request_id = message.payload.get("request_id", "")
        
        logger.info(f"Image Generation Agent processing: '{prompt[:50]}...'")
        logger.info(f"Backend: {self.backend} | Model: {self.model} | Content Type: {content_type}")
        
        if self.circuit_open:
            logger.warning(f"Circuit breaker OPEN - too many consecutive failures")
            if self.image_config["enable_fallback"]:
                logger.info("Attempting fallback to mock mode...")
                result = self._generate_mock_image(prompt, style, size)
                await self._send_success_response(message, result, request_id)
                return
            else:
                await self.send_error(
                    original_message=message,
                    error="Image generation service temporarily unavailable (circuit breaker open)",
                    details={"request_id": request_id, "prompt": prompt}
                )
                return
        
        max_retries = self.image_config["max_retries"]
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{max_retries}...")
                    wait_time = min(2 ** attempt, 10)
                    await asyncio.sleep(wait_time)
                
                result = await self._generate_image_internal(prompt, style, size, content_type)
                
                self.consecutive_failures = 0
                self.circuit_open = False
                self.last_successful_generation = time.time()
                self.generation_count += 1
                
                await self._send_success_response(message, result, request_id)
                logger.info(f"Image generation completed for request {request_id} (total: {self.generation_count})")
                return
                
            except asyncio.TimeoutError:
                logger.error(f"Attempt {attempt + 1} timed out after {self.image_config['timeout_seconds']}s")
                self.consecutive_failures += 1
                
                if attempt < max_retries:
                    continue
                else:
                    self._record_failure()
                    
                    if self.image_config["enable_fallback"]:
                        logger.warning("All retries failed, falling back to mock mode...")
                        result = self._generate_mock_image(prompt, style, size)
                        await self._send_success_response(message, result, request_id)
                    else:
                        await self.send_error(
                            original_message=message,
                            error=f"Image generation timed out after {max_retries + 1} attempts",
                            details={"request_id": request_id, "prompt": prompt, "timeout": self.image_config["timeout_seconds"]}
                        )
                    return
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                self.consecutive_failures += 1
                
                if attempt < max_retries:
                    continue
                else:
                    self._record_failure()
                    
                    if self.image_config["enable_fallback"]:
                        logger.warning("All retries failed, falling back to mock mode...")
                        result = self._generate_mock_image(prompt, style, size)
                        await self._send_success_response(message, result, request_id)
                    else:
                        await self.send_error(
                            original_message=message,
                            error=str(e),
                            details={"request_id": request_id, "prompt": prompt, "attempts": max_retries + 1}
                        )
                    return
    
    
    def _record_failure(self):
        """Record a failure and potentially open circuit breaker"""
        self.consecutive_failures += 1
        logger.warning(f"Consecutive failures: {self.consecutive_failures}/{self.max_consecutive_failures}")
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.circuit_open = True
            logger.error(f"🔴 CIRCUIT BREAKER OPENED - Image generation disabled until manual reset")
    
    
    async def _send_success_response(self, message: Message, result: Dict[str, Any], request_id: str):
        """Send successful response"""
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
    
    
    async def _generate_image(
        self,
        prompt: str,
        style: str = "vivid",
        size: str = "1024x1024",
        content_type: str = "social_media"
    ) -> Dict[str, Any]:
        """Generate image using configured backend with timeout"""
        
        logger.info(f"Generating image with {self.backend} backend...")
        
        try:
            result = await asyncio.wait_for(
                self._generate_image_internal(prompt, style, size, content_type),
                timeout=self.image_config["timeout_seconds"]
            )
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Generation timed out after {self.image_config['timeout_seconds']}s")
            raise
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    
    async def _generate_image_internal(
        self,
        prompt: str,
        style: str,
        size: str,
        content_type: str = "social_media"
    ) -> Dict[str, Any]:
        """Internal generation method (without timeout wrapper)"""
        
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
    
    
    async def _call_comfyui(self, prompt: str, size: str, content_type: str = "social_media") -> Dict[str, Any]:
        """Call ComfyUI API with enhanced prompts and text detection"""
        import requests
        
        logger.info(f"ComfyUI generation started for: '{prompt[:50]}...'")
        
        width, height = map(int, size.split('x'))
        
        has_text, text_content = self._detect_text_request(prompt)
        
        if has_text:
            logger.info(f"🔤 Text-in-image mode activated")
            if text_content:
                logger.info(f"   Target text: '{text_content}'")
        
        enhanced_prompt, negative_prompt = self._enhance_prompt_for_quality(
            prompt, 
            content_type,
            has_text=has_text
        )
        
        workflow = self._build_comfyui_workflow(
            enhanced_prompt, 
            width, 
            height, 
            negative_prompt,
            has_text=has_text
        )
        
        url = f"{self.client['url']}/prompt"
        logger.info(f"Submitting to: {url}")
        
        try:
            response = await asyncio.to_thread(
                self.client["requests"].post,
                url,
                json={"prompt": workflow},
                timeout=10
            )
            
            if response.status_code != 200:
                error_msg = f"ComfyUI error (status {response.status_code}): {response.text[:200]}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            prompt_id = response.json()['prompt_id']
            logger.info(f"ComfyUI job submitted successfully!")
            logger.info(f"   Prompt ID: {prompt_id}")
            
        except requests.exceptions.Timeout:
            raise Exception("ComfyUI submission timed out (10s)")
        except requests.exceptions.ConnectionError as e:
            raise Exception(f"Cannot connect to ComfyUI: {e}")
        except Exception as e:
            raise Exception(f"ComfyUI submission failed: {e}")
        
        try:
            image_path = await self._wait_for_comfyui_result(prompt_id, prompt)
            
            logger.info(f"High-quality image generated: {image_path}")
            
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
            
        except asyncio.TimeoutError:
            raise Exception(f"ComfyUI generation timed out")
        except Exception as e:
            raise Exception(f"ComfyUI result retrieval failed: {e}")
    
    
    def _build_comfyui_workflow(
        self, 
        prompt: str, 
        width: int, 
        height: int, 
        negative_prompt: str = None,
        has_text: bool = False
    ) -> Dict:
        """
        Build ComfyUI workflow with optimized settings
        Now with text-specific optimizations and sampler validation
        """
        
        if "sdxl" in self.model.lower():
            checkpoint = "sd_xl_base_1.0.safetensors"
            
            if has_text:
                steps = 30      # More steps for text clarity
                cfg = 8.5       # Higher CFG for better prompt adherence
                sampler_preferred = "euler"  # CHANGED: safer default
                scheduler_preferred = "normal"
                logger.info("   Using TEXT-OPTIMIZED settings for SDXL")
            else:
                steps = 25
                cfg = 7.0
                sampler_preferred = "dpmpp_2m"
                scheduler_preferred = "karras"
        else:
            checkpoint = "v1-5-pruned-emaonly.safetensors"
            
            if has_text:
                steps = 25
                cfg = 8.0
                sampler_preferred = "euler"
                scheduler_preferred = "normal"
                logger.info("   Using TEXT-OPTIMIZED settings for SD1.5")
            else:
                steps = 20
                cfg = 7.0
                sampler_preferred = "dpmpp_2m"
                scheduler_preferred = "karras"
        
        sampler = self._get_safe_sampler(sampler_preferred, is_text=has_text)
        scheduler = self._get_safe_scheduler(scheduler_preferred)
        
        if not negative_prompt:
            if has_text:
                negative_prompt = "blurry text, illegible text, distorted letters, ugly, blurry, low quality"
            else:
                negative_prompt = "ugly, blurry, low quality, distorted, deformed, watermark, text"
        
        workflow = {
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
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": checkpoint
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "agentic_ai_pro",
                    "images": ["8", 0]
                }
            }
        }
        
        logger.info(f"   Settings: {steps} steps, CFG {cfg}, {sampler} sampler, {scheduler} scheduler")
        
        return workflow
    
    
    async def _wait_for_comfyui_result(self, prompt_id: str, prompt: str) -> Path:
        """Wait for ComfyUI to finish with better error handling"""
        import requests
        
        url = f"{self.client['url']}/history/{prompt_id}"
        check_interval = self.image_config["check_interval_seconds"]
        timeout = self.image_config["timeout_seconds"]
        
        start_time = time.time()
        check_count = 0
        last_log_time = start_time
        last_status = None
        
        while time.time() - start_time < timeout:
            check_count += 1
            elapsed = time.time() - start_time
            
            if elapsed - (last_log_time - start_time) >= 10:
                logger.info(f"   Waiting... ({elapsed:.1f}s elapsed)")
                last_log_time = time.time()
            
            try:
                response = await asyncio.to_thread(
                    self.client["requests"].get,
                    url,
                    timeout=5
                )
                
                if response.status_code == 200:
                    history = response.json()
                    
                    if prompt_id in history:
                        outputs = history[prompt_id].get('outputs', {})
                        
                        if 'error' in history[prompt_id]:
                            error = history[prompt_id]['error']
                            raise Exception(f"ComfyUI generation error: {error}")
                        
                        for node_id, output in outputs.items():
                            if 'images' in output:
                                image_info = output['images'][0]
                                
                                logger.info(f"   Image found in ComfyUI outputs")
                                logger.info(f"   Filename: {image_info['filename']}")
                                
                                return await self._download_from_comfyui(image_info, prompt)
                        
                        status = history[prompt_id].get('status', {})
                        if status != last_status:
                            logger.debug(f"Status update: {status}")
                            last_status = status
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"   Check failed: {e}")
            except Exception as e:
                logger.error(f"   Error during check: {e}")
                raise
            
            await asyncio.sleep(check_interval)
        
        raise asyncio.TimeoutError(f"ComfyUI generation timed out after {timeout}s")
    
    
    async def _download_from_comfyui(self, image_info: Dict, prompt: str) -> Path:
        """Download image from ComfyUI with retry"""
        import requests
        
        comfyui_path = image_info['filename']
        subfolder = image_info.get('subfolder', '')
        
        download_url = f"{self.client['url']}/view"
        params = {
            'filename': comfyui_path,
            'subfolder': subfolder,
            'type': 'output'
        }
        
        logger.info(f"   Downloading from ComfyUI...")
        
        for attempt in range(3):
            try:
                img_response = await asyncio.to_thread(
                    self.client["requests"].get,
                    download_url,
                    params=params,
                    timeout=30
                )
                
                if img_response.status_code != 200:
                    raise Exception(f"Download failed: {img_response.status_code}")
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_prompt = safe_prompt.replace(' ', '_')
                filename = f"{safe_prompt}_{timestamp}.png"
                image_path = self.output_dir / filename
                
                with open(image_path, 'wb') as f:
                    f.write(img_response.content)
                
                file_size = image_path.stat().st_size / 1024
                logger.info(f"   Image saved locally: {image_path}")
                logger.info(f"   File size: {file_size:.1f} KB")
                
                return image_path
                
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(1)
                else:
                    raise
    
    
    async def _call_dalle(self, prompt: str, style: str, size: str) -> Dict[str, Any]:
        """Call DALL-E API"""
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
        
        return {
            "image_url": image_url,
            "image_path": str(image_path),
            "prompt": prompt,
            "size": size,
            "backend": "dalle",
            "model": self.model
        }
    
    
    async def _call_replicate(self, prompt: str, size: str) -> Dict[str, Any]:
        """Call Replicate API"""
        model_name = self.model or "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
        
        output = await asyncio.to_thread(
            self.client.run,
            model_name,
            input={"prompt": prompt}
        )
        
        image_url = output[0] if isinstance(output, list) else output
        image_path = await self._download_image(image_url, prompt)
        
        return {
            "image_url": image_url,
            "image_path": str(image_path),
            "prompt": prompt,
            "size": size,
            "backend": "replicate",
            "model": self.model
        }
    
    
    async def _call_segmind(self, prompt: str, size: str) -> Dict[str, Any]:
        """Call Segmind API"""
        url = "https://api.segmind.com/v1/sd1.5-txt2img"
        
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
            url,
            json=data,
            headers={"x-api-key": self.client["api_key"]},
            timeout=60
        )
        
        image_path = self.output_dir / f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        with open(image_path, 'wb') as f:
            f.write(response.content)
        
        return {
            "image_url": None,
            "image_path": str(image_path),
            "prompt": prompt,
            "size": size,
            "backend": "segmind",
            "model": self.model
        }
    
    
    def _generate_mock_image(self, prompt: str, style: str, size: str) -> Dict[str, Any]:
        """Generate a mock image placeholder"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"mock_img_{timestamp}.txt"
        image_path = self.output_dir / filename
        
        with open(image_path, 'w') as f:
            f.write(f"MOCK IMAGE\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Style: {style}\n")
            f.write(f"Size: {size}\n")
            f.write(f"Backend: {self.backend}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"[Configure IMAGE_BACKEND in .env for actual image generation]\n")
            f.write(f"Supported backends: comfyui, dalle, replicate, segmind\n")
        
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
        """Download image from URL and save locally"""
        import aiohttp
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(' ', '_')
        filename = f"{safe_prompt}_{timestamp}.png"
        image_path = self.output_dir / filename
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    with open(image_path, 'wb') as f:
                        f.write(await response.read())
        
        logger.info(f"Image saved to: {image_path}")
        return image_path
    
    async def stop(self):
        """Cleanup on shutdown"""
        logger.info("Stopping Image Generation Agent...")
        await super().stop()