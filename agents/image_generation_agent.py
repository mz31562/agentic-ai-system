# C:\Users\MohammedZaid\Desktop\agentic-ai-system\agents\image_generation_agent.py
import asyncio
import os
import json
import uuid
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from pathlib import Path

from core.agent_base import BaseAgent
from core.message_bus import MessageBus, Message

logger = logging.getLogger(__name__)


class ImageGenerationAgent(BaseAgent):
    """
    Enhanced Image Generation Agent - Creates images with robust error handling.
    Supports: ComfyUI (local), DALL-E, Replicate, Segmind, Mock
    Features: Automatic retry, fallback, circuit breaker, timeout handling
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
                "fallback_support"
            ]
        )
        
        # Default config - merge with provided config
        default_config = {
            "backend": os.getenv("IMAGE_BACKEND", "comfyui"),
            "model": os.getenv("IMAGE_MODEL", "sdxl"),
            "api_key": os.getenv("IMAGE_API_KEY"),
            "output_dir": os.getenv("IMAGE_OUTPUT_DIR", "generated_images"),
            "comfyui_url": os.getenv("COMFYUI_URL", "http://127.0.0.1:8188"),
            "default_size": os.getenv("DEFAULT_IMAGE_SIZE", "1024x1024"),
            "default_style": os.getenv("DEFAULT_IMAGE_STYLE", "vivid"),
            
            # Robustness settings
            "max_retries": int(os.getenv("IMAGE_MAX_RETRIES", "2")),
            "timeout_seconds": int(os.getenv("IMAGE_TIMEOUT_SECONDS", "180")),  # 3 minutes
            "enable_fallback": os.getenv("IMAGE_ENABLE_FALLBACK", "true").lower() == "true",
            "check_interval_seconds": 2  # How often to check ComfyUI status
        }
        
        if image_config:
            default_config.update(image_config)
        
        self.image_config = default_config
        self.backend = self.image_config["backend"]
        self.model = self.image_config["model"]
        self.output_dir = Path(self.image_config["output_dir"])
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track failures for circuit breaker
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.circuit_open = False
        
        # Initialize the appropriate client
        self.client = None
        self._initialize_backend()
    
    
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
        """Initialize ComfyUI client with health check"""
        try:
            import requests
            
            comfyui_url = self.image_config["comfyui_url"]
            
            self.client = {
                "url": comfyui_url,
                "requests": requests
            }
            
            # Test connection with timeout
            logger.info(f"Testing ComfyUI connection at {comfyui_url}...")
            try:
                response = requests.get(
                    f"{comfyui_url}/system_stats", 
                    timeout=5
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ ComfyUI initialized at {comfyui_url}")
                    self.consecutive_failures = 0  # Reset on success
                else:
                    raise Exception(f"ComfyUI returned status {response.status_code}")
                    
            except requests.exceptions.ConnectionError as e:
                logger.error(f"❌ Cannot connect to ComfyUI at {comfyui_url}")
                raise Exception(f"ComfyUI connection failed: {e}")
                
        except Exception as e:
            logger.error(f"ComfyUI initialization failed: {e}")
            raise
    
    
    def _init_dalle(self):
        """Initialize OpenAI DALL-E client"""
        try:
            from openai import OpenAI
            api_key = self.image_config["api_key"] or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not provided")
            self.client = OpenAI(api_key=api_key)
            logger.info(f"✅ DALL-E initialized")
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
            logger.info(f"✅ Replicate initialized")
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
            logger.info(f"✅ Segmind initialized")
        except ImportError:
            logger.error("Requests package not installed. Run: pip install requests")
            raise
    
    
    async def setup(self):
        """Subscribe to image generation requests"""
        self.message_bus.subscribe(
            topic="image_request",
            agent_id=self.agent_id,
            callback=self._process_message
        )
        
        logger.info(f"Image Generation Agent subscribed to topic: image_request")
        logger.info(f"Backend: {self.backend} | Model: {self.model}")
        logger.info(f"Max retries: {self.image_config['max_retries']} | Timeout: {self.image_config['timeout_seconds']}s")
    
    
    async def handle_message(self, message: Message):
        """Route incoming messages to appropriate handlers"""
        if message.topic == "image_request":
            await self.handle_image_request(message)
        else:
            logger.warning(f"Image Generation Agent received unknown topic: {message.topic}")
    
    
    async def handle_image_request(self, message: Message):
        """Handle image generation request with retry logic"""
        prompt = message.payload.get("prompt", "")
        style = message.payload.get("style", self.image_config["default_style"])
        size = message.payload.get("size", self.image_config["default_size"])
        request_id = message.payload.get("request_id", "")
        
        logger.info(f"Image Generation Agent processing: '{prompt[:50]}...'")
        logger.info(f"Backend: {self.backend} | Model: {self.model}")
        
        # Check circuit breaker
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
        
        # Try generation with retries
        max_retries = self.image_config["max_retries"]
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{max_retries}...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                
                # Generate image
                result = await self._generate_image(prompt, style, size)
                
                # Success! Reset failure counter
                self.consecutive_failures = 0
                self.circuit_open = False
                
                await self._send_success_response(message, result, request_id)
                logger.info(f"✅ Image generation completed for request {request_id}")
                return
                
            except asyncio.TimeoutError:
                logger.error(f"❌ Attempt {attempt + 1} timed out after {self.image_config['timeout_seconds']}s")
                
                if attempt < max_retries:
                    continue
                else:
                    # All retries exhausted
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
                logger.error(f"❌ Attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries:
                    continue
                else:
                    # All retries exhausted
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
        size: str = "1024x1024"
    ) -> Dict[str, Any]:
        """Generate image using configured backend with timeout"""
        
        logger.info(f"Generating image with {self.backend} backend...")
        
        try:
            # Wrap generation in timeout
            result = await asyncio.wait_for(
                self._generate_image_internal(prompt, style, size),
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
        size: str
    ) -> Dict[str, Any]:
        """Internal generation method (without timeout wrapper)"""
        
        if self.backend == "comfyui":
            return await self._call_comfyui(prompt, size)
            
        elif self.backend == "dalle":
            return await self._call_dalle(prompt, style, size)
            
        elif self.backend == "replicate":
            return await self._call_replicate(prompt, size)
            
        elif self.backend == "segmind":
            return await self._call_segmind(prompt, size)
            
        else:
            return self._generate_mock_image(prompt, style, size)
    
    
    async def _call_comfyui(self, prompt: str, size: str) -> Dict[str, Any]:
        """Call ComfyUI API with improved error handling"""
        import requests
        
        logger.info(f"ComfyUI generation started for prompt: '{prompt[:50]}...'")
        
        # Parse size
        width, height = map(int, size.split('x'))
        logger.info(f"Image size: {width}x{height}")
        
        # Build workflow
        workflow = self._build_comfyui_workflow(prompt, width, height)
        
        # Submit prompt
        url = f"{self.client['url']}/prompt"
        logger.info(f"Submitting to: {url}")
        
        try:
            response = await asyncio.to_thread(
                requests.post,
                url,
                json={"prompt": workflow},
                timeout=10  # 10s for submission
            )
            
            if response.status_code != 200:
                error_msg = f"ComfyUI error (status {response.status_code}): {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            prompt_id = response.json()['prompt_id']
            logger.info(f"✅ ComfyUI job submitted successfully!")
            logger.info(f"   Prompt ID: {prompt_id}")
            
        except requests.exceptions.Timeout:
            raise Exception("ComfyUI submission timed out (10s)")
        except requests.exceptions.ConnectionError as e:
            raise Exception(f"Cannot connect to ComfyUI: {e}")
        except Exception as e:
            raise Exception(f"ComfyUI submission failed: {e}")
        
        # Wait for completion with timeout
        try:
            image_path = await self._wait_for_comfyui_result(prompt_id, prompt)
            
            logger.info(f"✅ Image generated successfully: {image_path}")
            
            return {
                "image_url": None,
                "image_path": str(image_path),
                "prompt": prompt,
                "size": size,
                "backend": "comfyui_local",
                "model": self.model
            }
            
        except asyncio.TimeoutError:
            raise Exception(f"ComfyUI generation timed out")
        except Exception as e:
            raise Exception(f"ComfyUI result retrieval failed: {e}")
    
    
    def _build_comfyui_workflow(self, prompt: str, width: int, height: int) -> Dict:
        """Build a simple ComfyUI workflow"""
        
        # Determine checkpoint
        if "sdxl" in self.model.lower():
            checkpoint = "sd_xl_base_1.0.safetensors"
        else:
            checkpoint = "v1-5-pruned-emaonly.safetensors"
        
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": int(datetime.now().timestamp()),
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
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
                    "text": "ugly, blurry, low quality, distorted",
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
                    "filename_prefix": "agentic_ai",
                    "images": ["8", 0]
                }
            }
        }
        
        return workflow
    
    
    async def _wait_for_comfyui_result(self, prompt_id: str, prompt: str) -> Path:
        """Wait for ComfyUI to finish with better progress tracking"""
        import requests
        import time
        
        url = f"{self.client['url']}/history/{prompt_id}"
        check_interval = self.image_config["check_interval_seconds"]
        timeout = self.image_config["timeout_seconds"]
        
        start_time = time.time()
        check_count = 0
        last_log_time = start_time
        
        while time.time() - start_time < timeout:
            check_count += 1
            elapsed = time.time() - start_time
            
            # Log progress every 10 seconds
            if elapsed - (last_log_time - start_time) >= 10:
                logger.info(f"   Waiting... ({elapsed:.1f}s elapsed)")
                last_log_time = time.time()
            
            try:
                response = await asyncio.to_thread(
                    requests.get,
                    url,
                    timeout=5
                )
                
                if response.status_code == 200:
                    history = response.json()
                    
                    if prompt_id in history:
                        outputs = history[prompt_id].get('outputs', {})
                        
                        # Find the SaveImage node output
                        for node_id, output in outputs.items():
                            if 'images' in output:
                                image_info = output['images'][0]
                                
                                logger.info(f"   ✓ Image found in ComfyUI outputs")
                                logger.info(f"   Filename: {image_info['filename']}")
                                
                                # Download from ComfyUI
                                return await self._download_from_comfyui(image_info, prompt)
                
            except Exception as e:
                logger.warning(f"   Check failed: {e}")
            
            await asyncio.sleep(check_interval)
        
        raise asyncio.TimeoutError(f"ComfyUI generation timed out after {timeout}s")
    
    
    async def _download_from_comfyui(self, image_info: Dict, prompt: str) -> Path:
        """Download image from ComfyUI"""
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
        
        img_response = await asyncio.to_thread(
            requests.get,
            download_url,
            params=params,
            timeout=30
        )
        
        # Save locally
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(' ', '_')
        filename = f"{safe_prompt}_{timestamp}.png"
        image_path = self.output_dir / filename
        
        with open(image_path, 'wb') as f:
            f.write(img_response.content)
        
        file_size = image_path.stat().st_size / 1024  # KB
        logger.info(f"   ✓ Image saved locally: {image_path}")
        logger.info(f"   File size: {file_size:.1f} KB")
        
        return image_path
    
    
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