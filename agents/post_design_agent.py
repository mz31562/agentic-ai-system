import asyncio
import re
from typing import Dict, Any, Optional, List
import logging
import os

from core.agent_base import BaseAgent
from core.message_bus import MessageBus, Message

logger = logging.getLogger(__name__)


class PostDesignAgent(BaseAgent):
    """
    PostDesign Agent - Specialist in creating post designs and content.
    Supports multiple LLM backends: Claude, Ollama, Groq, HuggingFace
    """
    
    def __init__(self, message_bus: MessageBus, llm_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="post_design_agent",
            name="PostDesign Agent",
            message_bus=message_bus,
            capabilities=[
                "content_generation",
                "design_creation",
                "math_integration",
                "creative_writing"
            ]
        )
        
        # Default config
        self.llm_config = llm_config or {
            "backend": os.getenv("LLM_BACKEND", "ollama"),  # ollama, claude, groq, huggingface, mock
            "model": os.getenv("LLM_MODEL", "llama3"),
            "api_key": os.getenv("LLM_API_KEY"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        }
        
        self.backend = self.llm_config["backend"]
        self.model = self.llm_config["model"]
        
        # Initialize the appropriate client
        self.client = None
        self._initialize_backend()
        
        # Math-related keywords for detection
        self.math_keywords = [
            "fibonacci", "sequence", "calculate", "equation", "solve",
            "formula", "series", "pattern", "number", "math"
        ]
    
    
    def _initialize_backend(self):
        """Initialize the LLM backend based on configuration"""
        try:
            if self.backend == "ollama":
                self._init_ollama()
            elif self.backend == "claude":
                self._init_claude()
            elif self.backend == "groq":
                self._init_groq()
            elif self.backend == "huggingface":
                self._init_huggingface()
            elif self.backend == "mock":
                logger.info("Using mock mode (no LLM)")
                self.client = None
            else:
                logger.warning(f"Unknown backend '{self.backend}', falling back to mock mode")
                self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize {self.backend} backend: {e}")
            logger.info("Falling back to mock mode")
            self.backend = "mock"
            self.client = None
    
    
    def _init_ollama(self):
        """Initialize Ollama client"""
        try:
            import ollama
            self.client = ollama
            logger.info(f"✅ Ollama initialized with model: {self.model}")
            logger.info(f"   Base URL: {self.llm_config['base_url']}")
        except ImportError:
            logger.error("Ollama package not installed. Run: pip install ollama")
            raise
    
    
    def _init_claude(self):
        """Initialize Anthropic Claude client"""
        try:
            from anthropic import Anthropic
            api_key = self.llm_config["api_key"] or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not provided")
            self.client = Anthropic(api_key=api_key)
            logger.info(f"✅ Claude initialized with model: {self.model}")
        except ImportError:
            logger.error("Anthropic package not installed. Run: pip install anthropic")
            raise
    
    
    def _init_groq(self):
        """Initialize Groq client"""
        try:
            from groq import Groq
            api_key = self.llm_config["api_key"] or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not provided")
            self.client = Groq(api_key=api_key)
            logger.info(f"✅ Groq initialized with model: {self.model}")
        except ImportError:
            logger.error("Groq package not installed. Run: pip install groq")
            raise
    
    
    def _init_huggingface(self):
        """Initialize HuggingFace client"""
        try:
            from huggingface_hub import InferenceClient
            api_key = self.llm_config["api_key"] or os.getenv("HUGGINGFACE_API_KEY")
            self.client = InferenceClient(token=api_key)
            logger.info(f"✅ HuggingFace initialized with model: {self.model}")
        except ImportError:
            logger.error("HuggingFace Hub package not installed. Run: pip install huggingface-hub")
            raise
    
    
    async def setup(self):
        """Subscribe to design requests"""
        self.message_bus.subscribe(
            topic="design_request",
            agent_id=self.agent_id,
            callback=self._process_message
        )
        
        logger.info(f"PostDesign Agent subscribed to topic: design_request")
        logger.info(f"Using backend: {self.backend} with model: {self.model}")
    
    
    async def handle_message(self, message: Message):
        """Route incoming messages to appropriate handlers"""
        
        if message.topic == "design_request":
            await self.handle_design_request(message)
        else:
            logger.warning(f"PostDesign Agent received unknown topic: {message.topic}")
    
    
    async def handle_design_request(self, message: Message):
        """Handle design request from Host Agent"""
        user_message = message.payload.get("user_message", "")
        request_id = message.payload.get("request_id", "")
        
        logger.info(f"PostDesign Agent processing: '{user_message}'")
        
        try:
            # Check if math is needed
            needs_math = self._detect_math_requirement(user_message)
            
            math_data = None
            if needs_math:
                logger.info("Math requirement detected, querying Math MCP Server...")
                math_data = await self._query_math_server(user_message, message.correlation_id)
            
            # Generate design using configured LLM backend
            design_result = await self._generate_design(user_message, math_data)
            
            # Send response back to Host Agent
            await self.send_response(
                original_message=message,
                payload={
                    "design_result": design_result["content"],
                    "metadata": design_result["metadata"],
                    "math_data": math_data,
                    "request_id": request_id,
                    "status": "completed",
                    "backend_used": self.backend
                },
                topic="design_response"
            )
            
            logger.info(f"Design completed for request {request_id} using {self.backend}")
            
        except Exception as e:
            logger.error(f"Error processing design request: {e}", exc_info=True)
            await self.send_error(
                original_message=message,
                error=str(e),
                details={"request_id": request_id}
            )
    
    
    def _detect_math_requirement(self, user_message: str) -> bool:
        """Detect if the request requires mathematical calculations"""
        message_lower = user_message.lower()
        
        for keyword in self.math_keywords:
            if keyword in message_lower:
                logger.info(f"Math keyword detected: '{keyword}'")
                return True
        
        return False
    
    
    async def _query_math_server(
        self,
        user_message: str,
        correlation_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Query Math MCP Server for calculations"""
        try:
            # Send request to Math MCP Server
            await self.send_request(
                topic="math_request",
                recipient="math_mcp_server",
                payload={
                    "query": user_message,
                    "operation": "auto_detect"
                },
                correlation_id=correlation_id
            )
            
            # Wait for response (with timeout)
            await asyncio.sleep(0.5)
            
            # Mock math response for now (we'll replace when we build proper response handling)
            if "fibonacci" in user_message.lower():
                return {
                    "sequence": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
                    "formula": "F(n) = F(n-1) + F(n-2)",
                    "explanation": "Each number is the sum of the two preceding ones"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error querying Math MCP Server: {e}")
            return None
    
    
    async def _generate_design(
        self,
        user_message: str,
        math_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate design using configured LLM backend"""
        
        # Build prompt
        prompt = self._build_design_prompt(user_message, math_data)
        
        try:
            if self.backend == "ollama":
                content = await self._call_ollama(prompt)
            elif self.backend == "claude":
                content = await self._call_claude(prompt)
            elif self.backend == "groq":
                content = await self._call_groq(prompt)
            elif self.backend == "huggingface":
                content = await self._call_huggingface(prompt)
            else:
                # Mock mode
                content = self._generate_mock_design(user_message, math_data)
        except Exception as e:
            logger.error(f"LLM call failed: {e}, falling back to mock")
            content = self._generate_mock_design(user_message, math_data)
        
        # Extract metadata
        metadata = self._extract_metadata(user_message, math_data)
        metadata["backend"] = self.backend
        metadata["model"] = self.model
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    
    def _build_design_prompt(
        self,
        user_message: str,
        math_data: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for LLM"""
        
        base_prompt = f"""You are a creative social media post designer. 
Create an engaging, visually-descriptive post based on this request: "{user_message}"

Requirements:
- Make it engaging and shareable
- Use emojis where appropriate
- Keep it concise but informative (2-4 paragraphs max)
- Include a call-to-action if relevant
- Add relevant hashtags at the end
"""
        
        if math_data:
            base_prompt += f"""

Mathematical context to incorporate:
{math_data}

Make sure to explain the math in an accessible, interesting way that fits naturally into the post.
"""
        
        return base_prompt
    
    
    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        response = await asyncio.to_thread(
            self.client.chat,
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    
    
    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API"""
        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model or "claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    
    async def _call_groq(self, prompt: str) -> str:
        """Call Groq API"""
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model or "llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )
        return response.choices[0].message.content
    
    
    async def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API"""
        response = await asyncio.to_thread(
            self.client.text_generation,
            prompt,
            model=self.model or "meta-llama/Llama-2-7b-chat-hf",
            max_new_tokens=512
        )
        return response
    
    
    def _generate_mock_design(
        self,
        user_message: str,
        math_data: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a mock design (fallback)"""
        if math_data and "fibonacci" in user_message.lower():
            return f"""🌟 The Magic of Fibonacci Sequence! 🌟

Did you know? The Fibonacci sequence appears everywhere in nature! 🍃

The sequence: {', '.join(map(str, math_data.get('sequence', [])))}

Each number is the sum of the two before it:
{math_data.get('formula', 'F(n) = F(n-1) + F(n-2)')}

From spirals in seashells 🐚 to the arrangement of sunflower seeds 🌻, 
this mathematical pattern is nature's secret code!

✨ Want to see more math in nature? Follow for daily insights!

#Fibonacci #Mathematics #NatureIsAmazing #MathInNature #STEM"""
        
        else:
            return f"""✨ Here's a creative post based on your request! ✨

{user_message}

[Generated in {self.backend} mode - Configure LLM backend for AI-generated content]

🎨 Engaging • 📱 Shareable • 💡 Informative

#Creative #Design #Content"""
    
    
    def _extract_metadata(
        self,
        user_message: str,
        math_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract metadata about the design"""
        return {
            "style": "modern",
            "tone": "educational" if math_data else "engaging",
            "contains_math": math_data is not None,
            "colors": ["#3498db", "#2ecc71", "#f39c12"],
            "emojis_used": True,
            "word_count": len(user_message.split())
        }