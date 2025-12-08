# C:\Users\MohammedZaid\Desktop\agentic-ai-system\agents\post_design_agent.py
import asyncio
import re
from typing import Dict, Any, Optional, List
import logging
import os

from core.agent_base import BaseAgent
from core.message_bus import MessageBus, Message
from core.llm_manager import LLMManager, TaskComplexity

logger = logging.getLogger(__name__)


class PostDesignAgent(BaseAgent):
    """
    Enhanced PostDesign Agent with multi-LLM support.
    Intelligently routes requests to the best available LLM backend.
    """
    
    def __init__(self, message_bus: MessageBus, llm_manager: LLMManager):
        super().__init__(
            agent_id="post_design_agent",
            name="PostDesign Agent",
            message_bus=message_bus,
            capabilities=[
                "content_generation",
                "design_creation",
                "math_integration",
                "creative_writing",
                "multi_llm_support"
            ]
        )
        
        self.llm_manager = llm_manager
        
        # Math-related keywords for detection
        self.math_keywords = [
            "fibonacci", "sequence", "calculate", "equation", "solve",
            "formula", "series", "pattern", "number", "math"
        ]
    
    
    async def setup(self):
        """Subscribe to design requests"""
        self.message_bus.subscribe(
            topic="design_request",
            agent_id=self.agent_id,
            callback=self._process_message
        )
        
        logger.info(f"PostDesign Agent subscribed to topic: design_request")
        logger.info(f"Using Multi-LLM Manager with {len(self.llm_manager.backends)} backends")
    
    
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
            
            # Generate design using intelligent LLM selection
            design_result = await self._generate_design(user_message, math_data)
            
            # Send response back to Host Agent
            await self.send_response(
                original_message=message,
                payload={
                    "design_result": design_result["content"],
                    "metadata": design_result["metadata"],
                    "math_data": math_data,
                    "request_id": request_id,
                    "status": "completed"
                },
                topic="design_response"
            )
            
            backend_used = design_result["metadata"].get("backend_used", "unknown")
            logger.info(f"Design completed for request {request_id} using {backend_used}")
            
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
        """Generate design using intelligent LLM selection"""
        
        # Build prompt
        prompt = self._build_design_prompt(user_message, math_data)
        
        # Determine task characteristics
        task_type = self._classify_task_type(user_message)
        complexity = self._assess_complexity(user_message)
        
        logger.info(f"Task classification: type={task_type}, complexity={complexity.value}")
        
        try:
            # Let the LLM manager select the best backend
            result = await self.llm_manager.complete(
                prompt=prompt,
                task_type=task_type,
                complexity=complexity,
                max_tokens=1024,
                temperature=0.8 if task_type == "creative" else 0.7,
                prefer_local=True  # Prefer free/local when possible
            )
            
            content = result["content"]
            
            # Build comprehensive metadata
            metadata = {
                "backend_used": result["backend"],
                "provider": result["provider"],
                "model": result["model"],
                "tokens_used": result["tokens_used"],
                "latency_ms": result["latency_ms"],
                "cost": result["cost"],
                "task_type": task_type,
                "complexity": complexity.value,
                "style": "modern",
                "tone": "educational" if math_data else "engaging",
                "contains_math": math_data is not None,
                "colors": ["#3498db", "#2ecc71", "#f39c12"],
                "emojis_used": True,
                "word_count": len(user_message.split())
            }
            
            logger.info(
                f"Generation complete: {result['tokens_used']} tokens, "
                f"{result['latency_ms']}ms, ${result['cost']:.6f}"
            )
            
            return {
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            
            # Fallback to mock
            content = self._generate_mock_design(user_message, math_data)
            metadata = {
                "backend_used": "mock",
                "provider": "fallback",
                "model": "none",
                "tokens_used": 0,
                "latency_ms": 0,
                "cost": 0.0,
                "error": str(e),
                "style": "modern",
                "tone": "educational" if math_data else "engaging",
                "contains_math": math_data is not None
            }
            
            return {
                "content": content,
                "metadata": metadata
            }
    
    
    def _classify_task_type(self, user_message: str) -> str:
        """
        Classify the type of task based on user message
        
        Returns:
            Task type: creative, code, reasoning, analytical, or general
        """
        message_lower = user_message.lower()
        
        # Creative writing indicators
        creative_keywords = ["story", "poem", "creative", "imagine", "describe", "write"]
        if any(word in message_lower for word in creative_keywords):
            return "creative"
        
        # Code-related indicators
        code_keywords = ["code", "program", "function", "script", "debug", "algorithm"]
        if any(word in message_lower for word in code_keywords):
            return "code"
        
        # Reasoning/logic indicators
        reasoning_keywords = ["analyze", "explain", "why", "how", "compare", "evaluate"]
        if any(word in message_lower for word in reasoning_keywords):
            return "reasoning"
        
        # Analytical indicators
        analytical_keywords = ["data", "statistics", "trends", "patterns", "insights"]
        if any(word in message_lower for word in analytical_keywords):
            return "analytical"
        
        return "general"
    
    
    def _assess_complexity(self, user_message: str) -> TaskComplexity:
        """
        Assess the complexity of the task
        
        Returns:
            TaskComplexity enum value
        """
        word_count = len(user_message.split())
        
        # Simple: Short, straightforward requests
        if word_count < 10:
            return TaskComplexity.SIMPLE
        
        # Expert: Very long or contains complexity indicators
        if word_count > 50 or any(word in user_message.lower() 
                                   for word in ["complex", "detailed", "comprehensive", "in-depth"]):
            return TaskComplexity.EXPERT
        
        # Complex: Moderate length with reasoning indicators
        if word_count > 30 or any(word in user_message.lower() 
                                   for word in ["analyze", "compare", "evaluate", "explain"]):
            return TaskComplexity.COMPLEX
        
        # Medium: Default for moderate requests
        return TaskComplexity.MEDIUM
    
    
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
    
    
    def _generate_mock_design(
        self,
        user_message: str,
        math_data: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a mock design (fallback when no LLM available)"""
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

[Generated in MOCK mode - Enable LLM backends in .env for AI-generated content]

🎨 Engaging • 📱 Shareable • 💡 Informative

#Creative #Design #Content"""