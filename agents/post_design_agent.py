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
    Enhanced PostDesign Agent with advanced marketing & sales capabilities.
    Supports multiple content types, tones, and platforms.
    """
    
    def __init__(self, message_bus: MessageBus, llm_manager: LLMManager):
        super().__init__(
            agent_id="post_design_agent",
            name="PostDesign Agent",
            message_bus=message_bus,
            capabilities=[
                "content_generation",
                "marketing_copy",
                "sales_content",
                "social_media_posts",
                "email_campaigns",
                "design_creation",
                "math_integration",
                "creative_writing",
                "multi_llm_support"
            ]
        )
        
        self.llm_manager = llm_manager
        
        self.math_keywords = [
            "fibonacci", "sequence", "calculate", "equation", "solve",
            "formula", "series", "pattern", "number", "math"
        ]
        
        self.content_types = {
            "linkedin": ["linkedin", "professional post", "career"],
            "instagram": ["instagram", "insta", "ig post"],
            "twitter": ["twitter", "tweet", "x post"],
            "facebook": ["facebook", "fb post"],
            "email": ["email", "newsletter", "campaign"],
            "blog": ["blog", "article", "long-form"],
            "ad_copy": ["ad", "advertisement", "promotional"],
            "product_description": ["product", "description", "listing"],
            "sales_pitch": ["pitch", "sales", "proposal"],
        }
        
        self.tone_keywords = {
            "professional": ["professional", "formal", "corporate"],
            "casual": ["casual", "friendly", "relaxed"],
            "enthusiastic": ["exciting", "energetic", "enthusiastic"],
            "educational": ["educational", "informative", "teach"],
            "persuasive": ["persuasive", "convince", "compelling"],
            "inspiring": ["inspiring", "motivational", "uplifting"],
            "humorous": ["funny", "humorous", "witty"],
            "urgent": ["urgent", "limited time", "act now"],
        }
    
    
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
            content_type = self._detect_content_type(user_message)
            tone = self._detect_tone(user_message)
            needs_math = self._detect_math_requirement(user_message)
            
            logger.info(f"Content analysis: type={content_type}, tone={tone}, needs_math={needs_math}")
            
            math_data = None
            if needs_math:
                logger.info("Math requirement detected, querying Math MCP Server...")
                math_data = await self._query_math_server(user_message, message.correlation_id)
            
            design_result = await self._generate_design(
                user_message, 
                math_data,
                content_type=content_type,
                tone=tone
            )
            
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
    
    
    def _detect_content_type(self, user_message: str) -> str:
        """Detect the type of content being requested"""
        message_lower = user_message.lower()
        
        for content_type, keywords in self.content_types.items():
            for keyword in keywords:
                if keyword in message_lower:
                    logger.info(f"Content type detected: {content_type}")
                    return content_type
        
        return "social_media"
    
    
    def _detect_tone(self, user_message: str) -> str:
        """Detect the desired tone"""
        message_lower = user_message.lower()
        
        for tone, keywords in self.tone_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    logger.info(f"Tone detected: {tone}")
                    return tone
        
        return "professional"
    
    
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
            await self.send_request(
                topic="math_request",
                recipient="math_mcp_server",
                payload={
                    "query": user_message,
                    "operation": "auto_detect"
                },
                correlation_id=correlation_id
            )
            
            await asyncio.sleep(0.5)
            
            # Mock math response (replace with actual response handling)
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
        math_data: Optional[Dict[str, Any]] = None,
        content_type: str = "social_media",
        tone: str = "professional"
    ) -> Dict[str, Any]:
        """Generate design using intelligent LLM selection"""
        
        prompt = self._build_enhanced_prompt(
            user_message, 
            math_data, 
            content_type, 
            tone
        )
        
        task_type = self._classify_task_type(user_message, content_type)
        complexity = self._assess_complexity(user_message, content_type)
        
        logger.info(f"Task classification: type={task_type}, complexity={complexity.value}")
        
        try:
            result = await self.llm_manager.complete(
                prompt=prompt,
                task_type=task_type,
                complexity=complexity,
                max_tokens=2048,
                temperature=0.85 if task_type == "creative" else 0.7,
                prefer_local=False
            )
            
            content = result["content"]
            
            metadata = {
                "backend_used": result["backend"],
                "provider": result["provider"],
                "model": result["model"],
                "tokens_used": result["tokens_used"],
                "latency_ms": result["latency_ms"],
                "cost": result["cost"],
                "task_type": task_type,
                "complexity": complexity.value,
                "content_type": content_type,
                "tone": tone,
                "style": "modern",
                "contains_math": math_data is not None,
                "word_count": len(content.split())
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
            content = self._generate_mock_design(user_message, math_data, content_type, tone)
            metadata = {
                "backend_used": "mock",
                "provider": "fallback",
                "model": "none",
                "tokens_used": 0,
                "latency_ms": 0,
                "cost": 0.0,
                "error": str(e),
                "content_type": content_type,
                "tone": tone,
                "contains_math": math_data is not None
            }
            
            return {
                "content": content,
                "metadata": metadata
            }
    
    
    def _build_enhanced_prompt(
        self,
        user_message: str,
        math_data: Optional[Dict[str, Any]],
        content_type: str,
        tone: str
    ) -> str:
        """Build comprehensive prompt based on content type and tone"""
        
        # Platform-specific guidelines
        platform_guides = {
            "linkedin": {
                "format": "Professional post with 2-4 paragraphs",
                "style": "Thought leadership, industry insights, professional accomplishments",
                "length": "150-300 words",
                "hashtags": "3-5 relevant professional hashtags",
                "cta": "Encourage professional discussion or connection",
                "emojis": "Use sparingly and professionally"
            },
            "instagram": {
                "format": "Engaging caption with line breaks",
                "style": "Visual storytelling, relatable, authentic",
                "length": "100-200 words (can be longer for carousels)",
                "hashtags": "10-15 relevant hashtags in first comment",
                "cta": "Ask questions, encourage saves/shares",
                "emojis": "Use liberally to break up text"
            },
            "twitter": {
                "format": "Concise thread (1-5 tweets)",
                "style": "Punchy, quotable, conversation-starting",
                "length": "280 characters per tweet",
                "hashtags": "1-2 trending hashtags",
                "cta": "Ask for retweets, replies",
                "emojis": "1-2 per tweet maximum"
            },
            "email": {
                "format": "Subject line + 3-5 paragraph body",
                "style": "Personal, conversational, value-driven",
                "length": "200-400 words",
                "hashtags": "None",
                "cta": "Clear single action (click, reply, purchase)",
                "emojis": "In subject line only if appropriate"
            },
            "ad_copy": {
                "format": "Headline + body + CTA",
                "style": "Benefit-focused, urgency-driven, clear value prop",
                "length": "50-150 words",
                "hashtags": "None typically",
                "cta": "Strong action verb (Get, Start, Join, Claim)",
                "emojis": "Use strategically for attention"
            },
            "sales_pitch": {
                "format": "Problem-Agitate-Solution structure",
                "style": "Consultative, benefit-focused, credibility-building",
                "length": "300-500 words",
                "hashtags": "None",
                "cta": "Schedule call, request demo, start trial",
                "emojis": "Minimal to none"
            },
            "social_media": {
                "format": "2-3 paragraphs with line breaks",
                "style": "Engaging, shareable, on-brand",
                "length": "100-250 words",
                "hashtags": "3-7 relevant hashtags",
                "cta": "Like, comment, share",
                "emojis": "Use naturally throughout"
            }
        }
        
        # Tone-specific instructions
        tone_guides = {
            "professional": "Maintain formal language, industry terminology, authoritative voice",
            "casual": "Conversational, friendly, relatable, like talking to a friend",
            "enthusiastic": "High energy, exclamation points, exciting language, positive vibes",
            "educational": "Clear explanations, step-by-step, teach concepts, provide value",
            "persuasive": "Strong benefits, social proof, overcome objections, create urgency",
            "inspiring": "Aspirational language, emotional connection, storytelling, motivation",
            "humorous": "Witty, clever wordplay, light-hearted, entertaining",
            "urgent": "Time-sensitive language, scarcity, FOMO, immediate action"
        }
        
        guide = platform_guides.get(content_type, platform_guides["social_media"])
        tone_instruction = tone_guides.get(tone, tone_guides["professional"])
        
        # Build the comprehensive prompt
        base_prompt = f"""You are an expert marketing and sales copywriter with 10+ years of experience creating high-converting content across all platforms.

USER REQUEST: "{user_message}"

CONTENT TYPE: {content_type.upper().replace('_', ' ')}
TONE: {tone.upper()}

PLATFORM GUIDELINES:
- Format: {guide['format']}
- Style: {guide['style']}
- Length: {guide['length']}
- Hashtags: {guide['hashtags']}
- Call-to-Action: {guide['cta']}
- Emojis: {guide['emojis']}

TONE INSTRUCTIONS:
{tone_instruction}

MARKETING BEST PRACTICES:
1. Hook: Start with an attention-grabbing opening (question, stat, bold statement)
2. Value: Clearly communicate the benefit/value to the reader
3. Engagement: Make it shareable, relatable, or actionable
4. Authenticity: Sound human, not robotic
5. Clarity: One main message, easy to understand
6. Visual: Use formatting (line breaks, emojis) for readability

COPYWRITING FORMULAS TO CONSIDER:
- AIDA: Attention, Interest, Desire, Action
- PAS: Problem, Agitate, Solution
- BAB: Before, After, Bridge
- 4 P's: Picture, Promise, Prove, Push

REQUIREMENTS:
Match the exact tone and platform style
Include appropriate hashtags at the end
Add a compelling call-to-action
Use emojis according to platform guidelines
Make it scroll-stopping and shareable
Focus on benefits, not just features
Create emotional connection
Sound natural and authentic

"""
        
        if math_data:
            base_prompt += f"""
MATHEMATICAL CONTEXT TO INCORPORATE:
{math_data}

IMPORTANT: Integrate the math naturally into the narrative. Make it accessible and interesting, not dry or academic. Use real-world examples and analogies.
"""
        
        base_prompt += f"""
Now create compelling {content_type} content that:
1. Immediately grabs attention
2. Delivers clear value
3. Encourages engagement/action
4. Sounds authentic and human
5. Follows all platform guidelines above

GENERATE THE CONTENT NOW:"""
        
        return base_prompt
    
    
    def _classify_task_type(self, user_message: str, content_type: str) -> str:
        """Classify the type of task based on user message and content type"""
        
        # Content type influences task type
        if content_type in ["ad_copy", "sales_pitch", "email"]:
            return "reasoning"  # Persuasive writing requires reasoning
        
        if content_type in ["blog", "linkedin"]:
            return "reasoning"  # Thought leadership requires reasoning
        
        if content_type in ["instagram", "twitter"]:
            return "creative"  # Social posts are creative
        
        message_lower = user_message.lower()
        
        # Creative writing indicators
        creative_keywords = ["story", "engaging", "creative", "fun", "exciting"]
        if any(word in message_lower for word in creative_keywords):
            return "creative"
        
        # Analytical indicators
        analytical_keywords = ["analyze", "data", "statistics", "trends", "insights"]
        if any(word in message_lower for word in analytical_keywords):
            return "analytical"
        
        return "creative"  # Default to creative for marketing
    
    
    def _assess_complexity(self, user_message: str, content_type: str) -> TaskComplexity:
        """Assess the complexity of the task"""
        
        # Content type influences complexity
        if content_type in ["sales_pitch", "email", "blog"]:
            return TaskComplexity.COMPLEX  # Longer form requires more complexity
        
        if content_type in ["twitter", "ad_copy"]:
            return TaskComplexity.MEDIUM  # Short form, but needs to be punchy
        
        word_count = len(user_message.split())
        
        # Simple: Very short requests
        if word_count < 5:
            return TaskComplexity.SIMPLE
        
        # Expert: Very detailed or requires deep strategy
        if word_count > 50 or any(word in user_message.lower() 
                                   for word in ["comprehensive", "detailed", "strategy", "campaign"]):
            return TaskComplexity.EXPERT
        
        # Complex: Moderate detail with specific requirements
        if word_count > 20 or any(word in user_message.lower() 
                                   for word in ["persuasive", "convert", "sales"]):
            return TaskComplexity.COMPLEX
        
        # Medium: Default for most marketing requests
        return TaskComplexity.MEDIUM
    
    
    def _generate_mock_design(
        self,
        user_message: str,
        math_data: Optional[Dict[str, Any]],
        content_type: str,
        tone: str
    ) -> str:
        """Generate a mock design (fallback when no LLM available)"""
        
        mock_templates = {
            "linkedin": """Professional Perspective for {topic}

Here's what I've learned about {topic} in my career:

Key insight #1: [Relevant point]
Key insight #2: [Valuable takeaway]  
Key insight #3: [Actionable advice]

What's your experience with this? Drop your thoughts below!

#Professional #CareerGrowth #Leadership #Industry""",

            "instagram": """Let's talk about {topic}!

You know that feeling when... [relatable moment]

Here's what I discovered:
- Point one
- Point two
- Point three

Tag someone who needs to see this!

Save this for later!

#Inspo #Motivation #LifeHacks #Trending""",

            "twitter": """Hot take on {topic}:

[Attention-grabbing statement]

Here's why:
[Reason 1]
[Reason 2]

Thoughts?

#Twitter #Trending""",

            "email": """Subject: You'll want to see this about {topic}

Hey there!

Quick question: [Relevant question about their pain point]

I recently discovered something that could help...

[Brief value proposition]

[2-3 sentences of benefit]

Want to learn more? [CTA]

Best,
[Your name]""",

            "ad_copy": """{topic} - Transform Your [Outcome]!

Stop [Pain Point]. Start [Desired Result].

Benefit #1
Benefit #2
Benefit #3

Limited Time: [Offer]

[CTA Button Text] - Click Now!""",

            "sales_pitch": """Re: {topic} - Solution for [Company]

Hi [Name],

I noticed [specific observation about their company/situation].

Many [industry] companies face this challenge: [pain point]

What if you could [desired outcome] without [common objection]?

We've helped [social proof] achieve [specific result].

Can we schedule 15 minutes to explore if this fits your needs?

[CTA]""",
        }
        
        template = mock_templates.get(content_type, mock_templates.get("social_media", """
{topic}

[Mock content - Configure LLM backends in .env for AI-generated content]

Point one
Point two
Point three

#Marketing #Content
"""))
        
        # Extract topic from user message
        topic = user_message[:50] + ("..." if len(user_message) > 50 else "")
        
        content = template.format(topic=topic)
        
        if math_data and "fibonacci" in user_message.lower():
            content = f"""The Fibonacci Sequence in {topic}! 

Nature's secret code appears everywhere:

The sequence: {', '.join(map(str, math_data.get('sequence', [])))}

Formula: {math_data.get('formula', 'F(n) = F(n-1) + F(n-2)')}

This pattern is universal!

{content}

[Generated in MOCK mode - Enable LLM backends for AI content]

#Mathematics #Nature #Science"""
        
        return content