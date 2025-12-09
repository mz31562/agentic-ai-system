# C:\Users\MohammedZaid\Desktop\agentic-ai-system\agents\post_design_agent.py
import asyncio
import re
from typing import Dict, Any, Optional, List, Tuple
import logging
import os

from core.agent_base import BaseAgent
from core.message_bus import MessageBus, Message
from core.llm_manager import LLMManager, TaskComplexity

logger = logging.getLogger(__name__)


class PostDesignAgent(BaseAgent):
    """
    Enhanced PostDesign Agent with advanced marketing & sales capabilities.
    Supports multiple content types, tones, and platforms with intelligent model selection.
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
                "creative_writing",
                "multi_llm_support",
                "intelligent_model_selection"
            ]
        )
        
        self.llm_manager = llm_manager
        
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
        """Handle design request from Host Agent with intelligent model selection"""
        user_message = message.payload.get("user_message", "")
        request_id = message.payload.get("request_id", "")
        
        logger.info(f"PostDesign Agent processing: '{user_message}'")
        
        try:
            # Enhanced content analysis
            content_type = self._detect_content_type(user_message)
            tone = self._detect_tone(user_message)
            
            # NEW: Extract task metadata for smarter model selection
            task_type, task_metadata = self._classify_task_type(user_message, content_type)
            complexity = self._assess_complexity(user_message, content_type, task_metadata)
            
            logger.info(f"Content analysis: type={content_type}, tone={tone}")
            logger.info(f"Task analysis: type={task_type}, complexity={complexity.value}")
            logger.info(f"Task metadata: high_stakes={task_metadata.get('is_high_stakes')}, "
                       f"needs_speed={task_metadata.get('needs_speed')}, "
                       f"word_estimate={task_metadata.get('word_count_estimate')}")
            
            design_result = await self._generate_design(
                user_message, 
                content_type=content_type,
                tone=tone,
                task_type=task_type,
                complexity=complexity,
                task_metadata=task_metadata
            )
            
            await self.send_response(
                original_message=message,
                payload={
                    "design_result": design_result["content"],
                    "metadata": design_result["metadata"],
                    "request_id": request_id,
                    "status": "completed"
                },
                topic="design_response"
            )
            
            backend_used = design_result["metadata"].get("backend_used", "unknown")
            cost = design_result["metadata"].get("cost", 0.0)
            logger.info(f"Design completed for request {request_id} using {backend_used} (${cost:.4f})")
            
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
    
    
    def _extract_model_preference(self, user_message: str) -> Optional[str]:
        """
        Extract explicit model preference from user message
        Users can specify: [use-best-model], [use-fast-model], [use-free-model]
        """
        message_lower = user_message.lower()
        
        if "[use-best-model]" in message_lower or "[best-quality]" in message_lower:
            return "quality"
        elif "[use-fast-model]" in message_lower or "[fast]" in message_lower:
            return "speed"
        elif "[use-free-model]" in message_lower or "[free]" in message_lower:
            return "free"
        
        return None
    
    
    async def _generate_design(
        self,
        user_message: str,
        content_type: str = "social_media",
        tone: str = "professional",
        task_type: str = "creative",
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        task_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate design using intelligent LLM selection with task metadata"""
        
        task_metadata = task_metadata or {}
        
        prompt = self._build_enhanced_prompt(
            user_message, 
            content_type, 
            tone
        )
        
        # Check for explicit model preference
        model_preference = self._extract_model_preference(user_message)
        
        # Prepare kwargs for LLM manager
        llm_kwargs = {
            "prefer_local": model_preference == "free",
            "prefer_fast": task_metadata.get("needs_speed", False) or model_preference == "speed",
            "task_metadata": task_metadata
        }
        
        # Override for quality preference
        if model_preference == "quality":
            llm_kwargs["max_cost"] = 0.10  # Allow expensive models
            llm_kwargs["prefer_fast"] = False
        
        logger.info(f"Task classification: type={task_type}, complexity={complexity.value}")
        
        try:
            result = await self.llm_manager.complete(
                prompt=prompt,
                task_type=task_type,
                complexity=complexity,
                max_tokens=2048,
                temperature=0.85 if task_type == "creative" else 0.7,
                **llm_kwargs
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
                "word_count": len(content.split()),
                "task_metadata": task_metadata,
                "model_preference": model_preference
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
            content = self._generate_mock_design(user_message, content_type, tone)
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
            }
            
            return {
                "content": content,
                "metadata": metadata
            }
    
    
    def _build_enhanced_prompt(
        self,
        user_message: str,
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

Now create compelling {content_type} content that:
1. Immediately grabs attention
2. Delivers clear value
3. Encourages engagement/action
4. Sounds authentic and human
5. Follows all platform guidelines above

GENERATE THE CONTENT NOW:"""
        
        return base_prompt
    
    
    def _classify_task_type(self, user_message: str, content_type: str) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced classification that returns task type AND metadata
        
        Returns:
            Tuple of (task_type, metadata_dict)
        """
        message_lower = user_message.lower()
        
        # Determine if this is high-stakes content
        high_stakes_keywords = [
            "important", "urgent", "campaign", "launch", "announcement",
            "pitch", "proposal", "presentation", "keynote", "investor",
            "client", "executive", "board", "ceo", "critical"
        ]
        is_high_stakes = any(word in message_lower for word in high_stakes_keywords)
        
        # Determine if speed matters
        speed_keywords = ["quick", "fast", "asap", "urgent", "now", "immediately", "hurry"]
        needs_speed = any(word in message_lower for word in speed_keywords)
        
        # Estimate word count based on content type
        word_count_estimates = {
            "twitter": 50,
            "instagram": 150,
            "facebook": 200,
            "linkedin": 250,
            "ad_copy": 100,
            "email": 300,
            "blog": 500,
            "sales_pitch": 400,
            "product_description": 150,
            "social_media": 150
        }
        
        estimated_words = word_count_estimates.get(content_type, 150)
        
        # Build metadata
        task_metadata = {
            "is_high_stakes": is_high_stakes,
            "needs_speed": needs_speed,
            "requires_research": "blog" in content_type or "article" in content_type,
            "word_count_estimate": estimated_words,
            "content_type": content_type
        }
        
        # Determine task type based on content
        if content_type in ["blog", "article", "email", "sales_pitch", "linkedin"]:
            task_type = "reasoning"  # Strategic content needs reasoning
        elif content_type in ["ad_copy", "product_description"]:
            task_type = "reasoning"  # Persuasive content needs reasoning
        elif content_type in ["instagram", "twitter", "facebook"]:
            task_type = "creative"  # Social posts are creative
        else:
            # Check message for creative vs analytical
            creative_keywords = ["story", "engaging", "creative", "fun", "exciting", "emotional"]
            analytical_keywords = ["analyze", "data", "statistics", "trends", "insights", "report"]
            
            if any(word in message_lower for word in creative_keywords):
                task_type = "creative"
            elif any(word in message_lower for word in analytical_keywords):
                task_type = "analytical"
            else:
                task_type = "creative"  # Default for marketing
        
        # Override task type for high-stakes content
        if is_high_stakes and task_type == "creative":
            task_type = "reasoning"  # High-stakes needs better reasoning
            logger.info("Upgraded task type to 'reasoning' due to high-stakes flag")
        
        return task_type, task_metadata
    
    
    def _assess_complexity(
        self, 
        user_message: str, 
        content_type: str,
        task_metadata: Dict[str, Any]
    ) -> TaskComplexity:
        """Enhanced complexity assessment using metadata"""
        
        # High-stakes content always uses better models
        if task_metadata.get("is_high_stakes"):
            logger.info("Complexity set to COMPLEX due to high-stakes flag")
            return TaskComplexity.COMPLEX
        
        # Long-form content needs more sophistication
        word_estimate = task_metadata.get("word_count_estimate", 0)
        if word_estimate > 300:
            return TaskComplexity.COMPLEX
        elif word_estimate > 200:
            return TaskComplexity.MEDIUM
        
        # Content type specific complexity
        if content_type in ["sales_pitch", "blog", "email"]:
            return TaskComplexity.COMPLEX
        elif content_type in ["ad_copy", "linkedin", "product_description"]:
            return TaskComplexity.MEDIUM
        elif content_type in ["twitter", "instagram"]:
            # Social can be simple unless high-stakes
            return TaskComplexity.SIMPLE
        
        # Check prompt detail level
        word_count = len(user_message.split())
        
        if word_count < 5:
            return TaskComplexity.SIMPLE
        elif word_count > 40:
            return TaskComplexity.COMPLEX
        elif word_count > 15:
            return TaskComplexity.MEDIUM
        
        return TaskComplexity.MEDIUM
    
    
    def _generate_mock_design(
        self,
        user_message: str,
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

            "instagram": """Let's talk about {topic}! ✨

You know that feeling when... [relatable moment]

Here's what I discovered:
💡 Point one
💡 Point two
💡 Point three

Tag someone who needs to see this! 👇

Save this for later! 📌

#Inspo #Motivation #LifeHacks #Trending""",

            "twitter": """Hot take on {topic}:

[Attention-grabbing statement]

Here's why:
- [Reason 1]
- [Reason 2]

Thoughts? 🤔

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

✓ Benefit #1
✓ Benefit #2
✓ Benefit #3

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
        
        return content