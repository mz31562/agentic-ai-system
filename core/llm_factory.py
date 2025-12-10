import os
from typing import Optional
import logging

from core.llm_manager import LLMManager, LLMConfig, LLMProvider, TaskComplexity

logger = logging.getLogger(__name__)


def create_llm_manager() -> LLMManager:
    """
    Create and configure LLM Manager with all enabled backends
    Enhanced with better quality scores for intelligent model selection
    """
    manager = LLMManager()
    
    if os.getenv("LLM_OLLAMA_ENABLED", "false").lower() == "true":
        base_url = os.getenv("LLM_OLLAMA_BASE_URL", "http://localhost:11434")
        models = os.getenv("LLM_OLLAMA_MODELS", "llama3").split(",")
        
        for model in models:
            model = model.strip()
            
            if "llama3" in model.lower() or "llama-3" in model.lower():
                reasoning_quality = 7
                creative_quality = 8
                code_quality = 7
                speed_score = 6
            elif "codellama" in model.lower():
                reasoning_quality = 6
                creative_quality = 5
                code_quality = 9
                speed_score = 6
            elif "mistral" in model.lower():
                reasoning_quality = 7
                creative_quality = 7
                code_quality = 7
                speed_score = 7
            else:
                reasoning_quality = 6
                creative_quality = 6
                code_quality = 6
                speed_score = 6
            
            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model=model,
                base_url=base_url,
                cost_per_1k_tokens=0.0,
                avg_latency_ms=2500,
                max_tokens=4096,
                reasoning_quality=reasoning_quality,
                creative_quality=creative_quality,
                code_quality=code_quality,
                speed_score=speed_score,
                is_local=True,
                requires_internet=False,
                supports_streaming=True
            )
            
            manager.register_backend(f"ollama_{model}", config)
    
    if os.getenv("LLM_GROQ_ENABLED", "false").lower() == "true":
        api_key = os.getenv("LLM_GROQ_API_KEY")
        if api_key:
            models = os.getenv("LLM_GROQ_MODELS", "llama-3.1-8b-instant").split(",")
            
            for model in models:
                model = model.strip()
                
                model_lower = model.lower()
                
                if "70b" in model_lower or "3.3" in model_lower:
                    reasoning_quality = 9
                    creative_quality = 9
                    code_quality = 9
                    max_tokens = 8192
                    latency = 800
                    
                elif "8b" in model_lower:
                    reasoning_quality = 7
                    creative_quality = 7
                    code_quality = 7
                    max_tokens = 8192
                    latency = 400
                    
                elif "mixtral" in model_lower:
                    reasoning_quality = 8
                    creative_quality = 8
                    code_quality = 8
                    max_tokens = 32768
                    latency = 600
                    
                else:
                    reasoning_quality = 8
                    creative_quality = 8
                    code_quality = 8
                    max_tokens = 8192
                    latency = 500
                
                config = LLMConfig(
                    provider=LLMProvider.GROQ,
                    model=model,
                    api_key=api_key,
                    cost_per_1k_tokens=0.0,
                    avg_latency_ms=latency,
                    max_tokens=max_tokens,
                    reasoning_quality=reasoning_quality,
                    creative_quality=creative_quality,
                    code_quality=code_quality,
                    speed_score=10,
                    is_local=False,
                    requires_internet=True,
                    supports_streaming=True
                )
                
                backend_name = f"groq_{model.replace('-', '_').replace('.', '_')}"
                manager.register_backend(backend_name, config)
    
    if os.getenv("LLM_CLAUDE_ENABLED", "false").lower() == "true":
        api_key = os.getenv("LLM_CLAUDE_API_KEY")
        if api_key:
            models = os.getenv("LLM_CLAUDE_MODELS", "claude-sonnet-4-20250514").split(",")
            
            for model in models:
                model = model.strip()
                model_lower = model.lower()
                
                if "opus" in model_lower:
                    reasoning_quality = 10
                    creative_quality = 10
                    code_quality = 10
                    cost_per_1k = 0.075
                    latency = 2000
                    
                elif "sonnet" in model_lower:
                    reasoning_quality = 10
                    creative_quality = 10
                    code_quality = 9
                    cost_per_1k = 0.015
                    latency = 1500
                    
                elif "haiku" in model_lower:
                    reasoning_quality = 8
                    creative_quality = 8
                    code_quality = 8
                    cost_per_1k = 0.0025
                    latency = 800
                    
                else:
                    reasoning_quality = 9
                    creative_quality = 9
                    code_quality = 9
                    cost_per_1k = 0.015
                    latency = 1500
                
                config = LLMConfig(
                    provider=LLMProvider.CLAUDE,
                    model=model,
                    api_key=api_key,
                    cost_per_1k_tokens=cost_per_1k,
                    avg_latency_ms=latency,
                    max_tokens=8192,
                    reasoning_quality=reasoning_quality,
                    creative_quality=creative_quality,
                    code_quality=code_quality,
                    speed_score=7 if "haiku" in model_lower else 6,
                    is_local=False,
                    requires_internet=True,
                    supports_streaming=True,
                    supports_functions=True,
                    supports_vision=True
                )
                
                manager.register_backend(f"claude_{model.replace('-', '_')}", config)
    
    if os.getenv("LLM_OPENAI_ENABLED", "false").lower() == "true":
        api_key = os.getenv("LLM_OPENAI_API_KEY")
        if api_key:
            models = os.getenv("LLM_OPENAI_MODELS", "gpt-4").split(",")
            
            for model in models:
                model = model.strip()
                model_lower = model.lower()
                
                if "gpt-4" in model_lower and "turbo" not in model_lower and "mini" not in model_lower:
                    reasoning_quality = 10
                    creative_quality = 9
                    code_quality = 10
                    cost_per_1k = 0.03
                    latency = 2500
                    max_tokens = 8192
                    
                elif "gpt-4-turbo" in model_lower or "gpt-4o" in model_lower:
                    reasoning_quality = 10
                    creative_quality = 9
                    code_quality = 10
                    cost_per_1k = 0.01
                    latency = 1500
                    max_tokens = 128000 if "turbo" in model_lower else 8192
                    
                elif "gpt-4o-mini" in model_lower:
                    reasoning_quality = 8
                    creative_quality = 8
                    code_quality = 8
                    cost_per_1k = 0.00015
                    latency = 800
                    max_tokens = 16384
                    
                elif "gpt-3.5" in model_lower:
                    reasoning_quality = 7
                    creative_quality = 7
                    code_quality = 7
                    cost_per_1k = 0.0015
                    latency = 600
                    max_tokens = 16384
                    
                else:
                    reasoning_quality = 8
                    creative_quality = 8
                    code_quality = 8
                    cost_per_1k = 0.002
                    latency = 1000
                    max_tokens = 4096
                
                if latency < 1000:
                    speed_score = 9
                elif latency < 2000:
                    speed_score = 7
                else:
                    speed_score = 6
                
                config = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model=model,
                    api_key=api_key,
                    cost_per_1k_tokens=cost_per_1k,
                    avg_latency_ms=latency,
                    max_tokens=max_tokens,
                    reasoning_quality=reasoning_quality,
                    creative_quality=creative_quality,
                    code_quality=code_quality,
                    speed_score=speed_score,
                    is_local=False,
                    requires_internet=True,
                    supports_streaming=True,
                    supports_functions=True,
                    supports_vision="vision" in model_lower or "gpt-4" in model_lower
                )
                
                manager.register_backend(f"openai_{model.replace('-', '_')}", config)
    
    initialized = []
    failed = []
    
    for name, config in manager.backends.items():
        if name in manager.clients:
            quality_avg = (config.reasoning_quality + config.creative_quality + config.code_quality) / 3
            cost_label = "FREE" if config.cost_per_1k_tokens == 0 else f"${config.cost_per_1k_tokens:.4f}/1K"
            speed_label = "FAST" if config.avg_latency_ms < 1000 else "MODERATE" if config.avg_latency_ms < 2000 else "SLOW"  # CHANGED: Removed ⚡
            
            initialized.append({
                "name": name,
                "provider": config.provider.value,
                "model": config.model,
                "quality": quality_avg,
                "cost": cost_label,
                "speed": speed_label,
                "is_local": config.is_local
            })
        else:
            failed.append(f"{name} ({config.provider.value})")
    
    logger.info(f"=" * 60)
    logger.info(f"LLM Manager Initialization Complete")
    logger.info(f"=" * 60)
    logger.info(f"Total registered: {len(manager.backends)}")
    logger.info(f"Successfully initialized: {len(initialized)}")

    if initialized:
        logger.info(f"=" * 60)
        
        by_provider = {}
        for backend in initialized:
            provider = backend["provider"]
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(backend)
        
        for provider, backends in sorted(by_provider.items()):
            logger.info(f"")
            logger.info(f"{provider.upper()} ({len(backends)} model{'s' if len(backends) > 1 else ''}):")
            
            for backend in backends:
                location = "LOCAL" if backend["is_local"] else "CLOUD"
                logger.info(f"   {backend['name']}")
                logger.info(f"      Model: {backend['model']}")
                logger.info(f"      Quality: {backend['quality']:.1f}/10 | Cost: {backend['cost']} | Speed: {backend['speed']}")
                logger.info(f"      Location: {location}")

    if failed:
        logger.info(f"=" * 60)
        logger.warning(f"Failed to initialize: {len(failed)}")
        for backend in failed:
            logger.warning(f"   {backend}")

    logger.info(f"=" * 60)
    
    daily_budget = os.getenv("LLM_DAILY_BUDGET", "5.00")
    per_request = os.getenv("LLM_MAX_COST_PER_REQUEST", "0.05")
    logger.info(f"")
    logger.info(f"Budget Controls:")
    logger.info(f"   Daily Budget: ${daily_budget}")
    logger.info(f"   Per-Request Limit: ${per_request}")
    logger.info(f"")
    
    return manager