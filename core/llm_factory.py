# core/llm_factory.py
import os
from typing import Optional
import logging

from core.llm_manager import LLMManager, LLMConfig, LLMProvider, TaskComplexity

logger = logging.getLogger(__name__)


def create_llm_manager() -> LLMManager:
    """
    Create and configure LLM Manager with all enabled backends
    """
    manager = LLMManager()
    
    # Register Ollama backends
    if os.getenv("LLM_OLLAMA_ENABLED", "false").lower() == "true":
        base_url = os.getenv("LLM_OLLAMA_BASE_URL", "http://localhost:11434")
        models = os.getenv("LLM_OLLAMA_MODELS", "llama3").split(",")
        
        for model in models:
            model = model.strip()
            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model=model,
                base_url=base_url,
                cost_per_1k_tokens=0.0,  # Free!
                avg_latency_ms=2000,
                max_tokens=4096,
                reasoning_quality=7 if "llama3" in model else 6,
                creative_quality=8 if "llama3" in model else 6,
                code_quality=9 if "codellama" in model else 6,
                speed_score=6,
                is_local=True,
                requires_internet=False,
                supports_streaming=True
            )
            
            manager.register_backend(f"ollama_{model}", config)
    
    # Register Groq backends
    if os.getenv("LLM_GROQ_ENABLED", "false").lower() == "true":
        api_key = os.getenv("LLM_GROQ_API_KEY")
        if api_key:
            models = os.getenv("LLM_GROQ_MODELS", "llama-3.1-70b-versatile").split(",")
            
            for model in models:
                model = model.strip()
                config = LLMConfig(
                    provider=LLMProvider.GROQ,
                    model=model,
                    api_key=api_key,
                    cost_per_1k_tokens=0.0,  # Free tier!
                    avg_latency_ms=500,  # Very fast!
                    max_tokens=8192 if "70b" in model else 4096,
                    reasoning_quality=9 if "70b" in model else 7,
                    creative_quality=8 if "70b" in model else 7,
                    code_quality=8,
                    speed_score=10,  # Extremely fast
                    is_local=False,
                    requires_internet=True,
                    supports_streaming=True
                )
                
                manager.register_backend(f"groq_{model.replace('-', '_')}", config)
    
    # Register Claude backends
    if os.getenv("LLM_CLAUDE_ENABLED", "false").lower() == "true":
        api_key = os.getenv("LLM_CLAUDE_API_KEY")
        if api_key:
            models = os.getenv("LLM_CLAUDE_MODELS", "claude-sonnet-4-20250514").split(",")
            
            for model in models:
                model = model.strip()
                config = LLMConfig(
                    provider=LLMProvider.CLAUDE,
                    model=model,
                    api_key=api_key,
                    cost_per_1k_tokens=0.015 if "sonnet" in model else 0.0025,
                    avg_latency_ms=1500,
                    max_tokens=8192,
                    reasoning_quality=10 if "sonnet" in model else 8,
                    creative_quality=10,
                    code_quality=9,
                    speed_score=7,
                    is_local=False,
                    requires_internet=True,
                    supports_streaming=True,
                    supports_functions=True,
                    supports_vision=True
                )
                
                manager.register_backend(f"claude_{model.replace('-', '_')}", config)
    
    # Register OpenAI backends
    if os.getenv("LLM_OPENAI_ENABLED", "false").lower() == "true":
        api_key = os.getenv("LLM_OPENAI_API_KEY")
        if api_key:
            models = os.getenv("LLM_OPENAI_MODELS", "gpt-4").split(",")
            
            for model in models:
                model = model.strip()
                config = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model=model,
                    api_key=api_key,
                    cost_per_1k_tokens=0.03 if "gpt-4" in model else 0.002,
                    avg_latency_ms=2000 if "gpt-4" in model else 1000,
                    max_tokens=8192 if "gpt-4" in model else 4096,
                    reasoning_quality=10 if "gpt-4" in model else 8,
                    creative_quality=9,
                    code_quality=10 if "gpt-4" in model else 8,
                    speed_score=6 if "gpt-4" in model else 8,
                    is_local=False,
                    requires_internet=True,
                    supports_streaming=True,
                    supports_functions=True,
                    supports_vision="vision" in model
                )
                
                manager.register_backend(f"openai_{model.replace('-', '_')}", config)
    
    # Show what actually initialized
    initialized = []
    failed = []
    
    for name, config in manager.backends.items():
        if name in manager.clients:
            initialized.append(f"{name} ({config.provider.value})")
        else:
            failed.append(f"{name} ({config.provider.value})")
    
    logger.info(f"LLM Manager initialized:")
    logger.info(f"   Total registered: {len(manager.backends)}")
    logger.info(f"   Successfully initialized: {len(initialized)}")
    if initialized:
        for backend in initialized:
            logger.info(f"      {backend}")
    
    if failed:
        logger.warning(f"   Failed to initialize: {len(failed)}")
        for backend in failed:
            logger.warning(f"      ✗ {backend}")
    
    return manager