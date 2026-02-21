import os
import logging
from typing import Optional

from core.llm_manager import LLMManager, LLMConfig, LLMProvider, TaskComplexity

logger = logging.getLogger(__name__)


def create_llm_manager() -> LLMManager:
    """
    Create and configure LLM Manager with all enabled backends.
    Validates required env vars before registering each backend.
    HuggingFace is now fully wired up (was registered but never initialized).
    """
    manager = LLMManager()

    _register_ollama(manager)
    _register_groq(manager)
    _register_claude(manager)
    _register_openai(manager)
    _register_huggingface(manager)   # was previously missing

    _log_summary(manager)
    return manager


# -----------------------------------------------------------------------------
# Per-provider registration helpers
# -----------------------------------------------------------------------------

def _register_ollama(manager: LLMManager):
    if os.getenv("LLM_OLLAMA_ENABLED", "false").lower() != "true":
        return

    base_url = os.getenv("LLM_OLLAMA_BASE_URL", "http://localhost:11434")
    models_raw = os.getenv("LLM_OLLAMA_MODELS", "llama3")
    models = [m.strip() for m in models_raw.split(",") if m.strip()]

    for model in models:
        ml = model.lower()
        if "llama3" in ml or "llama-3" in ml:
            r, c, code, speed = 7, 8, 7, 6
        elif "codellama" in ml:
            r, c, code, speed = 6, 5, 9, 6
        elif "mistral" in ml:
            r, c, code, speed = 7, 7, 7, 7
        else:
            r, c, code, speed = 6, 6, 6, 6

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model=model,
            base_url=base_url,
            cost_per_1k_tokens=0.0,
            avg_latency_ms=2500,
            max_tokens=4096,
            reasoning_quality=r,
            creative_quality=c,
            code_quality=code,
            speed_score=speed,
            is_local=True,
            requires_internet=False,
            supports_streaming=True
        )
        manager.register_backend(f"ollama_{model}", config)


def _register_groq(manager: LLMManager):
    if os.getenv("LLM_GROQ_ENABLED", "false").lower() != "true":
        return

    api_key = os.getenv("LLM_GROQ_API_KEY", "").strip()
    if not api_key:
        logger.warning("LLM_GROQ_ENABLED=true but LLM_GROQ_API_KEY is missing — skipping Groq")
        return

    models_raw = os.getenv("LLM_GROQ_MODELS", "llama-3.1-8b-instant")
    models = [m.strip() for m in models_raw.split(",") if m.strip()]

    for model in models:
        ml = model.lower()
        if "70b" in ml or "3.3" in ml:
            r, c, code, tokens, latency = 9, 9, 9, 8192, 800
        elif "8b" in ml:
            r, c, code, tokens, latency = 7, 7, 7, 8192, 400
        elif "mixtral" in ml:
            r, c, code, tokens, latency = 8, 8, 8, 32768, 600
        else:
            r, c, code, tokens, latency = 8, 8, 8, 8192, 500

        config = LLMConfig(
            provider=LLMProvider.GROQ,
            model=model,
            api_key=api_key,
            cost_per_1k_tokens=0.0,
            avg_latency_ms=latency,
            max_tokens=tokens,
            reasoning_quality=r,
            creative_quality=c,
            code_quality=code,
            speed_score=10,
            is_local=False,
            requires_internet=True,
            supports_streaming=True
        )
        name = f"groq_{model.replace('-', '_').replace('.', '_')}"
        manager.register_backend(name, config)


def _register_claude(manager: LLMManager):
    if os.getenv("LLM_CLAUDE_ENABLED", "false").lower() != "true":
        return

    api_key = os.getenv("LLM_CLAUDE_API_KEY", "").strip()
    if not api_key:
        logger.warning("LLM_CLAUDE_ENABLED=true but LLM_CLAUDE_API_KEY is missing — skipping Claude")
        return

    models_raw = os.getenv("LLM_CLAUDE_MODELS", "claude-haiku-4-5-20251001")
    models = [m.strip() for m in models_raw.split(",") if m.strip()]

    for model in models:
        ml = model.lower()
        if "opus" in ml:
            r, c, code, cost, latency, speed = 10, 10, 10, 0.075, 2000, 5
        elif "sonnet" in ml:
            r, c, code, cost, latency, speed = 10, 10, 9, 0.015, 1500, 6
        elif "haiku" in ml:
            r, c, code, cost, latency, speed = 8, 8, 8, 0.0025, 800, 7
        else:
            r, c, code, cost, latency, speed = 9, 9, 9, 0.015, 1500, 6

        config = LLMConfig(
            provider=LLMProvider.CLAUDE,
            model=model,
            api_key=api_key,
            cost_per_1k_tokens=cost,
            avg_latency_ms=latency,
            max_tokens=8192,
            reasoning_quality=r,
            creative_quality=c,
            code_quality=code,
            speed_score=speed,
            is_local=False,
            requires_internet=True,
            supports_streaming=True,
            supports_functions=True,
            supports_vision=True
        )
        name = f"claude_{model.replace('-', '_')}"
        manager.register_backend(name, config)


def _register_openai(manager: LLMManager):
    if os.getenv("LLM_OPENAI_ENABLED", "false").lower() != "true":
        return

    api_key = os.getenv("LLM_OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("LLM_OPENAI_ENABLED=true but LLM_OPENAI_API_KEY is missing — skipping OpenAI")
        return

    models_raw = os.getenv("LLM_OPENAI_MODELS", "gpt-4o-mini")
    models = [m.strip() for m in models_raw.split(",") if m.strip()]

    for model in models:
        ml = model.lower()
        if "gpt-4o-mini" in ml:
            r, c, code, cost, latency, tokens = 8, 8, 8, 0.00015, 800, 16384
        elif "gpt-4o" in ml or "gpt-4-turbo" in ml:
            r, c, code, cost, latency, tokens = 10, 9, 10, 0.01, 1500, 128000
        elif "gpt-4" in ml:
            r, c, code, cost, latency, tokens = 10, 9, 10, 0.03, 2500, 8192
        elif "gpt-3.5" in ml:
            r, c, code, cost, latency, tokens = 7, 7, 7, 0.0015, 600, 16384
        else:
            r, c, code, cost, latency, tokens = 8, 8, 8, 0.002, 1000, 4096

        speed = 9 if latency < 1000 else (7 if latency < 2000 else 6)

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model=model,
            api_key=api_key,
            cost_per_1k_tokens=cost,
            avg_latency_ms=latency,
            max_tokens=tokens,
            reasoning_quality=r,
            creative_quality=c,
            code_quality=code,
            speed_score=speed,
            is_local=False,
            requires_internet=True,
            supports_streaming=True,
            supports_functions=True,
            supports_vision="vision" in ml or "gpt-4" in ml
        )
        name = f"openai_{model.replace('-', '_')}"
        manager.register_backend(name, config)


def _register_huggingface(manager: LLMManager):
    """
    HuggingFace backend — was listed in LLMProvider enum and _initialize_client
    but was never registered in llm_factory.py. Now fully wired up.
    """
    if os.getenv("LLM_HUGGINGFACE_ENABLED", "false").lower() != "true":
        return

    api_key = os.getenv("LLM_HUGGINGFACE_API_KEY", "").strip()
    if not api_key:
        logger.warning("LLM_HUGGINGFACE_ENABLED=true but LLM_HUGGINGFACE_API_KEY is missing — skipping HuggingFace")
        return

    models_raw = os.getenv("LLM_HUGGINGFACE_MODELS", "mistralai/Mistral-7B-Instruct-v0.2")
    models = [m.strip() for m in models_raw.split(",") if m.strip()]

    for model in models:
        ml = model.lower()

        # Quality heuristics based on known HF model families
        if "70b" in ml or "mixtral" in ml:
            r, c, code, latency = 8, 8, 7, 3000
        elif "13b" in ml or "mistral" in ml:
            r, c, code, latency = 7, 7, 6, 2500
        elif "7b" in ml or "falcon" in ml:
            r, c, code, latency = 6, 6, 5, 2000
        else:
            r, c, code, latency = 6, 6, 5, 2500

        config = LLMConfig(
            provider=LLMProvider.HUGGINGFACE,
            model=model,
            api_key=api_key,
            cost_per_1k_tokens=0.0,   # free inference API (rate limited)
            avg_latency_ms=latency,
            max_tokens=2048,
            reasoning_quality=r,
            creative_quality=c,
            code_quality=code,
            speed_score=4,             # HF inference API is slow
            is_local=False,
            requires_internet=True,
            supports_streaming=False   # HF inference API doesn't stream reliably
        )
        name = f"huggingface_{model.replace('/', '_').replace('-', '_')}"
        manager.register_backend(name, config)


# -----------------------------------------------------------------------------
# Startup summary logger
# -----------------------------------------------------------------------------

def _log_summary(manager: LLMManager):
    """Log a clean summary of what was initialized vs what failed"""
    initialized = []
    failed = []

    for name, config in manager.backends.items():
        if name in manager.clients:
            avg_quality = (config.reasoning_quality + config.creative_quality + config.code_quality) / 3
            cost_label = "FREE" if config.cost_per_1k_tokens == 0 else f"${config.cost_per_1k_tokens:.4f}/1K"
            speed_label = "FAST" if config.avg_latency_ms < 1000 else ("MODERATE" if config.avg_latency_ms < 2000 else "SLOW")
            initialized.append({
                "name": name,
                "provider": config.provider.value,
                "model": config.model,
                "quality": avg_quality,
                "cost": cost_label,
                "speed": speed_label,
                "local": config.is_local
            })
        else:
            failed.append(f"{name} ({config.provider.value})")

    logger.info("=" * 60)
    logger.info("LLM Manager — Initialization Complete")
    logger.info(f"Registered: {len(manager.backends)} | Active: {len(initialized)}")

    by_provider = {}
    for b in initialized:
        by_provider.setdefault(b["provider"], []).append(b)

    for provider, backends in sorted(by_provider.items()):
        logger.info(f"\n{provider.upper()} ({len(backends)} model{'s' if len(backends) > 1 else ''}):")
        for b in backends:
            location = "LOCAL" if b["local"] else "CLOUD"
            logger.info(f"   {b['name']}")
            logger.info(f"      Model: {b['model']} | Quality: {b['quality']:.1f}/10 | Cost: {b['cost']} | Speed: {b['speed']} | {location}")

    if failed:
        logger.warning(f"\nFailed to initialize ({len(failed)}):")
        for f in failed:
            logger.warning(f"   {f}")

    daily = os.getenv("LLM_DAILY_BUDGET", "5.00")
    per_req = os.getenv("LLM_MAX_COST_PER_REQUEST", "0.05")
    logger.info(f"\nBudget: ${per_req}/request | ${daily}/day")
    logger.info("=" * 60)