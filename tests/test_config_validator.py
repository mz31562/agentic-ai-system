# ============================================================
# tests/test_config_validator.py
# ============================================================
import os
import pytest
from unittest.mock import patch
from core.config_validator import validate_config


def test_no_backends_enabled():
    """Should fail validation if no LLM backends are enabled"""
    env_overrides = {
        "LLM_GROQ_ENABLED": "false",
        "LLM_OPENAI_ENABLED": "false",
        "LLM_CLAUDE_ENABLED": "false",
        "LLM_OLLAMA_ENABLED": "false",
        "LLM_HUGGINGFACE_ENABLED": "false",
        "IMAGE_BACKEND": "mock",
    }
    with patch.dict(os.environ, env_overrides, clear=False):
        result = validate_config()
    assert not result.valid
    assert any("No LLM backends" in e for e in result.errors)


def test_backend_enabled_without_key():
    """Should fail if backend is enabled but key is missing"""
    env_overrides = {
        "LLM_GROQ_ENABLED": "true",
        "LLM_GROQ_API_KEY": "",
        "LLM_OPENAI_ENABLED": "false",
        "LLM_CLAUDE_ENABLED": "false",
        "LLM_OLLAMA_ENABLED": "false",
        "LLM_HUGGINGFACE_ENABLED": "false",
        "IMAGE_BACKEND": "mock",
    }
    with patch.dict(os.environ, env_overrides, clear=False):
        result = validate_config()
    assert not result.valid
    assert any("LLM_GROQ_API_KEY" in e for e in result.errors)


def test_invalid_image_backend():
    """Should fail on unknown image backend"""
    env_overrides = {
        "LLM_GROQ_ENABLED": "true",
        "LLM_GROQ_API_KEY": "test_key",
        "IMAGE_BACKEND": "nonexistent_backend",
        "LLM_OPENAI_ENABLED": "false",
        "LLM_CLAUDE_ENABLED": "false",
        "LLM_OLLAMA_ENABLED": "false",
        "LLM_HUGGINGFACE_ENABLED": "false",
    }
    with patch.dict(os.environ, env_overrides, clear=False):
        result = validate_config()
    assert not result.valid
    assert any("IMAGE_BACKEND" in e for e in result.errors)


def test_invalid_image_size():
    """Should fail on malformed DEFAULT_IMAGE_SIZE"""
    env_overrides = {
        "LLM_GROQ_ENABLED": "true",
        "LLM_GROQ_API_KEY": "test_key",
        "IMAGE_BACKEND": "mock",
        "DEFAULT_IMAGE_SIZE": "badformat",
        "LLM_OPENAI_ENABLED": "false",
        "LLM_CLAUDE_ENABLED": "false",
        "LLM_OLLAMA_ENABLED": "false",
        "LLM_HUGGINGFACE_ENABLED": "false",
    }
    with patch.dict(os.environ, env_overrides, clear=False):
        result = validate_config()
    assert not result.valid
    assert any("DEFAULT_IMAGE_SIZE" in e for e in result.errors)


def test_valid_minimal_config():
    """Should pass with a minimal valid config"""
    env_overrides = {
        "LLM_GROQ_ENABLED": "true",
        "LLM_GROQ_API_KEY": "test_key_that_is_set",
        "LLM_OPENAI_ENABLED": "false",
        "LLM_CLAUDE_ENABLED": "false",
        "LLM_OLLAMA_ENABLED": "false",
        "LLM_HUGGINGFACE_ENABLED": "false",
        "IMAGE_BACKEND": "mock",
        "DEFAULT_IMAGE_SIZE": "1024x1024",
        "LLM_DAILY_BUDGET": "5.00",
        "LLM_MAX_COST_PER_REQUEST": "0.05",
    }
    with patch.dict(os.environ, env_overrides, clear=False):
        result = validate_config()
    assert result.valid
    assert not result.errors