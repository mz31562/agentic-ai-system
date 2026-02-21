"""
core/config_validator.py

Validates .env configuration at startup and gives clear, actionable
error messages instead of failing silently deep inside the system.

Call validate_config() from main.py before initializing anything else.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def error(self, msg: str):
        self.errors.append(msg)
        self.valid = False

    def warn(self, msg: str):
        self.warnings.append(msg)

    def print_report(self):
        if self.warnings:
            print("\nConfiguration warnings:")
            for w in self.warnings:
                print(f"   ⚠  {w}")

        if self.errors:
            print("\nConfiguration errors (system cannot start):")
            for e in self.errors:
                print(f"   ✗  {e}")
            print()
        elif self.warnings:
            print()


def validate_config() -> ValidationResult:
    """
    Validate all .env configuration at startup.
    Returns a ValidationResult — call .print_report() then check .valid.
    """
    result = ValidationResult()

    _validate_llm_backends(result)
    _validate_image_backend(result)
    _validate_budget(result)
    _validate_directories(result)

    return result


# -----------------------------------------------------------------------------
# LLM backend validation
# -----------------------------------------------------------------------------

def _validate_llm_backends(result: ValidationResult):
    """At least one LLM backend must be enabled and have its key set"""
    providers = {
        "GROQ":         ("LLM_GROQ_ENABLED",        "LLM_GROQ_API_KEY"),
        "OPENAI":       ("LLM_OPENAI_ENABLED",       "LLM_OPENAI_API_KEY"),
        "CLAUDE":       ("LLM_CLAUDE_ENABLED",       "LLM_CLAUDE_API_KEY"),
        "HUGGINGFACE":  ("LLM_HUGGINGFACE_ENABLED",  "LLM_HUGGINGFACE_API_KEY"),
        "OLLAMA":       ("LLM_OLLAMA_ENABLED",       None),  # no key needed
    }

    enabled_count = 0

    for provider, (enabled_var, key_var) in providers.items():
        enabled = os.getenv(enabled_var, "false").lower() == "true"
        if not enabled:
            continue

        enabled_count += 1

        if key_var:
            key = os.getenv(key_var, "").strip()
            if not key:
                result.error(
                    f"{provider} is enabled ({enabled_var}=true) "
                    f"but {key_var} is missing or empty"
                )

        if provider == "OLLAMA":
            base_url = os.getenv("LLM_OLLAMA_BASE_URL", "http://localhost:11434")
            models = os.getenv("LLM_OLLAMA_MODELS", "").strip()
            if not models:
                result.warn(
                    "LLM_OLLAMA_MODELS not set — defaulting to 'llama3'. "
                    "Make sure that model is pulled in your Ollama instance."
                )
            result.warn(
                f"Ollama is enabled — make sure the Ollama service is running at {base_url}"
            )

    if enabled_count == 0:
        result.error(
            "No LLM backends are enabled. "
            "Set at least one of: LLM_GROQ_ENABLED, LLM_OPENAI_ENABLED, "
            "LLM_CLAUDE_ENABLED, LLM_HUGGINGFACE_ENABLED, LLM_OLLAMA_ENABLED to 'true'"
        )


# -----------------------------------------------------------------------------
# Image backend validation
# -----------------------------------------------------------------------------

def _validate_image_backend(result: ValidationResult):
    backend = os.getenv("IMAGE_BACKEND", "comfyui").lower().strip()
    valid_backends = {"comfyui", "dalle", "replicate", "segmind", "mock"}

    if backend not in valid_backends:
        result.error(
            f"IMAGE_BACKEND='{backend}' is not valid. "
            f"Choose from: {', '.join(sorted(valid_backends))}"
        )
        return

    if backend == "comfyui":
        comfyui_url = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
        model = os.getenv("IMAGE_MODEL", "sdxl")
        result.warn(
            f"ComfyUI backend selected — ensure ComfyUI is running at {comfyui_url} "
            f"with model '{model}' loaded before generating images"
        )

    elif backend == "dalle":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            # also check the LLM key since some people share it
            key = os.getenv("LLM_OPENAI_API_KEY", "").strip()
        if not key:
            result.error(
                "IMAGE_BACKEND=dalle requires OPENAI_API_KEY to be set"
            )

    elif backend == "replicate":
        key = os.getenv("REPLICATE_API_KEY", "").strip()
        if not key:
            key = os.getenv("IMAGE_API_KEY", "").strip()
        if not key:
            result.error(
                "IMAGE_BACKEND=replicate requires REPLICATE_API_KEY (or IMAGE_API_KEY) to be set"
            )

    elif backend == "segmind":
        key = os.getenv("SEGMIND_API_KEY", "").strip()
        if not key:
            key = os.getenv("IMAGE_API_KEY", "").strip()
        if not key:
            result.error(
                "IMAGE_BACKEND=segmind requires SEGMIND_API_KEY (or IMAGE_API_KEY) to be set"
            )

    elif backend == "mock":
        result.warn(
            "IMAGE_BACKEND=mock — no real images will be generated. "
            "This is fine for testing."
        )

    # Size format check
    size = os.getenv("DEFAULT_IMAGE_SIZE", "1024x1024")
    try:
        w, h = size.split("x")
        int(w), int(h)
    except Exception:
        result.error(
            f"DEFAULT_IMAGE_SIZE='{size}' is invalid. "
            "Use format WIDTHxHEIGHT e.g. 1024x1024"
        )


# -----------------------------------------------------------------------------
# Budget validation
# -----------------------------------------------------------------------------

def _validate_budget(result: ValidationResult):
    for var, default in [("LLM_DAILY_BUDGET", "5.00"), ("LLM_MAX_COST_PER_REQUEST", "0.05")]:
        val = os.getenv(var, default)
        try:
            parsed = float(val)
            if parsed < 0:
                result.error(f"{var}={val} must be a positive number")
        except ValueError:
            result.error(f"{var}='{val}' is not a valid number")

    warn_pct = os.getenv("LLM_WARN_AT_PERCENT", "80")
    try:
        pct = float(warn_pct)
        if not (0 < pct <= 100):
            result.warn(f"LLM_WARN_AT_PERCENT={warn_pct} should be between 1 and 100")
    except ValueError:
        result.warn(f"LLM_WARN_AT_PERCENT='{warn_pct}' is not a valid number, defaulting to 80")


# -----------------------------------------------------------------------------
# Directory validation
# -----------------------------------------------------------------------------

def _validate_directories(result: ValidationResult):
    output_dir = os.getenv("IMAGE_OUTPUT_DIR", "generated_images")
    try:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        result.error(f"Cannot create IMAGE_OUTPUT_DIR='{output_dir}': {e}")