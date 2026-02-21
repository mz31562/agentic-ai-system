# ============================================================
# tests/test_backend_selection.py
# ============================================================
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from core.llm_factory import create_llm_manager
from core.llm_manager import TaskComplexity


async def test_backend_selection():
    """Test that backend scoring and ranking works correctly"""
    manager = create_llm_manager()

    print("=" * 70)
    print("BACKEND SELECTION TEST")
    print("=" * 70)

    scenarios = [
        ("creative",   TaskComplexity.SIMPLE,  "Instagram post"),
        ("creative",   TaskComplexity.MEDIUM,  "Blog article"),
        ("reasoning",  TaskComplexity.COMPLEX, "Market analysis"),
        ("code",       TaskComplexity.MEDIUM,  "Python script"),
    ]

    for task_type, complexity, description in scenarios:
        print(f"\nScenario : {description}")
        print(f"Task     : {task_type} | Complexity: {complexity.value}")
        print(f"Settings : PREFER_FAST={os.getenv('LLM_PREFER_FAST')} | PRIORITY={os.getenv('LLM_BACKEND_PRIORITY')}")

        ranked = manager._rank_backends(task_type, complexity, {})

        if not ranked:
            print("   No backends available — check your .env configuration")
            continue

        print("Rankings:")
        for i, (name, score, config) in enumerate(ranked, 1):
            medal = ["1st", "2nd", "3rd"][i - 1] if i <= 3 else f"{i}th"
            print(f"   {medal}. {name}")
            print(f"        Score: {score:.1f} | Provider: {config.provider.value} | Cost: ${config.cost_per_1k_tokens:.4f}/1K")


if __name__ == "__main__":
    asyncio.run(test_backend_selection())
