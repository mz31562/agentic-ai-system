import asyncio
import os
from dotenv import load_dotenv
from core.llm_factory import create_llm_manager
from core.llm_manager import TaskComplexity

load_dotenv()

async def test_selection():
    manager = create_llm_manager()
    
    print("=" * 70)
    print("BACKEND SELECTION TEST (UNBIASED)")
    print("=" * 70)
    
    scenarios = [
        ("creative", TaskComplexity.SIMPLE, "Instagram post"),
        ("creative", TaskComplexity.MEDIUM, "Blog article"),
        ("reasoning", TaskComplexity.COMPLEX, "Market analysis"),
        ("code", TaskComplexity.MEDIUM, "Python script"),
    ]
    
    for task_type, complexity, description in scenarios:
        print(f"\n📝 Scenario: {description}")
        print(f"   Task: {task_type}, Complexity: {complexity.value}")
        print(f"   Settings: PREFER_FAST={os.getenv('LLM_PREFER_FAST')}, PRIORITY={os.getenv('LLM_BACKEND_PRIORITY')}")
        
        ranked = manager._rank_backends(task_type, complexity, {})
        
        print(f"\n   Rankings:")
        for i, (name, score, config) in enumerate(ranked, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"   {emoji} {i}. {name}")
            print(f"       Score: {score:.1f} | Provider: {config.provider.value} | Cost: ${config.cost_per_1k_tokens:.4f}")

asyncio.run(test_selection())