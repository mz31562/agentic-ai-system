import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

# Single call to setup_logging — no more duplicated logging config in this file
from core.logging_config import setup_logging, toggle_debug

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
log_file = setup_logging(debug=DEBUG_MODE)

from core.message_bus import MessageBus, Message
from agents.host_agent import HostAgent
from agents.post_design_agent import PostDesignAgent
from agents.image_generation_agent import ImageGenerationAgent
from core.llm_factory import create_llm_manager

logger = logging.getLogger(__name__)


class CLIApp:
    """Command-line interface for the agentic system"""

    def __init__(self):
        self.message_bus: Optional[MessageBus] = None
        self.host_agent: Optional[HostAgent] = None
        self.post_design_agent: Optional[PostDesignAgent] = None
        self.image_generation_agent: Optional[ImageGenerationAgent] = None
        self.llm_manager = None
        self.response_queue = []
        self.running = True
        self.shutdown_in_progress = False
        self.log_file = log_file
        self.debug_mode = DEBUG_MODE


    async def initialize(self):
        """Initialize the agentic system"""
        print("\n" + "=" * 70)
        print("AGENTIC AI SYSTEM - CLI Interface")
        print("=" * 70 + "\n")

        if not self.debug_mode:
            print(f"Log file: {self.log_file}")
            print(f"Tip: set DEBUG_MODE=true in .env for verbose output\n")
        else:
            print(f"DEBUG MODE ENABLED — Verbose logging active")
            print(f"Logs: {self.log_file}\n")

        print("Initializing system components...")

        self.message_bus = MessageBus()

        # ComfyUI connectivity check
        image_backend = os.getenv("IMAGE_BACKEND", "comfyui")
        if image_backend == "comfyui":
            print("\nVerifying ComfyUI connection...")
            comfyui_url = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
            try:
                import requests
                response = requests.get(f"{comfyui_url}/system_stats", timeout=3)
                if response.status_code == 200:
                    print(f"   ComfyUI connected at {comfyui_url}")
                else:
                    print(f"   Warning: ComfyUI not responding at {comfyui_url}")
                    print(f"   Start ComfyUI or set IMAGE_BACKEND=dalle in .env")
            except Exception:
                print(f"   Warning: Cannot connect to ComfyUI at {comfyui_url}")
                print(f"   Start ComfyUI or set IMAGE_BACKEND=dalle in .env")

        # LLM backends
        print("\nInitializing LLM backends...")
        self.llm_manager = create_llm_manager()

        stats = self.llm_manager.get_stats()
        available_backends = [n for n, info in stats["backends"].items() if info["available"]]

        if available_backends:
            print(f"Active LLM backends: {len(available_backends)}")
            for name in available_backends:
                info = stats["backends"][name]["config"]
                print(f"   {name}")
                print(f"      Model: {info['model']}")
                if info['is_local']:
                    print(f"      Location: Local (Free)")
                else:
                    print(f"      Cost: ${info['cost_per_1k_tokens']:.4f}/1K tokens")
        else:
            print("Warning: No LLM backends available — mock mode active")
            print("   Configure backends in .env")

        image_config = {
            "backend": os.getenv("IMAGE_BACKEND", "comfyui"),
            "model": os.getenv("IMAGE_MODEL", "sdxl"),
            "api_key": os.getenv("IMAGE_API_KEY"),
            "output_dir": os.getenv("IMAGE_OUTPUT_DIR", "generated_images"),
            "comfyui_url": os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
        }

        # Pass llm_manager to HostAgent so it can use LLM-based intent classification
        self.host_agent = HostAgent(self.message_bus, llm_manager=self.llm_manager)
        self.post_design_agent = PostDesignAgent(self.message_bus, self.llm_manager)
        self.image_generation_agent = ImageGenerationAgent(self.message_bus, image_config=image_config)

        self.message_bus.subscribe(
            topic="user_response",
            agent_id="cli_app",
            callback=self.handle_response
        )

        await self.host_agent.start()
        await self.post_design_agent.start()
        await self.image_generation_agent.start()

        print("\nSystem ready")
        print(f"   LLM Backends : {len(available_backends)} active")
        print(f"   Image Backend: {image_config['backend']}")
        print(f"   Image Model  : {image_config['model']}")
        print(f"   Saga         : Enabled")
        print(f"   Intent       : {'LLM-based' if self.llm_manager and available_backends else 'Keyword matching'}\n")

        self.print_help()
        return True


    async def handle_response(self, message: Message):
        """Collect agent responses"""
        content = message.payload.get("result", message.payload.get("message", "No response"))
        self.response_queue.append(content)


    def print_help(self):
        print("=" * 70)
        print("COMMANDS:")
        print("=" * 70)
        print("  Type your request to interact with the agents")
        print("  /status  — System status and backend metrics")
        print("  /sagas   — Active saga workflow details")
        print("  /logs    — Recent log entries")
        print("  /debug   — Toggle verbose logging")
        print("  /help    — Show this message")
        print("  /clear   — Clear screen")
        print("  /exit    — Quit")
        print("=" * 70)


    async def send_message(self, user_message: str):
        """Send a user message and wait for response"""
        message = Message(
            type="request",
            sender="cli_app",
            topic="user_request",
            payload={"user_id": "cli_user", "message": user_message}
        )
        await self.message_bus.publish(message)
        print("\nProcessing request...\n")

        spinner = ['|', '/', '-', '\\']
        spinner_idx = 0

        for i in range(600):  # max 300s
            await asyncio.sleep(0.5)

            if self.response_queue:
                print("\rDone" + " " * 30, flush=True)
                print()
                break

            if not self.debug_mode and i % 2 == 0:
                elapsed = i * 0.5
                print(f"\r{spinner[spinner_idx]} Processing... ({elapsed:.0f}s)", end="", flush=True)
                spinner_idx = (spinner_idx + 1) % len(spinner)

            # Progress update every 30s for long saga workflows
            if i > 0 and i % 60 == 0 and self.host_agent:
                active_sagas = self.host_agent.saga_coordinator.get_all_active_sagas()
                for saga in active_sagas:
                    print(f"\n   [{saga['name']}] Step {saga['current_step']}/{saga['total_steps']}", end="")
        else:
            print("\rRequest timed out" + " " * 20, flush=True)
            print()

        if self.response_queue:
            while self.response_queue:
                response = self.response_queue.pop(0)
                print("=" * 70)
                print("RESPONSE:")
                print("=" * 70)
                print(response)
                print("=" * 70 + "\n")
        else:
            print(f"No response received. Check {self.log_file} for details.\n")


    async def show_status(self):
        print("\n" + "=" * 70)
        print("SYSTEM STATUS")
        print("=" * 70 + "\n")

        if self.llm_manager:
            stats = self.llm_manager.get_stats()
            available_count = sum(1 for i in stats["backends"].values() if i["available"])
            print(f"LLM Manager: {available_count}/{len(stats['backends'])} backends active")

            budget = stats.get("budget", {})
            if budget:
                print(f"\n   Budget:")
                print(f"      Spend   : ${budget.get('daily_spend', 0):.4f} / ${budget.get('daily_budget', 5.0):.2f} ({budget.get('daily_percent_used', 0):.1f}%)")
                print(f"      Remaining: ${budget.get('daily_remaining', 0):.4f}")
                print(f"      Requests : {budget.get('total_requests_today', 0)}")

            for name, info in stats["backends"].items():
                if not info["available"]:
                    continue
                config = info["config"]
                usage = info.get("stats", {})
                print(f"\n   {name}")
                print(f"      Provider: {config['provider']} | Model: {config['model']}")
                status = "Circuit Open" if info["circuit_open"] else "Active"
                print(f"      Status  : {status}")
                if usage and usage.get("total_requests", 0) > 0:
                    rate = usage["successful_requests"] / usage["total_requests"] * 100
                    print(f"      Requests: {usage['successful_requests']}/{usage['total_requests']} ({rate:.1f}%)")
                    print(f"      Tokens  : {usage['total_tokens']:,} | Cost: ${usage['total_cost']:.6f}")
                    print(f"      Latency : {usage['avg_latency_ms']:.0f}ms avg")
                else:
                    print(f"      Requests: 0 (unused)")

        for label, agent in [
            ("Host Agent", self.host_agent),
            ("PostDesign Agent", self.post_design_agent),
            ("Image Agent", self.image_generation_agent),
        ]:
            if agent:
                s = agent.get_status()
                print(f"\n{label}: {s['status']} | processed={s['processed_count']} | running={s['is_running']}")

        if self.host_agent:
            active = self.host_agent.get_active_requests_summary()
            print(f"\nActive Requests: {active['total_active']}")
            for req in active["requests"]:
                print(f"   {req['request_id'][:8]}... — {req['status']} ({req.get('workflow_type', 'single_agent')})")

            saga_summary = self.host_agent.get_saga_status_summary()
            print(f"\nActive Sagas: {saga_summary['active_sagas']}")
            for saga in saga_summary.get("sagas", []):
                print(f"   {saga['name']} ({saga['saga_id'][:8]}...) — {saga['status']} — Step {saga['current_step']}/{saga['total_steps']}")

        print("\n" + "=" * 70 + "\n")


    async def show_sagas(self):
        print("\n" + "=" * 70)
        print("SAGA WORKFLOWS")
        print("=" * 70 + "\n")

        if not self.host_agent or not hasattr(self.host_agent, 'saga_coordinator'):
            print("Saga coordinator not available\n")
            return

        sagas = self.host_agent.get_saga_status_summary().get("sagas", [])
        if not sagas:
            print("No active sagas\n")
            return

        for saga in sagas:
            print(f"Saga : {saga['name']}")
            print(f"  ID : {saga['saga_id']}")
            print(f"  Status   : {saga['status']}")
            print(f"  Progress : {saga['current_step']}/{saga['total_steps']} steps")
            if saga.get("started_at"):
                print(f"  Started  : {saga['started_at']}")
            for step_name, step_result in saga.get("step_results", {}).items():
                print(f"\n  Step: {step_name}")
                print(f"    Status : {step_result['status']}")
                if step_result.get("retries"):
                    print(f"    Retries: {step_result['retries']}")
                if step_result.get("error"):
                    print(f"    Error  : {step_result['error']}")
            print("\n" + "-" * 70 + "\n")

        print("=" * 70 + "\n")


    async def show_logs(self, lines: int = 20):
        print("\n" + "=" * 70)
        print(f"RECENT LOGS (last {lines} lines from {self.log_file})")
        print("=" * 70 + "\n")
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                recent = f.readlines()[-lines:]
            for line in recent:
                print(line.rstrip())
        except Exception as e:
            print(f"Cannot read log file: {e}")
        print("\n" + "=" * 70 + "\n")


    async def toggle_debug_mode(self):
        """Toggle debug logging — delegates to logging_config.toggle_debug()"""
        self.debug_mode = not self.debug_mode
        toggle_debug(self.debug_mode, self.log_file)
        state = "enabled" if self.debug_mode else "disabled"
        print(f"\nDebug mode {state}\n")


    async def run(self):
        """Main application loop"""
        try:
            await self.initialize()

            while self.running and not self.shutdown_in_progress:
                try:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue

                    if user_input.startswith("/"):
                        cmd = user_input.lower()
                        if cmd == "/exit":
                            print("\nShutting down...\n")
                            self.running = False
                        elif cmd == "/status":
                            await self.show_status()
                        elif cmd == "/sagas":
                            await self.show_sagas()
                        elif cmd == "/logs":
                            await self.show_logs()
                        elif cmd == "/debug":
                            await self.toggle_debug_mode()
                        elif cmd == "/help":
                            self.print_help()
                        elif cmd == "/clear":
                            os.system('cls' if os.name == 'nt' else 'clear')
                            self.print_help()
                        else:
                            print(f"Unknown command: {user_input}. Type /help for commands.\n")
                    else:
                        await self.send_message(user_input)

                except KeyboardInterrupt:
                    print("\n\nInterrupted. Shutting down...\n")
                    self.running = False
                except EOFError:
                    print("\n\nEOF. Shutting down...\n")
                    self.running = False
                except Exception as e:
                    print(f"\nError: {e}\n")
                    if self.debug_mode:
                        import traceback
                        traceback.print_exc()

        except KeyboardInterrupt:
            print("\n\nShutting down...\n")
        finally:
            await self.shutdown()


    async def shutdown(self):
        if self.shutdown_in_progress:
            return
        self.shutdown_in_progress = True
        print("Stopping agents...")

        agents = [a for a in [self.host_agent, self.post_design_agent, self.image_generation_agent] if a]
        try:
            await asyncio.wait_for(
                asyncio.gather(*[a.stop() for a in agents], return_exceptions=True),
                timeout=3.0
            )
        except asyncio.TimeoutError:
            print("Warning: Some agents did not stop within timeout")

        if self.message_bus:
            try:
                await asyncio.wait_for(self.message_bus.shutdown(), timeout=2.0)
            except asyncio.TimeoutError:
                print("Warning: Message bus shutdown timeout")

        print("Shutdown complete\n")


def main():
    import signal
    app = CLIApp()

    def signal_handler(sig, frame):
        print("\n\nSignal received. Shutting down...\n")
        app.running = False
        app.shutdown_in_progress = True

    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nExiting...\n")
    except Exception as e:
        print(f"\nUnexpected error: {e}\n")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()