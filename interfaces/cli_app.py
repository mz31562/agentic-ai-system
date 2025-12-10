import asyncio
import os
import sys
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"system_{datetime.now().strftime('%Y%m%d')}.log"

if DEBUG_MODE:
    console_level = logging.DEBUG
    console_format = '%(levelname)-8s | %(name)-25s | %(message)s'
else:
    console_level = logging.WARNING
    console_format = '%(message)s'

file_format = '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s'

logging.basicConfig(
    level=logging.DEBUG,
    format=file_format,
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

root_logger = logging.getLogger()
if not DEBUG_MODE:
    root_logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(console_format))
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

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
        print("\n" + "="*70)
        print("AGENTIC AI SYSTEM - CLI Interface")
        print("="*70 + "\n")
        
        if not self.debug_mode:
            print(f"Log file: {self.log_file}")
            print(f"Enable DEBUG_MODE=true in .env for verbose output\n")
        else:
            print(f"DEBUG MODE ENABLED - Verbose logging active")
            print(f"Logs saved to: {self.log_file}\n")
        
        print("Initializing system components...")
        
        self.message_bus = MessageBus()
        
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
                    print(f"   Start ComfyUI or configure IMAGE_BACKEND=dalle in .env")
            except:
                print(f"   Warning: Cannot connect to ComfyUI at {comfyui_url}")
                print(f"   Start ComfyUI or configure IMAGE_BACKEND=dalle in .env")
        
        print("\nInitializing LLM backends...")
        self.llm_manager = create_llm_manager()
        
        stats = self.llm_manager.get_stats()
        available_backends = [name for name, info in stats["backends"].items() 
                             if info["available"]]
        
        if available_backends:
            print(f"Active LLM backends: {len(available_backends)}")
            for backend_name in available_backends:
                backend_info = stats["backends"][backend_name]
                config = backend_info["config"]
                
                print(f"   {backend_name}")
                print(f"      Model: {config['model']}")
                if config['is_local']:
                    print(f"      Location: Local (Free)")
                else:
                    print(f"      Cost: ${config['cost_per_1k_tokens']:.4f}/1K tokens")
        else:
            print("Warning: No LLM backends available")
            print("   System will operate in mock mode")
            print("   Configure backends in .env file")
        
        image_config = {
            "backend": os.getenv("IMAGE_BACKEND", "comfyui"),
            "model": os.getenv("IMAGE_MODEL", "sdxl"),
            "api_key": os.getenv("IMAGE_API_KEY"),
            "output_dir": os.getenv("IMAGE_OUTPUT_DIR", "generated_images"),
            "comfyui_url": os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
        }
        
        self.host_agent = HostAgent(self.message_bus)
        self.post_design_agent = PostDesignAgent(
            self.message_bus,
            self.llm_manager
        )
        self.image_generation_agent = ImageGenerationAgent(
            self.message_bus,
            image_config=image_config
)
        
        self.message_bus.subscribe(
            topic="user_response",
            agent_id="cli_app",
            callback=self.handle_response
        )
        
        await self.host_agent.start()
        await self.post_design_agent.start()
        await self.image_generation_agent.start()
        
        print("\nSystem ready")
        if available_backends:
            print(f"   LLM Backends: {len(available_backends)} active")
        print(f"   Image Backend: {image_config['backend']}")
        print(f"   Image Model: {image_config['model']}")
        print(f"   Saga Orchestration: Enabled\n")
        
        self.print_help()
        
        return True
    
    
    async def handle_response(self, message: Message):
        """Handle responses from agents"""
        response_content = message.payload.get("result", message.payload.get("message", "No response"))
        self.response_queue.append(response_content)
    
    
    def print_help(self):
        """Print help message"""
        print("="*70)
        print("AVAILABLE COMMANDS:")
        print("="*70)
        print("  Type your message to interact with agents")
        print("  /status    - Display system status (including active sagas)")
        print("  /sagas     - Display detailed saga information")
        print("  /logs      - Display recent log entries")
        print("  /debug     - Toggle debug mode (verbose output)")
        print("  /help      - Display this help message")
        print("  /clear     - Clear screen")
        print("  /exit      - Exit application")
        print("  Ctrl+C     - Quick exit")
        print("="*70)
    
    
    async def send_message(self, user_message: str):
        """Send message to the system"""
        message = Message(
            type="request",
            sender="cli_app",
            topic="user_request",
            payload={
                "user_id": "cli_user",
                "message": user_message
            }
        )
        
        await self.message_bus.publish(message)
        
        print("\nProcessing request...\n")
        
        spinner = ['|', '/', '-', '\\']
        spinner_idx = 0
        
        for i in range(600):  # 600 * 0.5 = 300 seconds
            await asyncio.sleep(0.5)
            
            if self.response_queue:
                print("\rProcessing complete" + " "*20, flush=True)
                print()
                break
            
            if not self.debug_mode:
                if i % 2 == 0:
                    elapsed = i * 0.5
                    print(f"\r{spinner[spinner_idx]} Processing... ({elapsed:.0f}s)", end="", flush=True)
                    spinner_idx = (spinner_idx + 1) % len(spinner)
            
            if i > 0 and i % 60 == 0:
                if not self.debug_mode:
                    print()
                
                if self.host_agent and hasattr(self.host_agent, 'saga_coordinator'):
                    active_sagas = self.host_agent.saga_coordinator.get_all_active_sagas()
                    if active_sagas:
                        for saga in active_sagas:
                            print(f"   {saga['name']}: Step {saga['current_step']}/{saga['total_steps']}")
        else:
            if not self.debug_mode:
                print("\rRequest timeout" + " "*20, flush=True)
            print()
        
        if self.response_queue:
            while self.response_queue:
                response = self.response_queue.pop(0)
                print("="*70)
                print("RESPONSE:")
                print("="*70)
                print(response)
                print("="*70 + "\n")
        else:
            print("No response received (timeout)")
            print(f"Check {self.log_file} for details\n")
    
    
    async def show_status(self):
        """Display system status"""
        print("\n" + "="*70)
        print("SYSTEM STATUS")
        print("="*70 + "\n")
        
        if self.llm_manager:
            print(f"LLM Manager:")
            stats = self.llm_manager.get_stats()
            
            available_count = sum(1 for info in stats["backends"].values() if info["available"])
            print(f"   Total Backends: {len(stats['backends'])}")
            print(f"   Available: {available_count}")
            
            budget = stats.get("budget", {})
            if budget:
                daily_spend = budget.get("daily_spend", 0)
                daily_budget = budget.get("daily_budget", 5.0)
                percent_used = budget.get("daily_percent_used", 0)
                
                print(f"\n   Budget Status:")
                print(f"      Daily Spend: ${daily_spend:.4f} / ${daily_budget:.2f} ({percent_used:.1f}%)")
                print(f"      Remaining: ${budget.get('daily_remaining', 0):.4f}")
                print(f"      Total Requests Today: {budget.get('total_requests_today', 0)}")
            
            for backend_name, backend_info in stats["backends"].items():
                if backend_info["available"]:
                    config = backend_info["config"]
                    usage = backend_info.get("stats", {})
                    
                    print(f"\n   {backend_name}:")
                    print(f"      Provider: {config['provider']}")
                    print(f"      Model: {config['model']}")
                    print(f"      Status: {'Active' if not backend_info['circuit_open'] else 'Circuit Open'}")
                    
                    if usage and usage.get('total_requests', 0) > 0:
                        success_rate = (usage.get('successful_requests', 0) / usage.get('total_requests', 1)) * 100
                        print(f"      Requests: {usage.get('successful_requests', 0)}/{usage.get('total_requests', 0)} ({success_rate:.1f}% success)")
                        print(f"      Tokens: {usage.get('total_tokens', 0):,}")
                        print(f"      Cost: ${usage.get('total_cost', 0):.6f}")
                        print(f"      Avg Latency: {usage.get('avg_latency_ms', 0):.0f}ms")
                    else:
                        print(f"      Requests: 0 (unused)")
            
            print()
        
        if self.host_agent:
            host_status = self.host_agent.get_status()
            print(f"Host Agent:")
            print(f"   Status: {host_status['status']}")
            print(f"   Running: {'Yes' if host_status['is_running'] else 'No'}")
            print(f"   Processed: {host_status['processed_count']} messages\n")
        
        if self.post_design_agent:
            design_status = self.post_design_agent.get_status()
            print(f"PostDesign Agent:")
            print(f"   Status: {design_status['status']}")
            print(f"   Running: {'Yes' if design_status['is_running'] else 'No'}")
            print(f"   Processed: {design_status['processed_count']} messages\n")
        
        if self.image_generation_agent:
            image_status = self.image_generation_agent.get_status()
            print(f"Image Generation Agent:")
            print(f"   Status: {image_status['status']}")
            print(f"   Running: {'Yes' if image_status['is_running'] else 'No'}")
            print(f"   Processed: {image_status['processed_count']} messages\n")
        
        if self.host_agent:
            active = self.host_agent.get_active_requests_summary()
            print(f"Active Requests: {active['total_active']}")
            
            if active['total_active'] > 0:
                for req in active['requests']:
                    print(f"   {req['request_id'][:8]}... - {req['status']} ({req.get('workflow_type', 'single_agent')})")
            
            print()
        
        if self.host_agent and hasattr(self.host_agent, 'saga_coordinator'):
            saga_summary = self.host_agent.get_saga_status_summary()
            active_sagas = saga_summary.get('active_sagas', 0)
            
            print(f"Active Sagas: {active_sagas}")
            
            if active_sagas > 0:
                for saga in saga_summary.get('sagas', []):
                    print(f"\n   Saga: {saga['name']} ({saga['saga_id'][:8]}...)")
                    print(f"      Status: {saga['status']}")
                    print(f"      Progress: Step {saga['current_step']}/{saga['total_steps']}")
                    
                    if saga.get('step_results'):
                        print(f"      Steps:")
                        for step_name, step_result in saga['step_results'].items():
                            status_text = step_result['status'].title()
                            print(f"         {step_name}: {status_text}", end="")
                            
                            if step_result.get('retries', 0) > 0:
                                print(f" (retried {step_result['retries']}x)", end="")
                            
                            if step_result.get('error'):
                                print(f"\n            Error: {step_result['error'][:50]}...", end="")
                            
                            print()
            
            print()
        
        print("="*70 + "\n")
    
    
    async def show_sagas(self):
        """Display detailed saga information"""
        print("\n" + "="*70)
        print("SAGA WORKFLOWS")
        print("="*70 + "\n")
        
        if not self.host_agent or not hasattr(self.host_agent, 'saga_coordinator'):
            print("Saga coordinator not available\n")
            return
        
        saga_summary = self.host_agent.get_saga_status_summary()
        active_sagas = saga_summary.get('sagas', [])
        
        if not active_sagas:
            print("No active sagas\n")
            return
        
        for saga in active_sagas:
            print(f"Saga: {saga['name']}")
            print(f"   ID: {saga['saga_id']}")
            print(f"   Status: {saga['status']}")
            print(f"   Progress: {saga['current_step']}/{saga['total_steps']} steps")
            
            if saga.get('started_at'):
                print(f"   Started: {saga['started_at']}")
            
            if saga.get('step_results'):
                print(f"\n   Step Details:")
                for step_name, step_result in saga['step_results'].items():
                    status_text = step_result['status'].title()
                    
                    print(f"\n   Step: {step_name}")
                    print(f"      Status: {status_text}")
                    
                    if step_result.get('retries', 0) > 0:
                        print(f"      Retries: {step_result['retries']}")
                    
                    if step_result.get('error'):
                        print(f"      Error: {step_result['error']}")
            
            print("\n" + "-"*70 + "\n")
        
        print("="*70 + "\n")
    
    
    async def show_logs(self, lines: int = 20):
        """Display recent log entries"""
        print("\n" + "="*70)
        print(f"RECENT LOGS (last {lines} lines)")
        print("="*70 + "\n")
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
                recent = log_lines[-lines:]
                
                for line in recent:
                    print(line.rstrip())
        
        except Exception as e:
            print(f"Cannot read log file: {e}")
        
        print("\n" + "="*70 + "\n")
    
    
    async def toggle_debug(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        if self.debug_mode:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(logging.Formatter(
                '%(levelname)-8s | %(name)-25s | %(message)s'
            ))
            root_logger.addHandler(console_handler)
            print("\nDebug mode enabled - Verbose logging active\n")
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(logging.Formatter('%(message)s'))
            root_logger.addHandler(console_handler)
            print("\nDebug mode disabled\n")
        
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.addHandler(file_handler)
    
    
    async def run(self):
        """Main application loop"""
        try:
            init_result = await self.initialize()
            if init_result == False:
                return
            
            while self.running and not self.shutdown_in_progress:
                try:
                    user_input = input("You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.startswith("/"):
                        command = user_input.lower()
                        
                        if command == "/exit":
                            print("\nShutting down...\n")
                            self.running = False
                            break
                        
                        elif command == "/status":
                            await self.show_status()
                        
                        elif command == "/sagas":
                            await self.show_sagas()
                        
                        elif command == "/logs":
                            await self.show_logs()
                        
                        elif command == "/debug":
                            await self.toggle_debug()
                        
                        elif command == "/help":
                            self.print_help()
                        
                        elif command == "/clear":
                            os.system('cls' if os.name == 'nt' else 'clear')
                            self.print_help()
                        
                        else:
                            print(f"Unknown command: {user_input}")
                            print("Type /help for available commands\n")
                    
                    else:
                        await self.send_message(user_input)
                
                except KeyboardInterrupt:
                    print("\n\nInterrupted. Shutting down...\n")
                    self.running = False
                    break
                
                except EOFError:
                    print("\n\nEOF detected. Shutting down...\n")
                    self.running = False
                    break
                
                except Exception as e:
                    print(f"\nError: {e}\n")
                    if self.debug_mode:
                        import traceback
                        traceback.print_exc()
        
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt. Shutting down...\n")
        
        finally:
            await self.shutdown()
    
    
    async def shutdown(self):
        """Shutdown the system"""
        if self.shutdown_in_progress:
            return
        
        self.shutdown_in_progress = True
        print("Stopping agents...")
        
        try:
            shutdown_tasks = []
            
            if self.host_agent:
                shutdown_tasks.append(self.host_agent.stop())
            if self.post_design_agent:
                shutdown_tasks.append(self.post_design_agent.stop())
            if self.image_generation_agent:
                shutdown_tasks.append(self.image_generation_agent.stop())
            
            if shutdown_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*shutdown_tasks, return_exceptions=True),
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
            
        except Exception as e:
            print(f"Error during shutdown: {e}\n")


def main():
    """Main entry point"""
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