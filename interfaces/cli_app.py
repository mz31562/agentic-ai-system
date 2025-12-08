# C:\Users\MohammedZaid\Desktop\agentic-ai-system\interfaces\cli_app.py
import asyncio
import os
import sys
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from core.message_bus import MessageBus, Message
from agents.host_agent import HostAgent
from agents.post_design_agent import PostDesignAgent
from agents.image_generation_agent import ImageGenerationAgent
from mcp_server.math_server import MathMCPServer
from utils.comfyui_manager import ComfyUIManager
from core.llm_factory import create_llm_manager

load_dotenv()


class CLIApp:
    """Command-line interface for the agentic system"""
    
    def __init__(self):
        self.message_bus: Optional[MessageBus] = None
        self.host_agent: Optional[HostAgent] = None
        self.post_design_agent: Optional[PostDesignAgent] = None
        self.image_generation_agent: Optional[ImageGenerationAgent] = None
        self.math_server: Optional[MathMCPServer] = None
        self.comfyui_manager: Optional[ComfyUIManager] = None
        self.llm_manager = None  # NEW: Multi-LLM Manager
        self.response_queue = []
        self.running = True
        self.shutdown_in_progress = False
    
    
    async def initialize(self):
        """Initialize the agentic system"""
        print("\n" + "="*70)
        print("🤖 AGENTIC AI SYSTEM - CLI Interface")
        print("="*70 + "\n")
        
        print("🚀 Starting system...")
        
        # START COMFYUI IF NEEDED
        image_backend = os.getenv("IMAGE_BACKEND", "comfyui")
        if image_backend == "comfyui":
            print("\n📦 Checking ComfyUI status...")
            
            self.comfyui_manager = ComfyUIManager(
                comfyui_path=os.getenv("COMFYUI_PATH"),
                conda_env=os.getenv("COMFYUI_CONDA_ENV", "comfyui"),
                host=os.getenv("COMFYUI_HOST", "127.0.0.1"),
                port=int(os.getenv("COMFYUI_PORT", "8188"))
            )
            
            if not self.comfyui_manager.start(wait_for_ready=True, timeout=60):
                print("\n⚠️  WARNING: ComfyUI failed to start!")
                print("   Image generation will fall back to mock mode.")
                print("   You can:")
                print("   1. Start ComfyUI manually in another terminal")
                print("   2. Check that conda environment 'comfyui' exists")
                print("   3. Set IMAGE_BACKEND=mock in .env to skip ComfyUI\n")
                
                response = input("Continue anyway? [y/N]: ").strip().lower()
                if response != 'y':
                    print("Exiting...")
                    return False
        
        # Create message bus
        self.message_bus = MessageBus()
        
        # CREATE MULTI-LLM MANAGER (NEW!)
        print("\n🧠 Initializing LLM backends...")
        self.llm_manager = create_llm_manager()
        
        # Show available backends
        stats = self.llm_manager.get_stats()
        available_backends = [name for name, info in stats["backends"].items() 
                             if info["available"]]
        
        if available_backends:
            print(f"   ✅ Available LLM backends: {len(available_backends)}")
            for backend_name in available_backends:
                backend_info = stats["backends"][backend_name]
                config = backend_info["config"]
                provider_emoji = {
                    "ollama": "🏠",
                    "groq": "⚡",
                    "claude": "🧠",
                    "openai": "🤖"
                }.get(config['provider'], "📡")
                
                print(f"     {provider_emoji} {backend_name}")
                print(f"        Model: {config['model']}")
                if config['is_local']:
                    print(f"        Status: Local & Free ✅")
                else:
                    print(f"        Cost: ${config['cost_per_1k_tokens']:.4f}/1K tokens")
        else:
            print("   ⚠️  No LLM backends available!")
            print("   System will run in MOCK mode")
            print("   Enable backends in .env file")
        
        # Configure image backend
        image_config = {
            "backend": os.getenv("IMAGE_BACKEND", "comfyui"),
            "model": os.getenv("IMAGE_MODEL", "sdxl"),
            "api_key": os.getenv("IMAGE_API_KEY"),
            "output_dir": os.getenv("IMAGE_OUTPUT_DIR", "generated_images"),
            "comfyui_url": os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
        }
        
        # Create agents
        self.host_agent = HostAgent(self.message_bus)
        self.post_design_agent = PostDesignAgent(
            self.message_bus,
            self.llm_manager  # Pass the multi-LLM manager!
        )
        self.image_generation_agent = ImageGenerationAgent(
            self.message_bus,
            image_config=image_config
        )
        self.math_server = MathMCPServer(self.message_bus)
        
        # Subscribe to user responses
        self.message_bus.subscribe(
            topic="user_response",
            agent_id="cli_app",
            callback=self.handle_response
        )
        
        # Start all agents
        await self.host_agent.start()
        await self.post_design_agent.start()
        await self.image_generation_agent.start()
        await self.math_server.start()
        
        print("\n✅ System ready!")
        if available_backends:
            print(f"   LLM Backends: {len(available_backends)} active")
        print(f"   Image Backend: {image_config['backend']}")
        print(f"   Image Model: {image_config['model']}\n")
        
        self.print_help()
        
        return True
    
    
    async def handle_response(self, message: Message):
        """Handle responses from agents"""
        response_content = message.payload.get("result", message.payload.get("message", "No response"))
        self.response_queue.append(response_content)
    
    
    def print_help(self):
        """Print help message"""
        print("="*70)
        print("📖 COMMANDS:")
        print("="*70)
        print("  • Type your message to interact with agents")
        print("  • /status    - Show system status")
        print("  • /help      - Show this help message")
        print("  • /clear     - Clear screen")
        print("  • /exit      - Exit the application")
        print("  • Ctrl+C     - Quick exit")
        print("="*70 + "\n")
    
    
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
        
        # Wait for response
        print("\n⏳ Processing...\n")
        
        # Wait up to 5 minutes for response
        for i in range(600):  # 600 * 0.5 = 300 seconds
            await asyncio.sleep(0.5)
            if self.response_queue:
                break
            
            # Show progress every 30 seconds
            if i > 0 and i % 60 == 0:
                print(f"   Still processing... ({(i * 0.5):.0f}s elapsed)")
        
        # Display responses
        if self.response_queue:
            while self.response_queue:
                response = self.response_queue.pop(0)
                print("="*70)
                print("🤖 AGENT RESPONSE:")
                print("="*70)
                print(response)
                print("="*70 + "\n")
        else:
            print("⚠️ No response received (timeout)\n")
    
    
    async def show_status(self):
        """Show system status"""
        print("\n" + "="*70)
        print("📊 SYSTEM STATUS")
        print("="*70 + "\n")
        
        # ComfyUI Status
        if self.comfyui_manager:
            status = self.comfyui_manager.get_status()
            print(f"🖼️  ComfyUI Server:")
            print(f"   Running: {status['running']}")
            print(f"   URL: {status['url']}\n")
        
        # LLM Manager Stats (NEW!)
        if self.llm_manager:
            print(f"🧠 LLM Manager:")
            stats = self.llm_manager.get_stats()
            
            available_count = sum(1 for info in stats["backends"].values() if info["available"])
            print(f"   Total Backends: {len(stats['backends'])}")
            print(f"   Available: {available_count}")
            
            for backend_name, backend_info in stats["backends"].items():
                if backend_info["available"]:
                    config = backend_info["config"]
                    usage = backend_info.get("stats", {})
                    
                    provider_emoji = {
                        "ollama": "🏠",
                        "groq": "⚡",
                        "claude": "🧠",
                        "openai": "🤖"
                    }.get(config['provider'], "📡")
                    
                    print(f"\n   {provider_emoji} {backend_name}:")
                    print(f"      Provider: {config['provider']}")
                    print(f"      Model: {config['model']}")
                    print(f"      Status: {'🟢 Active' if not backend_info['circuit_open'] else '🔴 Circuit Open'}")
                    
                    if usage and usage.get('total_requests', 0) > 0:
                        success_rate = (usage.get('successful_requests', 0) / usage.get('total_requests', 1)) * 100
                        print(f"      Requests: {usage.get('successful_requests', 0)}/{usage.get('total_requests', 0)} ({success_rate:.1f}% success)")
                        print(f"      Tokens: {usage.get('total_tokens', 0):,}")
                        print(f"      Cost: ${usage.get('total_cost', 0):.6f}")
                        print(f"      Avg Latency: {usage.get('avg_latency_ms', 0):.0f}ms")
                    else:
                        print(f"      Requests: 0 (not yet used)")
            
            print()
        
        # Host Agent
        if self.host_agent:
            host_status = self.host_agent.get_status()
            print(f"🎯 Host Agent:")
            print(f"   Status: {host_status['status']}")
            print(f"   Running: {host_status['is_running']}")
            print(f"   Processed: {host_status['processed_count']} messages\n")
        
        # PostDesign Agent
        if self.post_design_agent:
            design_status = self.post_design_agent.get_status()
            print(f"🎨 PostDesign Agent:")
            print(f"   Status: {design_status['status']}")
            print(f"   Running: {design_status['is_running']}")
            print(f"   Processed: {design_status['processed_count']} messages\n")
        
        # Image Generation Agent
        if self.image_generation_agent:
            image_status = self.image_generation_agent.get_status()
            print(f"🖼️  Image Generation Agent:")
            print(f"   Status: {image_status['status']}")
            print(f"   Running: {image_status['is_running']}")
            print(f"   Processed: {image_status['processed_count']} messages\n")
        
        # Math Server
        if self.math_server:
            math_status = self.math_server.get_status()
            print(f"🔢 Math MCP Server:")
            print(f"   Status: {math_status['status']}")
            print(f"   Running: {math_status['is_running']}")
            print(f"   Processed: {math_status['processed_count']} messages\n")
        
        # Active Requests
        if self.host_agent:
            active = self.host_agent.get_active_requests_summary()
            print(f"📋 Active Requests: {active['total_active']}\n")
        
        print("="*70 + "\n")
    
    
    async def run(self):
        """Main application loop"""
        try:
            # Initialize system
            init_result = await self.initialize()
            if init_result == False:
                return
            
            while self.running and not self.shutdown_in_progress:
                try:
                    # Get user input
                    user_input = input("💬 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith("/"):
                        command = user_input.lower()
                        
                        if command == "/exit":
                            print("\n👋 Shutting down...\n")
                            self.running = False
                            break
                        
                        elif command == "/status":
                            await self.show_status()
                        
                        elif command == "/help":
                            self.print_help()
                        
                        elif command == "/clear":
                            os.system('cls' if os.name == 'nt' else 'clear')
                            self.print_help()
                        
                        else:
                            print(f"❌ Unknown command: {user_input}")
                            print("💡 Type /help for available commands\n")
                    
                    else:
                        # Regular message
                        await self.send_message(user_input)
                
                except KeyboardInterrupt:
                    print("\n\n👋 Interrupted. Shutting down...\n")
                    self.running = False
                    break
                
                except EOFError:
                    # Handle Ctrl+D or Ctrl+Z
                    print("\n\n👋 EOF detected. Shutting down...\n")
                    self.running = False
                    break
                
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Keyboard interrupt. Shutting down...\n")
        
        finally:
            # Cleanup
            await self.shutdown()
    
    
    async def shutdown(self):
        """Shutdown the system gracefully"""
        if self.shutdown_in_progress:
            return
        
        self.shutdown_in_progress = True
        print("🛑 Stopping agents...")
        
        try:
            # Stop all agents with timeout
            shutdown_tasks = []
            
            if self.host_agent:
                shutdown_tasks.append(self.host_agent.stop())
            if self.post_design_agent:
                shutdown_tasks.append(self.post_design_agent.stop())
            if self.image_generation_agent:
                shutdown_tasks.append(self.image_generation_agent.stop())
            if self.math_server:
                shutdown_tasks.append(self.math_server.stop())
            
            # Wait for all agents to stop with timeout
            if shutdown_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*shutdown_tasks, return_exceptions=True),
                        timeout=3.0
                    )
                except asyncio.TimeoutError:
                    print("⚠️  Some agents took too long to stop")
            
            # Shutdown message bus
            if self.message_bus:
                try:
                    await asyncio.wait_for(self.message_bus.shutdown(), timeout=2.0)
                except asyncio.TimeoutError:
                    print("⚠️  Message bus shutdown timed out")
            
            # STOP COMFYUI
            if self.comfyui_manager:
                print("🛑 Stopping ComfyUI server...")
                self.comfyui_manager.stop()
            
            print("✅ Shutdown complete!\n")
            
        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}\n")


def main():
    """Main entry point with signal handling"""
    app = CLIApp()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n👋 Signal received. Shutting down gracefully...\n")
        app.running = False
        app.shutdown_in_progress = True
    
    # Register signal handlers (Unix/Linux/Mac)
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\n👋 Exiting...\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()