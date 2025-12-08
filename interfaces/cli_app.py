import asyncio
import os
import sys
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

load_dotenv()


class CLIApp:
    """Command-line interface for the agentic system"""
    
    def __init__(self):
        self.message_bus: Optional[MessageBus] = None
        self.host_agent: Optional[HostAgent] = None
        self.post_design_agent: Optional[PostDesignAgent] = None
        self.image_generation_agent: Optional[ImageGenerationAgent] = None
        self.math_server: Optional[MathMCPServer] = None
        self.response_queue = []
        self.running = True
    
    
    async def initialize(self):
        """Initialize the agentic system"""
        print("\n" + "="*70)
        print("🤖 AGENTIC AI SYSTEM - CLI Interface")
        print("="*70 + "\n")
        
        print("🚀 Starting system...")
        
        # Create message bus
        self.message_bus = MessageBus()
        
        # Configure LLM backend
        llm_config = {
            "backend": os.getenv("LLM_BACKEND", "ollama"),
            "model": os.getenv("LLM_MODEL", "llama3"),
            "api_key": os.getenv("LLM_API_KEY"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        }
        
        # Configure image backend
        image_config = {
            "backend": os.getenv("IMAGE_BACKEND", "mock"),
            "model": os.getenv("IMAGE_MODEL", "dall-e-3"),
            "api_key": os.getenv("IMAGE_API_KEY"),
            "output_dir": os.getenv("IMAGE_OUTPUT_DIR", "generated_images")
        }
        
        # Create agents
        self.host_agent = HostAgent(self.message_bus)
        self.post_design_agent = PostDesignAgent(
            self.message_bus,
            llm_config=llm_config
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
        
        print("✅ System ready!")
        print(f"   LLM Backend: {llm_config['backend']}")
        print(f"   LLM Model: {llm_config['model']}")
        print(f"   Image Backend: {image_config['backend']}")
        print(f"   Image Model: {image_config['model']}\n")
        self.print_help()
    
    
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
        
        # Wait up to 10 seconds for responsey
        for _ in range(600):  # 600 * 0.5 = 300 seconds (5 minutes)
            await asyncio.sleep(0.5)
            if self.response_queue:
                break
            
            # Show progress every 30 seconds
            if _ > 0 and _ % 60 == 0:
                print(f"   Still processing... ({(_ * 0.5):.0f}s elapsed)")
        
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
        
        # Host Agent
        host_status = self.host_agent.get_status()
        print(f"🎯 Host Agent:")
        print(f"   Status: {host_status['status']}")
        print(f"   Running: {host_status['is_running']}")
        print(f"   Processed: {host_status['processed_count']} messages\n")
        
        # PostDesign Agent
        design_status = self.post_design_agent.get_status()
        print(f"🎨 PostDesign Agent:")
        print(f"   Status: {design_status['status']}")
        print(f"   Running: {design_status['is_running']}")
        print(f"   Processed: {design_status['processed_count']} messages\n")
        
        # Image Generation Agent
        image_status = self.image_generation_agent.get_status()
        print(f"🖼️  Image Generation Agent:")
        print(f"   Status: {image_status['status']}")
        print(f"   Running: {image_status['is_running']}")
        print(f"   Processed: {image_status['processed_count']} messages\n")
        
        # Math Server
        math_status = self.math_server.get_status()
        print(f"🔢 Math MCP Server:")
        print(f"   Status: {math_status['status']}")
        print(f"   Running: {math_status['is_running']}")
        print(f"   Processed: {math_status['processed_count']} messages\n")
        
        # Active Requests
        active = self.host_agent.get_active_requests_summary()
        print(f"📋 Active Requests: {active['total_active']}")
        
        # LLM Backend
        print(f"\n🧠 LLM Backend:")
        print(f"   Backend: {os.getenv('LLM_BACKEND', 'ollama')}")
        print(f"   Model: {os.getenv('LLM_MODEL', 'llama3')}")
        
        # Image Backend
        print(f"\n🎨 Image Backend:")
        print(f"   Backend: {os.getenv('IMAGE_BACKEND', 'mock')}")
        print(f"   Model: {os.getenv('IMAGE_MODEL', 'dall-e-3')}")
        print(f"   Output: {os.getenv('IMAGE_OUTPUT_DIR', 'generated_images')}")
        
        print("="*70 + "\n")
    
    
    async def run(self):
        """Main application loop"""
        await self.initialize()
        
        while self.running:
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
            
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
        # Cleanup
        await self.shutdown()
    
    
    async def shutdown(self):
        """Shutdown the system"""
        print("🛑 Stopping agents...")
        
        if self.host_agent:
            await self.host_agent.stop()
        if self.post_design_agent:
            await self.post_design_agent.stop()
        if self.image_generation_agent:
            await self.image_generation_agent.stop()
        if self.math_server:
            await self.math_server.stop()
        if self.message_bus:
            await self.message_bus.shutdown()
        
        print("✅ Shutdown complete!\n")


def main():
    """Main entry point"""
    app = CLIApp()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()