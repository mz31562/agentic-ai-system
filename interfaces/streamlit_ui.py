import streamlit as st
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from core.message_bus import MessageBus, Message
from agents.host_agent import HostAgent
from agents.post_design_agent import PostDesignAgent
from mcp_server.math_server import MathMCPServer  # Now imported

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-status {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-active {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .status-inactive {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .message-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .message-agent {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitUI:
    """Streamlit-based user interface for the agentic system"""
    
    def __init__(self):
        self.initialize_session_state()
    
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
            st.session_state.message_bus = None
            st.session_state.host_agent = None
            st.session_state.post_design_agent = None
            st.session_state.math_server = None  # Now active
            st.session_state.messages = []
            st.session_state.user_response_queue = []
    
    
    async def initialize_system(self):
        """Initialize the agentic system"""
        if not st.session_state.system_initialized:
            with st.spinner("🚀 Starting agentic system..."):
                # Create message bus
                st.session_state.message_bus = MessageBus()
                
                # Create agents
                st.session_state.host_agent = HostAgent(st.session_state.message_bus)
                
                # Configure LLM backend for PostDesignAgent
                llm_config = {
                    "backend": os.getenv("LLM_BACKEND", "ollama"),
                    "model": os.getenv("LLM_MODEL", "llama3"),
                    "api_key": os.getenv("LLM_API_KEY"),
                    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                }
                
                st.session_state.post_design_agent = PostDesignAgent(
                    st.session_state.message_bus,
                    llm_config=llm_config
                )
                st.session_state.math_server = MathMCPServer(st.session_state.message_bus)
                
                # Subscribe UI to user responses
                st.session_state.message_bus.subscribe(
                    topic="user_response",
                    agent_id="streamlit_ui",
                    callback=self.handle_user_response
                )
                
                # Start all agents
                await st.session_state.host_agent.start()
                await st.session_state.post_design_agent.start()
                await st.session_state.math_server.start()
                
                st.session_state.system_initialized = True
    
    
    async def handle_user_response(self, message: Message):
        """Handle responses from the system"""
        logger.info(f"UI received response: {message.payload}")
        response_data = {
            "type": "agent",
            "content": message.payload.get("result", message.payload.get("message", "No response")),
            "metadata": message.payload.get("metadata", {}),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.user_response_queue.append(response_data)
        logger.info(f"Response queued. Queue length: {len(st.session_state.user_response_queue)}")
    
    
    async def send_message(self, user_message: str):
        """Send user message to the system"""
        # Add user message to chat
        st.session_state.messages.append({
            "type": "user",
            "content": user_message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Create and publish message
        message = Message(
            type="request",
            sender="streamlit_ui",
            topic="user_request",
            payload={
                "user_id": "streamlit_user",
                "message": user_message
            }
        )
        
        await st.session_state.message_bus.publish(message)
        
        # Wait for response with longer timeout
        await asyncio.sleep(3)
        
        # Process any queued responses
        while st.session_state.user_response_queue:
            response = st.session_state.user_response_queue.pop(0)
            st.session_state.messages.append(response)
        
        # If no response received, add a timeout message
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["type"] == "user":
            st.session_state.messages.append({
                "type": "agent",
                "content": "⚠️ Processing... (Response pending)",
                "metadata": {},
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
    
    
    def render_sidebar(self):
        """Render the sidebar with system status"""
        with st.sidebar:
            st.markdown("### 🎛️ System Control")
            
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
            
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 📊 System Status")
            
            if st.session_state.system_initialized:
                # LLM Backend Info
                backend = os.getenv("LLM_BACKEND", "ollama")
                model = os.getenv("LLM_MODEL", "llama3")
                st.markdown(f"""
                <div class="agent-status status-active">
                    <strong>🧠 LLM Backend</strong><br>
                    Backend: {backend}<br>
                    Model: {model}
                </div>
                """, unsafe_allow_html=True)
                
                # Host Agent Status
                host_status = st.session_state.host_agent.get_status()
                status_class = "status-active" if host_status['is_running'] else "status-inactive"
                st.markdown(f"""
                <div class="agent-status {status_class}">
                    <strong>🎯 Host Agent</strong><br>
                    Status: {host_status['status']}<br>
                    Processed: {host_status['processed_count']} msgs
                </div>
                """, unsafe_allow_html=True)
                
                # PostDesign Agent Status
                design_status = st.session_state.post_design_agent.get_status()
                status_class = "status-active" if design_status['is_running'] else "status-inactive"
                st.markdown(f"""
                <div class="agent-status {status_class}">
                    <strong>🎨 PostDesign Agent</strong><br>
                    Status: {design_status['status']}<br>
                    Processed: {design_status['processed_count']} msgs
                </div>
                """, unsafe_allow_html=True)
                
                # Math Server Status
                math_status = st.session_state.math_server.get_status()
                status_class = "status-active" if math_status['is_running'] else "status-inactive"
                st.markdown(f"""
                <div class="agent-status {status_class}">
                    <strong>🔢 Math MCP Server</strong><br>
                    Status: {math_status['status']}<br>
                    Processed: {math_status['processed_count']} msgs
                </div>
                """, unsafe_allow_html=True)
                
                # Active Requests
                st.markdown("---")
                st.markdown("### 📋 Active Requests")
                active_requests = st.session_state.host_agent.get_active_requests_summary()
                st.metric("Total Active", active_requests['total_active'])
            
            else:
                st.info("System not initialized")
            
            st.markdown("---")
            st.markdown("### ℹ️ About")
            st.markdown("""
            **Agentic AI System**
            
            A multi-agent system with:
            - Host Agent (coordinator)
            - PostDesign Agent (creator)
            - Math MCP Server (calculator)
            
            Built with Python & Streamlit
            """)
    
    
    def render_chat(self):
        """Render the chat interface"""
        st.markdown('<div class="main-header">🤖 Agentic AI System</div>', unsafe_allow_html=True)
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display messages
            for msg in st.session_state.messages:
                if msg['type'] == 'user':
                    st.markdown(f"""
                    <div class="message-user">
                        <strong>👤 You</strong> <small>({msg['timestamp']})</small><br>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="message-agent">
                        <strong>🤖 Agent</strong> <small>({msg['timestamp']})</small><br>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Input area
        st.markdown("---")
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Type your message here... (e.g., 'Create a post about fibonacci')",
                label_visibility="collapsed",
                key="user_input"
            )
        
        with col2:
            send_button = st.button("Send 📤", use_container_width=True)
        
        # Handle send
        if send_button and user_input:
            # Use asyncio.run for proper async execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send_message(user_input))
            loop.close()
            st.rerun()
    
    
    def render_examples(self):
        """Render example prompts"""
        st.markdown("### 💡 Try these examples:")
        
        examples = [
            "Create a post about fibonacci sequence",
            "Design a post about machine learning",
            "Show me system status",
            "Calculate factorial of 5",
        ]
        
        cols = st.columns(2)
        for idx, example in enumerate(examples):
            with cols[idx % 2]:
                if st.button(example, use_container_width=True, key=f"example_{idx}"):
                    # Use asyncio.run for proper async execution
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.send_message(example))
                    loop.close()
                    st.rerun()


def main():
    """Main application entry point"""
    ui = StreamlitUI()
    
    # Initialize system
    if not st.session_state.system_initialized:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ui.initialize_system())
        loop.close()
        st.rerun()
    
    # Render UI
    ui.render_sidebar()
    ui.render_chat()
    
    # Show examples if no messages
    if len(st.session_state.messages) == 0:
        ui.render_examples()


if __name__ == "__main__":
    main()