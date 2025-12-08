Local AI Content Generator
A multi-agent system that generates social media content (text + images) entirely on your local machine. No API costs, no data leaving your computer, no rate limits.
Why This Exists
Got tired of paying for AI API calls for every little thing. Built this to run everything locally - text generation via Ollama, images via Stable Diffusion XL, all coordinated through a custom message bus architecture.
The Reality:

Zero operating costs after setup
Everything runs on your hardware (RTX 4080 in my case)
No external API dependencies
Generate unlimited content without worrying about bills

How It Works
Different AI agents communicate through a message bus to handle requests:
Message Bus
    │
    ├── Host Agent (routes requests)
    ├── Post Writer (Llama3/Ollama)
    ├── Image Generator (SDXL/ComfyUI)
    └── Math Server (calculations)
Ask for "a post about coffee with an image" and the system:

Routes to both text and image agents
Runs them in parallel
Combines results
Done

Requirements
Hardware:

NVIDIA GPU with 8GB+ VRAM
16GB+ RAM
~10GB disk space for models

Software:

Python 3.10+
Ollama
ComfyUI
CUDA-capable GPU drivers

Quick Start
1. Install Ollama:
bash# Download from ollama.ai
ollama pull llama3
2. Setup ComfyUI:
bashgit clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
conda create -n comfyui python=3.10 -y
conda activate comfyui
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
Download SDXL model (6.5GB) to ComfyUI/models/checkpoints/
3. Setup Main System:
bashpython -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Configure .env as needed
4. Run:
Terminal 1:
bashconda activate comfyui
cd ComfyUI
python main.py --listen --port 8188
Terminal 2:
bashsource venv/bin/activate
python interfaces/cli_app.py
```

## What It Does

- Generates social media posts with platform-specific formatting
- Creates custom images using Stable Diffusion XL
- Handles mathematical calculations for educational content
- Coordinates multiple operations in parallel
- All processing happens locally

**Example requests:**
- "Create a LinkedIn post about remote work"
- "Generate an image of a mountain landscape"
- "Create a post about fibonacci sequence with explanation"
- "Write a Twitter thread about AI trends with header image"

## Performance

On RTX 4080:
- Text generation: 5-10s
- Image generation: 10-20s
- Combined: 15-25s (parallel)
- Math operations: <1s

No network latency, no rate limits, no per-request costs.

## Architecture

Built with async Python and message-passing patterns:

- **BaseAgent**: Foundation for all agents
- **Message Bus**: Async pub-sub for agent communication
- **Host Agent**: Request routing and orchestration
- **Specialist Agents**: Text, images, calculations
- **Flexible Backends**: Swap between local/cloud as needed

Each agent operates independently, communicates through messages, and can be scaled or replaced without affecting others.

## Project Structure
```
├── agents/                  # Agent implementations
├── core/                    # Message bus, base classes
├── mcp_server/             # Mathematical operations
├── interfaces/             # CLI and UI
├── ComfyUI/                # Image generation engine
├── generated_images/       # Output directory
└── requirements.txt
Configuration
Edit .env to configure:

Text generation backend (Ollama/Groq/Claude/Mock)
Image generation backend (ComfyUI/DALL-E/Replicate/Mock)
Model parameters
Output directories

Supports multiple backends for flexibility and testing.
Extending
Add new agents by:

Inheriting from BaseAgent
Implementing message handlers
Registering with message bus

The architecture makes it straightforward to add new capabilities without modifying existing code.
Technical Stack

Core: Python 3.10+, AsyncIO
Text: Ollama (Llama3)
Images: ComfyUI, Stable Diffusion XL, PyTorch + CUDA
Architecture: Message bus, agent pattern, async coordination

Limitations

Requires decent GPU for image generation
Initial setup has multiple steps
Generation speed depends on hardware
Models require significant disk space

This is production-ish code that works but isn't perfect. Good for personal use, learning, or as a starting point for something bigger.
What I Learned Building This

Agent-based architectures for AI coordination
Async Python patterns that scale
GPU resource management
Integrating multiple AI frameworks
Building systems that are actually extensible

Future Ideas

Web interface for easier interaction
Database for content history
Additional agents (research, code, video)
Workflow automation
Multi-GPU support

Contributing
PRs welcome for:

Performance improvements
New agent implementations
Better documentation
Easier setup process
Bug fixes

https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
