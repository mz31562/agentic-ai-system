# 🎯 Marketing AI System

Multi-agent AI system for marketing content creation with intelligent LLM routing and image generation.

## Features

- **Smart Content Creation**: LinkedIn posts, Instagram captions, ad copy, email campaigns
- **Multi-LLM Support**: Automatic routing between Ollama (local), Groq (fast), Claude/GPT (premium)
- **Image Generation**: ComfyUI (local), DALL-E, Replicate support
- **Cost Optimization**: Intelligent backend selection based on task complexity

## Quick Start

### Prerequisites
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Setup Ollama (free local LLM)
# Download from: https://ollama.ai
ollama pull llama3
ollama pull mistral

# 3. Optional: Setup ComfyUI for images
# Or use DALL-E/Replicate APIs
```

### Configuration
```bash
# Copy and edit .env
cp .env.example .env

# Minimum config:
LLM_OLLAMA_ENABLED=true
IMAGE_BACKEND=dalle  # or comfyui
```

### Run
```bash
python main.py
```

## Example Usage
```
You: Create a LinkedIn post about AI in marketing

Agent: 🎯 AI is revolutionizing marketing...
[Generated content with proper formatting]

You: Generate an Instagram post with image about summer sale

Agent: [Creates post copy + generates image]
```

## Architecture
```
Core Components:
├── Host Agent      → Routes requests
├── PostDesign Agent → Content creation (multi-LLM)
└── Image Agent     → Visual generation

Message Bus → Event-driven communication
LLM Manager → Intelligent model selection
```

## Configuration Options

### LLM Backends

- **Ollama**: Free, local, unlimited
- **Groq**: Free tier, very fast
- **Claude**: Premium quality ($)
- **OpenAI**: GPT-4 ($)

### Image Backends

- **ComfyUI**: Local, SDXL/SD1.5
- **DALL-E**: OpenAI ($)
- **Replicate**: Cloud GPU ($)

## Development
```bash
# Run tests
pytest tests/

# Check types
mypy core/ agents/

# Format code
black .
```

## License

MIT