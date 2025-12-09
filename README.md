# 🤖 Agentic AI Marketing System

A sophisticated multi-agent AI system for generating high-quality marketing content and images. Intelligently routes requests between specialized AI agents and supports multiple LLM backends with automatic failover.

## ✨ Features

### 🧠 Multi-Agent Architecture
- **Host Agent**: Central coordinator that routes requests to specialist agents
- **PostDesign Agent**: Creates marketing copy, social media posts, email campaigns, and sales content
- **Image Generation Agent**: Generates images with ComfyUI, DALL-E, Replicate, or Segmind backends

### 🔄 Intelligent LLM Backend Selection
- **Multiple LLM Providers**: Groq, Ollama (local), OpenAI, Claude, HuggingFace
- **Unbiased Backend Routing**: Selects optimal backend based on task type, complexity, cost, and speed preferences
- **Automatic Failover**: Falls back to next best backend if one fails
- **Circuit Breaker**: Temporarily disables failing services
- **Cost-Aware**: Respects budget constraints per request

### 🎨 Advanced Image Generation
- **Multiple Backends**: ComfyUI (local SDXL), DALL-E, Replicate, Segmind
- **Text-in-Image Optimization**: Detects and optimizes prompts for text generation in images
- **Robust Retry Logic**: Automatic retries with exponential backoff
- **Quality Enhancement**: Intelligently enhances prompts for better results
- **Fallback Mode**: Mock generation if no backend available

### 📱 Content Type Support
- LinkedIn posts (professional)
- Instagram captions (visual storytelling)
- Twitter/X threads (concise, engaging)
- Email campaigns (conversational)
- Blog articles (long-form)
- Ad copy (benefit-driven)
- Product descriptions
- Sales pitches

### 🎯 Tone & Style Detection
Automatically detects or allows specification of:
- Professional, casual, enthusiastic
- Educational, persuasive, inspiring
- Humorous, urgent

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.9+
python --version

# Clone and navigate to project
cd agentic-ai-system
```

### Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure `.env` file** with your API keys and preferences:
```bash
# Core LLM backends (choose at least one)
LLM_GROQ_ENABLED=true
LLM_GROQ_API_KEY=your_groq_api_key

LLM_OPENAI_ENABLED=true
LLM_OPENAI_API_KEY=your_openai_api_key

# OR use free local Ollama
LLM_OLLAMA_ENABLED=true
LLM_OLLAMA_BASE_URL=http://localhost:11434

# Image generation
IMAGE_BACKEND=comfyui  # or dalle, replicate, segmind
COMFYUI_URL=http://127.0.0.1:8188  # if using ComfyUI
```

3. **Optional: Start ComfyUI** (if using local image generation):
```bash
cd ComfyUI
python main.py --listen --port 8188
```

4. **Run the CLI**:
```bash
python main.py
```

## 📖 Usage Examples

### Example 1: Generate a LinkedIn Post
```
You: Create a professional LinkedIn post about AI in marketing
Agent: [Generates thought-leadership post using optimal LLM]
```

### Example 2: Create Instagram Post with Image
```
You: Design an Instagram post with an image about digital marketing trends
Host Agent: Routes to both PostDesign and Image agents
Result: Post copy + generated image
```

### Example 3: Email Campaign
```
You: Write a persuasive email campaign about our new SaaS product
PostDesign Agent: Creates benefit-focused email with strong CTA
```

### Example 4: Generate Product Image
```
You: Create an image of a sleek modern laptop in a minimalist workspace
Image Agent: Generates high-quality product image using SDXL
```

## 🔧 Configuration

### LLM Backend Settings (`.env`)

```bash
# Backend Priority (comma-separated, first = highest priority)
LLM_BACKEND_PRIORITY=groq,ollama,openai

# Prefer fast responses (recommended for marketing)
LLM_PREFER_FAST=true

# Prefer local models (privacy-focused)
LLM_PREFER_LOCAL=false

# Maximum cost per request (USD)
LLM_MAX_COST_PER_REQUEST=0.01

# Allow paid backends when needed
LLM_ALLOW_PAID=true
```

### Available LLM Backends

| Backend | Cost | Speed | Quality | Local | Setup |
|---------|------|-------|---------|-------|-------|
| **Groq** | Free | ⚡⚡⚡ | ★★★★☆ | ❌ | API Key |
| **Ollama** | Free | ⚡⚡ | ★★★☆☆ | ✅ | Local Install |
| **OpenAI** | $$ | ⚡⚡ | ★★★★★ | ❌ | API Key |
| **Claude** | $$$ | ⚡ | ★★★★★ | ❌ | API Key |

### Image Generation Settings

```bash
IMAGE_BACKEND=comfyui  # comfyui, dalle, replicate, segmind
IMAGE_MODEL=sdxl       # Model to use
COMFYUI_URL=http://127.0.0.1:8188
DEFAULT_IMAGE_SIZE=1024x1024
IMAGE_MAX_RETRIES=2
IMAGE_TIMEOUT_SECONDS=300
ENABLE_TEXT_OPTIMIZATION=true  # Optimize for text-in-image
```

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI/User Interface                    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                    Message Bus                           │
│         (Async Pub/Sub Communication Layer)             │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┬──────────────┐
        │            │            │              │
┌───────▼──┐  ┌──────▼───┐  ┌────▼──────┐  ┌──▼──────────┐
│   Host   │  │ PostDesign│  │   Image    │  │   LLM       │
│  Agent   │  │  Agent    │  │ Generation │  │ Manager     │
│          │  │           │  │   Agent    │  │             │
│ Routes   │  │ Marketing │  │            │  │ Routes to   │
│ Requests │  │ Copy      │  │ ComfyUI    │  │ Best LLM    │
│          │  │ Design    │  │ DALL-E     │  │ Backend     │
└────────┬─┘  └───────┬───┘  └───┬────────┘  └─┬──────────┘
         │            │          │            │
         │            └─────────▼────────────┬┘
         │                      │
         └──────────────────────▼──────────────┐
                                               │
                        ┌──────────────────────▼─────┐
                        │  LLM Backends              │
                        ├────────────────────────────┤
                        │ • Groq (Free, Fast)       │
                        │ • Ollama (Free, Local)    │
                        │ • OpenAI (GPT-4)          │
                        │ • Claude (Creative)       │
                        └────────────────────────────┘
```

## 📦 Project Structure

```
agentic-ai-system/
├── main.py                          # Entry point
├── requirements.txt                 # Dependencies
├── .env                            # Configuration (API keys)
│
├── core/                           # Core system components
│   ├── message_bus.py             # Async pub/sub messaging
│   ├── agent_base.py              # Base agent class
│   ├── llm_manager.py             # LLM routing & selection
│   └── llm_factory.py             # LLM initialization
│
├── agents/                         # Specialized agents
│   ├── host_agent.py              # Request coordinator
│   ├── post_design_agent.py       # Marketing copy generator
│   └── image_generation_agent.py  # Image generator
│
├── interfaces/                     # User interfaces
│   └── cli_app.py                 # Command-line interface
│
├── ComfyUI/                        # Local SDXL image generation (optional)
│   └── main.py                    # ComfyUI server
│
└── generated_images/              # Output directory
```

## 🎮 CLI Commands

```
You: /help
     Shows available commands

You: /status
     Displays system status, available backends, usage statistics

You: /clear
     Clear the terminal

You: /exit
     Gracefully shutdown the system

You: (any message)
     Send a request to the agent system
```

## 📊 Monitoring & Status

Check system status and backend performance:
```
/status

Output:
═══════════════════════════════════════════════════════
SYSTEM STATUS
═══════════════════════════════════════════════════════

LLM Manager:
   Total Backends: 4
   Available: 3

   groq_llama_3_1_70b_versatile:
      Provider: groq
      Model: llama-3.1-70b-versatile
      Status: Active
      Requests: 5/5 (100.0% success)
      Tokens: 1,234
      Cost: $0.000612
      Avg Latency: 450ms
   ...
```

## 🧪 Testing

### Test Backend Selection
```bash
python test_unbiased.py
```

Shows how the system ranks and selects backends for different task types and complexities.

### Test ComfyUI Connection
```bash
python test_comfyui_connection.py
```

Verifies ComfyUI is running and accessible.

## 🔐 Security & Privacy

- **Local Processing**: Use Ollama for 100% local, private processing
- **Cost Control**: Set `LLM_MAX_COST_PER_REQUEST` to limit spending
- **API Keys**: Never commit `.env` to version control
- **Environment Variables**: All sensitive data in `.env` only

## 🚨 Troubleshooting

### "No LLM backends available"
- Ensure at least one backend is enabled in `.env`
- For Groq: Verify API key is valid
- For Ollama: Ensure it's running (`python main.py --listen`)
- For OpenAI: Check API key format and account status

### ComfyUI connection failed
```bash
# Start ComfyUI if using image generation
cd ComfyUI
python main.py --listen --port 8188
```

### "Image generation timed out"
- Increase `IMAGE_TIMEOUT_SECONDS` in `.env`
- Check ComfyUI is running and responsive
- Lower `DEFAULT_IMAGE_SIZE` for faster generation

### Circuit breaker open
- Wait 60 seconds for backend to recover
- Or restart the application
- Check backend logs for underlying issues

## 📚 Advanced Features

### Unbiased Backend Selection
The system uses sophisticated scoring to fairly rank backends:
- Base quality score (reasoning, creativity, code ability)
- Task complexity matching
- Speed preferences
- Cost considerations
- User-defined priority list

See [LLMManager._calculate_backend_score](core/llm_manager.py) for details.

### Text-in-Image Optimization
When detecting text requests, the system:
1. Identifies text generation intent
2. Extracts the target text
3. Uses text-optimized prompts
4. Applies enhanced prompts with text quality keywords
5. Selects optimal samplers for text clarity

## 🤝 Contributing

To extend the system:

1. **Add a new Agent**: Inherit from [BaseAgent](core/agent_base.py)
2. **Add LLM Provider**: Register in [llm_factory.py](core/llm_factory.py)
3. **Add Image Backend**: Extend [ImageGenerationAgent](agents/image_generation_agent.py)

## 📝 License

[Add your license here]

## 📞 Support

For issues or questions:
1. Check `.env` configuration
2. Review system logs in console output
3. Test individual components with test scripts
4. Check backend-specific documentation

## 🎯 Roadmap

- [ ] Web UI dashboard
- [ ] Prompt templates library
- [ ] Analytics & reporting
- [ ] A/B testing for copy variations
- [ ] Multi-language support
- [ ] Custom model fine-tuning
- [ ] Scheduled content generation

---

**Made with ❤️ for marketing professionals and AI enthusiasts**
