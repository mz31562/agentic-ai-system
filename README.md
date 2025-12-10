# Agentic AI Marketing System

A multi-agent AI system for automated marketing content generation and image synthesis. The system implements intelligent request routing between specialized agents and supports multiple LLM backends with automatic failover capabilities.

## Features

### Multi-Agent Architecture
- **Host Agent**: Central coordinator for request routing to specialist agents
- **PostDesign Agent**: Generates marketing copy, social media posts, email campaigns, and sales content
- **Image Generation Agent**: Produces images using ComfyUI, DALL-E, Replicate, or Segmind backends

### LLM Backend Management
- **Multiple Provider Support**: Groq, Ollama (local), OpenAI, Claude, HuggingFace
- **Dynamic Backend Selection**: Optimizes backend choice based on task type, complexity, cost, and performance metrics
- **Automatic Failover**: Seamless fallback to alternative backends on failure
- **Circuit Breaker Pattern**: Temporarily disables failing services to prevent cascading failures
- **Cost Management**: Enforces budget constraints per request

### Image Generation Capabilities
- **Multiple Backends**: ComfyUI (local SDXL), DALL-E, Replicate, Segmind
- **Text-in-Image Optimization**: Detects and optimizes prompts for text generation within images
- **Retry Logic**: Implements exponential backoff for failed requests
- **Prompt Enhancement**: Automatically improves prompts for higher quality output
- **Fallback Mode**: Mock generation when no backend is available

### Content Type Support
- LinkedIn posts
- Instagram captions
- Twitter/X threads
- Email campaigns
- Blog articles
- Advertisement copy
- Product descriptions
- Sales pitches

### Tone and Style Detection
Automatic detection and configuration of:
- Professional, casual, enthusiastic
- Educational, persuasive, inspiring
- Humorous, urgent

## Installation

### Prerequisites
- Python 3.9 or higher
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd agentic-ai-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:
```bash
# Core LLM backends (configure at least one)
LLM_GROQ_ENABLED=true
LLM_GROQ_API_KEY=your_groq_api_key

LLM_OPENAI_ENABLED=true
LLM_OPENAI_API_KEY=your_openai_api_key

# Local Ollama option
LLM_OLLAMA_ENABLED=true
LLM_OLLAMA_BASE_URL=http://localhost:11434

# Image generation configuration
IMAGE_BACKEND=comfyui  # Options: comfyui, dalle, replicate, segmind
COMFYUI_URL=http://127.0.0.1:8188
```

4. (Optional) Start ComfyUI for local image generation:
```bash
cd ComfyUI
python main.py --listen --port 8188
```

5. Launch the CLI:
```bash
python main.py
```

## Usage

### Generate LinkedIn Post
```
Input: Create a professional LinkedIn post about AI in marketing
Output: Generates thought-leadership content using optimal LLM
```

### Create Instagram Post with Image
```
Input: Design an Instagram post with an image about digital marketing trends
Output: Post copy and generated image
```

### Generate Email Campaign
```
Input: Write a persuasive email campaign about our new SaaS product
Output: Benefit-focused email with call-to-action
```

### Generate Product Image
```
Input: Create an image of a sleek modern laptop in a minimalist workspace
Output: High-quality product image using SDXL
```

## Configuration

### LLM Backend Settings

```bash
# Backend priority (comma-separated, highest priority first)
LLM_BACKEND_PRIORITY=groq,ollama,openai

# Performance preferences
LLM_PREFER_FAST=true
LLM_PREFER_LOCAL=false

# Cost control
LLM_MAX_COST_PER_REQUEST=0.01
LLM_ALLOW_PAID=true
```

### LLM Backend Comparison

| Backend | Cost | Speed | Quality | Local | Requirements |
|---------|------|-------|---------|-------|--------------|
| Groq | Free | High | Good | No | API Key |
| Ollama | Free | Medium | Medium | Yes | Local Install |
| OpenAI | Paid | Medium | Excellent | No | API Key |
| Claude | Paid | Low | Excellent | No | API Key |

### Image Generation Settings

```bash
IMAGE_BACKEND=comfyui
IMAGE_MODEL=sdxl
COMFYUI_URL=http://127.0.0.1:8188
DEFAULT_IMAGE_SIZE=1024x1024
IMAGE_MAX_RETRIES=2
IMAGE_TIMEOUT_SECONDS=300
ENABLE_TEXT_OPTIMIZATION=true
```

## System Architecture

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
└──────────┘  └───────────┘  └───────────┘  └─────────────┘
```

## Project Structure

```
agentic-ai-system/
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── .env                            # Configuration file
│
├── core/                           # Core system components
│   ├── message_bus.py             # Async pub/sub messaging
│   ├── agent_base.py              # Base agent class
│   ├── llm_manager.py             # LLM routing and selection
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
├── ComfyUI/                        # Local SDXL generation (optional)
│   └── main.py                    # ComfyUI server
│
└── generated_images/              # Output directory
```

## CLI Commands

```
/help     - Display available commands
/status   - Show system status and backend performance metrics
/clear    - Clear terminal
/exit     - Shutdown system
```

## Monitoring

View system status and backend performance:

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
```

## Testing

### Test Backend Selection
```bash
python test_unbiased.py
```

### Test ComfyUI Connection
```bash
python test_comfyui_connection.py
```

## Security Considerations

- **Local Processing**: Use Ollama for local, private processing
- **Cost Control**: Configure `LLM_MAX_COST_PER_REQUEST` to limit spending
- **API Keys**: Never commit `.env` file to version control
- **Environment Variables**: Store all sensitive data in `.env` only

## Troubleshooting

### No LLM backends available
- Verify at least one backend is enabled in `.env`
- Confirm API keys are valid
- For Ollama: Ensure service is running
- Check account status for paid services

### ComfyUI connection failed
```bash
cd ComfyUI
python main.py --listen --port 8188
```

### Image generation timeout
- Increase `IMAGE_TIMEOUT_SECONDS` in `.env`
- Verify ComfyUI is running and responsive
- Reduce `DEFAULT_IMAGE_SIZE` for faster generation

### Circuit breaker open
- Wait 60 seconds for automatic recovery
- Restart application if issue persists
- Review backend logs for root cause

## Advanced Features

### Backend Selection Algorithm
The system implements a scoring mechanism for backend selection based on:
- Base quality metrics (reasoning, creativity, code generation)
- Task complexity matching
- Performance requirements
- Cost constraints
- User-defined priorities

See `core/llm_manager.py:_calculate_backend_score` for implementation details.

### Text-in-Image Optimization
Text generation detection and optimization process:
1. Analyze prompt for text generation intent
2. Extract target text content
3. Apply text-optimized prompt templates
4. Enhance prompt with text quality keywords
5. Select optimal samplers for text clarity

## Contributing

To extend the system:

1. **New Agent**: Inherit from `BaseAgent` in `core/agent_base.py`
2. **LLM Provider**: Register in `core/llm_factory.py`
3. **Image Backend**: Extend `ImageGenerationAgent` in `agents/image_generation_agent.py`

## License

MIT License

## Support

For issues or questions:
1. Review `.env` configuration
2. Check console logs
3. Run component tests
4. Consult provider-specific documentation

## Roadmap

- Web UI dashboard
- Prompt template library
- Analytics and reporting
- A/B testing for content variations
- Multi-language support
- Custom model fine-tuning
- Scheduled content generation

---

**A professional AI system for automated marketing content generation**
