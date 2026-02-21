"""
mcp_server/__init__.py

MCP (Model Context Protocol) server for the Agentic AI System.

This exposes your agents as MCP tools, meaning external clients like
Claude Desktop, Cursor, or any MCP-compatible app can call your agents
directly as tools — without going through the CLI.

Example tools we'll expose:
  - generate_post(platform, topic, tone)     → calls PostDesignAgent
  - generate_image(prompt, size, style)      → calls ImageGenerationAgent
  - generate_post_with_image(...)            → triggers the full Saga workflow

We'll build this out after the FastAPI layer is done, since the MCP server
will call the same API endpoints — keeping everything consistent.

To use MCP with Claude Desktop once built:
  Add to claude_desktop_config.json:
  {
    "mcpServers": {
      "agentic-ai": {
        "command": "python",
        "args": ["-m", "mcp_server.server"],
        "env": { "API_URL": "http://localhost:8000" }
      }
    }
  }
"""

# mcp_server/server.py will go here
# mcp_server/tools.py will define the tool schemas