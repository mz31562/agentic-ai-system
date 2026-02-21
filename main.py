"""
main.py — Application entry point

Boot order:
  1. Windows event loop fix (must be before any asyncio import)
  2. Load .env
  3. Setup logging
  4. Validate config
  5. Launch CLI or API
"""

import sys
import os

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

from core.logging_config import setup_logging
log_file = setup_logging(debug=DEBUG_MODE)

from core.config_validator import validate_config


def main():
    # ── 1. Validate config before touching anything else ──────────────────
    result = validate_config()
    result.print_report()

    if not result.valid:
        print("Fix the errors above in your .env file and restart.\n")
        sys.exit(1)

    # ── 2. Decide what to launch ──────────────────────────────────────────
    mode = os.getenv("APP_MODE", "cli").lower().strip()

    if mode == "api":
        # Launch FastAPI server (for React frontend / MCP)
        import uvicorn
        print("Starting API server on http://localhost:8000\n")
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=DEBUG_MODE,
            log_level="debug" if DEBUG_MODE else "warning"
        )
    else:
        # Launch CLI (default)
        from interfaces.cli_app import main as run_cli
        run_cli()


if __name__ == "__main__":
    main()