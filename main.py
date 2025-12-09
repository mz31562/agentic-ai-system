# C:\Users\MohammedZaid\Desktop\agentic-ai-system\main.py
#!/usr/bin/env python3
"""
Marketing AI System - Main Entry Point
Simple launcher for the CLI interface
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from interfaces.cli_app import main

if __name__ == "__main__":
    print("Starting Marketing AI System...\n")
    main()