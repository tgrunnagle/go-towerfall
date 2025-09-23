#!/usr/bin/env python3
"""
Script to run the Bot API Server for integration with the Go game server.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the bot directory to the Python path
bot_dir = Path(__file__).parent
sys.path.insert(0, str(bot_dir))

from rl_bot_system.server.bot_api_server import main

if __name__ == "__main__":
    print("Starting Bot API Server...")
    print("This server provides HTTP API endpoints for bot management")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot API Server stopped.")
    except Exception as e:
        print(f"Error starting Bot API Server: {e}")
        sys.exit(1)