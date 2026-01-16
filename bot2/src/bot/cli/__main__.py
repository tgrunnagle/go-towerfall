"""Entry point for running the CLI as a module.

Usage:
    uv run python -m bot.cli --help
"""

from bot.cli.main import app

if __name__ == "__main__":
    app()
