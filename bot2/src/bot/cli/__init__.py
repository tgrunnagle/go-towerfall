"""CLI module for the TowerFall ML bot training pipeline.

This module provides command-line tools for managing training runs,
models, and configuration for the reinforcement learning bot.

Usage:
    uv run python -m bot.cli --help
    uv run python -m bot.cli train start --config config/training.yaml
    uv run python -m bot.cli model list
"""

from bot.cli.main import app

__all__ = ["app"]
