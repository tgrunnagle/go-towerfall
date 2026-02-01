"""Entry point for running the Bot Service.

Usage:
    uv run python -m bot.service

Environment variables:
    BOT_SERVICE_PORT: Port to listen on (default: 8080)
    BOT_SERVICE_HOST: Host to bind (default: 0.0.0.0)
    GAME_SERVER_HTTP_URL: Game server HTTP URL (default: http://localhost:4000)
    GAME_SERVER_WS_URL: Game server WebSocket URL (default: ws://localhost:4000/ws)
    MODEL_REGISTRY_PATH: Path to model registry (optional)
    DEFAULT_DEVICE: Device for inference (default: cpu)
"""

import logging
import os

import uvicorn

from bot.service.bot_manager import BotManager
from bot.service.bot_service import app, set_bot_manager
from bot.training.registry import ModelRegistry


def main() -> None:
    """Initialize and run the Bot Service."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load configuration from environment
    port_str = os.environ.get("BOT_SERVICE_PORT", "8080")
    try:
        port = int(port_str)
    except ValueError:
        logger.error(
            f"Invalid BOT_SERVICE_PORT value: '{port_str}'. Must be an integer."
        )
        raise
    host = os.environ.get("BOT_SERVICE_HOST", "0.0.0.0")
    http_url = os.environ.get("GAME_SERVER_HTTP_URL", "http://localhost:4000")
    ws_url = os.environ.get("GAME_SERVER_WS_URL", "ws://localhost:4000/ws")
    registry_path = os.environ.get("MODEL_REGISTRY_PATH")
    device = os.environ.get("DEFAULT_DEVICE", "cpu")

    # Initialize ModelRegistry if path is provided
    registry: ModelRegistry | None = None
    if registry_path:
        logger.info(f"Loading model registry from {registry_path}")
        registry = ModelRegistry(registry_path)
    else:
        logger.warning("MODEL_REGISTRY_PATH not set - neural network bots unavailable")

    # Initialize BotManager
    manager = BotManager(
        registry=registry,
        http_url=http_url,
        ws_url=ws_url,
        default_device=device,
    )
    set_bot_manager(manager)

    logger.info(f"Starting Bot Service on {host}:{port}")
    logger.info(f"Game server: HTTP={http_url}, WS={ws_url}")

    # Run uvicorn server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
