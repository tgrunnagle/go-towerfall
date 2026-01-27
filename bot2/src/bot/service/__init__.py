"""Bot service layer - WebSocket connectivity and bot lifecycle management."""

from bot.service.bot_manager import (
    BotConfig,
    BotInfo,
    BotManager,
    SpawnBotRequest,
    SpawnBotResponse,
)
from bot.service.websocket_bot import (
    BotRunnerProtocol,
    WebSocketBotClient,
    WebSocketBotClientConfig,
)

__all__ = [
    "BotRunnerProtocol",
    "WebSocketBotClient",
    "WebSocketBotClientConfig",
    "BotConfig",
    "BotInfo",
    "BotManager",
    "SpawnBotRequest",
    "SpawnBotResponse",
]
