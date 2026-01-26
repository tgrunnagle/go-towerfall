"""Bot service layer - WebSocket connectivity and bot lifecycle management."""

from bot.service.websocket_bot import (
    BotRunnerProtocol,
    WebSocketBotClient,
    WebSocketBotClientConfig,
)

__all__ = [
    "BotRunnerProtocol",
    "WebSocketBotClient",
    "WebSocketBotClientConfig",
]
