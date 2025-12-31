"""HTTP client module for go-towerfall bot.

This module provides an async HTTP client wrapper for REST API interactions
with the Go game server.

Usage:
    from bot.client import GameHTTPClient, GameAPIError, GameConnectionError

    async with GameHTTPClient() as client:
        maps = await client.get_maps()
        game = await client.create_game("BotName", "Room", "arena1")
"""

from bot.client.http_client import (
    GameAPIError,
    GameConnectionError,
    GameHTTPClient,
    GameHTTPClientError,
)

__all__ = [
    "GameHTTPClient",
    "GameHTTPClientError",
    "GameAPIError",
    "GameConnectionError",
]
