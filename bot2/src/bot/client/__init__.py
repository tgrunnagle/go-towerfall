"""Client module for go-towerfall bot.

This module provides game client classes for interacting with the Go game server:
- GameClient: Unified client supporting both WebSocket and REST modes
- GameHTTPClient: Low-level async HTTP client wrapper for REST API interactions

Usage:
    from bot.client import GameClient, ClientMode

    # WebSocket mode (real-time play)
    async with GameClient(mode=ClientMode.WEBSOCKET) as client:
        await client.join_game(player_name="Bot", room_code="ABC123")
        await client.send_keyboard_input("d", True)

    # REST mode (ML training)
    async with GameClient(mode=ClientMode.REST) as client:
        await client.create_game("MLBot", "Training", "arena1", training_mode=True)
        state = await client.get_game_state()
"""

from bot.client.game_client import ClientMode, GameClient, GameClientError
from bot.client.http_client import (
    GameAPIError,
    GameConnectionError,
    GameHTTPClient,
    GameHTTPClientError,
)

__all__ = [
    # Unified GameClient
    "GameClient",
    "ClientMode",
    "GameClientError",
    # HTTP Client
    "GameHTTPClient",
    "GameHTTPClientError",
    "GameAPIError",
    "GameConnectionError",
]
