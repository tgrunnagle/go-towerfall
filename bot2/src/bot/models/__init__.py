"""Pydantic models for the bot game client.

This package provides type-safe data validation and serialization for all interactions
between the Python bot and the Go game server.

Usage:
    from bot.models import GameState, PlayerState, GameUpdate
    from bot.models import CreateGameRequest, JoinGameResponse
    from bot.models import GAME_CONSTANTS, ObjectTypes
"""

# Base models
# HTTP API models
from bot.models.api import (
    BotAction,
    BotActionRequest,
    BotActionResponse,
    CreateGameRequest,
    CreateGameResponse,
    GetGameStateResponse,
    GetMapsResponse,
    GetRoomStatsResponse,
    JoinGameRequest,
    JoinGameResponse,
    MapInfo,
    PlayerStatsDTO,
    ResetGameRequest,
    ResetGameResponse,
)
from bot.models.base import BaseObjectState, Point

# Constants
from bot.models.constants import (
    GAME_CONSTANTS,
    GameConstants,
    ObjectTypes,
    StateKeys,
)

# Game object models
from bot.models.game_objects import (
    ArrowState,
    BlockState,
    BulletState,
    PlayerState,
)

# Game state models
from bot.models.game_state import GameState, GameUpdate, GameUpdateEvent

# WebSocket models
from bot.models.websocket import (
    ClientStateRequest,
    CreateGameWSRequest,
    CreateGameWSResponse,
    ErrorMessage,
    ExitGameRequest,
    ExitGameResponse,
    JoinGameWSRequest,
    JoinGameWSResponse,
    KeyStatusRequest,
    MessageTypes,
    PlayerClickRequest,
    RejoinGameRequest,
    RejoinGameResponse,
    SpectatorUpdate,
    WebSocketMessage,
)

__all__ = [
    # Base
    "Point",
    "BaseObjectState",
    # Game Objects
    "PlayerState",
    "ArrowState",
    "BlockState",
    "BulletState",
    # Game State
    "GameUpdateEvent",
    "GameUpdate",
    "GameState",
    # HTTP API
    "MapInfo",
    "GetMapsResponse",
    "CreateGameRequest",
    "CreateGameResponse",
    "JoinGameRequest",
    "JoinGameResponse",
    "PlayerStatsDTO",
    "GetRoomStatsResponse",
    "GetGameStateResponse",
    "ResetGameRequest",
    "ResetGameResponse",
    "BotAction",
    "BotActionRequest",
    "BotActionResponse",
    # WebSocket
    "WebSocketMessage",
    "KeyStatusRequest",
    "PlayerClickRequest",
    "ClientStateRequest",
    "RejoinGameRequest",
    "RejoinGameResponse",
    "ExitGameRequest",
    "ExitGameResponse",
    "CreateGameWSRequest",
    "CreateGameWSResponse",
    "JoinGameWSRequest",
    "JoinGameWSResponse",
    "ErrorMessage",
    "SpectatorUpdate",
    "MessageTypes",
    # Constants
    "GameConstants",
    "GAME_CONSTANTS",
    "ObjectTypes",
    "StateKeys",
]
