"""WebSocket message models.

This module contains Pydantic models for WebSocket message serialization and deserialization.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Base WebSocket Message
# =============================================================================


class WebSocketMessage(BaseModel):
    """Base WebSocket message structure.

    All WebSocket messages have a type and optional payload.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: str
    payload: dict[str, Any] | None = None


# =============================================================================
# Input Messages (Client -> Server)
# =============================================================================


class KeyStatusRequest(BaseModel):
    """Keyboard input message payload.

    Sent when a key is pressed or released.
    """

    model_config = ConfigDict(populate_by_name=True)

    key: str  # W, A, S, D
    is_down: bool = Field(alias="isDown")


class PlayerClickRequest(BaseModel):
    """Mouse click input message payload.

    Sent when a mouse button is pressed or released.
    """

    model_config = ConfigDict(populate_by_name=True)

    x: float
    y: float
    is_down: bool = Field(alias="isDown")
    button: int  # 0=left, 2=right


class ClientStateRequest(BaseModel):
    """Client state update message payload.

    Sent to update the player's facing direction.
    """

    model_config = ConfigDict(populate_by_name=True)

    direction: float = Field(alias="dir")  # Direction in radians


# =============================================================================
# Rejoin Game Messages
# =============================================================================


class RejoinGameRequest(BaseModel):
    """Rejoin game request payload.

    Sent after WebSocket connection to rejoin an existing game session.
    """

    model_config = ConfigDict(populate_by_name=True)

    room_id: str = Field(alias="roomId")
    player_id: str = Field(alias="playerId")
    player_token: str = Field(alias="playerToken")


class RejoinGameResponse(BaseModel):
    """Rejoin game response payload.

    Received after successfully rejoining a game.
    """

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    room_name: str | None = Field(default=None, alias="roomName")
    room_code: str | None = Field(default=None, alias="roomCode")
    room_password: str | None = Field(default=None, alias="roomPassword")
    player_name: str | None = Field(default=None, alias="playerName")
    player_id: str | None = Field(default=None, alias="playerId")
    error: str | None = None


# =============================================================================
# Exit Game Messages
# =============================================================================


class ExitGameRequest(BaseModel):
    """Exit game request payload."""

    model_config = ConfigDict(populate_by_name=True)


class ExitGameResponse(BaseModel):
    """Exit game response payload."""

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    error: str | None = None


# =============================================================================
# Create/Join Game via WebSocket (alternative to HTTP)
# =============================================================================


class CreateGameWSRequest(BaseModel):
    """Create game request payload via WebSocket."""

    model_config = ConfigDict(populate_by_name=True)

    room_name: str = Field(alias="roomName")
    player_name: str = Field(alias="playerName")


class CreateGameWSResponse(BaseModel):
    """Create game response payload via WebSocket."""

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    error: str | None = None
    room_id: str | None = Field(default=None, alias="roomId")
    room_name: str | None = Field(default=None, alias="roomName")
    room_code: str | None = Field(default=None, alias="roomCode")
    room_password: str | None = Field(default=None, alias="roomPassword")
    player_id: str | None = Field(default=None, alias="playerId")
    player_token: str | None = Field(default=None, alias="playerToken")


class JoinGameWSRequest(BaseModel):
    """Join game request payload via WebSocket."""

    model_config = ConfigDict(populate_by_name=True)

    room_code: str = Field(alias="roomCode")
    room_password: str = Field(alias="roomPassword")
    player_name: str = Field(alias="playerName")


class JoinGameWSResponse(BaseModel):
    """Join game response payload via WebSocket."""

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    error: str | None = None
    room_id: str | None = Field(default=None, alias="roomId")
    room_name: str | None = Field(default=None, alias="roomName")
    room_code: str | None = Field(default=None, alias="roomCode")
    player_id: str | None = Field(default=None, alias="playerId")
    player_token: str | None = Field(default=None, alias="playerToken")


# =============================================================================
# Error Message
# =============================================================================


class ErrorMessage(BaseModel):
    """Error message payload."""

    model_config = ConfigDict(populate_by_name=True)

    message: str


# =============================================================================
# Spectator Update
# =============================================================================


class SpectatorUpdate(BaseModel):
    """Spectator list update payload."""

    model_config = ConfigDict(populate_by_name=True)

    spectators: list[str]


# =============================================================================
# WebSocket Message Types (Constants)
# =============================================================================


class MessageTypes:
    """WebSocket message type constants."""

    # Client -> Server
    KEY = "Key"
    CLICK = "Click"
    DIRECTION = "Direction"
    CREATE_GAME = "CreateGame"
    JOIN_GAME = "JoinGame"
    REJOIN_GAME = "RejoinGame"
    EXIT_GAME = "ExitGame"

    # Server -> Client
    GAME_UPDATE = "GameUpdate"
    CREATE_GAME_RESPONSE = "CreateGameResponse"
    JOIN_GAME_RESPONSE = "JoinGameResponse"
    REJOIN_GAME_RESPONSE = "RejoinGameResponse"
    EXIT_GAME_RESPONSE = "ExitGameResponse"
    ERROR = "Error"
    SPECTATOR_UPDATE = "SpectatorUpdate"
