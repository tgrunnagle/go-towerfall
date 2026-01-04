"""HTTP API request/response models.

This module contains Pydantic models for HTTP API interactions with the game server.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Map API
# =============================================================================


class MapInfo(BaseModel):
    """Information about a game map."""

    model_config = ConfigDict(populate_by_name=True)

    type: str
    name: str
    canvas_size_x: int = Field(alias="canvas_size_x")
    canvas_size_y: int = Field(alias="canvas_size_y")


class GetMapsResponse(BaseModel):
    """Response from GET /api/maps endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    maps: list[MapInfo]


# =============================================================================
# Create Game API
# =============================================================================


class CreateGameRequest(BaseModel):
    """Request body for POST /api/createGame endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    room_name: str = Field(alias="roomName")
    player_name: str = Field(alias="playerName")
    map_type: str = Field(alias="mapType")
    # Training mode options (all optional)
    training_mode: bool | None = Field(default=None, alias="trainingMode")
    tick_multiplier: float | None = Field(default=None, alias="tickMultiplier")
    max_game_duration_sec: int | None = Field(default=None, alias="maxGameDurationSec")
    disable_respawn_timer: bool | None = Field(
        default=None, alias="disableRespawnTimer"
    )
    max_kills: int | None = Field(default=None, alias="maxKills")


class CreateGameResponse(BaseModel):
    """Response from POST /api/createGame endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    room_id: str = Field(alias="roomId")
    room_code: str = Field(alias="roomCode")
    room_name: str = Field(alias="roomName")
    room_password: str = Field(alias="roomPassword")
    player_id: str = Field(alias="playerId")
    player_token: str = Field(alias="playerToken")
    canvas_size_x: int = Field(alias="canvasSizeX")
    canvas_size_y: int = Field(alias="canvasSizeY")
    # Training mode settings (returned when training mode is enabled)
    training_mode: bool | None = Field(default=None, alias="trainingMode")
    tick_multiplier: float | None = Field(default=None, alias="tickMultiplier")
    max_game_duration_sec: int | None = Field(default=None, alias="maxGameDurationSec")
    disable_respawn_timer: bool | None = Field(
        default=None, alias="disableRespawnTimer"
    )
    max_kills: int | None = Field(default=None, alias="maxKills")


# =============================================================================
# Join Game API
# =============================================================================


class JoinGameRequest(BaseModel):
    """Request body for POST /api/joinGame endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    room_code: str = Field(alias="roomCode")
    player_name: str = Field(alias="playerName")
    room_password: str = Field(alias="roomPassword")
    is_spectator: bool | None = Field(default=None, alias="isSpectator")


class JoinGameResponse(BaseModel):
    """Response from POST /api/joinGame endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    room_id: str = Field(alias="roomId")
    room_code: str = Field(alias="roomCode")
    room_name: str | None = Field(default=None, alias="roomName")
    player_id: str = Field(alias="playerId")
    player_token: str = Field(alias="playerToken")
    is_spectator: bool = Field(alias="isSpectator")
    canvas_size_x: int = Field(alias="canvasSizeX")
    canvas_size_y: int = Field(alias="canvasSizeY")


# =============================================================================
# Room Stats API
# =============================================================================


class PlayerStatsDTO(BaseModel):
    """Kill/death statistics for a player."""

    model_config = ConfigDict(populate_by_name=True)

    player_id: str = Field(alias="playerId")
    player_name: str = Field(alias="playerName")
    kills: int
    deaths: int


class GetRoomStatsResponse(BaseModel):
    """Response from GET /api/rooms/{roomId}/stats endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    room_id: str | None = Field(default=None, alias="roomId")
    player_stats: dict[str, PlayerStatsDTO] | None = Field(
        default=None, alias="playerStats"
    )
    error: str | None = None


# =============================================================================
# Game State Snapshot API
# =============================================================================


class GetGameStateResponse(BaseModel):
    """Response from GET /api/rooms/{roomId}/state endpoint.

    The server returns game state data directly at the top level:
    - objectStates: dict of object ID -> object state
    - timestamp: server timestamp
    - trainingComplete: whether training episode is complete (optional)
    """

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    room_id: str | None = Field(default=None, alias="roomId")
    timestamp: str | None = None
    object_states: dict[str, dict[str, Any] | None] | None = Field(
        default=None, alias="objectStates"
    )
    training_complete: bool = Field(default=False, alias="trainingComplete")
    error: str | None = None


# =============================================================================
# Game Reset API
# =============================================================================


class ResetGameRequest(BaseModel):
    """Request body for POST /api/rooms/{roomId}/reset endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    map_type: str | None = Field(default=None, alias="mapType")


class ResetGameResponse(BaseModel):
    """Response from POST /api/rooms/{roomId}/reset endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    room_id: str | None = Field(default=None, alias="roomId")
    map_type: str | None = Field(default=None, alias="mapType")
    canvas_size_x: int | None = Field(default=None, alias="canvasSizeX")
    canvas_size_y: int | None = Field(default=None, alias="canvasSizeY")
    error: str | None = None


# =============================================================================
# Bot Action API
# =============================================================================


class BotAction(BaseModel):
    """A single bot action (key, click, or direction)."""

    model_config = ConfigDict(populate_by_name=True)

    type: str  # "key", "click", or "direction"
    key: str | None = None  # For key actions: W/A/S/D
    is_down: bool | None = Field(default=None, alias="isDown")  # For key/click actions
    x: float | None = None  # For click actions
    y: float | None = None  # For click actions
    button: int | None = None  # For click actions: 0=left, 2=right
    direction: float | None = None  # For direction actions (radians)


class BotActionRequest(BaseModel):
    """Request body for POST /api/rooms/{roomId}/players/{playerId}/actions endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    actions: list[BotAction]


class BotActionResponse(BaseModel):
    """Response from bot action submission endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    actions_processed: int | None = Field(default=None, alias="actionsProcessed")
    timestamp: int | None = None
    error: str | None = None
