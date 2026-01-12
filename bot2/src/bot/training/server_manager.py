"""Game server manager for ML training.

This module provides the GameServerManager class that manages game server instances
for the ML training pipeline. It handles creating training games, tracking their
lifecycle, and cleaning up resources.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx
from pydantic import ValidationError

from bot.models import GameState, GameUpdate, GetGameStateResponse
from bot.training.exceptions import (
    GameCreationError,
    GameNotFoundError,
    GameServerError,
    MaxGamesExceededError,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingGameConfig:
    """Configuration for a training game instance."""

    room_name: str
    map_type: str = "default"
    tick_multiplier: float = 10.0
    max_game_duration_sec: int = 60
    disable_respawn_timer: bool = True
    max_kills: int = 20


@dataclass
class GameInstance:
    """Represents an active game server instance."""

    room_id: str
    room_code: str
    room_name: str
    config: TrainingGameConfig
    player_id: str
    player_token: str
    canvas_size: tuple[int, int]
    created_at: float
    is_active: bool = field(default=True)


class GameServerManager:
    """Manages game server instances for ML training.

    This class provides lifecycle management for training game instances,
    including creation, status tracking, reset, and termination.

    Example:
        async with GameServerManager(
            http_url="http://localhost:4000",
            max_concurrent_games=4,
        ) as manager:
            config = TrainingGameConfig(room_name="Training 1")
            game = await manager.create_game(config, "Bot1")
            # Use game for training...
            await manager.terminate_game(game.room_id)
    """

    def __init__(
        self,
        http_url: str = "http://localhost:4000",
        max_concurrent_games: int = 10,
    ) -> None:
        """Initialize the game server manager.

        Args:
            http_url: Base URL of the game server HTTP API.
            max_concurrent_games: Maximum number of concurrent game instances.
        """
        self._http_url = http_url.rstrip("/")
        self._max_concurrent_games = max_concurrent_games
        self._client: httpx.AsyncClient | None = None
        self._games: dict[str, GameInstance] = {}

    async def __aenter__(self) -> GameServerManager:
        """Async context manager entry."""
        self._client = httpx.AsyncClient(base_url=self._http_url, timeout=30.0)
        logger.debug("GameServerManager connected to %s", self._http_url)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.terminate_all()
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.debug("GameServerManager closed")

    async def create_game(
        self,
        config: TrainingGameConfig,
        initial_player_name: str = "TrainingBot",
    ) -> GameInstance:
        """Create a new training game instance.

        Uses POST /api/createGame with training options.

        Args:
            config: Training game configuration.
            initial_player_name: Name for the initial player (bot controller).

        Returns:
            GameInstance with connection details.

        Raises:
            GameCreationError: If game creation fails.
            MaxGamesExceededError: If max_concurrent_games reached.
            GameServerError: If client is not connected.
        """
        if self._client is None:
            raise GameServerError("Client not connected. Use as async context manager.")

        active_count = len([g for g in self._games.values() if g.is_active])
        if active_count >= self._max_concurrent_games:
            raise MaxGamesExceededError(
                f"Maximum of {self._max_concurrent_games} concurrent games reached"
            )

        payload = {
            "playerName": initial_player_name,
            "roomName": config.room_name,
            "mapType": config.map_type,
            "trainingMode": True,
            "tickMultiplier": config.tick_multiplier,
            "maxGameDurationSec": config.max_game_duration_sec,
            "disableRespawnTimer": config.disable_respawn_timer,
            "maxKills": config.max_kills,
        }

        logger.info(
            "Creating training game '%s' with map=%s, tick_multiplier=%s",
            config.room_name,
            config.map_type,
            config.tick_multiplier,
        )

        try:
            response = await self._client.post("/api/createGame", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error creating game: %s", e)
            raise GameCreationError(f"HTTP error: {e}") from e
        except httpx.RequestError as e:
            logger.error("Network error creating game: %s", e)
            raise GameCreationError(f"Network error: {e}") from e

        if not data.get("success"):
            error_msg = data.get("error", "Unknown error")
            logger.error("Game creation failed: %s", error_msg)
            raise GameCreationError(error_msg)

        instance = GameInstance(
            room_id=data["roomId"],
            room_code=data["roomCode"],
            room_name=config.room_name,
            config=config,
            player_id=data["playerId"],
            player_token=data["playerToken"],
            canvas_size=(data["canvasSizeX"], data["canvasSizeY"]),
            created_at=time.time(),
        )

        self._games[instance.room_id] = instance
        logger.info(
            "Created game room_id=%s, room_code=%s, player_id=%s",
            instance.room_id,
            instance.room_code,
            instance.player_id,
        )

        return instance

    async def terminate_game(self, room_id: str) -> None:
        """Terminate a game instance.

        Removes the game from active tracking and cleans up resources.

        Args:
            room_id: The room ID to terminate.

        Raises:
            GameNotFoundError: If the game is not found.
        """
        instance = self._games.get(room_id)
        if not instance:
            raise GameNotFoundError(f"Game {room_id} not found")

        instance.is_active = False
        del self._games[room_id]
        logger.info("Terminated game room_id=%s", room_id)

    async def reset_game(self, room_id: str) -> GameState:
        """Reset a game instance for a new episode.

        Uses POST /api/rooms/{roomId}/reset. If the reset endpoint is not
        available, falls back to terminating and recreating the game.

        Args:
            room_id: The room ID to reset.

        Returns:
            Initial game state after reset.

        Raises:
            GameNotFoundError: If the game is not found.
            GameServerError: If client is not connected or reset fails.
        """
        if self._client is None:
            raise GameServerError("Client not connected. Use as async context manager.")

        instance = self._games.get(room_id)
        if not instance:
            raise GameNotFoundError(f"Game {room_id} not found")

        try:
            headers = {"X-Player-Token": instance.player_token}
            response = await self._client.post(
                f"/api/rooms/{room_id}/reset", json={}, headers=headers
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                raise GameServerError(data.get("error", "Reset failed"))

            # Fetch and return initial state after reset
            return await self._fetch_game_state(
                room_id, instance.canvas_size, instance.player_token
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Fallback: recreate the game
                logger.warning(
                    "Reset endpoint not available for room_id=%s, recreating game",
                    room_id,
                )
                config = instance.config
                player_name = "TrainingBot"
                await self.terminate_game(room_id)
                new_instance = await self.create_game(config, player_name)
                return await self._fetch_game_state(
                    new_instance.room_id,
                    new_instance.canvas_size,
                    new_instance.player_token,
                )
            logger.error("HTTP error resetting game: %s", e)
            raise GameServerError(f"Reset failed: {e}") from e

    async def _fetch_game_state(
        self, room_id: str, canvas_size: tuple[int, int], player_token: str
    ) -> GameState:
        """Fetch the current game state from the server.

        Args:
            room_id: The room ID to fetch state for.
            canvas_size: Canvas dimensions (width, height).
            player_token: The player token for authentication.

        Returns:
            Parsed GameState object.

        Raises:
            GameServerError: If fetching state fails.
        """
        if self._client is None:
            raise GameServerError("Client not connected")

        try:
            headers = {"X-Player-Token": player_token}
            response = await self._client.get(
                f"/api/rooms/{room_id}/state", headers=headers
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                raise GameServerError(data.get("error", "Failed to get game state"))

            # Parse the response using the Pydantic model
            state_response = GetGameStateResponse.model_validate(data)

            # Convert to GameState using GameUpdate
            # Using model_validate with aliased names for type checker compatibility
            update = GameUpdate.model_validate(
                {
                    "fullUpdate": True,
                    "objectStates": state_response.object_states or {},
                    "trainingComplete": state_response.training_complete,
                }
            )

            return GameState.from_update(
                update,
                canvas_size_x=canvas_size[0],
                canvas_size_y=canvas_size[1],
            )

        except httpx.HTTPStatusError as e:
            logger.error("HTTP error fetching game state: %s", e)
            raise GameServerError(f"Failed to fetch game state: {e}") from e
        except ValidationError as e:
            logger.error("Validation error parsing game state: %s", e)
            raise GameServerError(f"Invalid game state response: {e}") from e

    def get_game_status(self, room_id: str) -> GameInstance | None:
        """Get the current status of a game instance.

        Args:
            room_id: The room ID to query.

        Returns:
            GameInstance if active, None if not found.
        """
        return self._games.get(room_id)

    def get_active_games(self) -> list[GameInstance]:
        """Get all currently active game instances.

        Returns:
            List of active GameInstance objects.
        """
        return [g for g in self._games.values() if g.is_active]

    async def terminate_all(self) -> None:
        """Terminate all active game instances.

        Used for cleanup during shutdown or error recovery.
        """
        room_ids = list(self._games.keys())
        for room_id in room_ids:
            try:
                await self.terminate_game(room_id)
            except GameNotFoundError:
                pass  # Already removed
        logger.info("Terminated all %d games", len(room_ids))

    async def health_check(self) -> bool:
        """Check if the game server is reachable.

        Returns:
            True if the server responds, False otherwise.
        """
        if self._client is None:
            return False

        try:
            response = await self._client.get("/api/maps")
            logger.debug("Health check status: %d", response.status_code)
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.debug("Health check failed: %s", e)
            return False
