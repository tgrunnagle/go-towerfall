"""Async HTTP client wrapper for go-towerfall REST API.

This module provides a clean, type-safe interface for all REST API interactions
with the Go game server, including proper error handling, retry logic, and
response validation using Pydantic models.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TypeVar

from httpx import AsyncClient, HTTPError, TimeoutException
from pydantic import BaseModel, ValidationError

from bot.models import (
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

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Custom Exceptions
# =============================================================================


class GameHTTPClientError(Exception):
    """Base exception for HTTP client errors."""

    pass


class GameAPIError(GameHTTPClientError):
    """Raised when the game server returns an error response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class GameConnectionError(GameHTTPClientError):
    """Raised when connection to the game server fails."""

    pass


# =============================================================================
# HTTP Client
# =============================================================================


class GameHTTPClient:
    """Async HTTP client for go-towerfall REST API interactions.

    This client provides a clean interface for all REST API operations with
    proper error handling, configurable timeouts, and automatic retry logic
    for transient failures.

    Example:
        async with GameHTTPClient(base_url="http://localhost:4000") as client:
            maps = await client.get_maps()
            game = await client.create_game("Bot1", "TrainingRoom", "arena1")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:4000",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize the HTTP client.

        Args:
            base_url: Base URL for the game server (default: http://localhost:4000)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts for transient failures (default: 3)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: AsyncClient | None = None

    async def __aenter__(self) -> GameHTTPClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Initialize the HTTP client.

        Creates an httpx.AsyncClient for connection pooling and reuse.
        """
        if self._client is None:
            self._client = AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
            logger.debug("HTTP client connected to %s", self.base_url)

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("HTTP client closed")

    # =========================================================================
    # Public API Methods
    # =========================================================================

    async def get_maps(self) -> list[MapInfo]:
        """Get list of available maps.

        Returns:
            List of MapInfo objects describing available maps.

        Raises:
            GameAPIError: If the server returns an error.
            GameConnectionError: If connection fails.
        """
        response = await self._get("/api/maps", GetMapsResponse)
        return response.maps

    async def create_game(
        self,
        player_name: str,
        room_name: str,
        map_type: str,
        *,
        training_mode: bool | None = None,
        tick_multiplier: float | None = None,
        max_game_duration_sec: int | None = None,
        disable_respawn_timer: bool | None = None,
        max_kills: int | None = None,
    ) -> CreateGameResponse:
        """Create a new game room.

        Args:
            player_name: Name of the player creating the game.
            room_name: Display name for the game room.
            map_type: Type of map to use (e.g., "arena1").
            training_mode: Enable training mode for ML bots.
            tick_multiplier: Game speed multiplier (training mode).
            max_game_duration_sec: Maximum game duration in seconds.
            disable_respawn_timer: Disable respawn timer (training mode).
            max_kills: Maximum kills before game ends.

        Returns:
            CreateGameResponse with room details and player credentials.

        Raises:
            GameAPIError: If the server returns an error.
            GameConnectionError: If connection fails.
        """
        request_data: dict[str, str | bool | float | int] = {
            "playerName": player_name,
            "roomName": room_name,
            "mapType": map_type,
        }
        if training_mode is not None:
            request_data["trainingMode"] = training_mode
        if tick_multiplier is not None:
            request_data["tickMultiplier"] = tick_multiplier
        if max_game_duration_sec is not None:
            request_data["maxGameDurationSec"] = max_game_duration_sec
        if disable_respawn_timer is not None:
            request_data["disableRespawnTimer"] = disable_respawn_timer
        if max_kills is not None:
            request_data["maxKills"] = max_kills

        request = CreateGameRequest.model_validate(request_data)
        return await self._post("/api/createGame", request, CreateGameResponse)

    async def join_game(
        self,
        player_name: str,
        room_code: str,
        room_password: str = "",
        *,
        is_spectator: bool = False,
    ) -> JoinGameResponse:
        """Join an existing game room.

        Args:
            player_name: Name of the player joining.
            room_code: Room code to join.
            room_password: Password for the room (if required).
            is_spectator: Whether to join as spectator.

        Returns:
            JoinGameResponse with player credentials.

        Raises:
            GameAPIError: If the server returns an error (e.g., room not found).
            GameConnectionError: If connection fails.
        """
        request_data: dict[str, str | bool] = {
            "playerName": player_name,
            "roomCode": room_code,
            "roomPassword": room_password,
        }
        if is_spectator:
            request_data["isSpectator"] = is_spectator

        request = JoinGameRequest.model_validate(request_data)
        return await self._post("/api/joinGame", request, JoinGameResponse)

    # =========================================================================
    # Training Mode API Methods
    # =========================================================================

    async def get_game_state(self, room_id: str) -> GetGameStateResponse:
        """Get current game state snapshot (training mode).

        Args:
            room_id: The room ID to get state for.

        Returns:
            GetGameStateResponse with the current game state.

        Raises:
            GameAPIError: If the server returns an error.
            GameConnectionError: If connection fails.
        """
        return await self._get(f"/api/rooms/{room_id}/state", GetGameStateResponse)

    async def submit_action(
        self,
        room_id: str,
        player_id: str,
        actions: list[BotAction],
    ) -> BotActionResponse:
        """Submit bot actions via REST (training mode).

        Args:
            room_id: The room ID to submit actions for.
            player_id: The player ID submitting actions.
            actions: List of BotAction objects to submit.

        Returns:
            BotActionResponse with action processing results.

        Raises:
            GameAPIError: If the server returns an error.
            GameConnectionError: If connection fails.
        """
        request = BotActionRequest(actions=actions)
        return await self._post(
            f"/api/rooms/{room_id}/players/{player_id}/actions",
            request,
            BotActionResponse,
        )

    async def reset_game(
        self,
        room_id: str,
        map_type: str | None = None,
    ) -> ResetGameResponse:
        """Reset game for new training episode.

        Args:
            room_id: The room ID to reset.
            map_type: Optional new map type to use after reset.

        Returns:
            ResetGameResponse with reset results.

        Raises:
            GameAPIError: If the server returns an error.
            GameConnectionError: If connection fails.
        """
        request_data: dict[str, str] = {}
        if map_type is not None:
            request_data["mapType"] = map_type

        request = ResetGameRequest.model_validate(request_data)
        return await self._post(
            f"/api/rooms/{room_id}/reset", request, ResetGameResponse
        )

    async def get_game_stats(self, room_id: str) -> dict[str, PlayerStatsDTO]:
        """Get kill/death statistics for the room.

        Args:
            room_id: The room ID to get stats for.

        Returns:
            Dictionary mapping player IDs to their statistics.

        Raises:
            GameAPIError: If the server returns an error.
            GameConnectionError: If connection fails.
        """
        response = await self._get(f"/api/rooms/{room_id}/stats", GetRoomStatsResponse)
        return response.player_stats or {}

    # =========================================================================
    # Internal HTTP Methods
    # =========================================================================

    async def _get(
        self,
        endpoint: str,
        response_model: type[T],
    ) -> T:
        """Perform GET request with retry logic.

        Args:
            endpoint: API endpoint path (e.g., "/api/maps").
            response_model: Pydantic model class for response parsing.

        Returns:
            Parsed response as the specified model type.
        """
        return await self._request("GET", endpoint, response_model)

    async def _post(
        self,
        endpoint: str,
        request_data: BaseModel,
        response_model: type[T],
    ) -> T:
        """Perform POST request with retry logic.

        Args:
            endpoint: API endpoint path.
            request_data: Pydantic model instance for request body.
            response_model: Pydantic model class for response parsing.

        Returns:
            Parsed response as the specified model type.
        """
        return await self._request(
            "POST",
            endpoint,
            response_model,
            json=request_data.model_dump(by_alias=True, exclude_none=True),
        )

    async def _request(
        self,
        method: str,
        endpoint: str,
        response_model: type[T],
        **kwargs: Any,
    ) -> T:
        """Execute HTTP request with error handling and retries.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path.
            response_model: Pydantic model class for response parsing.
            **kwargs: Additional arguments passed to httpx request.

        Returns:
            Parsed response as the specified model type.

        Raises:
            GameAPIError: If the server returns an error (not retried).
            GameConnectionError: If connection fails after all retries.
            GameHTTPClientError: For unexpected errors.
        """
        if self._client is None:
            raise GameConnectionError("Client not connected. Call connect() first.")

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    "Request %s %s (attempt %d/%d)",
                    method,
                    endpoint,
                    attempt + 1,
                    self.max_retries,
                )

                response = await self._client.request(method, endpoint, **kwargs)

                # Check for HTTP-level errors
                if response.status_code >= 400:
                    error_text = response.text
                    logger.warning("API error %d: %s", response.status_code, error_text)
                    raise GameAPIError(
                        f"API error: {error_text}",
                        status_code=response.status_code,
                    )

                data = response.json()

                # Check for application-level error in response
                if isinstance(data, dict) and data.get("success") is False:
                    error_msg = data.get("error", "Unknown error")
                    logger.warning("API returned error: %s", error_msg)
                    raise GameAPIError(error_msg)

                return response_model.model_validate(data)

            except TimeoutException as e:
                last_error = GameConnectionError(f"Request timeout: {e}")
                logger.warning("Request timeout on attempt %d: %s", attempt + 1, e)

            except HTTPError as e:
                last_error = GameConnectionError(f"HTTP error: {e}")
                logger.warning("HTTP error on attempt %d: %s", attempt + 1, e)

            except GameAPIError:
                # Do not retry API errors - they are intentional server responses
                raise

            except ValidationError as e:
                # Do not retry validation errors - response format is unexpected
                raise GameHTTPClientError(f"Response validation error: {e}") from e

            except Exception as e:
                last_error = GameHTTPClientError(f"Unexpected error: {e}")
                logger.warning("Unexpected error on attempt %d: %s", attempt + 1, e)

            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                backoff = 2**attempt * 0.1
                logger.debug("Retrying in %.1f seconds...", backoff)
                await asyncio.sleep(backoff)

        raise last_error or GameConnectionError("Request failed after retries")
