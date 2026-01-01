"""Unified game client supporting both WebSocket and REST modes.

This module provides a GameClient class that can operate in two modes:
- WEBSOCKET mode: Real-time WebSocket-based communication for human-like gameplay
- REST mode: REST-based action submission for ML training scenarios

The client composes GameHTTPClient for all REST operations and optionally
manages a WebSocket connection for real-time mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from typing import Awaitable, Callable

import websockets
from websockets import ClientConnection
from websockets.exceptions import ConnectionClosed

from bot.client.http_client import GameHTTPClient, GameHTTPClientError
from bot.models import (
    BotAction,
    CreateGameResponse,
    GameState,
    GameUpdate,
    JoinGameResponse,
    PlayerStatsDTO,
)


class ClientMode(Enum):
    """Operating mode for the GameClient."""

    WEBSOCKET = "websocket"  # Real-time WebSocket-based communication
    REST = "rest"  # Training mode with REST-based actions


class GameClientError(GameHTTPClientError):
    """Base exception for GameClient errors."""

    pass


class GameClient:
    """Unified game client supporting both WebSocket and REST modes.

    In WEBSOCKET mode:
        - Actions sent via WebSocket messages
        - Game state received via WebSocket broadcasts
        - Suitable for real-time play and spectating

    In REST mode:
        - Actions submitted via REST API (POST /api/rooms/{roomId}/players/{playerId}/actions)
        - Game state polled via REST API (GET /api/rooms/{roomId}/state)
        - Suitable for ML training with synchronous stepping

    Example (WebSocket mode):
        async with GameClient(mode=ClientMode.WEBSOCKET) as client:
            await client.join_game(player_name="Bot", room_code="ABC123")
            await client.send_keyboard_input("d", True)  # Move right

    Example (REST mode):
        async with GameClient(mode=ClientMode.REST) as client:
            await client.create_game(
                player_name="MLBot",
                room_name="Training",
                map_type="arena1",
                training_mode=True,
            )
            state = await client.get_game_state()
            await client.send_keyboard_input("d", True)
    """

    def __init__(
        self,
        http_url: str = "http://localhost:4000",
        ws_url: str = "ws://localhost:4000/ws",
        mode: ClientMode = ClientMode.WEBSOCKET,
        timeout: float = 30.0,
    ):
        """Initialize the GameClient.

        Args:
            http_url: Base URL for REST API.
            ws_url: WebSocket endpoint URL.
            mode: Operating mode (WEBSOCKET or REST).
            timeout: Request timeout in seconds.
        """
        self.http_url = http_url
        self.ws_url = ws_url
        self.mode = mode
        self.timeout = timeout

        # HTTP client for REST operations
        self._http_client = GameHTTPClient(
            base_url=http_url,
            timeout=timeout,
        )

        # WebSocket state (only used in WEBSOCKET mode)
        self._websocket: ClientConnection | None = None
        self._listener_task: asyncio.Task[None] | None = None
        self._message_handlers: list[Callable[[dict], Awaitable[None]]] = []

        # Player/room state
        self.player_id: str | None = None
        self.player_token: str | None = None
        self.room_id: str | None = None
        self.room_code: str | None = None

        # Canvas dimensions (set after joining/creating game)
        self.canvas_size_x: int = 800
        self.canvas_size_y: int = 800

        # Cached game state (updated by WebSocket messages or REST polling)
        self._game_state: GameState | None = None

        self._logger = logging.getLogger(__name__)

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> GameClient:
        """Async context manager entry."""
        await self._http_client.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Connection Methods
    # =========================================================================

    async def create_game(
        self,
        player_name: str,
        room_name: str,
        map_type: str,
        training_mode: bool = False,
        tick_rate_multiplier: float = 1.0,
        max_game_duration_sec: int | None = None,
        disable_respawn_timer: bool = False,
        max_kills: int | None = None,
    ) -> CreateGameResponse:
        """Create a new game room.

        Args:
            player_name: Name of the player creating the game.
            room_name: Display name for the room.
            map_type: Map to use (e.g., "arena1").
            training_mode: Enable training mode options.
            tick_rate_multiplier: Speed multiplier for training (requires training_mode).
            max_game_duration_sec: Maximum game duration in seconds.
            disable_respawn_timer: Disable respawn timer (training mode).
            max_kills: Maximum kills before game ends.

        Returns:
            CreateGameResponse with room details and player credentials.
        """
        response = await self._http_client.create_game(
            player_name=player_name,
            room_name=room_name,
            map_type=map_type,
            training_mode=training_mode if training_mode else None,
            tick_multiplier=tick_rate_multiplier if training_mode else None,
            max_game_duration_sec=max_game_duration_sec,
            disable_respawn_timer=disable_respawn_timer
            if disable_respawn_timer
            else None,
            max_kills=max_kills,
        )

        self.player_id = response.player_id
        self.player_token = response.player_token
        self.room_id = response.room_id
        self.room_code = response.room_code
        self.canvas_size_x = response.canvas_size_x
        self.canvas_size_y = response.canvas_size_y

        if self.mode == ClientMode.WEBSOCKET:
            await self._connect_websocket()

        return response

    async def join_game(
        self,
        player_name: str,
        room_code: str,
        room_password: str = "",
        is_spectator: bool = False,
    ) -> JoinGameResponse:
        """Join an existing game room.

        Args:
            player_name: Name of the player joining.
            room_code: Room code to join.
            room_password: Optional room password.
            is_spectator: Join as spectator (watch only).

        Returns:
            JoinGameResponse with player credentials.
        """
        response = await self._http_client.join_game(
            player_name=player_name,
            room_code=room_code,
            room_password=room_password,
            is_spectator=is_spectator,
        )

        self.player_id = response.player_id
        self.player_token = response.player_token
        self.room_id = response.room_id
        self.room_code = room_code
        self.canvas_size_x = response.canvas_size_x
        self.canvas_size_y = response.canvas_size_y

        if self.mode == ClientMode.WEBSOCKET:
            await self._connect_websocket()

        return response

    # =========================================================================
    # WebSocket Methods (Real-time Mode)
    # =========================================================================

    async def _connect_websocket(self) -> None:
        """Establish WebSocket connection and rejoin game."""
        if self._websocket is not None:
            return

        self._logger.info("Connecting to WebSocket at %s", self.ws_url)
        self._websocket = await websockets.connect(self.ws_url)

        # Start message listener
        self._listener_task = asyncio.create_task(self._listen_for_messages())

        # Rejoin game with credentials
        rejoin_message = {
            "type": "RejoinGame",
            "payload": {
                "playerId": self.player_id,
                "playerToken": self.player_token,
                "roomId": self.room_id,
            },
        }
        await self._websocket.send(json.dumps(rejoin_message))
        self._logger.info("Rejoined game via WebSocket")

    async def _listen_for_messages(self) -> None:
        """Listen for incoming WebSocket messages."""
        if self._websocket is None:
            return

        try:
            async for message in self._websocket:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                await self._handle_message(message)
        except ConnectionClosed:
            self._logger.warning("WebSocket connection closed")
        except Exception as e:
            self._logger.exception("Error in WebSocket listener: %s", e)

    async def _handle_message(self, message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Update cached game state if this is a GameUpdate message
            if data.get("type") == "GameUpdate":
                payload = data.get("payload", {})
                game_update = GameUpdate.model_validate(payload)
                self._game_state = GameState.from_update(
                    game_update,
                    existing_state=self._game_state,
                    canvas_size_x=self.canvas_size_x,
                    canvas_size_y=self.canvas_size_y,
                )

            # Call registered handlers
            for handler in self._message_handlers:
                try:
                    await handler(data)
                except Exception as e:
                    self._logger.error("Error in message handler: %s", e)

        except json.JSONDecodeError:
            self._logger.error("Failed to parse message: %s", message)

    def register_message_handler(
        self,
        handler: Callable[[dict], Awaitable[None]],
    ) -> None:
        """Register a handler for WebSocket messages.

        Args:
            handler: Async function called with each message dict.
        """
        self._message_handlers.append(handler)

    def unregister_message_handler(
        self,
        handler: Callable[[dict], Awaitable[None]],
    ) -> None:
        """Remove a previously registered message handler."""
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)

    # =========================================================================
    # Action Methods (Mode-Aware)
    # =========================================================================

    async def send_keyboard_input(self, key: str, pressed: bool) -> None:
        """Send keyboard input (W, A, S, D keys).

        Args:
            key: Key character (w, a, s, d).
            pressed: True for key down, False for key up.
        """
        if self.mode == ClientMode.WEBSOCKET:
            await self._send_ws_keyboard(key, pressed)
        else:
            await self._send_rest_action(
                BotAction(type="key", key=key, is_down=pressed)
            )

    async def send_mouse_input(
        self,
        button: str,
        pressed: bool,
        x: float,
        y: float,
    ) -> None:
        """Send mouse input (click/aim).

        Args:
            button: Button name ("left" or "right").
            pressed: True for button down, False for button up.
            x: Mouse X coordinate.
            y: Mouse Y coordinate.
        """
        # Convert button name to button code (0=left, 2=right)
        button_code = 0 if button == "left" else 2

        if self.mode == ClientMode.WEBSOCKET:
            await self._send_ws_mouse(button_code, pressed, x, y)
        else:
            await self._send_rest_action(
                BotAction(
                    type="click",
                    button=button_code,
                    is_down=pressed,
                    x=x,
                    y=y,
                )
            )

    async def send_direction(self, direction: float) -> None:
        """Send aim direction update.

        Args:
            direction: Direction in radians (0 to 2π).
        """
        if self.mode == ClientMode.WEBSOCKET:
            await self._send_ws_direction(direction)
        else:
            await self._send_rest_action(
                BotAction(type="direction", direction=direction)
            )

    async def _send_ws_keyboard(self, key: str, pressed: bool) -> None:
        """Send keyboard input via WebSocket."""
        if self._websocket is None:
            raise GameClientError("WebSocket not connected")

        message = {"type": "Key", "payload": {"key": key, "isDown": pressed}}
        await self._websocket.send(json.dumps(message))

    async def _send_ws_mouse(
        self,
        button: int,
        pressed: bool,
        x: float,
        y: float,
    ) -> None:
        """Send mouse input via WebSocket.

        Args:
            button: Button code (0=left, 2=right).
            pressed: True for button down, False for button up.
            x: Mouse X coordinate.
            y: Mouse Y coordinate.
        """
        if self._websocket is None:
            raise GameClientError("WebSocket not connected")

        message = {
            "type": "Click",
            "payload": {"x": x, "y": y, "button": button, "isDown": pressed},
        }
        await self._websocket.send(json.dumps(message))

    async def _send_ws_direction(self, direction: float) -> None:
        """Send direction update via WebSocket.

        Args:
            direction: Direction in radians.
        """
        if self._websocket is None:
            raise GameClientError("WebSocket not connected")

        message = {"type": "ClientState", "payload": {"direction": direction}}
        await self._websocket.send(json.dumps(message))

    async def _send_rest_action(self, action: BotAction) -> None:
        """Send action via REST API (training mode)."""
        if self.room_id is None or self.player_id is None:
            raise GameClientError("Not connected to a game")

        await self._http_client.submit_action(
            room_id=self.room_id,
            player_id=self.player_id,
            actions=[action],
        )

    # =========================================================================
    # Game State Methods
    # =========================================================================

    async def get_game_state(self) -> GameState:
        """Get current game state.

        In WEBSOCKET mode: Returns cached state from last broadcast.
        In REST mode: Polls server for current state.

        Returns:
            Current GameState.

        Raises:
            GameClientError: If not connected or no state available.
        """
        if self.mode == ClientMode.WEBSOCKET:
            if self._game_state is None:
                raise GameClientError("No game state received yet")
            return self._game_state
        else:
            if self.room_id is None:
                raise GameClientError("Not connected to a game")
            response = await self._http_client.get_game_state(self.room_id)
            if response.game_update is None:
                raise GameClientError("No game state available from server")
            game_update = GameUpdate.model_validate(response.game_update)
            self._game_state = GameState.from_update(
                game_update,
                existing_state=self._game_state,
                canvas_size_x=self.canvas_size_x,
                canvas_size_y=self.canvas_size_y,
            )
            return self._game_state

    async def wait_for_game_state(
        self,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> GameState:
        """Wait for game state to be available.

        In WEBSOCKET mode: Waits for the first game state broadcast from server.
        In REST mode: Equivalent to get_game_state() (polls immediately).

        This is useful after joining a game in WebSocket mode, where the first
        game state may not be immediately available.

        Args:
            timeout: Maximum time to wait in seconds.
            poll_interval: Time between checks in seconds (WebSocket mode only).

        Returns:
            Current GameState once available.

        Raises:
            GameClientError: If not connected to a game.
            TimeoutError: If no game state received within timeout.

        Example:
            async with GameClient(mode=ClientMode.WEBSOCKET) as client:
                await client.join_game(player_name="Bot", room_code="ABC123")
                state = await client.wait_for_game_state(timeout=10.0)
        """
        if self.mode == ClientMode.REST:
            return await self.get_game_state()

        if self._websocket is None:
            raise GameClientError("Not connected to a game")

        elapsed = 0.0
        while elapsed < timeout:
            if self._game_state is not None:
                return self._game_state
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(
            f"No game state received within {timeout} seconds. "
            "Ensure the game has started and WebSocket is connected."
        )

    async def reset_game(self, map_type: str | None = None) -> None:
        """Reset game for new training episode (REST mode only).

        Args:
            map_type: Optional new map type to use after reset.

        Raises:
            GameClientError: If not in REST mode or not connected.
        """
        if self.mode != ClientMode.REST:
            raise GameClientError("reset_game() only available in REST mode")
        if self.room_id is None:
            raise GameClientError("Not connected to a game")

        response = await self._http_client.reset_game(self.room_id, map_type)
        if response.canvas_size_x is not None:
            self.canvas_size_x = response.canvas_size_x
        if response.canvas_size_y is not None:
            self.canvas_size_y = response.canvas_size_y
        # Clear cached state after reset
        self._game_state = None

    async def get_stats(self) -> dict[str, PlayerStatsDTO]:
        """Get kill/death statistics for reward calculation.

        Returns:
            Dict with player statistics.
        """
        if self.room_id is None:
            raise GameClientError("Not connected to a game")

        return await self._http_client.get_game_stats(self.room_id)

    # =========================================================================
    # Cleanup Methods
    # =========================================================================

    async def exit_game(self) -> None:
        """Exit the current game gracefully."""
        if self.mode == ClientMode.WEBSOCKET and self._websocket is not None:
            try:
                message = {"type": "ExitGame", "payload": {}}
                await self._websocket.send(json.dumps(message))
            except Exception as e:
                self._logger.warning("Error sending exit message: %s", e)

        await self.close()

    async def close(self) -> None:
        """Close all connections and cleanup."""
        # Cancel WebSocket listener
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        # Close WebSocket
        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception as e:
                self._logger.warning("Error closing WebSocket: %s", e)
            self._websocket = None

        # Close HTTP client
        await self._http_client.close()

        # Clear state
        self.player_id = None
        self.player_token = None
        self.room_id = None
        self.room_code = None
        self._game_state = None
        self._message_handlers.clear()
