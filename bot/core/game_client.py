import json
import asyncio
import aiohttp
from dataclasses import dataclass
import websockets
from enum import Enum
from typing import Dict, Optional, Any, Callable, Awaitable
import logging
from urllib.parse import urljoin
from core.game_state import (
    GameState,
    PlayerState,
    ArrowState,
    Block,
    player_state_from_dict,
    arrow_state_from_dict,
    block_from_dict,
)


class InputType(Enum):
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    DIRECTION = "direction"


class TrainingMode(Enum):
    """Training mode configurations for GameClient."""

    NORMAL = "normal"  # Standard WebSocket communication
    TRAINING = "training"  # Direct state access with speed control
    HEADLESS = "headless"  # Maximum speed with minimal communication


class GameClient:
    def __init__(
        self,
        ws_url: str = "ws://localhost:4000/ws",
        http_url: str = "http://localhost:4000",
    ):
        self.ws_url = ws_url
        self.http_url = http_url
        self.websocket = None
        self.game_state: GameState = GameState(
            player=None, enemies={}, blocks={}, arrows={}
        )
        self.player_id = None
        self.player_token = None
        self.room_id = None
        self._logger = logging.getLogger(__name__)
        self._message_handlers = []

        # Training mode extensions
        self.training_mode = TrainingMode.NORMAL
        self.speed_multiplier = 1.0
        self.direct_state_access = False
        self._state_cache = {}
        self._last_state_update = 0
        self._training_session_id = None
        self._state_update_callbacks: list[
            Callable[[Dict[str, Any]], Awaitable[None]]
        ] = []

    async def connect(
        self, room_code: str, player_name: str, room_password: Optional[str] = None
    ) -> None:
        """Connect to a game room"""
        try:
            # First join via HTTP API
            self._logger.info(
                f"Joining game via HTTP API {room_code} as player {player_name}"
            )

            join_data = {
                "playerName": player_name,
                "roomCode": room_code,
                "roomPassword": room_password or "",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    urljoin(self.http_url, "api/joinGame"), json=join_data
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            self.player_id = data["playerId"]
            self.player_token = data["playerToken"]
            self.room_id = data["roomId"]

            # Now connect to WebSocket
            self._logger.info(f"Connecting to websocket server at {self.ws_url}")
            self.websocket = await websockets.connect(self.ws_url)

            self.register_message_handler(self._handle_game_state_update)

            # Start listening for messages
            listener_task = asyncio.create_task(self._listen_for_messages())

            # Rejoin the game room with our token
            self._logger.info(f"Rejoining game room with token")
            rejoin_message = {
                "type": "RejoinGame",
                "payload": {
                    "playerId": self.player_id,
                    "playerToken": self.player_token,
                    "roomId": self.room_id,
                },
            }

            await self.websocket.send(json.dumps(rejoin_message))

            # Keep the listener task running
            return listener_task
        except Exception as e:
            self._logger.error(f"Failed to connect: {e}")
            raise

    async def send_keyboard_input(self, key: str, pressed: bool) -> None:
        """Send keyboard input (W, A, S, D)"""
        if not self.websocket:
            return

        message = {"type": "Key", "payload": {"key": key, "isDown": pressed}}
        await self.websocket.send(json.dumps(message))

    async def send_mouse_input(
        self, button: str, pressed: bool, x: float, y: float
    ) -> None:
        """Send mouse input (left/right click)

        Args:
            button: "left" or "right" (converted to 0 or 2 for Go server)
            pressed: True for mouse down, False for mouse up
            x: X coordinate of click
            y: Y coordinate of click
        """
        if not self.websocket:
            return

        # Convert button string to integer as expected by Go server
        # 0 for left click, 2 for right click
        button_lower = button.lower()
        if button_lower == "left":
            button_code = 0
        elif button_lower == "right":
            button_code = 2
        else:
            self._logger.warning(f"Unknown button '{button}', defaulting to left click")
            button_code = 0

        message = {
            "type": "Click",
            "payload": {
                "x": x,
                "y": y,
                "button": button_code,
                "isDown": pressed,
            },
        }
        await self.websocket.send(json.dumps(message))

    def register_message_handler(self, handler) -> None:
        self._message_handlers.append(handler)

    async def exit_game(self) -> None:
        if not self.websocket:
            return
        message = {"type": "ExitGame", "payload": {}}
        await self.websocket.send(json.dumps(message))
        await self.close()

    async def _handle_message(self, message: str) -> None:
        """Handle incoming websocket messages"""
        try:
            data = json.loads(message)
            for handler in self._message_handlers:
                try:
                    await handler(data)
                except Exception as e:
                    self._logger.error(f"Error in message handler: {e}")
        except json.JSONDecodeError:
            self._logger.error(f"Failed to parse message: {message}")

    async def _listen_for_messages(self) -> None:
        """Listen for incoming websocket messages"""
        while True:
            try:
                message = await self.websocket.recv()
                await self._handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                self._logger.error("Connection to server closed")
                break
            except Exception as e:
                self._logger.exception(f"Error in message listener", e)
                break

    async def close(self) -> None:
        """Close the WebSocket connection"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                self._logger.exception("Error closing WebSocket connection", e)
            self.websocket = None

    # Training mode extensions

    async def enable_training_mode(
        self,
        speed_multiplier: float = 10.0,
        headless: bool = False,
        training_session_id: Optional[str] = None,
    ) -> None:
        """
        Enable training mode with accelerated game speed and direct state access.

        Args:
            speed_multiplier: Game speed multiplier (1.0 = normal, 10.0 = 10x speed)
            headless: Enable headless mode for maximum speed
            training_session_id: Optional training session ID for room management
        """
        self.training_mode = (
            TrainingMode.HEADLESS if headless else TrainingMode.TRAINING
        )
        self.speed_multiplier = speed_multiplier
        self.direct_state_access = True
        self._training_session_id = training_session_id

        # Configure training room if we have a room
        if self.room_id:
            await self._configure_training_room()

        self._logger.info(
            f"Training mode enabled - Speed: {speed_multiplier}x, Headless: {headless}"
        )

    async def disable_training_mode(self) -> None:
        """Disable training mode and return to normal operation."""
        self.training_mode = TrainingMode.NORMAL
        self.speed_multiplier = 1.0
        self.direct_state_access = False
        self._training_session_id = None

        # Reset training room configuration if we have a room
        if self.room_id:
            await self._configure_training_room()

        self._logger.info("Training mode disabled")

    async def get_direct_state(self) -> Dict[str, Any]:
        """
        Get game state directly without WebSocket communication.
        This bypasses normal message handling for faster state retrieval.

        Returns:
            Current game state dictionary
        """
        if not self.direct_state_access:
            raise RuntimeError(
                "Direct state access not enabled. Call enable_training_mode() first."
            )

        try:
            # Make direct HTTP request to training API for state
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    urljoin(self.http_url, f"api/training/rooms/{self.room_id}/state"),
                    headers={"Authorization": f"Bearer {self.player_token}"},
                ) as response:
                    response.raise_for_status()
                    state_data = await response.json()

                    # Update cached state - handle both lowercase and capitalized field names
                    self._state_cache = state_data.get(
                        "state", state_data.get("State", {})
                    )
                    self._last_state_update = state_data.get(
                        "timestamp", state_data.get("Timestamp", 0)
                    )

                    # Notify callbacks
                    for callback in self._state_update_callbacks:
                        try:
                            await callback(self._state_cache)
                        except Exception as e:
                            self._logger.error(f"Error in state update callback: {e}")

                    return self._state_cache

        except Exception as e:
            self._logger.error(f"Failed to get direct state: {e}")
            # Fallback to cached state if available
            return self._state_cache

    async def set_room_speed(self, speed_multiplier: float) -> None:
        """
        Set the speed multiplier for the current training room.

        Args:
            speed_multiplier: New speed multiplier (1.0 = normal speed)
        """
        if not self.room_id:
            raise RuntimeError("No active room to configure")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    urljoin(self.http_url, f"api/training/rooms/{self.room_id}/speed"),
                    json={"speedMultiplier": speed_multiplier},
                    headers={"Authorization": f"Bearer {self.player_token}"},
                ) as response:
                    response.raise_for_status()

            self.speed_multiplier = speed_multiplier
            self._logger.info(f"Room speed set to {speed_multiplier}x")

        except Exception as e:
            self._logger.error(f"Failed to set room speed: {e}")
            raise

    async def create_training_room(
        self,
        room_name: str,
        player_name: str,
        speed_multiplier: float = 10.0,
        headless: bool = False,
        map_type: str = "default",
    ) -> Dict[str, Any]:
        """
        Create a new training room with speed control capabilities.

        Args:
            room_name: Name for the training room
            player_name: Name for the player/bot
            speed_multiplier: Initial speed multiplier
            headless: Enable headless mode
            map_type: Map type to use

        Returns:
            Room creation response with room details
        """
        try:
            training_config = {
                "speedMultiplier": speed_multiplier,
                "headlessMode": headless,
                "trainingMode": True,
                "sessionId": self._training_session_id,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    urljoin(self.http_url, "api/training/createRoom"),
                    json={
                        "playerName": player_name,
                        "roomName": room_name,
                        "mapType": map_type,
                        "trainingConfig": training_config,
                    },
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            self.player_id = data["playerId"]
            self.player_token = data["playerToken"]
            self.room_id = data["roomId"]
            self.training_mode = (
                TrainingMode.HEADLESS if headless else TrainingMode.TRAINING
            )
            self.speed_multiplier = speed_multiplier
            self.direct_state_access = True

            self._logger.info(
                f"Created training room {data['roomCode']} with {speed_multiplier}x speed"
            )
            return data

        except Exception as e:
            self._logger.error(f"Failed to create training room: {e}")
            raise

    async def join_training_room(
        self,
        room_code: str,
        player_name: str,
        room_password: str,
        enable_direct_access: bool = True,
    ) -> Dict[str, Any]:
        """
        Join an existing training room with training mode capabilities.

        Args:
            room_code: Room code to join
            player_name: Name for the player/bot
            room_password: Room password
            enable_direct_access: Enable direct state access

        Returns:
            Join response with room details
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    urljoin(self.http_url, "api/training/joinRoom"),
                    json={
                        "playerName": player_name,
                        "roomCode": room_code,
                        "roomPassword": room_password,
                        "enableDirectAccess": enable_direct_access,
                    },
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            self.player_id = data["playerId"]
            self.player_token = data["playerToken"]
            self.room_id = data["roomId"]

            # Check if room has training capabilities
            if data.get("trainingEnabled", False):
                self.training_mode = (
                    TrainingMode.HEADLESS
                    if data.get("headlessMode", False)
                    else TrainingMode.TRAINING
                )
                self.speed_multiplier = data.get("speedMultiplier", 1.0)
                self.direct_state_access = enable_direct_access

            self._logger.info(f"Joined training room {room_code}")
            return data

        except Exception as e:
            self._logger.error(f"Failed to join training room: {e}")
            raise

    def register_state_update_callback(
        self, callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Register a callback to be called when state is updated via direct access.

        Args:
            callback: Async function to call with updated state
        """
        self._state_update_callbacks.append(callback)

    def unregister_state_update_callback(
        self, callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Unregister a state update callback.

        Args:
            callback: Callback function to remove
        """
        if callback in self._state_update_callbacks:
            self._state_update_callbacks.remove(callback)

    async def _configure_training_room(self) -> None:
        """Configure the current room for training mode."""
        if not self.room_id:
            return

        try:
            config = {
                "trainingMode": self.training_mode.value,
                "speedMultiplier": self.speed_multiplier,
                "directStateAccess": self.direct_state_access,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    urljoin(
                        self.http_url, f"api/training/rooms/{self.room_id}/configure"
                    ),
                    json=config,
                    headers={"Authorization": f"Bearer {self.player_token}"},
                ) as response:
                    response.raise_for_status()

        except Exception as e:
            self._logger.error(f"Failed to configure training room: {e}")

    def is_training_mode(self) -> bool:
        """Check if client is in training mode."""
        return self.training_mode != TrainingMode.NORMAL

    def get_training_info(self) -> Dict[str, Any]:
        """Get current training mode information."""
        return {
            "training_mode": self.training_mode.value,
            "speed_multiplier": self.speed_multiplier,
            "direct_state_access": self.direct_state_access,
            "training_session_id": self._training_session_id,
            "last_state_update": self._last_state_update,
        }

    def _handle_game_state_update(self, message_data: Dict[str, Any]) -> None:
        """Handle game state update."""
        if not message_data.get("type") == "GameState":
            return

        if message_data["payload"].get("fullUpdate", False):
            self.game_state.player = None
            self.game_state.enemies = {}
            self.game_state.blocks = {}
            self.game_state.arrows = {}

        for obj_id, obj in message_data["payload"]["objectStates"].items():
            if obj["type"] == "player":
                if obj_id == self.player_id:
                    self.game_state.player = player_state_from_dict(obj)
                else:
                    self.game_state.enemies[obj_id] = player_state_from_dict(obj)
            elif obj["type"] == "block":
                self.game_state.blocks[obj_id] = block_from_dict(obj)
            elif obj["type"] == "arrow":
                self.game_state.arrows[obj_id] = arrow_from_dict(obj)
            else:
                self._logger.warning(f"Unknown object type: {obj['type']}")
