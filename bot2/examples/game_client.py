# Example implementation of a GameClient that connects to a game server via WebSocket and HTTP API.

import asyncio
import json
import logging
from enum import Enum
from typing import Optional
from urllib.parse import urljoin

import aiohttp
import websockets


class InputType(Enum):
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    DIRECTION = "direction"


class GameClient:
    def __init__(
        self,
        ws_url: str = "ws://localhost:4000/ws",
        http_url: str = "http://localhost:4000",
    ):
        self.ws_url = ws_url
        self.http_url = http_url
        self.websocket = None
        self.game_state = {}
        self.player_id = None
        self.player_token = None
        self.room_id = None
        self._logger = logging.getLogger(__name__)
        self._message_handlers = []

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
                "roomPassword": room_password,
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

            # Start listening for messages
            listener_task = asyncio.create_task(self._listen_for_messages())

            # Rejoin the game room with our token
            self._logger.info("Rejoining game room with token")
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
        """Send mouse input (left/right click)"""
        if not self.websocket:
            return

        message = {
            "type": "Click",
            "payload": {
                "x": x,
                "y": y,
                "button": button,
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
                self._logger.exception("Error in message listener", e)
                break

    async def close(self) -> None:
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
