import json
import asyncio
import websockets
from enum import Enum
from typing import Dict, Optional
import logging

class InputType(Enum):
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    DIRECTION = "direction"

class GameClient:
    def __init__(self, server_url: str = "ws://localhost:4000/ws"):
        self.server_url = server_url
        self.websocket = None
        self.game_state = {}
        self.player_id = None
        self.room_id = None
        self._logger = logging.getLogger(__name__)

    async def _handle_message(self, message: str) -> None:
        """Handle incoming websocket messages"""
        try:
            data = json.loads(message)
            self._logger.info(f"Received message: {data}")
            
            # TODO: Handle different message types
            message_type = data.get('type')
            if message_type == 'game_state':
                self.game_state = data.get('state', {})
            elif message_type == 'error':
                self._logger.error(f"Server error: {data.get('error')}")
        except json.JSONDecodeError:
            self._logger.error(f"Failed to parse message: {message}")
        except Exception as e:
            self._logger.error(f"Error handling message: {e}")

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
                self._logger.error(f"Error in message listener: {e}")
                break

    async def connect(self, room_code: str, player_name: str, room_password: Optional[str] = None) -> None:
        """Connect to a game room"""
        try:
            self._logger.info(f"Connecting to websocket server at {self.server_url}")
            self.websocket = await websockets.connect(self.server_url)

            # Start listening for messages
            listener_task = asyncio.create_task(self._listen_for_messages())

            # Join the game room
            self._logger.info(f"Joining game room {room_code} as player {player_name}")
            join_message = {
                "type": "join_game",
                "roomCode": room_code,
                "playerName": player_name
            }
            if room_password:
                join_message["roomPassword"] = room_password

            await self.websocket.send(json.dumps(join_message))

            # Keep the listener task running
            return listener_task
        except Exception as e:
            self._logger.error(f"Failed to connect: {e}")
            raise

    async def send_keyboard_input(self, key: str, pressed: bool) -> None:
        """Send keyboard input (W, A, S, D)"""
        if not self.websocket:
            return

        message = {
            "type": "player_input",
            "inputType": InputType.KEYBOARD.value,
            "key": key,
            "pressed": pressed
        }
        await self.websocket.send(json.dumps(message))

    async def send_mouse_input(self, button: str, pressed: bool, x: float, y: float) -> None:
        """Send mouse input (left/right click)"""
        if not self.websocket:
            return

        message = {
            "type": "player_input",
            "inputType": InputType.MOUSE.value,
            "button": button,
            "pressed": pressed,
            "x": x,
            "y": y
        }
        await self.websocket.send(json.dumps(message))

    async def send_direction(self, x: float, y: float) -> None:
        """Send player direction"""
        if not self.websocket:
            return

        message = {
            "type": "player_input",
            "inputType": InputType.DIRECTION.value,
            "x": x,
            "y": y
        }
        await self.websocket.send(json.dumps(message))

    async def _handle_game_state(self, state: Dict) -> None:
        """Handle incoming game state. Override this method to implement custom state handling."""
        pass

    async def close(self) -> None:
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
