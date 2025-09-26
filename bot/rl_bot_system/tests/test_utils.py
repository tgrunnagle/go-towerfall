"""
Test utilities for bot-game server integration tests.
"""

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import aiohttp
import websockets

# Import the original GameClient to subclass it
import sys
from pathlib import Path

# Add bot directory to path for imports
bot_dir = Path(__file__).parent.parent.parent
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

from core.game_client import GameClient


class EnhancedGameClient(GameClient):
    """Test-enhanced game client that extends GameClient with testing utilities."""
    
    def __init__(self, ws_url: str, http_url: str):
        super().__init__(ws_url, http_url)
        self.test_messages: List[Dict[str, Any]] = []
        self._test_message_handler_registered = False
    
    async def connect_to_room(self, room_code: str, player_name: str, room_password: str = "") -> bool:
        """Connect to a game room with test-friendly return value."""
        try:
            await self.connect(room_code, player_name, room_password)
            
            # Register test message handler if not already done
            if not self._test_message_handler_registered:
                self.register_message_handler(self._capture_test_messages)
                self._test_message_handler_registered = True
            
            return True
        except Exception as e:
            self._logger.error(f"Failed to connect to room: {e}")
            return False
    
    async def send_keyboard_input(self, key: str, pressed: bool) -> bool:
        """Send keyboard input with test-friendly return value."""
        try:
            await super().send_keyboard_input(key, pressed)
            return True
        except Exception as e:
            self._logger.error(f"Failed to send keyboard input: {e}")
            return False
    
    async def send_mouse_input(self, button: str, pressed: bool, x: float, y: float) -> bool:
        """Send mouse input with test-friendly return value."""
        try:
            await super().send_mouse_input(button, pressed, x, y)
            return True
        except Exception as e:
            self._logger.error(f"Failed to send mouse input: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the game."""
        try:
            await self.exit_game()
        except Exception as e:
            self._logger.error(f"Error during disconnect: {e}")
    
    async def _capture_test_messages(self, message: Dict[str, Any]):
        """Capture messages for test validation."""
        self.test_messages.append(message)
    
    def get_captured_messages(self) -> List[Dict[str, Any]]:
        """Get all messages captured during testing."""
        return self.test_messages.copy()
    
    def clear_captured_messages(self):
        """Clear captured messages."""
        self.test_messages.clear()
    
    def get_messages_by_type(self, message_type: str) -> List[Dict[str, Any]]:
        """Get captured messages of a specific type."""
        return [msg for msg in self.test_messages if msg.get("type") == message_type]
    
    def wait_for_message_type(self, message_type: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Wait for a specific message type to be received."""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            messages = self.get_messages_by_type(message_type)
            if messages:
                return messages[-1]  # Return the most recent message
            time.sleep(0.1)
        
        return None
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for debugging."""
        return {
            "player_id": self.player_id,
            "player_token": self.player_token,
            "room_id": self.room_id,
            "connected": self.websocket is not None,
            "messages_captured": len(self.test_messages)
        }
    
    async def send_test_sequence(self, actions: List[Tuple[str, Any]]) -> bool:
        """Send a sequence of test actions."""
        try:
            for action_type, action_data in actions:
                if action_type == "keyboard":
                    key, pressed = action_data
                    await self.send_keyboard_input(key, pressed)
                elif action_type == "mouse":
                    button, pressed, x, y = action_data
                    await self.send_mouse_input(button, pressed, x, y)
                elif action_type == "wait":
                    await asyncio.sleep(action_data)
                
                # Small delay between actions
                await asyncio.sleep(0.05)
            
            return True
        except Exception as e:
            self._logger.error(f"Failed to send test sequence: {e}")
            return False


class ServerHealthChecker:
    """Utility for checking server health."""
    
    @staticmethod
    async def check_go_server(url: str, timeout: float = 5.0) -> bool:
        """Check if Go server is healthy."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(f"{url}/api/maps") as response:
                    return response.status == 200
        except Exception:
            return False
    
    @staticmethod
    async def check_websocket_server(ws_url: str, timeout: float = 5.0) -> bool:
        """Check if WebSocket server is accessible."""
        try:
            websocket = await asyncio.wait_for(
                websockets.connect(ws_url),
                timeout=timeout
            )
            await websocket.close()
            return True
        except Exception:
            return False
    
    @staticmethod
    async def wait_for_server_ready(url: str, ws_url: str, max_wait: float = 30.0) -> bool:
        """Wait for both HTTP and WebSocket servers to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            http_ready = await ServerHealthChecker.check_go_server(url)
            ws_ready = await ServerHealthChecker.check_websocket_server(ws_url)
            
            if http_ready and ws_ready:
                return True
            
            await asyncio.sleep(0.5)
        
        return False


class GameRoomManager:
    """Utility for managing test game rooms."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.created_rooms: List[Tuple[str, str]] = []  # (room_id, room_code)
        self.logger = logging.getLogger(__name__)
    
    async def create_room(self, room_name: str = "TestRoom", map_type: str = "default") -> Tuple[str, str]:
        """Create a test room and return (room_id, room_code)."""
        try:
            create_data = {
                "playerName": "TestHost",
                "roomName": room_name,
                "mapType": map_type
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.server_url}/api/createGame", json=create_data) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if not data.get("success"):
                        raise RuntimeError(f"Failed to create room: {data.get('error')}")
                    
                    room_id = data["roomId"]
                    room_code = data["roomCode"]
                    
                    self.created_rooms.append((room_id, room_code))
                    self.logger.info(f"Created test room: {room_code} (ID: {room_id})")
                    
                    return room_id, room_code
                    
        except Exception as e:
            self.logger.error(f"Error creating test room: {e}")
            raise
    
    async def cleanup_rooms(self):
        """Clean up all created test rooms."""
        # Note: The Go server doesn't have a delete room API,
        # so rooms will be cleaned up when the server restarts
        self.created_rooms.clear()


def validate_message_structure(message: Dict[str, Any]) -> bool:
    """Validate that a message has the expected structure."""
    if not isinstance(message, dict):
        return False
    
    if "type" not in message:
        return False
    
    # Common message types and their expected structure
    message_type = message["type"]
    
    if message_type in ["GameState", "PlayerJoined", "PlayerLeft", "GameStarted", "GameEnded"]:
        return "payload" in message and isinstance(message["payload"], dict)
    
    return True


def extract_game_state_info(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract game state information from a list of messages."""
    game_states = [msg for msg in messages if msg.get("type") == "GameState"]
    
    if not game_states:
        return {}
    
    latest_state = game_states[-1]
    payload = latest_state.get("payload", {})
    
    return {
        "player_count": len(payload.get("players", [])),
        "game_objects": len(payload.get("gameObjects", [])),
        "has_players": "players" in payload,
        "has_game_objects": "gameObjects" in payload,
        "message_count": len(game_states)
    }