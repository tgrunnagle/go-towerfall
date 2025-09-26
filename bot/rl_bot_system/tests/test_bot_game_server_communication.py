"""
Bot-Game Server Communication Validation Tests

This module implements comprehensive integration tests that validate communication
between the Python bot server and Go game server. Tests cover bot spawning,
action execution, game state validation, message serialization, error handling,
and authentication.

Requirements covered: 6.1, 6.2, 7.1, 7.2
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
import pytest_asyncio
import websockets
from websockets.exceptions import ConnectionClosed

from game_client import GameClient, TrainingMode
from rl_bot_system.server.bot_server import (
    BotServer, BotServerConfig, BotConfig, BotType, BotStatus
)
from rl_bot_system.server.bot_server_api import BotServerApi
from rl_bot_system.rules_based.rules_based_bot import DifficultyLevel


class ServerManager:
    """Manages test server connections and bot server lifecycle."""
    
    def __init__(self):
        self.bot_server: Optional[BotServer] = None
        self.bot_server_api: Optional[BotServerApi] = None
        self.logger = logging.getLogger(__name__)
        
        # Server configuration
        self.go_server_port = 4000
        self.bot_server_port = 4002
        self.go_server_url = f"http://localhost:{self.go_server_port}"
        self.bot_server_url = f"http://localhost:{self.bot_server_port}"
        self.ws_url = f"ws://localhost:{self.go_server_port}/ws"
    
    async def check_go_server_running(self, timeout: float = 5.0) -> bool:
        """Check if the Go game server is running and accessible."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(f"{self.go_server_url}/api/maps") as response:
                    if response.status == 200:
                        self.logger.info("Go server is running and accessible")
                        return True
                    else:
                        self.logger.error(f"Go server returned status {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Go server is not accessible: {e}")
            return False
    
    async def start_bot_server(self) -> bool:
        """Start the Python bot server."""
        try:
            config = BotServerConfig(
                max_bots_per_room=5,
                max_total_bots=20,
                bot_timeout_seconds=120,
                cleanup_interval_seconds=30,
                game_server_url=self.go_server_url
            )
            
            self.bot_server = BotServer(config)
            self.bot_server_api = BotServerApi(config)
            
            await self.bot_server.start()
            await self.bot_server_api.initialize()
            
            self.logger.info("Bot server started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting bot server: {e}")
            return False
    
    async def stop_bot_server(self):
        """Stop the bot server."""
        # Stop bot server
        if self.bot_server_api:
            try:
                await self.bot_server_api.cleanup()
            except Exception as e:
                self.logger.error(f"Error stopping bot server API: {e}")
        
        if self.bot_server:
            try:
                await self.bot_server.stop()
            except Exception as e:
                self.logger.error(f"Error stopping bot server: {e}")
    
    async def create_test_room(self, room_name: str = "TestRoom") -> Tuple[str, str, str]:
        """Create a test room and return room_id, room_code, and room_password."""
        try:
            async with aiohttp.ClientSession() as session:
                create_data = {
                    "playerName": "TestHost",
                    "roomName": room_name,
                    "mapType": "default"
                }
                
                async with session.post(
                    f"{self.go_server_url}/api/createGame",
                    json=create_data
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if not data.get("success"):
                        raise RuntimeError(f"Failed to create room: {data.get('error')}")
                    
                    room_id = data["roomId"]
                    room_code = data["roomCode"]
                
                # Get room details to retrieve the password
                async with session.get(
                    f"{self.go_server_url}/api/rooms/{room_id}/details"
                ) as response:
                    response.raise_for_status()
                    details_data = await response.json()
                    
                    if not details_data.get("success"):
                        raise RuntimeError(f"Failed to get room details: {details_data.get('error')}")
                    
                    room_password = details_data["roomPassword"]
                    
                    return room_id, room_code, room_password
                    
        except Exception as e:
            self.logger.error(f"Error creating test room: {e}")
            raise
    
    async def wait_for_bot_server_health(self, timeout: float = 10.0) -> bool:
        """Wait for bot server to be healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check bot server
                if self.bot_server and self.bot_server._running:
                    return True
                    
            except Exception:
                pass
            
            await asyncio.sleep(0.5)
        
        return False


class TestBotGameServerCommunication:
    """Test suite for bot-game server communication validation."""
    
    @pytest.mark.asyncio
    async def test_go_server_accessibility(self):
        """Test that Go server is accessible (standalone test)."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
                async with session.get("http://localhost:4000/api/maps") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert "maps" in data
                    logging.info("Go server is accessible and responding correctly")
        except Exception as e:
            pytest.skip(f"Go server is not accessible: {e}")
    
    @pytest.mark.asyncio
    async def test_bot_configuration_validation(self):
        """Test bot configuration validation (unit test)."""
        # Valid configuration
        valid_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.INTERMEDIATE,
            name="ValidTestBot"
        )
        
        # This should not raise an exception
        assert valid_config.bot_type == BotType.RULES_BASED
        assert valid_config.difficulty == DifficultyLevel.INTERMEDIATE
        assert valid_config.name == "ValidTestBot"
        
        # Test different difficulty levels
        for difficulty in DifficultyLevel:
            config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=difficulty,
                name=f"TestBot_{difficulty.value}"
            )
            assert config.difficulty == difficulty
    
    @pytest_asyncio.fixture(scope="class")
    async def server_manager(self):
        """Fixture to manage test servers."""
        manager = ServerManager()
        
        # Check if Go server is running
        go_running = await manager.check_go_server_running()
        if not go_running:
            pytest.skip("Go server is not running. Please start the Go server on port 4000 before running tests.")
        
        # Start bot server
        bot_started = await manager.start_bot_server()
        if not bot_started:
            pytest.skip("Failed to start bot server")
        
        # Wait for bot server to be healthy
        if not await manager.wait_for_bot_server_health():
            await manager.stop_bot_server()
            pytest.skip("Bot server not healthy")
        
        yield manager
        
        # Cleanup
        await manager.stop_bot_server()
    
    @pytest_asyncio.fixture
    async def test_room(self, server_manager):
        """Fixture to create a test room."""
        room_id, room_code, room_password = await server_manager.create_test_room()
        return {"room_id": room_id, "room_code": room_code, "room_password": room_password}
    
    @pytest.mark.asyncio
    async def test_server_startup_and_health(self, server_manager):
        """Test that both servers start up and are healthy."""
        # Test Go server health
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_manager.go_server_url}/api/maps") as response:
                assert response.status == 200
                data = await response.json()
                assert "maps" in data
        
        # Test bot server health
        assert server_manager.bot_server._running
        status = server_manager.bot_server.get_server_status()
        assert status["running"] is True
    
    @pytest.mark.asyncio
    async def test_bot_spawning_via_api(self, server_manager, test_room):
        """Test bot spawning through API calls."""
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.INTERMEDIATE,
            name="TestBot_API"
        )
        
        room_info = {
            "room_code": test_room["room_code"],
            "room_password": test_room["room_password"]
        }
        
        # Spawn bot
        bot_id = await server_manager.bot_server.spawn_bot(bot_config, room_info)
        
        assert bot_id is not None
        assert bot_id in server_manager.bot_server._bots
        
        # Wait for bot to initialize
        await asyncio.sleep(2)
        
        # Check bot status
        bot_status = server_manager.bot_server.get_bot_status(bot_id)
        assert bot_status is not None
        assert bot_status["config"]["name"] == "TestBot_API"
        
        # Cleanup
        await server_manager.bot_server.terminate_bot(bot_id)
    
    @pytest.mark.asyncio
    async def test_bot_websocket_connection(self, server_manager, test_room):
        """Test bot WebSocket connections to game server."""
        # Create a game client (simulating bot connection)
        client = GameClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        try:
            # Connect to the test room
            await client.connect(
                room_code=test_room["room_code"],
                player_name="TestBotClient",
                room_password=test_room["room_password"]
            )
            
            # Verify connection
            assert client.websocket is not None
            assert client.player_id is not None
            assert client.player_token is not None
            assert client.room_id == test_room["room_id"]
            
            # Test basic message sending
            await client.send_keyboard_input("W", True)
            await asyncio.sleep(0.1)
            await client.send_keyboard_input("W", False)
            
            # Test mouse input
            await client.send_mouse_input("left", True, 100.0, 100.0)
            await asyncio.sleep(0.1)
            await client.send_mouse_input("left", False, 100.0, 100.0)
            
        finally:
            await client.exit_game()
    
    @pytest.mark.asyncio
    async def test_bot_action_execution_and_state_validation(self, server_manager, test_room):
        """Test bot action execution and game state response validation."""
        from rl_bot_system.tests.test_utils import EnhancedGameClient
        
        # Create game client with state tracking
        client = EnhancedGameClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        try:
            success = await client.connect_to_room(
                room_code=test_room["room_code"],
                player_name="StateTestBot",
                room_password=test_room["room_password"]
            )
            assert success, "Failed to connect to test room"
            
            # Execute a sequence of actions using the test utility
            test_actions = [
                ("keyboard", ("W", True)),    # Start moving up
                ("wait", 0.2),
                ("keyboard", ("A", True)),    # Start moving left  
                ("wait", 0.2),
                ("keyboard", ("W", False)),   # Stop moving up
                ("wait", 0.2),
                ("keyboard", ("A", False)),   # Stop moving left
                ("wait", 0.2),
                ("mouse", ("left", True, 150.0, 150.0)),   # Mouse click
                ("wait", 0.1),
                ("mouse", ("left", False, 150.0, 150.0)),  # Mouse release
            ]
            
            success = await client.send_test_sequence(test_actions)
            assert success, "Failed to send test action sequence"
            
            # Wait for state updates
            await asyncio.sleep(1)
            
            # Get captured messages and filter for game states
            all_messages = client.get_captured_messages()
            game_states = client.get_messages_by_type("GameState")
            
            # Validate that we received game state updates
            assert len(game_states) > 0, f"No game state updates received. Got {len(all_messages)} total messages"
            
            # Validate state structure (basic validation)
            for state in game_states:
                payload = state.get("payload", {})
                # Go server uses objectStates instead of players/gameObjects
                assert "objectStates" in payload or "players" in payload or "gameObjects" in payload
            
        finally:
            await client.exit_game()
    
    @pytest.mark.asyncio
    async def test_message_serialization_deserialization(self, server_manager, test_room):
        """Test proper message serialization/deserialization between servers."""
        from rl_bot_system.tests.test_utils import EnhancedGameClient
        
        client = EnhancedGameClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        try:
            success = await client.connect_to_room(
                room_code=test_room["room_code"],
                player_name="SerializationTestBot",
                room_password=test_room["room_password"]
            )
            assert success, "Failed to connect to test room"
            
            # Send various message types to test serialization
            test_messages = [
                # Keyboard input
                {"key": "W", "pressed": True},
                {"key": "W", "pressed": False},
                # Mouse input with coordinates
                {"button": "left", "pressed": True, "x": 123.456, "y": 789.012},
                {"button": "left", "pressed": False, "x": 123.456, "y": 789.012},
            ]
            
            for msg_data in test_messages:
                if "key" in msg_data:
                    await client.send_keyboard_input(msg_data["key"], msg_data["pressed"])
                elif "button" in msg_data:
                    await client.send_mouse_input(
                        msg_data["button"], msg_data["pressed"], 
                        msg_data["x"], msg_data["y"]
                    )
                
                await asyncio.sleep(0.1)
            
            # Wait for responses
            await asyncio.sleep(1)
            
            # Get captured messages from test client
            received_messages = client.get_captured_messages()
            
            # Validate message structure
            assert len(received_messages) > 0, "No messages received"
            
            for message in received_messages:
                # All messages should be valid JSON (already parsed)
                assert isinstance(message, dict)
                assert "type" in message
                
                # Validate common message structure
                if message.get("type") in ["GameState", "PlayerJoined", "PlayerLeft"]:
                    assert "payload" in message
                    assert isinstance(message["payload"], dict)
        
        finally:
            await client.exit_game()
    
    @pytest.mark.asyncio
    async def test_network_failure_handling(self, server_manager, test_room):
        """Test error handling for network failures and server disconnections."""
        client = GameClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        try:
            await client.connect(
                room_code=test_room["room_code"],
                player_name="NetworkFailureTestBot",
                room_password=test_room["room_password"]
            )
            
            # Verify connection is established
            assert client.websocket is not None
            
            # Simulate network failure by closing WebSocket
            await client.websocket.close()
            
            # Try to send a message after connection is closed
            with pytest.raises((ConnectionClosed, websockets.exceptions.ConnectionClosed)):
                await client.send_keyboard_input("W", True)
            
        except Exception as e:
            # Expected behavior - connection should fail gracefully
            assert "connection" in str(e).lower() or "closed" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_invalid_server_connection(self, server_manager):
        """Test connection to invalid server addresses."""
        # Test connection to non-existent server
        invalid_client = GameClient(
            ws_url="ws://localhost:9999/ws",
            http_url="http://localhost:9999"
        )
        
        with pytest.raises((aiohttp.ClientError, ConnectionRefusedError, OSError)):
            await invalid_client.connect(
                room_code="INVALID",
                player_name="InvalidTestBot"
            )
    
    @pytest.mark.asyncio
    async def test_bot_authentication_and_room_access(self, server_manager):
        """Test bot authentication and room access control."""
        # Create a room with password
        async with aiohttp.ClientSession() as session:
            create_data = {
                "playerName": "HostPlayer",
                "roomName": "PasswordProtectedRoom",
                "mapType": "default"
            }
            
            async with session.post(
                f"{server_manager.go_server_url}/api/createGame",
                json=create_data
            ) as response:
                response.raise_for_status()
                room_data = await response.json()
                
                room_code = room_data["roomCode"]
                room_id = room_data["roomId"]
        
        # Test 1: Valid room code without password
        client1 = GameClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        # Get room password
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_manager.go_server_url}/api/rooms/{room_id}/details") as response:
                response.raise_for_status()
                details_data = await response.json()
                room_password = details_data["roomPassword"]
        
        try:
            await client1.connect(
                room_code=room_code,
                player_name="ValidBot",
                room_password=room_password
            )
            
            # Should succeed
            assert client1.player_id is not None
            assert client1.room_id == room_id
            
        finally:
            await client1.exit_game()
        
        # Test 2: Invalid room code
        client2 = GameClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        with pytest.raises((aiohttp.ClientError, RuntimeError)):
            await client2.connect(
                room_code="INVALID123",
                player_name="InvalidBot"
            )
        
        # Test 3: Empty player name
        client3 = GameClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        with pytest.raises((aiohttp.ClientError, RuntimeError)):
            await client3.connect(
                room_code=room_code,
                player_name=""
            )
    
    @pytest.mark.asyncio
    async def test_multiple_bots_same_room(self, server_manager, test_room):
        """Test multiple bots connecting to the same room."""
        clients = []
        
        try:
            # Connect multiple bots to the same room
            for i in range(3):
                client = GameClient(
                    ws_url=server_manager.ws_url,
                    http_url=server_manager.go_server_url
                )
                
                await client.connect(
                    room_code=test_room["room_code"],
                    player_name=f"MultiBot_{i}",
                    room_password=test_room["room_password"]
                )
                
                clients.append(client)
                
                # Verify each connection
                assert client.player_id is not None
                assert client.room_id == test_room["room_id"]
            
            # Test that all bots can send actions simultaneously
            for i, client in enumerate(clients):
                await client.send_keyboard_input("W", True)
                await client.send_mouse_input("left", True, 100 + i * 50, 100 + i * 50)
                await asyncio.sleep(0.1)
            
            # Stop all actions
            for client in clients:
                await client.send_keyboard_input("W", False)
                await client.send_mouse_input("left", False, 0, 0)
            
            await asyncio.sleep(0.5)
            
        finally:
            # Cleanup all clients
            for client in clients:
                try:
                    await client.exit_game()
                except Exception as e:
                    logging.warning(f"Error cleaning up client: {e}")
    


if __name__ == "__main__":
    # Configure logging for test runs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])