"""
Tests for bot lifecycle management API integration.

This module tests the integration between the API layer and bot lifecycle management.
These tests run against a deployed game server.
"""

import asyncio
import pytest
import pytest_asyncio
import aiohttp
import logging

from bot.rl_bot_system.server.bot_server_api import BotServerApi
from bot.rl_bot_system.server.bot_server import (
    BotServerConfig, BotConfig, BotType, BotStatus, DifficultyLevel
)
from bot.rl_bot_system.tests.test_utils import ServerHealthChecker, GameRoomManager


@pytest_asyncio.fixture(scope="class")
async def api_manager(server_config):
    """Fixture to manage API server."""
    # Check if Go server is running
    go_running = await ServerHealthChecker.check_go_server(server_config["go_server_url"])
    if not go_running:
        pytest.skip("Go server is not running. Please start the Go server on port 4000 before running tests.")
    
    # Create API with bot server
    config = BotServerConfig(
        max_total_bots=10,
        max_bots_per_room=5,
        game_server_url=server_config["go_server_url"]
    )
    api = BotServerApi(config)
    
    await api.initialize()
    
    # Wait for initialization
    await asyncio.sleep(1)
    
    yield api
    
    # Cleanup
    await api.cleanup()


@pytest_asyncio.fixture
async def test_room(server_config):
    """Fixture to create a test room."""
    room_manager = GameRoomManager(server_config["go_server_url"])
    room_id, room_code = await room_manager.create_room("APITest")
    
    # Get room password
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{server_config['go_server_url']}/api/rooms/{room_id}/details") as response:
            response.raise_for_status()
            details_data = await response.json()
            room_password = details_data["roomPassword"]
    
    yield {"room_id": room_id, "room_code": room_code, "room_password": room_password}
    
    await room_manager.cleanup_rooms()


@pytest.mark.asyncio
async def test_bot_server_api_initialization(api_manager):
    """Test BotServerApi initialization and setup."""
    api = api_manager
    
    assert api.config is not None
    assert api.bot_server is not None
    assert api.router is not None
    assert api.bot_server._running is True


@pytest.mark.asyncio
async def test_api_integration_methods(api_manager):
    """Test the API integration methods."""
    api = api_manager
    
    # Test that the API has the expected router and configuration
    assert api.router is not None
    assert api.config.max_total_bots == 10
    
    # Test server status
    server_status = api.bot_server.get_server_status()
    assert server_status["running"] is True


@pytest.mark.asyncio
async def test_callback_handlers(api_manager):
    """Test that the API can handle basic operations without errors."""
    api = api_manager
    
    # Test that the API is properly initialized
    assert api.bot_server is not None
    assert api.router is not None
    
    # Test that the bot server is running
    server_status = api.bot_server.get_server_status()
    assert server_status["running"] is True


@pytest.mark.asyncio
async def test_api_with_real_bot_operations(api_manager, test_room):
    """Test API operations with real bot spawning and termination."""
    api = api_manager
    
    bot_config = BotConfig(
        bot_type=BotType.RULES_BASED,
        difficulty=DifficultyLevel.INTERMEDIATE,
        name="APITestBot"
    )
    
    room_info = {
        'room_id': test_room['room_id'],
        'room_code': test_room['room_code'],
        'room_password': test_room['room_password']
    }
    
    # Spawn a bot through the API's bot server
    bot_id = await api.bot_server.spawn_bot(bot_config, room_info)
    
    assert bot_id is not None
    assert bot_id in api.bot_server._bots
    
    # Wait for initialization
    await asyncio.sleep(2)
    
    # Check bot status
    bot_status = api.bot_server.get_bot_status(bot_id)
    assert bot_status is not None
    assert bot_status['config']['name'] == 'APITestBot'
    
    # Terminate the bot
    success = await api.bot_server.terminate_bot(bot_id)
    assert success is True
    assert bot_id not in api.bot_server._bots


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])