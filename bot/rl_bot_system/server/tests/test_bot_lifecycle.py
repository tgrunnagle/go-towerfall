"""
Tests for bot lifecycle management functionality.

This module tests the enhanced bot lifecycle management features including:
- Bot spawning and termination
- Real-time difficulty configuration
- Auto-cleanup when human players leave
- Bot health monitoring and reconnection logic

These tests run against a deployed game server.
"""

import asyncio
import pytest
import pytest_asyncio
import aiohttp
import logging
from datetime import datetime, timedelta

from bot.rl_bot_system.server.bot_server import (
    BotServer, BotServerConfig, BotConfig, BotType, BotStatus, DifficultyLevel
)
from bot.rl_bot_system.rules_based.rules_based_bot import RulesBasedBot
from bot.rl_bot_system.tests.test_utils import ServerHealthChecker, GameRoomManager

@pytest_asyncio.fixture(scope="class")
async def server_manager(server_config):
    """Fixture to manage test servers."""
    # Check if Go server is running
    go_running = await ServerHealthChecker.check_go_server(server_config["go_server_url"])
    if not go_running:
        pytest.skip("Go server is not running. Please start the Go server on port 4000 before running tests.")
    
    # Start bot server
    config = BotServerConfig(
        max_total_bots=10,
        max_bots_per_room=5,
        bot_timeout_seconds=60,
        cleanup_interval_seconds=10,
        game_server_url=server_config["go_server_url"]
    )
    
    server = BotServer(config)
    await server.start()
    
    # Wait for server to be ready
    await asyncio.sleep(1)
    
    yield server
    
    # Cleanup
    await server.stop()


@pytest_asyncio.fixture
async def test_room(server_config):
    """Fixture to create a test room."""
    room_manager = GameRoomManager(server_config["go_server_url"])
    room_id, room_code = await room_manager.create_room("BotLifecycleTest")
    
    # Get room password
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{server_config['go_server_url']}/api/rooms/{room_id}/details") as response:
            response.raise_for_status()
            details_data = await response.json()
            room_password = details_data["roomPassword"]
    
    yield {"room_id": room_id, "room_code": room_code, "room_password": room_password}
    
    await room_manager.cleanup_rooms()


@pytest.mark.asyncio
async def test_bot_spawning_and_termination(server_manager, test_room):
    """Test basic bot spawning and termination."""
    server = server_manager
    
    bot_config = BotConfig(
        bot_type=BotType.RULES_BASED,
        difficulty=DifficultyLevel.INTERMEDIATE,
        name="TestBot",
        auto_cleanup=True
    )
    
    room_info = {
        'room_id': test_room['room_id'],
        'room_code': test_room['room_code'],
        'room_password': test_room['room_password']
    }
    
    # Spawn a bot
    bot_id = await server.spawn_bot(bot_config, room_info)
    
    assert bot_id is not None
    assert bot_id in server._bots
    
    # Wait for initialization
    await asyncio.sleep(2)
    
    # Check bot status
    bot_status = server.get_bot_status(bot_id)
    assert bot_status is not None
    assert bot_status['config']['name'] == 'TestBot'
    
    # Terminate the bot
    success = await server.terminate_bot(bot_id)
    assert success is True
    assert bot_id not in server._bots

@pytest.mark.asyncio
async def test_real_time_difficulty_configuration(server_manager, test_room):
    """Test real-time difficulty configuration."""
    server = server_manager
    
    bot_config = BotConfig(
        bot_type=BotType.RULES_BASED,
        difficulty=DifficultyLevel.INTERMEDIATE,
        name="TestBot"
    )
    
    room_info = {
        'room_id': test_room['room_id'],
        'room_code': test_room['room_code'],
        'room_password': test_room['room_password']
    }
    
    # Spawn a bot
    bot_id = await server.spawn_bot(bot_config, room_info)
    
    # Wait for initialization
    await asyncio.sleep(2)
    
    # Change difficulty to Expert
    success = await server.configure_bot_difficulty(bot_id, DifficultyLevel.EXPERT)
    assert success is True
    
    # Verify the bot configuration was updated
    bot_status = server.get_bot_status(bot_id)
    assert bot_status["config"]["difficulty"] == DifficultyLevel.EXPERT.value
    
    # Clean up
    await server.terminate_bot(bot_id)
@pytest.mark.asyncio
async def test_bot_health_monitoring(server_manager, test_room):
    """Test bot health monitoring functionality."""
    server = server_manager
    
    bot_config = BotConfig(
        bot_type=BotType.RULES_BASED,
        difficulty=DifficultyLevel.INTERMEDIATE,
        name="TestBot"
    )
    
    room_info = {
        'room_id': test_room['room_id'],
        'room_code': test_room['room_code'],
        'room_password': test_room['room_password']
    }
    
    # Spawn a bot
    bot_id = await server.spawn_bot(bot_config, room_info)
    
    # Wait for initialization
    await asyncio.sleep(2)
    
    # Check health status
    health_status = await server.monitor_bot_health(bot_id)
    
    assert health_status['bot_id'] == bot_id
    assert 'healthy' in health_status
    assert 'connection_status' in health_status
    assert 'ai_status' in health_status
    assert 'performance_issues' in health_status
    
    # Clean up
    await server.terminate_bot(bot_id)


@pytest.mark.asyncio
async def test_auto_cleanup_empty_rooms(server_manager, test_room):
    """Test auto-cleanup when rooms become empty of human players."""
    server = server_manager
    server.config.auto_cleanup_empty_rooms = True
    
    room_info = {
        'room_id': test_room['room_id'],
        'room_code': test_room['room_code'],
        'room_password': test_room['room_password']
    }
    
    # Spawn multiple bots in the same room
    bot_ids = []
    for i in range(3):
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.INTERMEDIATE,
            name=f"TestBot{i}",
            auto_cleanup=True
        )
        bot_id = await server.spawn_bot(bot_config, room_info)
        bot_ids.append(bot_id)
    
    # Wait for initialization
    await asyncio.sleep(3)
    
    # Verify bots are in the room
    room_bots = server.get_room_bots(room_info['room_id'])
    assert len(room_bots) == 3
    
    # Manually trigger cleanup (simulating empty room detection)
    cleanup_count = await server.cleanup_room_bots(room_info['room_id'], "test_cleanup")
    
    assert cleanup_count == 3
    
    # Verify all bots were cleaned up
    for bot_id in bot_ids:
        assert bot_id not in server._bots


@pytest.mark.asyncio
async def test_manual_room_cleanup(server_manager, test_room):
    """Test manual cleanup of all bots in a room."""
    server = server_manager
    
    room_info = {
        'room_id': test_room['room_id'],
        'room_code': test_room['room_code'],
        'room_password': test_room['room_password']
    }
    
    # Spawn multiple bots in the same room
    bot_ids = []
    for i in range(3):
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.INTERMEDIATE,
            name=f"TestBot{i}"
        )
        bot_id = await server.spawn_bot(bot_config, room_info)
        bot_ids.append(bot_id)
    
    # Wait for initialization
    await asyncio.sleep(3)
    
    # Verify bots are in the room
    room_bots = server.get_room_bots(room_info['room_id'])
    assert len(room_bots) == 3
    
    # Manually clean up the room
    cleanup_count = await server.cleanup_room_bots(room_info['room_id'], "manual_test")
    
    assert cleanup_count == 3
    
    # Verify all bots were removed
    for bot_id in bot_ids:
        assert bot_id not in server._bots
    
    # Verify room is no longer tracked
    assert room_info['room_id'] not in server._room_bots


@pytest.mark.asyncio
async def test_get_all_bot_health(server_manager, test_room):
    """Test getting health status for all bots."""
    server = server_manager
    
    room_info = {
        'room_id': test_room['room_id'],
        'room_code': test_room['room_code'],
        'room_password': test_room['room_password']
    }
    
    # Spawn multiple bots
    bot_ids = []
    for i in range(3):
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.INTERMEDIATE,
            name=f"TestBot{i}"
        )
        bot_id = await server.spawn_bot(bot_config, room_info)
        bot_ids.append(bot_id)
    
    # Wait for initialization
    await asyncio.sleep(3)
    
    # Get health status for all bots
    all_health = await server.get_all_bot_health()
    
    assert len(all_health) == 3
    for bot_id in bot_ids:
        assert bot_id in all_health
        assert 'healthy' in all_health[bot_id]
        assert all_health[bot_id]['bot_id'] == bot_id
    
    # Clean up
    for bot_id in bot_ids:
        await server.terminate_bot(bot_id)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])