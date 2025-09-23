"""
Tests for bot lifecycle management functionality.

This module tests the enhanced bot lifecycle management features including:
- Bot spawning and termination
- Real-time difficulty configuration
- Auto-cleanup when human players leave
- Bot health monitoring and reconnection logic
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from bot.rl_bot_system.server.bot_server import (
    BotServer, BotServerConfig, BotConfig, BotType, BotStatus, DifficultyLevel
)
from bot.rl_bot_system.rules_based.rules_based_bot import RulesBasedBot

@pytest.mark.asyncio
async def test_bot_spawning_and_termination():
    """Test basic bot spawning and termination."""
    config = BotServerConfig(
        max_total_bots=10,
        max_bots_per_room=5,
        bot_timeout_seconds=60,
        cleanup_interval_seconds=10
    )
    
    server = BotServer(config)
    await server.start()
    
    try:
        # Mock the client pool and game client
        with patch.object(server.client_pool, 'get_client') as mock_get_client, \
             patch.object(server, '_load_bot_ai') as mock_load_ai:
            
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            mock_load_ai.return_value = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
            
            bot_config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name="TestBot",
                auto_cleanup=True
            )
            
            room_info = {
                'room_id': 'test_room_123',
                'room_code': 'TEST123',
                'room_password': ''
            }
            
            # Spawn a bot
            bot_id = await server.spawn_bot(bot_config, room_info)
            
            assert bot_id is not None
            assert bot_id in server._bots
            
            # Wait for initialization
            await asyncio.sleep(0.1)
            
            # Check bot status
            bot_status = server.get_bot_status(bot_id)
            assert bot_status is not None
            assert bot_status['config']['name'] == 'TestBot'
            
            # Terminate the bot
            success = await server.terminate_bot(bot_id)
            assert success is True
            assert bot_id not in server._bots
    
    finally:
        await server.stop()

@pytest.mark.asyncio
async def test_real_time_difficulty_configuration():
    """Test real-time difficulty configuration."""
    config = BotServerConfig(max_total_bots=10)
    server = BotServer(config)
    await server.start()
    
    try:
        with patch.object(server.client_pool, 'get_client') as mock_get_client, \
             patch.object(server, '_load_bot_ai') as mock_load_ai:
            
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            
            # Create a mock rules-based bot
            mock_bot_ai = MagicMock(spec=RulesBasedBot)
            mock_bot_ai.config = {'decision_frequency': 0.2, 'accuracy_modifier': 0.75}
            mock_load_ai.return_value = mock_bot_ai
            
            bot_config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name="TestBot"
            )
            
            room_info = {'room_id': 'test_room', 'room_code': 'TEST'}
            
            # Spawn a bot
            bot_id = await server.spawn_bot(bot_config, room_info)
            
            # Wait for initialization
            await asyncio.sleep(0.1)
            
            # Change difficulty to Expert
            success = await server.configure_bot_difficulty(bot_id, DifficultyLevel.EXPERT)
            assert success is True
            
            # Verify the bot AI was updated
            mock_bot_ai.set_difficulty_level.assert_called_with(DifficultyLevel.EXPERT)
            
            # Check that real-time changes were applied
            assert mock_bot_ai.config['decision_frequency'] == 0.1  # Expert frequency
            assert mock_bot_ai.config['accuracy_modifier'] == 0.95  # Expert accuracy
            
            # Clean up
            await server.terminate_bot(bot_id)
    
    finally:
        await server.stop()
@pytest.mark.asyncio
async def test_bot_health_monitoring():
    """Test bot health monitoring functionality."""
    config = BotServerConfig(max_total_bots=10)
    server = BotServer(config)
    await server.start()
    
    try:
        with patch.object(server.client_pool, 'get_client') as mock_get_client, \
             patch.object(server, '_load_bot_ai') as mock_load_ai:
            
            mock_client = AsyncMock()
            mock_client.websocket = MagicMock()
            mock_client.websocket.client_state.name = 'CONNECTED'
            mock_client.game_state = {}
            mock_client.get_direct_state.return_value = {}
            mock_get_client.return_value = mock_client
            mock_load_ai.return_value = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
            
            bot_config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name="TestBot"
            )
            
            room_info = {'room_id': 'test_room', 'room_code': 'TEST'}
            
            # Spawn a bot
            bot_id = await server.spawn_bot(bot_config, room_info)
            
            # Wait for initialization
            await asyncio.sleep(0.1)
            
            # Check health status
            health_status = await server.monitor_bot_health(bot_id)
            
            assert health_status['bot_id'] == bot_id
            assert health_status['healthy'] is True
            assert health_status['connection_status'] == 'connected'
            assert health_status['ai_status'] == 'loaded'
            assert len(health_status['performance_issues']) == 0
            
            # Simulate connection loss
            mock_client.websocket.client_state.name = 'DISCONNECTED'
            
            health_status = await server.monitor_bot_health(bot_id)
            assert health_status['healthy'] is False
            assert 'connection_lost' in health_status['performance_issues']
            
            # Clean up
            await server.terminate_bot(bot_id)
    
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_auto_cleanup_empty_rooms():
    """Test auto-cleanup when rooms become empty of human players."""
    config = BotServerConfig(
        max_total_bots=10,
        auto_cleanup_empty_rooms=True
    )
    server = BotServer(config)
    await server.start()
    
    try:
        with patch.object(server.client_pool, 'get_client') as mock_get_client, \
             patch.object(server, '_load_bot_ai') as mock_load_ai, \
             patch.object(server, 'check_room_human_players') as mock_check_humans:
            
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            mock_load_ai.return_value = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
            
            # Initially room has human players
            mock_check_humans.return_value = True
            
            room_info = {'room_id': 'test_room', 'room_code': 'TEST'}
            
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
            await asyncio.sleep(0.1)
            
            # Verify bots are in the room
            room_bots = server.get_room_bots(room_info['room_id'])
            assert len(room_bots) == 3
            
            # Simulate room becoming empty of human players
            mock_check_humans.return_value = False
            
            # Trigger cleanup check
            await server._check_and_cleanup_empty_rooms()
            
            # Verify all bots were cleaned up
            for bot_id in bot_ids:
                assert bot_id not in server._bots
            
            # Verify room is no longer tracked
            assert room_info['room_id'] not in server._room_bots
    
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_manual_room_cleanup():
    """Test manual cleanup of all bots in a room."""
    config = BotServerConfig(max_total_bots=10)
    server = BotServer(config)
    await server.start()
    
    try:
        with patch.object(server.client_pool, 'get_client') as mock_get_client, \
             patch.object(server, '_load_bot_ai') as mock_load_ai:
            
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            mock_load_ai.return_value = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
            
            room_info = {'room_id': 'test_room', 'room_code': 'TEST'}
            
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
            await asyncio.sleep(0.1)
            
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
    
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_get_all_bot_health():
    """Test getting health status for all bots."""
    config = BotServerConfig(max_total_bots=10)
    server = BotServer(config)
    await server.start()
    
    try:
        with patch.object(server.client_pool, 'get_client') as mock_get_client, \
             patch.object(server, '_load_bot_ai') as mock_load_ai:
            
            mock_client = AsyncMock()
            mock_client.websocket = MagicMock()
            mock_client.websocket.client_state.name = 'CONNECTED'
            mock_client.game_state = {}
            mock_client.get_direct_state.return_value = {}
            mock_get_client.return_value = mock_client
            mock_load_ai.return_value = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
            
            room_info = {'room_id': 'test_room', 'room_code': 'TEST'}
            
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
            await asyncio.sleep(0.1)
            
            # Get health status for all bots
            all_health = await server.get_all_bot_health()
            
            assert len(all_health) == 3
            for bot_id in bot_ids:
                assert bot_id in all_health
                assert all_health[bot_id]['healthy'] is True
                assert all_health[bot_id]['bot_id'] == bot_id
            
            # Clean up
            for bot_id in bot_ids:
                await server.terminate_bot(bot_id)
    
    finally:
        await server.stop()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])