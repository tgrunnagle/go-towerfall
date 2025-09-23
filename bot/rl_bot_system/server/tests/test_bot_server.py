"""
Tests for BotServer class and related components.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from rl_bot_system.server.bot_server import (
    BotServer, BotServerConfig, BotConfig, BotType, BotStatus, DifficultyLevel,
    GameClientPool, BotInstance
)
from rl_bot_system.rules_based.rules_based_bot import RulesBasedBot
import game_client


class TestGameClientPool:
    """Test cases for GameClientPool."""
    
    @pytest.fixture
    def pool(self):
        return GameClientPool(max_connections=5)
    
    @pytest.mark.asyncio
    async def test_get_client_creates_new(self, pool):
        """Test that get_client creates a new client when pool is empty."""
        client = await pool.get_client("bot1", "http://localhost:4000")
        
        assert client.__class__.__name__ == 'GameClient'
        assert "bot1" in pool._active_clients
        assert len(pool._available_clients) == 0
    
    @pytest.mark.asyncio
    async def test_get_client_reuses_available(self, pool):
        """Test that get_client reuses available clients."""
        # Create and return a client to populate the pool
        client1 = await pool.get_client("bot1", "http://localhost:4000")
        await pool.return_client("bot1")
        
        # Get a new client - should reuse the returned one
        client2 = await pool.get_client("bot2", "http://localhost:4000")
        
        assert client1 is client2
        assert "bot2" in pool._active_clients
        assert "bot1" not in pool._active_clients
    
    @pytest.mark.asyncio
    async def test_get_client_max_connections(self, pool):
        """Test that get_client raises error when max connections exceeded."""
        # Fill up the pool
        for i in range(5):
            await pool.get_client(f"bot{i}", "http://localhost:4000")
        
        # Try to get one more - should fail
        with pytest.raises(RuntimeError, match="Maximum connections"):
            await pool.get_client("bot5", "http://localhost:4000")
    
    @pytest.mark.asyncio
    async def test_return_client_cleans_state(self, pool):
        """Test that return_client properly cleans client state."""
        client = await pool.get_client("bot1", "http://localhost:4000")
        
        # Set some state
        client.player_id = "test_player"
        client.room_id = "test_room"
        client.game_state = {"test": "data"}
        
        await pool.return_client("bot1")
        
        # Client should be cleaned
        assert client.player_id is None
        assert client.room_id is None
        assert client.game_state == {}
    
    def test_get_pool_stats(self, pool):
        """Test pool statistics reporting."""
        stats = pool.get_pool_stats()
        
        assert stats['available_clients'] == 0
        assert stats['active_clients'] == 0
        assert stats['max_connections'] == 5
        assert stats['utilization_percent'] == 0


class TestBotServer:
    """Test cases for BotServer."""
    
    @pytest.fixture
    def config(self):
        return BotServerConfig(
            max_bots_per_room=3,
            max_total_bots=10,
            bot_timeout_seconds=60,
            cleanup_interval_seconds=30
        )
    
    @pytest.fixture
    def server(self, config):
        return BotServer(config)
    
    @pytest.mark.asyncio
    async def test_server_start_stop(self, server):
        """Test server startup and shutdown."""
        assert not server._running
        
        await server.start()
        assert server._running
        assert server._cleanup_task is not None
        
        await server.stop()
        assert not server._running
    
    @pytest.mark.asyncio
    async def test_spawn_bot_success(self, server):
        """Test successful bot spawning."""
        await server.start()
        
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.INTERMEDIATE,
            name="TestBot"
        )
        
        room_info = {
            'room_id': 'test_room',
            'room_code': 'TEST123',
            'room_password': ''
        }
        
        with patch.object(server, '_initialize_bot', new_callable=AsyncMock):
            bot_id = await server.spawn_bot(bot_config, room_info)
        
        assert bot_id in server._bots
        assert server._bots[bot_id].config.name == "TestBot"
        assert server._bots[bot_id].status == BotStatus.INITIALIZING
        assert 'test_room' in server._room_bots
        assert bot_id in server._room_bots['test_room']
        
        await server.stop()
    
    @pytest.mark.asyncio
    async def test_spawn_bot_max_total_limit(self, server):
        """Test that spawning fails when max total bots exceeded."""
        await server.start()
        
        # Fill up to max capacity
        for i in range(server.config.max_total_bots):
            bot_config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.BEGINNER,
                name=f"Bot{i}"
            )
            room_info = {'room_id': f'room{i}'}
            
            with patch.object(server, '_initialize_bot', new_callable=AsyncMock):
                await server.spawn_bot(bot_config, room_info)
        
        # Try to spawn one more - should fail
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.BEGINNER,
            name="ExtraBot"
        )
        room_info = {'room_id': 'extra_room'}
        
        with pytest.raises(RuntimeError, match="Maximum total bots"):
            await server.spawn_bot(bot_config, room_info)
        
        await server.stop()
    
    @pytest.mark.asyncio
    async def test_spawn_bot_max_room_limit(self, server):
        """Test that spawning fails when max bots per room exceeded."""
        await server.start()
        
        room_info = {
            'room_id': 'test_room',
            'room_code': 'TEST123'
        }
        
        # Fill up room to max capacity
        for i in range(server.config.max_bots_per_room):
            bot_config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.BEGINNER,
                name=f"Bot{i}"
            )
            
            with patch.object(server, '_initialize_bot', new_callable=AsyncMock):
                await server.spawn_bot(bot_config, room_info)
        
        # Try to spawn one more in same room - should fail
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.BEGINNER,
            name="ExtraBot"
        )
        
        with pytest.raises(RuntimeError, match="Maximum bots per room"):
            await server.spawn_bot(bot_config, room_info)
        
        await server.stop()
    
    @pytest.mark.asyncio
    async def test_terminate_bot_success(self, server):
        """Test successful bot termination."""
        await server.start()
        
        # Create a bot
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.INTERMEDIATE,
            name="TestBot"
        )
        room_info = {'room_id': 'test_room'}
        
        with patch.object(server, '_initialize_bot', new_callable=AsyncMock):
            bot_id = await server.spawn_bot(bot_config, room_info)
        
        # Mock game client
        mock_client = AsyncMock()
        server._bots[bot_id].game_client = mock_client
        
        # Terminate the bot
        result = await server.terminate_bot(bot_id)
        
        assert result is True
        assert bot_id not in server._bots
        assert 'test_room' not in server._room_bots  # Room should be cleaned up
        mock_client.exit_game.assert_called_once()
        
        await server.stop()
    
    @pytest.mark.asyncio
    async def test_terminate_bot_not_found(self, server):
        """Test terminating non-existent bot."""
        await server.start()
        
        result = await server.terminate_bot("nonexistent")
        assert result is False
        
        await server.stop()
    
    @pytest.mark.asyncio
    async def test_configure_bot_difficulty(self, server):
        """Test configuring bot difficulty."""
        await server.start()
        
        # Create a bot
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.BEGINNER,
            name="TestBot"
        )
        room_info = {'room_id': 'test_room'}
        
        with patch.object(server, '_initialize_bot', new_callable=AsyncMock):
            bot_id = await server.spawn_bot(bot_config, room_info)
        
        # For this test, let's just test that the difficulty is updated in the config
        # The isinstance check is complex to mock, so we'll test the core functionality
        
        # Configure difficulty
        result = await server.configure_bot_difficulty(bot_id, DifficultyLevel.EXPERT)
        
        assert result is True
        assert server._bots[bot_id].config.difficulty == DifficultyLevel.EXPERT
        

        
        await server.stop()
    
    def test_get_bot_status(self, server):
        """Test getting bot status."""
        # Create a mock bot instance
        bot_id = "test_bot"
        bot_instance = BotInstance(
            bot_id=bot_id,
            config=BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name="TestBot"
            ),
            status=BotStatus.ACTIVE,
            room_id="test_room",
            game_client=None,
            bot_ai=None,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            performance_stats={}
        )
        
        server._bots[bot_id] = bot_instance
        
        status = server.get_bot_status(bot_id)
        
        assert status is not None
        assert status['bot_id'] == bot_id
        assert status['config']['name'] == "TestBot"
        assert status['status'] == BotStatus.ACTIVE.value
    
    def test_get_bot_status_not_found(self, server):
        """Test getting status for non-existent bot."""
        status = server.get_bot_status("nonexistent")
        assert status is None
    
    def test_get_room_bots(self, server):
        """Test getting bots in a room."""
        # Create mock bot instances
        room_id = "test_room"
        bot_ids = ["bot1", "bot2"]
        
        for bot_id in bot_ids:
            bot_instance = BotInstance(
                bot_id=bot_id,
                config=BotConfig(
                    bot_type=BotType.RULES_BASED,
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    name=f"Bot{bot_id}"
                ),
                status=BotStatus.ACTIVE,
                room_id=room_id,
                game_client=None,
                bot_ai=None,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                performance_stats={}
            )
            server._bots[bot_id] = bot_instance
        
        server._room_bots[room_id] = set(bot_ids)
        
        room_bots = server.get_room_bots(room_id)
        
        assert len(room_bots) == 2
        assert all(bot['room_id'] == room_id for bot in room_bots)
    
    def test_get_room_bots_empty_room(self, server):
        """Test getting bots for empty room."""
        room_bots = server.get_room_bots("empty_room")
        assert room_bots == []
    
    def test_get_server_status(self, server):
        """Test getting server status."""
        # Add some mock bots
        for i, status in enumerate([BotStatus.ACTIVE, BotStatus.INITIALIZING, BotStatus.ERROR]):
            bot_instance = BotInstance(
                bot_id=f"bot{i}",
                config=BotConfig(
                    bot_type=BotType.RULES_BASED,
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    name=f"Bot{i}"
                ),
                status=status,
                room_id=f"room{i}",
                game_client=None,
                bot_ai=None,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                performance_stats={}
            )
            server._bots[f"bot{i}"] = bot_instance
        
        status = server.get_server_status()
        
        assert status['total_bots'] == 3
        assert status['bots_by_status']['active'] == 1
        assert status['bots_by_status']['initializing'] == 1
        assert status['bots_by_status']['error'] == 1
        assert 'client_pool_stats' in status
    
    def test_get_available_bot_types(self, server):
        """Test getting available bot types."""
        bot_types = server.get_available_bot_types()
        
        assert len(bot_types) >= 1  # At least rules-based should be available
        
        rules_based = next((bt for bt in bot_types if bt['type'] == 'rules_based'), None)
        assert rules_based is not None
        assert 'difficulties' in rules_based
        assert rules_based['supports_training_mode'] is True
    
    @pytest.mark.asyncio
    async def test_bot_status_callback(self, server):
        """Test bot status change callbacks."""
        callback_calls = []
        
        async def status_callback(bot_id: str, status: BotStatus):
            callback_calls.append((bot_id, status))
        
        server.register_bot_status_callback(status_callback)
        
        # Trigger a status change
        bot_id = "test_bot"
        bot_instance = BotInstance(
            bot_id=bot_id,
            config=BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name="TestBot"
            ),
            status=BotStatus.INITIALIZING,
            room_id="test_room",
            game_client=None,
            bot_ai=None,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            performance_stats={}
        )
        server._bots[bot_id] = bot_instance
        
        await server._update_bot_status(bot_id, BotStatus.ACTIVE)
        
        assert len(callback_calls) == 1
        assert callback_calls[0] == (bot_id, BotStatus.ACTIVE)
    
    @pytest.mark.asyncio
    async def test_room_empty_callback(self, server):
        """Test room empty callbacks."""
        callback_calls = []
        
        async def room_empty_callback(room_id: str):
            callback_calls.append(room_id)
        
        server.register_room_empty_callback(room_empty_callback)
        server.config.auto_cleanup_empty_rooms = True
        
        # Create and terminate a bot to trigger room empty
        bot_id = "test_bot"
        room_id = "test_room"
        
        bot_instance = BotInstance(
            bot_id=bot_id,
            config=BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name="TestBot",
                auto_cleanup=True
            ),
            status=BotStatus.ACTIVE,
            room_id=room_id,
            game_client=AsyncMock(),
            bot_ai=None,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            performance_stats={}
        )
        
        server._bots[bot_id] = bot_instance
        server._room_bots[room_id] = {bot_id}
        
        await server.terminate_bot(bot_id)
        
        assert len(callback_calls) == 1
        assert callback_calls[0] == room_id


if __name__ == "__main__":
    pytest.main([__file__])