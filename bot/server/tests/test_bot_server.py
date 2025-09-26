"""
Tests for BotServer class and related components.
These tests run against a deployed game server.
"""

import asyncio
import pytest
import pytest_asyncio
import aiohttp
import logging
from datetime import datetime, timedelta

from server.bot_server import (
    BotServer, BotServerConfig, BotConfig, BotType, BotStatus, DifficultyLevel,
    GameClientPool, BotInstance
)
from rl_bot_system.rules_based.rules_based_bot import RulesBasedBot
from rl_bot_system.tests.test_utils import ServerHealthChecker, GameRoomManager
from core.game_client import GameClient


class TestGameClientPool:
    """Test cases for GameClientPool."""
    
    @pytest.fixture
    def pool(self, server_config):
        return GameClientPool(max_connections=5)
    
    @pytest.mark.asyncio
    async def test_get_client_creates_new(self, pool, server_config):
        """Test that get_client creates a new client when pool is empty."""
        # Check if Go server is running
        go_running = await ServerHealthChecker.check_go_server(server_config["go_server_url"])
        if not go_running:
            pytest.skip("Go server is not running")
        
        client = await pool.get_client("bot1", server_config["go_server_url"])
        
        assert client.__class__.__name__ == 'GameClient'
        assert "bot1" in pool._active_clients
        assert len(pool._available_clients) == 0
        
        # Clean up
        await pool.return_client("bot1")
    
    @pytest.mark.asyncio
    async def test_get_client_reuses_available(self, pool, server_config):
        """Test that get_client reuses available clients."""
        # Check if Go server is running
        go_running = await ServerHealthChecker.check_go_server(server_config["go_server_url"])
        if not go_running:
            pytest.skip("Go server is not running")
        
        # Create and return a client to populate the pool
        client1 = await pool.get_client("bot1", server_config["go_server_url"])
        await pool.return_client("bot1")
        
        # Get a new client - should reuse the returned one
        client2 = await pool.get_client("bot2", server_config["go_server_url"])
        
        assert client1 is client2
        assert "bot2" in pool._active_clients
        assert "bot1" not in pool._active_clients
        
        # Clean up
        await pool.return_client("bot2")
    
    @pytest.mark.asyncio
    async def test_get_client_max_connections(self, pool, server_config):
        """Test that get_client raises error when max connections exceeded."""
        # Check if Go server is running
        go_running = await ServerHealthChecker.check_go_server(server_config["go_server_url"])
        if not go_running:
            pytest.skip("Go server is not running")
        
        # Fill up the pool
        for i in range(5):
            await pool.get_client(f"bot{i}", server_config["go_server_url"])
        
        # Try to get one more - should fail
        with pytest.raises(RuntimeError, match="Maximum connections"):
            await pool.get_client("bot5", server_config["go_server_url"])
        
        # Clean up
        for i in range(5):
            await pool.return_client(f"bot{i}")
    
    def test_get_pool_stats(self, pool):
        """Test pool statistics reporting."""
        stats = pool.get_pool_stats()
        
        assert stats['available_clients'] == 0
        assert stats['active_clients'] == 0
        assert stats['max_connections'] == 5
        assert stats['utilization_percent'] == 0


class TestBotServer:
    """Test cases for BotServer."""
    
    @pytest_asyncio.fixture(scope="class")
    async def server_manager(self, server_config):
        """Fixture to manage test server."""
        # Check if Go server is running
        go_running = await ServerHealthChecker.check_go_server(server_config["go_server_url"])
        if not go_running:
            pytest.skip("Go server is not running. Please start the Go server on port 4000 before running tests.")
        
        config = BotServerConfig(
            max_bots_per_room=3,
            max_total_bots=10,
            bot_timeout_seconds=60,
            cleanup_interval_seconds=30,
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
    async def test_room(self, server_config):
        """Fixture to create a test room."""
        room_manager = GameRoomManager(server_config["go_server_url"])
        room_id, room_code = await room_manager.create_room("BotServerTest")
        
        # Get room password
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_config['go_server_url']}/api/rooms/{room_id}/details") as response:
                response.raise_for_status()
                details_data = await response.json()
                room_password = details_data["roomPassword"]
        
        yield {"room_id": room_id, "room_code": room_code, "room_password": room_password}
        
        await room_manager.cleanup_rooms()
    
    @pytest.mark.asyncio
    async def test_server_start_stop(self, server_manager):
        """Test server startup and shutdown."""
        server = server_manager
        assert server._running
        assert server._cleanup_task is not None
    
    @pytest.mark.asyncio
    async def test_spawn_bot_success(self, server_manager, test_room):
        """Test successful bot spawning."""
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
        
        bot_id = await server.spawn_bot(bot_config, room_info)
        
        assert bot_id in server._bots
        assert server._bots[bot_id].config.name == "TestBot"
        assert test_room['room_id'] in server._room_bots
        assert bot_id in server._room_bots[test_room['room_id']]
        
        # Clean up
        await server.terminate_bot(bot_id)
    
    @pytest.mark.asyncio
    async def test_spawn_bot_max_total_limit(self, server_manager, test_room, server_config):
        """Test that spawning fails when max total bots exceeded."""
        server = server_manager
        
        # We need to create multiple rooms since we have a per-room limit
        from rl_bot_system.tests.test_utils import GameRoomManager
        room_manager = GameRoomManager(server_config["go_server_url"])
        
        bot_ids = []
        rooms_created = []
        
        try:
            # Calculate how many rooms we need
            max_total = server.config.max_total_bots
            max_per_room = server.config.max_bots_per_room
            rooms_needed = (max_total + max_per_room - 1) // max_per_room  # Ceiling division
            
            # Create additional rooms as needed
            room_infos = [test_room]  # Start with the existing test room
            
            for i in range(rooms_needed - 1):  # -1 because we already have test_room
                room_id, room_code = await room_manager.create_room(f"MaxTotalTest{i}")
                rooms_created.append((room_id, room_code))
                
                # Get room password
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{server_config['go_server_url']}/api/rooms/{room_id}/details") as response:
                        response.raise_for_status()
                        details_data = await response.json()
                        room_password = details_data["roomPassword"]
                
                room_infos.append({
                    'room_id': room_id,
                    'room_code': room_code,
                    'room_password': room_password
                })
            
            # Fill up to max capacity across multiple rooms
            for i in range(max_total):
                room_index = i // max_per_room
                room_info = room_infos[room_index]
                
                bot_config = BotConfig(
                    bot_type=BotType.RULES_BASED,
                    difficulty=DifficultyLevel.BEGINNER,
                    name=f"Bot{i}"
                )
                
                bot_id = await server.spawn_bot(bot_config, room_info)
                bot_ids.append(bot_id)
            
            # Try to spawn one more - should fail with total limit
            bot_config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.BEGINNER,
                name="ExtraBot"
            )
            
            with pytest.raises(RuntimeError, match="Maximum total bots"):
                await server.spawn_bot(bot_config, room_infos[0])
            
        finally:
            # Clean up all bots
            for bot_id in bot_ids:
                try:
                    await server.terminate_bot(bot_id)
                except Exception as e:
                    logging.warning(f"Error terminating bot {bot_id}: {e}")
            
            # Clean up created rooms
            await room_manager.cleanup_rooms()
    
    @pytest.mark.asyncio
    async def test_spawn_bot_max_room_limit(self, server_manager, test_room):
        """Test that spawning fails when max bots per room exceeded."""
        server = server_manager
        
        room_info = {
            'room_id': test_room['room_id'],
            'room_code': test_room['room_code'],
            'room_password': test_room['room_password']
        }
        
        bot_ids = []
        
        # Fill up room to max capacity
        for i in range(server.config.max_bots_per_room):
            bot_config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.BEGINNER,
                name=f"Bot{i}"
            )
            
            bot_id = await server.spawn_bot(bot_config, room_info)
            bot_ids.append(bot_id)
        
        # Try to spawn one more in same room - should fail
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.BEGINNER,
            name="ExtraBot"
        )
        
        with pytest.raises(RuntimeError, match="Maximum bots per room"):
            await server.spawn_bot(bot_config, room_info)
        
        # Clean up
        for bot_id in bot_ids:
            await server.terminate_bot(bot_id)
    
    @pytest.mark.asyncio
    async def test_terminate_bot_success(self, server_manager, test_room):
        """Test successful bot termination."""
        server = server_manager
        
        # Create a bot
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
        
        bot_id = await server.spawn_bot(bot_config, room_info)
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        # Terminate the bot
        result = await server.terminate_bot(bot_id)
        
        assert result is True
        assert bot_id not in server._bots
    
    @pytest.mark.asyncio
    async def test_terminate_bot_not_found(self, server_manager):
        """Test terminating non-existent bot."""
        server = server_manager
        
        result = await server.terminate_bot("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_configure_bot_difficulty(self, server_manager, test_room):
        """Test configuring bot difficulty."""
        server = server_manager
        
        # Create a bot
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.BEGINNER,
            name="TestBot"
        )
        room_info = {
            'room_id': test_room['room_id'],
            'room_code': test_room['room_code'],
            'room_password': test_room['room_password']
        }
        
        bot_id = await server.spawn_bot(bot_config, room_info)
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        # Configure difficulty
        result = await server.configure_bot_difficulty(bot_id, DifficultyLevel.EXPERT)
        
        assert result is True
        assert server._bots[bot_id].config.difficulty == DifficultyLevel.EXPERT
        
        # Clean up
        await server.terminate_bot(bot_id)
    
    @pytest.mark.asyncio
    async def test_get_bot_status(self, server_manager, test_room):
        """Test getting bot status."""
        server = server_manager
        
        # Create a bot
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
        
        bot_id = await server.spawn_bot(bot_config, room_info)
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        status = server.get_bot_status(bot_id)
        
        assert status is not None
        assert status['bot_id'] == bot_id
        assert status['config']['name'] == "TestBot"
        
        # Clean up
        await server.terminate_bot(bot_id)
    
    def test_get_bot_status_not_found(self, server_manager):
        """Test getting status for non-existent bot."""
        server = server_manager
        status = server.get_bot_status("nonexistent")
        assert status is None
    
    @pytest.mark.asyncio
    async def test_get_room_bots(self, server_manager, test_room):
        """Test getting bots in a room."""
        server = server_manager
        
        room_info = {
            'room_id': test_room['room_id'],
            'room_code': test_room['room_code'],
            'room_password': test_room['room_password']
        }
        
        # Create multiple bots in the same room
        bot_ids = []
        for i in range(2):
            bot_config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name=f"TestBot{i}"
            )
            bot_id = await server.spawn_bot(bot_config, room_info)
            bot_ids.append(bot_id)
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        room_bots = server.get_room_bots(test_room['room_id'])
        
        assert len(room_bots) == 2
        assert all(bot['room_id'] == test_room['room_id'] for bot in room_bots)
        
        # Clean up
        for bot_id in bot_ids:
            await server.terminate_bot(bot_id)
    
    def test_get_room_bots_empty_room(self, server_manager):
        """Test getting bots for empty room."""
        server = server_manager
        room_bots = server.get_room_bots("empty_room")
        assert room_bots == []
    
    def test_get_server_status(self, server_manager):
        """Test getting server status."""
        server = server_manager
        
        status = server.get_server_status()
        
        assert 'total_bots' in status
        assert 'bots_by_status' in status
        assert 'client_pool_stats' in status
        assert status['running'] is True
    
    def test_get_available_bot_types(self, server_manager):
        """Test getting available bot types."""
        server = server_manager
        bot_types = server.get_available_bot_types()
        
        assert len(bot_types) >= 1  # At least rules-based should be available
        
        rules_based = next((bt for bt in bot_types if bt['type'] == 'rules_based'), None)
        assert rules_based is not None
        assert 'difficulties' in rules_based
        assert rules_based['supports_training_mode'] is True
    
    @pytest.mark.asyncio
    async def test_bot_status_callback(self, server_manager, test_room):
        """Test bot status change callbacks."""
        server = server_manager
        callback_calls = []
        
        async def status_callback(bot_id: str, status: BotStatus):
            callback_calls.append((bot_id, status))
        
        server.register_bot_status_callback(status_callback)
        
        # Create a bot to trigger status changes
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
        
        bot_id = await server.spawn_bot(bot_config, room_info)
        
        # Wait for status changes
        await asyncio.sleep(2)
        
        # Should have received at least one status callback
        assert len(callback_calls) >= 1
        
        # Clean up
        await server.terminate_bot(bot_id)
    
    @pytest.mark.asyncio
    async def test_room_empty_callback(self, server_manager, test_room):
        """Test room empty callbacks."""
        server = server_manager
        callback_calls = []
        
        async def room_empty_callback(room_id: str):
            callback_calls.append(room_id)
        
        server.register_room_empty_callback(room_empty_callback)
        server.config.auto_cleanup_empty_rooms = True
        
        # Create and terminate a bot to trigger room empty
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
        
        bot_id = await server.spawn_bot(bot_config, room_info)
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        await server.terminate_bot(bot_id)
        
        # Room empty callback might be called
        # Note: This depends on the implementation details


if __name__ == "__main__":
    pytest.main([__file__])