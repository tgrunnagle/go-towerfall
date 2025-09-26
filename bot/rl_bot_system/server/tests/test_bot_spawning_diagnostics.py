"""
Tests for enhanced bot spawning diagnostics.

This module tests the detailed logging and diagnostic tracking
implemented for bot spawning process as part of task 2.
"""

import pytest
import pytest_asyncio
import asyncio
import logging
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from rl_bot_system.server.bot_server import BotServer, BotServerConfig, BotConfig, BotType
from rl_bot_system.rules_based.rules_based_bot import DifficultyLevel
from rl_bot_system.server.diagnostics import (
    BotLifecycleEvent, DiagnosticLevel, ConnectionStatus, AIStatus
)


@pytest.fixture
def bot_server_config():
    """Create a test bot server configuration."""
    return BotServerConfig(
        max_bots_per_room=4,
        max_total_bots=10,
        bot_timeout_seconds=60,
        cleanup_interval_seconds=30,
        game_server_url="http://localhost:4000",
        models_dir="test_models",
        enable_performance_tracking=True
    )


@pytest_asyncio.fixture
async def bot_server(bot_server_config):
    """Create and start a test bot server."""
    server = BotServer(bot_server_config)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
def bot_config():
    """Create a test bot configuration."""
    return BotConfig(
        bot_type=BotType.RULES_BASED,
        difficulty=DifficultyLevel.INTERMEDIATE,
        name="TestBot",
        training_mode=False,
        auto_cleanup=True
    )


@pytest.fixture
def room_info():
    """Create test room information."""
    return {
        'room_id': 'test_room_123',
        'room_code': 'TEST123',
        'room_password': ''
    }


class TestBotSpawningDiagnostics:
    """Test enhanced bot spawning diagnostics."""
    
    @pytest.mark.asyncio
    async def test_spawn_bot_detailed_logging(self, bot_server, bot_config, room_info, caplog):
        """Test that bot spawning includes detailed logging."""
        with caplog.at_level(logging.INFO):
            # Mock the client pool and initialization to avoid actual connections
            with patch.object(bot_server, '_initialize_bot', new_callable=AsyncMock) as mock_init:
                bot_id = await bot_server.spawn_bot(bot_config, room_info)
                
                # Verify bot was created
                assert bot_id in bot_server._bots
                
                # Check that detailed logging occurred
                log_messages = [record.message for record in caplog.records]
                
                # Verify spawn request logging
                spawn_request_logs = [msg for msg in log_messages if "Bot spawn requested" in msg]
                assert len(spawn_request_logs) > 0
                
                spawn_log = spawn_request_logs[0]
                assert "Type: rules_based" in spawn_log
                assert "Difficulty: intermediate" in spawn_log
                assert "Name: TestBot" in spawn_log
                assert "Room: test_room_123" in spawn_log
                assert "Training: False" in spawn_log
                
                # Verify bot creation logging
                creation_logs = [msg for msg in log_messages if "Generated bot ID" in msg]
                assert len(creation_logs) > 0
                
                # Verify completion logging
                completion_logs = [msg for msg in log_messages if "Bot spawn setup completed successfully" in msg]
                assert len(completion_logs) > 0
                
                completion_log = completion_logs[0]
                assert f"ID: {bot_id}" in completion_log
                assert "Type: rules_based" in completion_log
                assert "Duration:" in completion_log
                
                # Verify initialization was called with correlation ID
                mock_init.assert_called_once()
                args, kwargs = mock_init.call_args
                assert args[0] == bot_id
                assert args[1] == room_info
                assert len(args) >= 3  # correlation_id should be passed
    
    @pytest.mark.asyncio
    async def test_spawn_bot_validation_logging(self, bot_server, room_info, caplog):
        """Test that bot configuration validation failures are logged."""
        # Create invalid bot config
        invalid_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.INTERMEDIATE,
            name="",  # Invalid empty name
            training_mode=False,
            auto_cleanup=True
        )
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="Bot configuration validation failed"):
                await bot_server.spawn_bot(invalid_config, room_info)
            
            # Check that validation failure was logged
            error_logs = [record.message for record in caplog.records if record.levelno >= logging.ERROR]
            validation_errors = [msg for msg in error_logs if "Bot spawn failed" in msg and "validation failed" in msg]
            assert len(validation_errors) > 0
    
    @pytest.mark.asyncio
    async def test_spawn_bot_limit_checking_logging(self, bot_server, bot_config, room_info, caplog):
        """Test that bot limit checking includes detailed logging."""
        # Fill up the bot server to capacity using different rooms to avoid room limits
        bot_ids = []
        for i in range(bot_server.config.max_total_bots):
            config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name=f"TestBot{i}",
                training_mode=False,
                auto_cleanup=True
            )
            
            # Use different room for each bot to avoid room limit
            room_info_i = {
                'room_id': f'test_room_{i}',
                'room_code': f'TEST{i}',
                'room_password': ''
            }
            
            with patch.object(bot_server, '_initialize_bot', new_callable=AsyncMock):
                bot_id = await bot_server.spawn_bot(config, room_info_i)
                bot_ids.append(bot_id)
        
        # Try to spawn one more bot (should fail)
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="Maximum total bots"):
                await bot_server.spawn_bot(bot_config, room_info)
            
            # Check that limit exceeded error was logged
            error_logs = [record.message for record in caplog.records if record.levelno >= logging.ERROR]
            limit_errors = [msg for msg in error_logs if "Maximum total bots" in msg and "exceeded" in msg]
            assert len(limit_errors) > 0
    
    @pytest.mark.asyncio
    async def test_spawn_bot_diagnostic_events(self, bot_server, bot_config, room_info):
        """Test that bot spawning creates appropriate diagnostic events."""
        with patch.object(bot_server, '_initialize_bot', new_callable=AsyncMock):
            bot_id = await bot_server.spawn_bot(bot_config, room_info)
            
            # Check that bot was registered with diagnostic tracker
            diagnostic_info = bot_server.diagnostic_tracker.get_bot_diagnostics(bot_id)
            assert diagnostic_info is not None
            assert diagnostic_info.bot_name == "TestBot"
            assert diagnostic_info.bot_type == "rules_based"
            assert diagnostic_info.difficulty == "intermediate"
            assert diagnostic_info.room_id == "test_room_123"
            
            # Check that spawn events were logged
            events = bot_server.diagnostic_tracker.get_diagnostic_events(bot_id)
            spawn_events = [e for e in events if e.event_type == BotLifecycleEvent.BOT_SPAWN_REQUESTED]
            assert len(spawn_events) >= 2  # Should have registration and completion events
            
            # Find the completion event (has spawn_duration_ms)
            completion_event = None
            for event in spawn_events:
                if 'spawn_duration_ms' in event.details:
                    completion_event = event
                    break
            
            assert completion_event is not None
            assert completion_event.level == DiagnosticLevel.INFO
            assert "spawn setup completed" in completion_event.message
            assert completion_event.details['bot_type'] == 'rules_based'
            assert completion_event.details['difficulty'] == 'intermediate'
            assert completion_event.details['room_id'] == 'test_room_123'
            assert 'spawn_duration_ms' in completion_event.details
            assert 'total_bots_after' in completion_event.details
    
    @pytest.mark.asyncio
    async def test_bot_config_validation_comprehensive(self, bot_server, room_info):
        """Test comprehensive bot configuration validation."""
        # Test various invalid configurations
        test_cases = [
            # Invalid bot name cases
            (BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name="",
                training_mode=False,
                auto_cleanup=True
            ), "Bot name is required"),
            (BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name="x" * 51,
                training_mode=False,
                auto_cleanup=True
            ), "Bot name must be 50 characters or less"),
            
            # Invalid RL generation cases
            (BotConfig(
                bot_type=BotType.RL_GENERATION,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name="TestBot",
                training_mode=False,
                auto_cleanup=True
            ), "Generation is required for RL bots"),
            (BotConfig(
                bot_type=BotType.RL_GENERATION,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name="TestBot",
                generation=-1,
                training_mode=False,
                auto_cleanup=True
            ), "Generation must be a non-negative integer"),
        ]
        
        for invalid_config, expected_error in test_cases:
            with pytest.raises(RuntimeError) as exc_info:
                await bot_server.spawn_bot(invalid_config, room_info)
            
            assert "Bot configuration validation failed" in str(exc_info.value)
            assert expected_error in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_room_bot_limit_logging(self, bot_server, bot_config, room_info, caplog):
        """Test that room bot limit checking includes detailed logging."""
        # Fill up the room to capacity
        bot_ids = []
        for i in range(bot_server.config.max_bots_per_room):
            config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.INTERMEDIATE,
                name=f"RoomBot{i}",
                training_mode=False,
                auto_cleanup=True
            )
            
            with patch.object(bot_server, '_initialize_bot', new_callable=AsyncMock):
                bot_id = await bot_server.spawn_bot(config, room_info)
                bot_ids.append(bot_id)
        
        # Try to spawn one more bot in the same room (should fail)
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="Maximum bots per room"):
                await bot_server.spawn_bot(bot_config, room_info)
            
            # Check that room limit exceeded error was logged
            error_logs = [record.message for record in caplog.records if record.levelno >= logging.ERROR]
            room_limit_errors = [msg for msg in error_logs if "Maximum bots per room" in msg and "exceeded" in msg]
            assert len(room_limit_errors) > 0
            
            # Verify the error includes room information
            room_error = room_limit_errors[0]
            assert "test_room_123" in room_error
    
    @pytest.mark.asyncio
    async def test_spawn_bot_correlation_tracking(self, bot_server, bot_config, room_info):
        """Test that bot spawning uses correlation IDs for event tracking."""
        with patch.object(bot_server, '_initialize_bot', new_callable=AsyncMock) as mock_init:
            bot_id = await bot_server.spawn_bot(bot_config, room_info)
            
            # Verify initialization was called with correlation ID
            mock_init.assert_called_once()
            args, kwargs = mock_init.call_args
            correlation_id = args[2] if len(args) > 2 else None
            assert correlation_id is not None
            
            # Check that diagnostic events include the correlation ID
            events = bot_server.diagnostic_tracker.get_diagnostic_events(bot_id)
            spawn_events = [e for e in events if e.event_type == BotLifecycleEvent.BOT_SPAWN_REQUESTED]
            assert len(spawn_events) >= 2
            
            # Find the completion event (should have correlation ID)
            completion_event = None
            for event in spawn_events:
                if event.correlation_id == correlation_id:
                    completion_event = event
                    break
            
            assert completion_event is not None
            assert completion_event.correlation_id == correlation_id
    
    @pytest.mark.asyncio
    async def test_spawn_bot_performance_metrics(self, bot_server, bot_config, room_info):
        """Test that bot spawning tracks performance metrics."""
        start_time = datetime.now()
        
        with patch.object(bot_server, '_initialize_bot', new_callable=AsyncMock):
            bot_id = await bot_server.spawn_bot(bot_config, room_info)
            
            end_time = datetime.now()
            
            # Check that spawn event includes timing information
            events = bot_server.diagnostic_tracker.get_diagnostic_events(bot_id)
            spawn_events = [e for e in events if e.event_type == BotLifecycleEvent.BOT_SPAWN_REQUESTED]
            assert len(spawn_events) >= 2
            
            # Find the completion event (has spawn_duration_ms)
            completion_event = None
            for event in spawn_events:
                if 'spawn_duration_ms' in event.details:
                    completion_event = event
                    break
            
            assert completion_event is not None
            assert 'spawn_duration_ms' in completion_event.details
            assert isinstance(completion_event.details['spawn_duration_ms'], (int, float))
            assert completion_event.details['spawn_duration_ms'] >= 0
            
            # Check that bot count metrics are included
            assert 'total_bots_after' in completion_event.details
            assert completion_event.details['total_bots_after'] == 1
            assert 'room_bots_after' in completion_event.details
            assert completion_event.details['room_bots_after'] == 1