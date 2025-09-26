"""
Tests for the bot diagnostic infrastructure.

This module tests the diagnostic tracking system for bot lifecycle events,
health monitoring, and activity metrics.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from rl_bot_system.server.diagnostics import (
    BotDiagnosticTracker, BotLifecycleEvent, DiagnosticLevel,
    ConnectionStatus, AIStatus, BotDiagnosticInfo, ConnectionHealth,
    BotActivityMetrics, DiagnosticEvent
)


@pytest_asyncio.fixture
async def diagnostic_tracker():
    """Create a diagnostic tracker for testing."""
    tracker = BotDiagnosticTracker(max_events_per_bot=100, cleanup_interval_hours=1)
    await tracker.start()
    try:
        yield tracker
    finally:
        await tracker.stop()


@pytest.fixture
def sample_bot_id():
    """Sample bot ID for testing."""
    return "test-bot-123"


@pytest.fixture
def sample_bot_data():
    """Sample bot data for testing."""
    return {
        'bot_name': 'TestBot',
        'bot_type': 'rules_based',
        'difficulty': 'intermediate',
        'room_id': 'test-room-456'
    }


class TestBotDiagnosticTracker:
    """Test the BotDiagnosticTracker class."""
    
    @pytest.mark.asyncio
    async def test_register_bot(self, diagnostic_tracker, sample_bot_id, sample_bot_data):
        """Test bot registration creates diagnostic tracking."""
        # Register bot
        diagnostic_tracker.register_bot(
            bot_id=sample_bot_id,
            bot_name=sample_bot_data['bot_name'],
            bot_type=sample_bot_data['bot_type'],
            difficulty=sample_bot_data['difficulty'],
            room_id=sample_bot_data['room_id']
        )
        
        # Verify bot is registered
        diagnostic_info = diagnostic_tracker.get_bot_diagnostics(sample_bot_id)
        assert diagnostic_info is not None
        assert diagnostic_info.bot_id == sample_bot_id
        assert diagnostic_info.bot_name == sample_bot_data['bot_name']
        assert diagnostic_info.bot_type == sample_bot_data['bot_type']
        assert diagnostic_info.difficulty == sample_bot_data['difficulty']
        assert diagnostic_info.room_id == sample_bot_data['room_id']
        assert diagnostic_info.status == "initializing"
        assert diagnostic_info.connection_status == ConnectionStatus.NOT_CONNECTED
        assert diagnostic_info.ai_status == AIStatus.NOT_LOADED
        
        # Verify connection health is created
        connection_health = diagnostic_tracker.get_connection_health(sample_bot_id)
        assert connection_health is not None
        assert connection_health.websocket_connected is False
        assert connection_health.connection_status == ConnectionStatus.NOT_CONNECTED
        
        # Verify activity metrics are created
        activity_metrics = diagnostic_tracker.get_activity_metrics(sample_bot_id)
        assert activity_metrics is not None
        assert activity_metrics.decisions_made == 0
        assert activity_metrics.actions_executed == 0
        
        # Verify initial event was logged
        events = diagnostic_tracker.get_diagnostic_events(sample_bot_id)
        assert len(events) > 0
        assert events[0].event_type == BotLifecycleEvent.BOT_SPAWN_REQUESTED
    
    @pytest.mark.asyncio
    async def test_unregister_bot(self, diagnostic_tracker, sample_bot_id, sample_bot_data):
        """Test bot unregistration cleans up diagnostic data."""
        # Register bot first
        diagnostic_tracker.register_bot(
            bot_id=sample_bot_id,
            bot_name=sample_bot_data['bot_name'],
            bot_type=sample_bot_data['bot_type'],
            difficulty=sample_bot_data['difficulty']
        )
        
        # Verify bot is registered
        assert diagnostic_tracker.get_bot_diagnostics(sample_bot_id) is not None
        
        # Unregister bot
        diagnostic_tracker.unregister_bot(sample_bot_id)
        
        # Verify bot data is cleaned up
        assert diagnostic_tracker.get_bot_diagnostics(sample_bot_id) is None
        assert diagnostic_tracker.get_connection_health(sample_bot_id) is None
        assert diagnostic_tracker.get_activity_metrics(sample_bot_id) is None
        assert diagnostic_tracker.get_diagnostic_events(sample_bot_id) == []
    
    @pytest.mark.asyncio
    async def test_log_event(self, diagnostic_tracker, sample_bot_id, sample_bot_data):
        """Test logging diagnostic events."""
        # Register bot
        diagnostic_tracker.register_bot(
            bot_id=sample_bot_id,
            bot_name=sample_bot_data['bot_name'],
            bot_type=sample_bot_data['bot_type'],
            difficulty=sample_bot_data['difficulty']
        )
        
        # Log a test event
        test_message = "Test diagnostic event"
        test_details = {'test_key': 'test_value'}
        
        diagnostic_tracker.log_event(
            bot_id=sample_bot_id,
            event_type=BotLifecycleEvent.BOT_AI_LOADED,
            level=DiagnosticLevel.INFO,
            message=test_message,
            details=test_details
        )
        
        # Verify event was logged
        events = diagnostic_tracker.get_diagnostic_events(sample_bot_id)
        ai_loaded_events = [e for e in events if e.event_type == BotLifecycleEvent.BOT_AI_LOADED]
        assert len(ai_loaded_events) > 0
        
        event = ai_loaded_events[0]
        assert event.message == test_message
        assert event.details == test_details
        assert event.level == DiagnosticLevel.INFO
    
    @pytest.mark.asyncio
    async def test_update_bot_status(self, diagnostic_tracker, sample_bot_id, sample_bot_data):
        """Test updating bot status."""
        # Register bot
        diagnostic_tracker.register_bot(
            bot_id=sample_bot_id,
            bot_name=sample_bot_data['bot_name'],
            bot_type=sample_bot_data['bot_type'],
            difficulty=sample_bot_data['difficulty']
        )
        
        # Update status
        new_status = "active"
        diagnostic_tracker.update_bot_status(sample_bot_id, new_status)
        
        # Verify status was updated
        diagnostic_info = diagnostic_tracker.get_bot_diagnostics(sample_bot_id)
        assert diagnostic_info.status == new_status
        
        # Verify event was logged
        events = diagnostic_tracker.get_diagnostic_events(sample_bot_id)
        active_events = [e for e in events if e.event_type == BotLifecycleEvent.BOT_ACTIVE]
        assert len(active_events) > 0
    
    @pytest.mark.asyncio
    async def test_update_connection_status(self, diagnostic_tracker, sample_bot_id, sample_bot_data):
        """Test updating connection status."""
        # Register bot
        diagnostic_tracker.register_bot(
            bot_id=sample_bot_id,
            bot_name=sample_bot_data['bot_name'],
            bot_type=sample_bot_data['bot_type'],
            difficulty=sample_bot_data['difficulty']
        )
        
        # Update connection status
        websocket_url = "ws://localhost:4000/ws"
        diagnostic_tracker.update_connection_status(
            bot_id=sample_bot_id,
            connection_status=ConnectionStatus.CONNECTED,
            websocket_connected=True,
            websocket_url=websocket_url
        )
        
        # Verify connection status was updated
        diagnostic_info = diagnostic_tracker.get_bot_diagnostics(sample_bot_id)
        assert diagnostic_info.connection_status == ConnectionStatus.CONNECTED
        assert diagnostic_info.websocket_connected is True
        assert diagnostic_info.websocket_url == websocket_url
        
        connection_health = diagnostic_tracker.get_connection_health(sample_bot_id)
        assert connection_health.connection_status == ConnectionStatus.CONNECTED
        assert connection_health.websocket_connected is True
        
        # Verify event was logged
        events = diagnostic_tracker.get_diagnostic_events(sample_bot_id)
        connected_events = [e for e in events if e.event_type == BotLifecycleEvent.BOT_WEBSOCKET_CONNECTED]
        assert len(connected_events) > 0
    
    @pytest.mark.asyncio
    async def test_update_ai_status(self, diagnostic_tracker, sample_bot_id, sample_bot_data):
        """Test updating AI status."""
        # Register bot
        diagnostic_tracker.register_bot(
            bot_id=sample_bot_id,
            bot_name=sample_bot_data['bot_name'],
            bot_type=sample_bot_data['bot_type'],
            difficulty=sample_bot_data['difficulty']
        )
        
        # Update AI status
        diagnostic_tracker.update_ai_status(sample_bot_id, AIStatus.LOADED)
        
        # Verify AI status was updated
        diagnostic_info = diagnostic_tracker.get_bot_diagnostics(sample_bot_id)
        assert diagnostic_info.ai_status == AIStatus.LOADED
        
        # Verify event was logged
        events = diagnostic_tracker.get_diagnostic_events(sample_bot_id)
        ai_loaded_events = [e for e in events if e.event_type == BotLifecycleEvent.BOT_AI_LOADED]
        assert len(ai_loaded_events) > 0
    
    @pytest.mark.asyncio
    async def test_record_bot_decision(self, diagnostic_tracker, sample_bot_id, sample_bot_data):
        """Test recording bot decisions."""
        # Register bot
        diagnostic_tracker.register_bot(
            bot_id=sample_bot_id,
            bot_name=sample_bot_data['bot_name'],
            bot_type=sample_bot_data['bot_type'],
            difficulty=sample_bot_data['difficulty']
        )
        
        # Record decisions
        decision_details = {'action': 'move_left', 'confidence': 0.8}
        diagnostic_tracker.record_bot_decision(sample_bot_id, decision_details)
        diagnostic_tracker.record_bot_decision(sample_bot_id)
        
        # Verify decisions were recorded
        diagnostic_info = diagnostic_tracker.get_bot_diagnostics(sample_bot_id)
        assert diagnostic_info.decisions_made == 2
        assert diagnostic_info.last_decision_time is not None
        
        activity_metrics = diagnostic_tracker.get_activity_metrics(sample_bot_id)
        assert activity_metrics.decisions_made == 2
        assert activity_metrics.last_decision_time is not None
    
    @pytest.mark.asyncio
    async def test_record_bot_action(self, diagnostic_tracker, sample_bot_id, sample_bot_data):
        """Test recording bot actions."""
        # Register bot
        diagnostic_tracker.register_bot(
            bot_id=sample_bot_id,
            bot_name=sample_bot_data['bot_name'],
            bot_type=sample_bot_data['bot_type'],
            difficulty=sample_bot_data['difficulty']
        )
        
        # Record successful action
        diagnostic_tracker.record_bot_action(
            bot_id=sample_bot_id,
            action_type="move_left",
            success=True,
            action_details={'key': 'a', 'duration': 0.1}
        )
        
        # Record failed action
        diagnostic_tracker.record_bot_action(
            bot_id=sample_bot_id,
            action_type="shoot",
            success=False,
            error_message="No target found"
        )
        
        # Verify actions were recorded
        diagnostic_info = diagnostic_tracker.get_bot_diagnostics(sample_bot_id)
        assert diagnostic_info.actions_sent == 2
        assert len(diagnostic_info.error_messages) == 1
        assert "No target found" in diagnostic_info.error_messages
        
        activity_metrics = diagnostic_tracker.get_activity_metrics(sample_bot_id)
        assert activity_metrics.actions_executed == 1
        assert activity_metrics.actions_failed == 1
        assert activity_metrics.errors_encountered == 1
    
    @pytest.mark.asyncio
    async def test_get_diagnostic_events_filtering(self, diagnostic_tracker, sample_bot_id, sample_bot_data):
        """Test filtering diagnostic events."""
        # Register bot
        diagnostic_tracker.register_bot(
            bot_id=sample_bot_id,
            bot_name=sample_bot_data['bot_name'],
            bot_type=sample_bot_data['bot_type'],
            difficulty=sample_bot_data['difficulty']
        )
        
        # Log various events
        diagnostic_tracker.log_event(
            bot_id=sample_bot_id,
            event_type=BotLifecycleEvent.BOT_AI_LOADED,
            level=DiagnosticLevel.INFO,
            message="AI loaded"
        )
        
        diagnostic_tracker.log_event(
            bot_id=sample_bot_id,
            event_type=BotLifecycleEvent.BOT_ERROR,
            level=DiagnosticLevel.ERROR,
            message="Bot error occurred"
        )
        
        diagnostic_tracker.log_event(
            bot_id=sample_bot_id,
            event_type=BotLifecycleEvent.BOT_AI_DECISION_MADE,
            level=DiagnosticLevel.DEBUG,
            message="Decision made"
        )
        
        # Test filtering by event type
        ai_events = diagnostic_tracker.get_diagnostic_events(
            sample_bot_id, 
            event_type=BotLifecycleEvent.BOT_AI_LOADED
        )
        assert len(ai_events) == 1
        assert ai_events[0].event_type == BotLifecycleEvent.BOT_AI_LOADED
        
        # Test filtering by level
        error_events = diagnostic_tracker.get_diagnostic_events(
            sample_bot_id,
            level=DiagnosticLevel.ERROR
        )
        assert len(error_events) == 1
        assert error_events[0].level == DiagnosticLevel.ERROR
        
        # Test limit
        limited_events = diagnostic_tracker.get_diagnostic_events(sample_bot_id, limit=2)
        assert len(limited_events) <= 2
    
    @pytest.mark.asyncio
    async def test_get_all_bot_diagnostics(self, diagnostic_tracker, sample_bot_data):
        """Test getting diagnostics for all bots."""
        # Register multiple bots
        bot_ids = ["bot-1", "bot-2", "bot-3"]
        
        for bot_id in bot_ids:
            diagnostic_tracker.register_bot(
                bot_id=bot_id,
                bot_name=f"TestBot-{bot_id}",
                bot_type=sample_bot_data['bot_type'],
                difficulty=sample_bot_data['difficulty']
            )
        
        # Get all diagnostics
        all_diagnostics = diagnostic_tracker.get_all_bot_diagnostics()
        
        # Verify all bots are included
        assert len(all_diagnostics) == len(bot_ids)
        for bot_id in bot_ids:
            assert bot_id in all_diagnostics
            assert all_diagnostics[bot_id] is not None
    
    @pytest.mark.asyncio
    async def test_event_callbacks(self, diagnostic_tracker, sample_bot_id, sample_bot_data):
        """Test event callbacks are triggered."""
        # Register bot
        diagnostic_tracker.register_bot(
            bot_id=sample_bot_id,
            bot_name=sample_bot_data['bot_name'],
            bot_type=sample_bot_data['bot_type'],
            difficulty=sample_bot_data['difficulty']
        )
        
        # Register callback
        callback_called = False
        received_event = None
        
        async def test_callback(event):
            nonlocal callback_called, received_event
            callback_called = True
            received_event = event
        
        diagnostic_tracker.register_event_callback(test_callback)
        
        # Log an event
        diagnostic_tracker.log_event(
            bot_id=sample_bot_id,
            event_type=BotLifecycleEvent.BOT_ACTIVE,
            level=DiagnosticLevel.INFO,
            message="Bot is now active"
        )
        
        # Give callback time to execute
        await asyncio.sleep(0.1)
        
        # Verify callback was called
        assert callback_called
        assert received_event is not None
        assert received_event.event_type == BotLifecycleEvent.BOT_ACTIVE


class TestDiagnosticDataModels:
    """Test the diagnostic data model classes."""
    
    def test_bot_diagnostic_info_to_dict(self):
        """Test BotDiagnosticInfo serialization."""
        current_time = datetime.now()
        
        diagnostic_info = BotDiagnosticInfo(
            bot_id="test-bot",
            bot_name="TestBot",
            bot_type="rules_based",
            difficulty="intermediate",
            status="active",
            connection_status=ConnectionStatus.CONNECTED,
            ai_status=AIStatus.ACTIVE,
            room_id="test-room",
            created_at=current_time,
            last_activity=current_time,
            last_decision_time=current_time
        )
        
        data_dict = diagnostic_info.to_dict()
        
        assert data_dict['bot_id'] == "test-bot"
        assert data_dict['bot_name'] == "TestBot"
        assert data_dict['connection_status'] == "connected"
        assert data_dict['ai_status'] == "active"
        assert data_dict['websocket_connected'] is False
        assert data_dict['actions_sent'] == 0
    
    def test_connection_health_to_dict(self):
        """Test ConnectionHealth serialization."""
        current_time = datetime.now()
        
        connection_health = ConnectionHealth(
            websocket_connected=True,
            connection_status=ConnectionStatus.CONNECTED,
            last_ping=current_time,
            last_pong=current_time,
            reconnection_attempts=2,
            messages_sent=10,
            messages_received=15
        )
        
        data_dict = connection_health.to_dict()
        
        assert data_dict['websocket_connected'] is True
        assert data_dict['connection_status'] == "connected"
        assert data_dict['reconnection_attempts'] == 2
        assert data_dict['messages_sent'] == 10
        assert data_dict['messages_received'] == 15
    
    def test_bot_activity_metrics_to_dict(self):
        """Test BotActivityMetrics serialization."""
        current_time = datetime.now()
        
        activity_metrics = BotActivityMetrics(
            decisions_made=50,
            actions_executed=45,
            actions_failed=5,
            game_state_updates=100,
            errors_encountered=3,
            uptime_seconds=300.5,
            last_decision_time=current_time,
            decision_success_rate=0.9,
            action_success_rate=0.9
        )
        
        data_dict = activity_metrics.to_dict()
        
        assert data_dict['decisions_made'] == 50
        assert data_dict['actions_executed'] == 45
        assert data_dict['actions_failed'] == 5
        assert data_dict['decision_success_rate'] == 0.9
        assert data_dict['action_success_rate'] == 0.9
        assert data_dict['uptime_seconds'] == 300.5
    
    def test_diagnostic_event_to_dict(self):
        """Test DiagnosticEvent serialization."""
        current_time = datetime.now()
        
        event = DiagnosticEvent(
            event_id="event-123",
            bot_id="bot-456",
            event_type=BotLifecycleEvent.BOT_ACTIVE,
            timestamp=current_time,
            level=DiagnosticLevel.INFO,
            message="Bot became active",
            details={'test': 'value'},
            correlation_id="corr-789"
        )
        
        data_dict = event.to_dict()
        
        assert data_dict['event_id'] == "event-123"
        assert data_dict['bot_id'] == "bot-456"
        assert data_dict['event_type'] == "bot_active"
        assert data_dict['level'] == "info"
        assert data_dict['message'] == "Bot became active"
        assert data_dict['details'] == {'test': 'value'}
        assert data_dict['correlation_id'] == "corr-789"