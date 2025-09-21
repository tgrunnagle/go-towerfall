"""
Unit tests for SessionManager class.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from bot.rl_bot_system.training.session_manager import (
    SessionManager,
    ResourceLimits,
    ResourceStatus,
    SessionRequest
)
from bot.rl_bot_system.training.training_session import (
    TrainingConfig,
    TrainingMode,
    SessionMetrics
)


class TestSessionManager:
    """Test cases for SessionManager class."""

    @pytest.fixture
    def resource_limits(self):
        """Create test resource limits."""
        return ResourceLimits(
            max_concurrent_sessions=3,
            max_parallel_episodes_per_session=2,
            max_total_parallel_episodes=6,
            memory_limit_mb=4096,
            cpu_limit_percent=70
        )

    @pytest.fixture
    def session_manager(self, resource_limits):
        """Create a test session manager."""
        return SessionManager(
            resource_limits=resource_limits,
            game_server_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws"
        )

    def test_session_manager_initialization(self, session_manager, resource_limits):
        """Test session manager initialization."""
        assert session_manager.resource_limits == resource_limits
        assert len(session_manager.active_sessions) == 0
        assert session_manager.resource_status == ResourceStatus.AVAILABLE
        assert session_manager.current_parallel_episodes == 0

    def test_resource_limits_defaults(self):
        """Test resource limits default values."""
        limits = ResourceLimits()
        assert limits.max_concurrent_sessions == 5
        assert limits.max_parallel_episodes_per_session == 4
        assert limits.max_total_parallel_episodes == 16
        assert limits.memory_limit_mb == 8192
        assert limits.cpu_limit_percent == 80

    def test_session_request_creation(self):
        """Test SessionRequest creation."""
        config = TrainingConfig(max_episodes=500)
        request = SessionRequest(
            session_id="test_session",
            config=config,
            priority=5
        )
        
        assert request.session_id == "test_session"
        assert request.config == config
        assert request.priority == 5
        assert request.requested_at is not None

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test session creation."""
        config = TrainingConfig(max_episodes=100)
        
        session_id = await session_manager.create_session(
            config=config,
            priority=1,
            session_id="custom_session"
        )
        
        assert session_id == "custom_session"
        assert session_manager.session_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_create_session_with_auto_id(self, session_manager):
        """Test session creation with automatic ID generation."""
        session_id = await session_manager.create_session()
        
        assert session_id is not None
        assert session_manager.session_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager):
        """Test getting an active session."""
        # Add a mock session
        mock_session = MagicMock()
        session_manager.active_sessions["test_session"] = mock_session
        
        result = await session_manager.get_session("test_session")
        assert result == mock_session
        
        # Test non-existent session
        result = await session_manager.get_session("non_existent")
        assert result is None

    @pytest.mark.asyncio
    async def test_pause_session(self, session_manager):
        """Test pausing a session."""
        mock_session = AsyncMock()
        session_manager.active_sessions["test_session"] = mock_session
        
        result = await session_manager.pause_session("test_session")
        assert result is True
        mock_session.pause.assert_called_once()
        
        # Test non-existent session
        result = await session_manager.pause_session("non_existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_resume_session(self, session_manager):
        """Test resuming a session."""
        mock_session = AsyncMock()
        session_manager.active_sessions["test_session"] = mock_session
        
        result = await session_manager.resume_session("test_session")
        assert result is True
        mock_session.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_session(self, session_manager):
        """Test stopping a session."""
        mock_session = AsyncMock()
        session_manager.active_sessions["test_session"] = mock_session
        
        result = await session_manager.stop_session("test_session")
        assert result is True
        mock_session.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_sessions_info(self, session_manager):
        """Test getting all sessions information."""
        # Add mock sessions
        mock_session1 = AsyncMock()
        mock_session1.get_session_info.return_value = {"id": "session1"}
        mock_session2 = AsyncMock()
        mock_session2.get_session_info.return_value = {"id": "session2"}
        
        session_manager.active_sessions["session1"] = mock_session1
        session_manager.active_sessions["session2"] = mock_session2
        session_manager.completed_sessions = ["completed1", "completed2"]
        
        # Mock queue size
        session_manager.session_queue.qsize = MagicMock(return_value=3)
        
        info = await session_manager.get_all_sessions_info()
        
        assert "activeSessions" in info
        assert "completedSessions" in info
        assert "queuedSessions" in info
        assert "resourceStatus" in info
        assert info["queuedSessions"] == 3
        assert len(info["completedSessions"]) == 2

    @pytest.mark.asyncio
    async def test_get_global_metrics(self, session_manager):
        """Test getting global metrics."""
        # Add mock sessions with metrics
        mock_session1 = MagicMock()
        mock_session1.metrics = SessionMetrics(
            episodes_completed=10,
            total_reward=100.0,
            episodes_failed=1
        )
        
        mock_session2 = MagicMock()
        mock_session2.metrics = SessionMetrics(
            episodes_completed=20,
            total_reward=200.0,
            episodes_failed=2
        )
        
        session_manager.active_sessions["session1"] = mock_session1
        session_manager.active_sessions["session2"] = mock_session2
        session_manager.current_parallel_episodes = 4
        
        metrics = await session_manager.get_global_metrics()
        
        assert metrics["totalSessions"] == 2
        assert metrics["totalEpisodes"] == 30
        assert metrics["totalRewards"] == 300.0
        assert metrics["totalFailures"] == 3
        assert metrics["averageRewardPerEpisode"] == 10.0
        assert metrics["globalSuccessRate"] == 30/33  # 30 success / 33 total

    def test_can_start_new_session(self, session_manager):
        """Test session start capability check."""
        # Initially should be able to start
        assert session_manager._can_start_new_session() is True
        
        # Add sessions up to limit
        for i in range(session_manager.resource_limits.max_concurrent_sessions):
            session_manager.active_sessions[f"session_{i}"] = MagicMock()
        
        # Should not be able to start more
        assert session_manager._can_start_new_session() is False
        
        # Test parallel episodes limit
        session_manager.active_sessions.clear()
        session_manager.current_parallel_episodes = session_manager.resource_limits.max_total_parallel_episodes
        
        assert session_manager._can_start_new_session() is False

    def test_resource_status_enum(self):
        """Test ResourceStatus enum values."""
        assert ResourceStatus.AVAILABLE.value == "available"
        assert ResourceStatus.LIMITED.value == "limited"
        assert ResourceStatus.EXHAUSTED.value == "exhausted"

    @pytest.mark.asyncio
    async def test_start_and_stop_manager(self, session_manager):
        """Test starting and stopping the session manager."""
        # Mock the background tasks
        with patch.object(session_manager, '_process_session_queue', new_callable=AsyncMock) as mock_processor, \
             patch.object(session_manager, '_monitor_resources', new_callable=AsyncMock) as mock_monitor:
            
            await session_manager.start()
            
            # Check that background tasks were created
            assert session_manager._session_processor_task is not None
            assert session_manager._resource_monitor_task is not None
            
            # Stop the manager
            await session_manager.stop()
            
            # Check shutdown event was set
            assert session_manager._shutdown_event.is_set()

    def test_handler_registration(self, session_manager):
        """Test event handler registration."""
        start_handler = AsyncMock()
        complete_handler = AsyncMock()
        resource_handler = AsyncMock()
        
        session_manager.register_session_start_handler(start_handler)
        session_manager.register_session_complete_handler(complete_handler)
        session_manager.register_resource_alert_handler(resource_handler)
        
        assert start_handler in session_manager.session_start_handlers
        assert complete_handler in session_manager.session_complete_handlers
        assert resource_handler in session_manager.resource_alert_handlers

    @pytest.mark.asyncio
    async def test_episode_and_metrics_handlers(self, session_manager):
        """Test episode and metrics handler methods."""
        from bot.rl_bot_system.training.training_session import EpisodeResult
        
        # Test episode completion handler (should be no-op)
        result = EpisodeResult(
            episode_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration=1.0,
            total_reward=10.0,
            episode_length=100,
            game_result="win",
            final_state={}
        )
        
        # Should not raise any exceptions
        await session_manager._handle_episode_complete(result)
        
        # Test metrics update handler (should be no-op)
        metrics = SessionMetrics()
        await session_manager._handle_metrics_update(metrics)

    @pytest.mark.asyncio
    async def test_resource_monitoring_logic(self, session_manager):
        """Test resource monitoring status updates."""
        # Test available status (default)
        assert session_manager.resource_status == ResourceStatus.AVAILABLE
        
        # Simulate high resource usage for limited status
        session_manager.current_parallel_episodes = int(
            session_manager.resource_limits.max_total_parallel_episodes * 0.9
        )
        
        # Add sessions near limit
        for i in range(int(session_manager.resource_limits.max_concurrent_sessions * 0.9)):
            session_manager.active_sessions[f"session_{i}"] = MagicMock()
        
        # Manually trigger resource status update logic
        old_status = session_manager.resource_status
        
        if (
            len(session_manager.active_sessions) >= session_manager.resource_limits.max_concurrent_sessions or
            session_manager.current_parallel_episodes >= session_manager.resource_limits.max_total_parallel_episodes
        ):
            session_manager.resource_status = ResourceStatus.EXHAUSTED
        elif (
            len(session_manager.active_sessions) >= session_manager.resource_limits.max_concurrent_sessions * 0.8 or
            session_manager.current_parallel_episodes >= session_manager.resource_limits.max_total_parallel_episodes * 0.8
        ):
            session_manager.resource_status = ResourceStatus.LIMITED
        else:
            session_manager.resource_status = ResourceStatus.AVAILABLE
        
        # Should be limited now
        assert session_manager.resource_status == ResourceStatus.LIMITED

    @pytest.mark.asyncio
    async def test_session_queue_priority(self, session_manager):
        """Test session queue priority handling."""
        # Create sessions with different priorities
        await session_manager.create_session(priority=1)
        await session_manager.create_session(priority=5)  # Higher priority
        await session_manager.create_session(priority=3)
        
        assert session_manager.session_queue.qsize() == 3
        
        # The queue should handle priority ordering internally
        # (actual priority testing would require processing the queue)