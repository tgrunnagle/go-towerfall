"""
Unit tests for TrainingSession class.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from rl_bot_system.training.training_session import (
    TrainingSession,
    TrainingConfig,
    TrainingMode,
    SessionStatus,
    SessionMetrics,
    EpisodeResult
)


class TestTrainingSession:
    """Test cases for TrainingSession class."""

    @pytest.fixture
    def training_config(self):
        """Create a test training configuration."""
        return TrainingConfig(
            speed_multiplier=10.0,
            headless_mode=True,
            max_episodes=100,
            episode_timeout=60,
            parallel_episodes=2,
            training_mode=TrainingMode.TRAINING,
            spectator_enabled=False,
            auto_cleanup=True
        )

    @pytest.fixture
    def training_session(self, training_config):
        """Create a test training session."""
        return TrainingSession(
            session_id="test_session",
            config=training_config,
            game_server_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws"
        )

    def test_training_session_initialization(self, training_session, training_config):
        """Test training session initialization."""
        assert training_session.session_id == "test_session"
        assert training_session.config == training_config
        assert training_session.status == SessionStatus.INITIALIZING
        assert training_session.metrics.episodes_completed == 0
        assert training_session.room_code is None
        assert len(training_session.active_episodes) == 0

    def test_training_config_defaults(self):
        """Test training configuration default values."""
        config = TrainingConfig()
        assert config.speed_multiplier == 1.0
        assert config.headless_mode is False
        assert config.max_episodes == 1000
        assert config.episode_timeout == 300
        assert config.parallel_episodes == 1
        assert config.training_mode == TrainingMode.REALTIME
        assert config.spectator_enabled is False
        assert config.auto_cleanup is True

    @pytest.mark.asyncio
    async def test_session_initialization(self, training_session):
        """Test session initialization process."""
        with patch.object(training_session, '_create_training_room', new_callable=AsyncMock) as mock_create_room:
            mock_create_room.return_value = None
            
            await training_session.initialize()
            
            assert training_session.status == SessionStatus.RUNNING
            assert training_session.start_time is not None
            mock_create_room.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_initialization_failure(self, training_session):
        """Test session initialization failure handling."""
        with patch.object(training_session, '_create_training_room', new_callable=AsyncMock) as mock_create_room:
            mock_create_room.side_effect = Exception("Room creation failed")
            
            with pytest.raises(Exception, match="Room creation failed"):
                await training_session.initialize()
            
            assert training_session.status == SessionStatus.FAILED

    @pytest.mark.asyncio
    async def test_request_training_room(self, training_session):
        """Test training room request."""
        mock_response_data = {
            "roomCode": "TR123456",
            "roomId": "room_tr123456",
            "config": {"speedMultiplier": 10.0, "headlessMode": True}
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = mock_response_data
            
            # Fix the mock chain for aiohttp
            mock_session_instance = AsyncMock()
            mock_post_context = AsyncMock()
            mock_post_context.__aenter__.return_value = mock_response
            mock_session_instance.post.return_value = mock_post_context
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            result = await training_session.request_training_room(10.0, True)
            
            assert result == mock_response_data
            assert training_session.room_code == "TR123456"
            assert training_session.room_id == "room_tr123456"

    @pytest.mark.asyncio
    async def test_pause_and_resume(self, training_session):
        """Test session pause and resume functionality."""
        training_session.status = SessionStatus.RUNNING
        
        await training_session.pause()
        assert training_session.status == SessionStatus.PAUSED
        
        await training_session.resume()
        assert training_session.status == SessionStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stop_session(self, training_session):
        """Test session stop functionality."""
        # Add mock active episodes - use actual asyncio Tasks
        async def dummy_episode():
            await asyncio.sleep(0.1)
        
        mock_task1 = asyncio.create_task(dummy_episode())
        mock_task2 = asyncio.create_task(dummy_episode())
        training_session.active_episodes = {
            "episode_1": mock_task1,
            "episode_2": mock_task2
        }
        
        await training_session.stop()
        
        assert training_session.status == SessionStatus.COMPLETED
        assert training_session.end_time is not None
        assert mock_task1.cancelled() or mock_task1.done()
        assert mock_task2.cancelled() or mock_task2.done()

    def test_episode_result_creation(self):
        """Test EpisodeResult data structure."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        result = EpisodeResult(
            episode_id="test_episode",
            start_time=start_time,
            end_time=end_time,
            duration=10.5,
            total_reward=25.0,
            episode_length=150,
            game_result="win",
            final_state={"health": 100, "score": 200}
        )
        
        assert result.episode_id == "test_episode"
        assert result.duration == 10.5
        assert result.total_reward == 25.0
        assert result.game_result == "win"
        assert result.error is None

    def test_session_metrics_initialization(self):
        """Test SessionMetrics initialization."""
        metrics = SessionMetrics()
        
        assert metrics.episodes_completed == 0
        assert metrics.episodes_failed == 0
        assert metrics.total_reward == 0.0
        assert metrics.average_reward == 0.0
        assert metrics.best_reward == float('-inf')
        assert metrics.worst_reward == float('inf')
        assert metrics.success_rate == 0.0

    @pytest.mark.asyncio
    async def test_episode_completion_handling(self, training_session):
        """Test episode completion handling."""
        # Create a mock episode result
        result = EpisodeResult(
            episode_id="test_episode",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration=10.0,
            total_reward=15.0,
            episode_length=100,
            game_result="win",
            final_state={}
        )
        
        # Mock episode completion handler
        handler = AsyncMock()
        training_session.register_episode_handler(handler)
        
        await training_session._handle_episode_completion(result)
        
        # Check metrics were updated
        assert training_session.metrics.episodes_completed == 1
        assert training_session.metrics.total_reward == 15.0
        assert training_session.metrics.average_reward == 15.0
        assert training_session.metrics.best_reward == 15.0
        assert len(training_session.completed_episodes) == 1
        
        # Check handler was called
        handler.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_episode_failure_handling(self, training_session):
        """Test episode failure handling."""
        await training_session._handle_episode_failure("failed_episode", "Test error")
        
        assert training_session.metrics.episodes_failed == 1

    @pytest.mark.asyncio
    async def test_get_session_info(self, training_session, training_config):
        """Test session info retrieval."""
        training_session.status = SessionStatus.RUNNING
        training_session.room_code = "TR123456"
        training_session.room_id = "room_tr123456"
        training_session.start_time = datetime.now()
        
        info = await training_session.get_session_info()
        
        assert info["sessionId"] == "test_session"
        assert info["status"] == "running"
        assert info["config"]["speedMultiplier"] == 10.0
        assert info["config"]["headlessMode"] is True
        assert info["roomInfo"]["roomCode"] == "TR123456"
        assert info["metrics"]["episodesCompleted"] == 0

    def test_training_mode_enum(self):
        """Test TrainingMode enum values."""
        assert TrainingMode.REALTIME.value == "realtime"
        assert TrainingMode.TRAINING.value == "training"
        assert TrainingMode.HEADLESS.value == "headless"
        assert TrainingMode.BATCH.value == "batch"

    def test_session_status_enum(self):
        """Test SessionStatus enum values."""
        assert SessionStatus.INITIALIZING.value == "initializing"
        assert SessionStatus.RUNNING.value == "running"
        assert SessionStatus.PAUSED.value == "paused"
        assert SessionStatus.COMPLETED.value == "completed"
        assert SessionStatus.FAILED.value == "failed"
        assert SessionStatus.CANCELLED.value == "cancelled"

    @pytest.mark.asyncio
    async def test_simulate_episode_placeholder(self, training_session):
        """Test the placeholder episode simulation."""
        mock_game_client = MagicMock()
        
        result = await training_session._simulate_episode(mock_game_client, "test_episode")
        
        assert result["reward"] == 10.0
        assert result["length"] == 100
        assert result["result"] == "win"
        assert "final_state" in result

    @pytest.mark.asyncio
    async def test_metrics_update_handlers(self, training_session):
        """Test metrics update handler registration and calling."""
        handler = AsyncMock()
        training_session.register_metrics_handler(handler)
        
        # Simulate metrics update
        for metrics_handler in training_session.metrics_update_handlers:
            await metrics_handler(training_session.metrics)
        
        handler.assert_called_once_with(training_session.metrics)

    @pytest.mark.asyncio
    async def test_cleanup_with_auto_cleanup_disabled(self, training_session):
        """Test cleanup when auto_cleanup is disabled."""
        training_session.config.auto_cleanup = False
        training_session.room_id = "test_room"
        
        with patch.object(training_session, '_cleanup_training_room', new_callable=AsyncMock) as mock_cleanup:
            await training_session._cleanup()
            mock_cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_with_auto_cleanup_enabled(self, training_session):
        """Test cleanup when auto_cleanup is enabled."""
        training_session.config.auto_cleanup = True
        training_session.room_id = "test_room"
        
        with patch.object(training_session, '_cleanup_training_room', new_callable=AsyncMock) as mock_cleanup:
            await training_session._cleanup()
            mock_cleanup.assert_called_once()

    def test_session_id_generation(self):
        """Test automatic session ID generation."""
        session = TrainingSession()
        assert session.session_id is not None
        assert len(session.session_id) > 0
        
        # Test custom session ID
        custom_session = TrainingSession(session_id="custom_id")
        assert custom_session.session_id == "custom_id"