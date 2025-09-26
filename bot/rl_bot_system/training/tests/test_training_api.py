"""
Unit tests for TrainingAPI class.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rl_bot_system.training.training_api import TrainingAPI
from rl_bot_system.training.session_manager import SessionManager
from rl_bot_system.training.batch_episode_manager import BatchEpisodeManager
from rl_bot_system.training.training_session import TrainingConfig, TrainingMode


class TestTrainingAPI:
    """Test cases for TrainingAPI class."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        return AsyncMock(spec=SessionManager)

    @pytest.fixture
    def mock_batch_manager(self):
        """Create a mock batch manager."""
        return AsyncMock(spec=BatchEpisodeManager)

    @pytest.fixture
    def training_api(self, mock_session_manager, mock_batch_manager):
        """Create a test training API."""
        return TrainingAPI(
            session_manager=mock_session_manager,
            batch_manager=mock_batch_manager,
            game_server_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws"
        )

    @pytest.mark.asyncio
    async def test_training_api_initialization(self, training_api, mock_session_manager, mock_batch_manager):
        """Test training API initialization."""
        assert training_api.session_manager == mock_session_manager
        assert training_api.batch_manager == mock_batch_manager

    @pytest.mark.asyncio
    async def test_start_and_stop_api(self, training_api, mock_session_manager, mock_batch_manager):
        """Test starting and stopping the training API."""
        await training_api.start()
        
        mock_session_manager.start.assert_called_once()
        mock_batch_manager.start.assert_called_once()
        
        await training_api.stop()
        
        mock_session_manager.stop.assert_called_once()
        mock_batch_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_training_session_success(self, training_api, mock_session_manager):
        """Test successful training session creation."""
        mock_session_manager.create_session.return_value = "test_session_123"
        
        request_data = {
            "config": {
                "speedMultiplier": 10.0,
                "headlessMode": True,
                "maxEpisodes": 500,
                "parallelEpisodes": 2,
                "trainingMode": "training",
                "spectatorEnabled": False
            },
            "priority": 1,
            "sessionId": "custom_session"
        }
        
        result = await training_api.create_training_session(request_data)
        
        assert result["success"] is True
        assert result["sessionId"] == "test_session_123"
        assert "created and queued" in result["message"]
        
        # Verify session manager was called with correct config
        mock_session_manager.create_session.assert_called_once()
        call_args = mock_session_manager.create_session.call_args
        assert call_args.kwargs["priority"] == 1
        assert call_args.kwargs["session_id"] == "custom_session"
        
        config = call_args.kwargs["config"]
        assert config.speed_multiplier == 10.0
        assert config.headless_mode is True
        assert config.max_episodes == 500
        assert config.parallel_episodes == 2
        assert config.training_mode == TrainingMode.TRAINING

    @pytest.mark.asyncio
    async def test_create_training_session_with_defaults(self, training_api, mock_session_manager):
        """Test training session creation with default values."""
        mock_session_manager.create_session.return_value = "default_session"
        
        request_data = {}  # Empty request, should use defaults
        
        result = await training_api.create_training_session(request_data)
        
        assert result["success"] is True
        
        # Verify defaults were used
        call_args = mock_session_manager.create_session.call_args
        config = call_args.kwargs["config"]
        assert config.speed_multiplier == 1.0
        assert config.headless_mode is False
        assert config.max_episodes == 1000
        assert config.parallel_episodes == 1
        assert config.training_mode == TrainingMode.REALTIME

    @pytest.mark.asyncio
    async def test_create_training_session_failure(self, training_api, mock_session_manager):
        """Test training session creation failure."""
        mock_session_manager.create_session.side_effect = Exception("Creation failed")
        
        request_data = {"config": {"speedMultiplier": 5.0}}
        
        result = await training_api.create_training_session(request_data)
        
        assert result["success"] is False
        assert result["error"] == "Creation failed"
        assert "Failed to create" in result["message"]

    @pytest.mark.asyncio
    async def test_get_session_info_success(self, training_api, mock_session_manager):
        """Test successful session info retrieval."""
        mock_session = AsyncMock()
        mock_session.get_session_info.return_value = {
            "sessionId": "test_session",
            "status": "running",
            "metrics": {"episodesCompleted": 10}
        }
        mock_session_manager.get_session.return_value = mock_session
        
        result = await training_api.get_session_info("test_session")
        
        assert result["success"] is True
        assert result["session"]["sessionId"] == "test_session"
        assert result["session"]["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_session_info_not_found(self, training_api, mock_session_manager):
        """Test session info retrieval for non-existent session."""
        mock_session_manager.get_session.return_value = None
        
        result = await training_api.get_session_info("nonexistent")
        
        assert result["success"] is False
        assert result["error"] == "Session not found"

    @pytest.mark.asyncio
    async def test_list_all_sessions(self, training_api, mock_session_manager):
        """Test listing all sessions."""
        mock_sessions_info = {
            "activeSessions": {"session1": {"status": "running"}},
            "completedSessions": ["session2"],
            "queuedSessions": 2
        }
        mock_global_metrics = {
            "totalSessions": 1,
            "totalEpisodes": 50
        }
        
        mock_session_manager.get_all_sessions_info.return_value = mock_sessions_info
        mock_session_manager.get_global_metrics.return_value = mock_global_metrics
        
        result = await training_api.list_all_sessions()
        
        assert result["success"] is True
        assert result["sessions"] == mock_sessions_info
        assert result["globalMetrics"] == mock_global_metrics

    @pytest.mark.asyncio
    async def test_pause_session_success(self, training_api, mock_session_manager):
        """Test successful session pause."""
        mock_session_manager.pause_session.return_value = True
        
        result = await training_api.pause_session("test_session")
        
        assert result["success"] is True
        assert "paused" in result["message"]

    @pytest.mark.asyncio
    async def test_pause_session_not_found(self, training_api, mock_session_manager):
        """Test pausing non-existent session."""
        mock_session_manager.pause_session.return_value = False
        
        result = await training_api.pause_session("nonexistent")
        
        assert result["success"] is False
        assert result["error"] == "Session not found"

    @pytest.mark.asyncio
    async def test_resume_session_success(self, training_api, mock_session_manager):
        """Test successful session resume."""
        mock_session_manager.resume_session.return_value = True
        
        result = await training_api.resume_session("test_session")
        
        assert result["success"] is True
        assert "resumed" in result["message"]

    @pytest.mark.asyncio
    async def test_stop_session_success(self, training_api, mock_session_manager):
        """Test successful session stop."""
        mock_session_manager.stop_session.return_value = True
        
        result = await training_api.stop_session("test_session")
        
        assert result["success"] is True
        assert "stopped" in result["message"]

    @pytest.mark.asyncio
    async def test_request_training_room(self, training_api):
        """Test training room request."""
        request_data = {
            "speedMultiplier": 15.0,
            "headlessMode": True,
            "maxPlayers": 6,
            "password": "secret",
            "spectatorEnabled": True
        }
        
        result = await training_api.request_training_room(request_data)
        
        assert result["success"] is True
        assert "roomCode" in result
        assert "roomId" in result
        assert result["config"]["speedMultiplier"] == 15.0
        assert result["config"]["headlessMode"] is True

    @pytest.mark.asyncio
    async def test_request_training_room_with_defaults(self, training_api):
        """Test training room request with default values."""
        request_data = {}
        
        result = await training_api.request_training_room(request_data)
        
        assert result["success"] is True
        assert result["config"]["speedMultiplier"] == 1.0
        assert result["config"]["headlessMode"] is False
        assert result["config"]["maxPlayers"] == 8

    @pytest.mark.asyncio
    async def test_get_training_room_info(self, training_api):
        """Test getting training room information."""
        result = await training_api.get_training_room_info("TR123456")
        
        assert result["success"] is True
        assert result["roomCode"] == "TR123456"
        assert "status" in result
        assert "playerCount" in result

    @pytest.mark.asyncio
    async def test_submit_episode_batch_success(self, training_api, mock_batch_manager):
        """Test successful episode batch submission."""
        request_data = {
            "batchId": "batch_001",
            "episodeIds": ["ep_001", "ep_002", "ep_003"],
            "roomCode": "TR123456",
            "roomPassword": "secret",
            "maxParallel": 2,
            "timeoutSeconds": 120
        }
        
        result = await training_api.submit_episode_batch(request_data)
        
        assert result["success"] is True
        assert result["batchId"] == "batch_001"
        assert result["episodeCount"] == 3
        
        mock_batch_manager.submit_batch.assert_called_once_with(
            batch_id="batch_001",
            episode_ids=["ep_001", "ep_002", "ep_003"],
            room_code="TR123456",
            room_password="secret",
            max_parallel=2,
            timeout_seconds=120
        )

    @pytest.mark.asyncio
    async def test_submit_episode_batch_failure(self, training_api, mock_batch_manager):
        """Test episode batch submission failure."""
        mock_batch_manager.submit_batch.side_effect = Exception("Batch submission failed")
        
        request_data = {
            "batchId": "batch_001",
            "episodeIds": ["ep_001"],
            "roomCode": "TR123456"
        }
        
        result = await training_api.submit_episode_batch(request_data)
        
        assert result["success"] is False
        assert result["error"] == "Batch submission failed"

    @pytest.mark.asyncio
    async def test_get_batch_status_success(self, training_api, mock_batch_manager):
        """Test successful batch status retrieval."""
        mock_status = {
            "batchId": "batch_001",
            "totalEpisodes": 3,
            "completed": 1,
            "running": 1,
            "queued": 1
        }
        mock_batch_manager.get_batch_status.return_value = mock_status
        
        result = await training_api.get_batch_status("batch_001")
        
        assert result["success"] is True
        assert result["batch"] == mock_status

    @pytest.mark.asyncio
    async def test_get_batch_status_not_found(self, training_api, mock_batch_manager):
        """Test batch status retrieval for non-existent batch."""
        mock_batch_manager.get_batch_status.return_value = None
        
        result = await training_api.get_batch_status("nonexistent")
        
        assert result["success"] is False
        assert result["error"] == "Batch not found"

    @pytest.mark.asyncio
    async def test_cancel_batch_success(self, training_api, mock_batch_manager):
        """Test successful batch cancellation."""
        mock_batch_manager.cancel_batch.return_value = True
        
        result = await training_api.cancel_batch("batch_001")
        
        assert result["success"] is True
        assert "cancelled" in result["message"]

    @pytest.mark.asyncio
    async def test_cancel_batch_not_found(self, training_api, mock_batch_manager):
        """Test cancelling non-existent batch."""
        mock_batch_manager.cancel_batch.return_value = False
        
        result = await training_api.cancel_batch("nonexistent")
        
        assert result["success"] is False
        assert result["error"] == "Batch not found"

    @pytest.mark.asyncio
    async def test_get_system_status(self, training_api, mock_session_manager):
        """Test system status retrieval."""
        mock_sessions_info = {
            "activeSessions": {"session1": {}},
            "completedSessions": ["session2", "session3"],
            "queuedSessions": 1,
            "resourceStatus": {"status": "available"}
        }
        mock_global_metrics = {"totalSessions": 1}
        
        mock_session_manager.get_all_sessions_info.return_value = mock_sessions_info
        mock_session_manager.get_global_metrics.return_value = mock_global_metrics
        
        result = await training_api.get_system_status()
        
        assert result["success"] is True
        assert result["status"] == "running"
        assert result["sessions"]["active"] == 1
        assert result["sessions"]["completed"] == 2
        assert result["sessions"]["queued"] == 1

    @pytest.mark.asyncio
    async def test_get_training_metrics_for_session(self, training_api, mock_session_manager):
        """Test getting training metrics for specific session."""
        mock_session = AsyncMock()
        mock_session.get_session_info.return_value = {
            "metrics": {"episodesCompleted": 10, "averageReward": 15.5}
        }
        mock_session_manager.get_session.return_value = mock_session
        
        result = await training_api.get_training_metrics("test_session")
        
        assert result["success"] is True
        assert result["sessionId"] == "test_session"
        assert result["metrics"]["episodesCompleted"] == 10

    @pytest.mark.asyncio
    async def test_get_training_metrics_global(self, training_api, mock_session_manager):
        """Test getting global training metrics."""
        mock_global_metrics = {"totalEpisodes": 100, "totalRewards": 1500.0}
        mock_session_manager.get_global_metrics.return_value = mock_global_metrics
        
        result = await training_api.get_training_metrics()
        
        assert result["success"] is True
        assert result["globalMetrics"] == mock_global_metrics

    @pytest.mark.asyncio
    async def test_get_default_config(self, training_api):
        """Test getting default configuration."""
        result = await training_api.get_default_config()
        
        assert result["success"] is True
        assert "config" in result
        assert result["config"]["speedMultiplier"] == 1.0
        assert result["config"]["headlessMode"] is False
        assert result["config"]["maxEpisodes"] == 1000

    @pytest.mark.asyncio
    async def test_validate_config_valid(self, training_api):
        """Test configuration validation with valid config."""
        config_data = {
            "speedMultiplier": 10.0,
            "maxEpisodes": 500,
            "parallelEpisodes": 4,
            "trainingMode": "training"
        }
        
        result = await training_api.validate_config(config_data)
        
        assert result["success"] is True
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_config_invalid(self, training_api):
        """Test configuration validation with invalid config."""
        config_data = {
            "speedMultiplier": -5.0,  # Invalid: negative
            "maxEpisodes": -100,      # Invalid: negative
            "parallelEpisodes": 20,   # Invalid: too high
            "trainingMode": "invalid" # Invalid: not a valid mode
        }
        
        result = await training_api.validate_config(config_data)
        
        assert result["success"] is True
        assert result["valid"] is False
        assert len(result["errors"]) == 4  # Should have 4 validation errors

    @pytest.mark.asyncio
    async def test_validate_config_edge_cases(self, training_api):
        """Test configuration validation edge cases."""
        # Test boundary values
        config_data = {
            "speedMultiplier": 100.0,  # Max allowed
            "maxEpisodes": 1,          # Min allowed
            "parallelEpisodes": 16,    # Max allowed
            "trainingMode": "headless" # Valid mode
        }
        
        result = await training_api.validate_config(config_data)
        
        assert result["success"] is True
        assert result["valid"] is True
        
        # Test just over boundaries
        config_data = {
            "speedMultiplier": 101.0,  # Over max
            "parallelEpisodes": 17     # Over max
        }
        
        result = await training_api.validate_config(config_data)
        
        assert result["success"] is True
        assert result["valid"] is False
        assert len(result["errors"]) == 2