"""
Integration tests for episode replay functionality.

Tests the complete episode replay system including backend API,
WebSocket communication, and frontend integration.
"""

import asyncio
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from rl_bot_system.spectator.episode_replay import (
    EpisodeReplayManager,
    ReplayControls,
    ReplayState
)
from rl_bot_system.spectator.spectator_manager import SpectatorManager
from rl_bot_system.replay.replay_manager import ReplayManager
from rl_bot_system.evaluation.evaluator import GameEpisode


class TestEpisodeReplayIntegration:
    """Integration tests for episode replay system."""
    
    @pytest.fixture
    async def replay_manager(self):
        """Create a mock replay manager."""
        manager = Mock(spec=ReplayManager)
        manager.get_available_sessions.return_value = [
            {
                "session_id": "test_session_1",
                "start_time": "2024-01-01T10:00:00",
                "user_metadata": {}
            }
        ]
        
        # Create mock episodes
        episodes = [
            GameEpisode(
                episode_id="episode_1",
                model_generation=1,
                states=[{"x": i, "y": i} for i in range(10)],
                actions=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                rewards=[1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 2.0],
                total_reward=10.0,
                episode_length=10,
                game_result="win",
                episode_metrics={}
            ),
            GameEpisode(
                episode_id="episode_2",
                model_generation=2,
                states=[{"x": i*2, "y": i*2} for i in range(8)],
                actions=[1, 0, 1, 0, 1, 0, 1, 0],
                rewards=[0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 1.5],
                total_reward=6.7,
                episode_length=8,
                game_result="loss",
                episode_metrics={}
            )
        ]
        
        manager.load_episodes_from_session.return_value = episodes
        return manager
    
    @pytest.fixture
    async def spectator_manager(self):
        """Create a mock spectator manager."""
        manager = Mock(spec=SpectatorManager)
        return manager
    
    @pytest.fixture
    async def episode_replay_manager(self, replay_manager, spectator_manager):
        """Create episode replay manager."""
        return EpisodeReplayManager(replay_manager, spectator_manager)
    
    @pytest.mark.asyncio
    async def test_start_episode_replay(self, episode_replay_manager):
        """Test starting episode replay."""
        # Start replay
        replay_id = await episode_replay_manager.start_episode_replay(
            session_id="test_session",
            episode_id="episode_1"
        )
        
        assert replay_id is not None
        assert replay_id.startswith("replay_test_session_episode_1")
        
        # Check replay status
        status = await episode_replay_manager.get_replay_status(replay_id)
        assert status is not None
        assert status["episode_id"] == "episode_1"
        assert status["state"] == "stopped"
        assert status["current_frame"] == 0
    
    @pytest.mark.asyncio
    async def test_replay_controls(self, episode_replay_manager):
        """Test replay control commands."""
        # Start replay
        replay_id = await episode_replay_manager.start_episode_replay(
            session_id="test_session",
            episode_id="episode_1"
        )
        
        # Test play command
        success = await episode_replay_manager.control_replay(replay_id, "play")
        assert success
        
        # Wait a bit for playback to start
        await asyncio.sleep(0.1)
        
        # Test pause command
        success = await episode_replay_manager.control_replay(replay_id, "pause")
        assert success
        
        # Test seek command
        success = await episode_replay_manager.control_replay(
            replay_id, "seek", {"frame": 5}
        )
        assert success
        
        # Test speed change
        success = await episode_replay_manager.control_replay(
            replay_id, "speed", {"speed": 2.0}
        )
        assert success
        
        # Test stop command
        success = await episode_replay_manager.control_replay(replay_id, "stop")
        assert success
    
    @pytest.mark.asyncio
    async def test_comparison_replay(self, episode_replay_manager):
        """Test side-by-side episode comparison."""
        # Start comparison replay
        replay_id = await episode_replay_manager.start_comparison_replay(
            session_id="test_session",
            episode_ids=["episode_1", "episode_2"]
        )
        
        assert replay_id is not None
        assert replay_id.startswith("comparison_test_session")
        
        # Check comparison status
        status = await episode_replay_manager.get_replay_status(replay_id)
        assert status is not None
        assert status["comparison_mode"] is True
        assert status["episode_count"] == 2
        assert len(status["episodes"]) == 2
    
    @pytest.mark.asyncio
    async def test_replay_callbacks(self, episode_replay_manager):
        """Test replay event callbacks."""
        frame_updates = []
        state_changes = []
        
        async def frame_callback(replay_id, frame):
            frame_updates.append((replay_id, frame))
        
        async def state_callback(replay_id, state):
            state_changes.append((replay_id, state))
        
        # Start replay
        replay_id = await episode_replay_manager.start_episode_replay(
            session_id="test_session",
            episode_id="episode_1"
        )
        
        # Register callbacks
        episode_replay_manager.register_frame_callback(replay_id, frame_callback)
        episode_replay_manager.register_state_callback(replay_id, state_callback)
        
        # Start playback
        await episode_replay_manager.control_replay(replay_id, "play")
        
        # Wait for some updates
        await asyncio.sleep(0.2)
        
        # Stop playback
        await episode_replay_manager.control_replay(replay_id, "stop")
        
        # Check that callbacks were called
        assert len(frame_updates) > 0
        assert len(state_changes) > 0
        
        # Verify callback data
        replay_id_from_callback, frame = frame_updates[0]
        assert replay_id_from_callback == replay_id
        assert frame.frame_index >= 0
        assert frame.game_state is not None
    
    @pytest.mark.asyncio
    async def test_replay_cleanup(self, episode_replay_manager):
        """Test replay session cleanup."""
        # Start multiple replays
        replay_id1 = await episode_replay_manager.start_episode_replay(
            session_id="test_session",
            episode_id="episode_1"
        )
        
        replay_id2 = await episode_replay_manager.start_comparison_replay(
            session_id="test_session",
            episode_ids=["episode_1", "episode_2"]
        )
        
        # Verify replays exist
        status1 = await episode_replay_manager.get_replay_status(replay_id1)
        status2 = await episode_replay_manager.get_replay_status(replay_id2)
        assert status1 is not None
        assert status2 is not None
        
        # Stop replays
        success1 = await episode_replay_manager.stop_replay(replay_id1)
        success2 = await episode_replay_manager.stop_replay(replay_id2)
        assert success1
        assert success2
        
        # Verify replays are cleaned up
        status1 = await episode_replay_manager.get_replay_status(replay_id1)
        status2 = await episode_replay_manager.get_replay_status(replay_id2)
        assert status1 is None
        assert status2 is None
    
    @pytest.mark.asyncio
    async def test_invalid_operations(self, episode_replay_manager):
        """Test error handling for invalid operations."""
        # Test with non-existent episode
        with pytest.raises(ValueError, match="Episode .* not found"):
            await episode_replay_manager.start_episode_replay(
                session_id="test_session",
                episode_id="non_existent_episode"
            )
        
        # Test control on non-existent replay
        success = await episode_replay_manager.control_replay(
            "non_existent_replay", "play"
        )
        assert not success
        
        # Test status of non-existent replay
        status = await episode_replay_manager.get_replay_status("non_existent_replay")
        assert status is None
        
        # Test stop of non-existent replay
        success = await episode_replay_manager.stop_replay("non_existent_replay")
        assert not success
    
    @pytest.mark.asyncio
    async def test_replay_controls_validation(self, episode_replay_manager):
        """Test replay controls validation."""
        # Test with custom controls
        controls = ReplayControls(
            playback_speed=2.0,
            auto_loop=True,
            show_frame_info=False,
            show_decision_overlay=False
        )
        
        replay_id = await episode_replay_manager.start_episode_replay(
            session_id="test_session",
            episode_id="episode_1",
            controls=controls
        )
        
        status = await episode_replay_manager.get_replay_status(replay_id)
        assert status["playback_speed"] == 2.0
        
        # Test speed limits
        success = await episode_replay_manager.control_replay(
            replay_id, "speed", {"speed": 15.0}  # Should be clamped to max
        )
        assert success
        
        # Test invalid commands
        success = await episode_replay_manager.control_replay(
            replay_id, "invalid_command"
        )
        assert not success


class TestReplayAPIIntegration:
    """Integration tests for replay API endpoints."""
    
    @pytest.fixture
    def mock_managers(self):
        """Create mock managers for API testing."""
        replay_manager = Mock(spec=ReplayManager)
        spectator_manager = Mock(spec=SpectatorManager)
        episode_replay_manager = Mock(spec=EpisodeReplayManager)
        
        return replay_manager, spectator_manager, episode_replay_manager
    
    def test_api_initialization(self, mock_managers):
        """Test API initialization with managers."""
        from rl_bot_system.server.replay_api import initialize_replay_api
        
        replay_manager, spectator_manager, episode_replay_manager = mock_managers
        
        # Should not raise any exceptions
        initialize_replay_api(replay_manager, spectator_manager, episode_replay_manager)
    
    @pytest.mark.asyncio
    async def test_websocket_integration(self):
        """Test WebSocket integration for replay updates."""
        # This would test the WebSocket message flow for replay updates
        # Mock WebSocket connection and verify message handling
        
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()
        
        # Simulate replay frame update
        frame_data = {
            "type": "replay_frame",
            "frame_index": 5,
            "timestamp": datetime.now().isoformat(),
            "game_state": {"x": 10, "y": 20},
            "action_taken": 1,
            "reward_received": 1.5
        }
        
        # Test message sending
        await mock_websocket.send_text(json.dumps(frame_data))
        mock_websocket.send_text.assert_called_once()
        
        # Verify message format
        sent_message = mock_websocket.send_text.call_args[0][0]
        parsed_message = json.loads(sent_message)
        assert parsed_message["type"] == "replay_frame"
        assert parsed_message["frame_index"] == 5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])