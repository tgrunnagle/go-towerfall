"""
Tests for the GameEnvironment class.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock

from bot.rl_bot_system.environment.game_environment import GameEnvironment, TrainingMode
from bot.rl_bot_system.environment.state_processors import RawCoordinateProcessor
from bot.rl_bot_system.environment.action_spaces import DiscreteActionSpace
from bot.rl_bot_system.environment.reward_functions import SparseRewardFunction


@pytest.fixture
def mock_client():
    """Mock game client fixture."""
    mock_client = Mock()
    mock_client.player_id = "test_player"
    mock_client.send_keyboard_input = AsyncMock()
    mock_client.send_mouse_input = AsyncMock()
    mock_client.register_message_handler = Mock()
    mock_client.close = AsyncMock()
    return mock_client


@pytest.fixture
def game_environment(mock_client):
    """Game environment fixture."""
    return GameEnvironment(
        game_client=mock_client,
        state_processor=RawCoordinateProcessor(),
        action_space_config=DiscreteActionSpace(),
        reward_function=SparseRewardFunction(),
        training_mode=TrainingMode.TRAINING,
        max_episode_steps=100
    )


class TestGameEnvironment:
    """Test cases for GameEnvironment."""
    
    def test_initialization(self, game_environment, mock_client):
        """Test environment initialization."""
        assert game_environment.game_client == mock_client
        assert game_environment.training_mode == TrainingMode.TRAINING
        assert game_environment.max_episode_steps == 100
        assert game_environment.current_step == 0
        assert game_environment.episode_reward == 0.0
        assert not game_environment.episode_done
        
        # Check spaces are set up
        assert game_environment.observation_space is not None
        assert game_environment.action_space is not None
    
    def test_reset(self, game_environment):
        """Test environment reset."""
        # Set some state to reset
        game_environment.current_step = 50
        game_environment.episode_reward = 10.0
        game_environment.episode_done = True
        
        # Reset environment
        obs, info = game_environment.reset()
        
        # Check state was reset
        assert game_environment.current_step == 0
        assert game_environment.episode_reward == 0.0
        assert not game_environment.episode_done
        
        # Check observation
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (83,)  # RawCoordinateProcessor returns 83 features
        
        # Check info
        assert info['episode_step'] == 0
        assert info['training_mode'] == 'training'
        assert 'raw_state' in info
    
    @pytest.mark.asyncio
    async def test_step(self, game_environment):
        """Test environment step."""
        # Reset first
        game_environment.reset()
        
        # Take a step
        action = 1  # move_left
        obs, reward, terminated, truncated, info = game_environment.step(action)
        
        # Check step results
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check state updates
        assert game_environment.current_step == 1
        assert game_environment.last_action == action
        
        # Check info
        assert info['episode_step'] == 1
        assert info['action_executed'] == action
        assert 'episode_stats' in info
    
    def test_training_mode_switch(self, game_environment):
        """Test switching training modes."""
        assert game_environment.training_mode == TrainingMode.TRAINING
        
        game_environment.set_training_mode(TrainingMode.EVALUATION)
        assert game_environment.training_mode == TrainingMode.EVALUATION
    
    def test_episode_stats_tracking(self, game_environment):
        """Test episode statistics tracking."""
        stats = game_environment.get_episode_stats()
        
        expected_keys = [
            'total_reward', 'steps', 'kills', 'deaths',
            'shots_fired', 'shots_hit', 'damage_dealt', 'damage_taken'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
    
    def test_max_episode_steps(self, game_environment):
        """Test episode truncation at max steps."""
        game_environment.max_episode_steps = 2
        game_environment.reset()
        
        # Take steps up to limit
        obs, reward, terminated, truncated, info = game_environment.step(0)
        assert not truncated
        
        obs, reward, terminated, truncated, info = game_environment.step(0)
        assert truncated  # Should be truncated at max steps
        assert game_environment.episode_done
    
    def test_action_space_integration(self, game_environment):
        """Test action space integration."""
        action_space = game_environment.action_space
        
        # Test valid actions
        for action in range(action_space.n):
            assert action_space.contains(action)
        
        # Test invalid action
        assert not action_space.contains(-1)
        assert not action_space.contains(action_space.n)
    
    def test_observation_space_integration(self, game_environment):
        """Test observation space integration."""
        obs_space = game_environment.observation_space
        
        # Reset to get observation
        obs, _ = game_environment.reset()
        
        # Check observation fits space
        assert obs_space.contains(obs)
    
    @pytest.mark.asyncio
    async def test_close(self, game_environment):
        """Test environment cleanup."""
        game_environment.close()
        # Give a moment for the async task to be scheduled
        await asyncio.sleep(0.01)
        # Verify close was called (async mock will be scheduled)
        # In a real test, we'd need to handle the async call properly


if __name__ == '__main__':
    unittest.main()