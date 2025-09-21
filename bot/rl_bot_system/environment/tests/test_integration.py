"""
Integration tests for the complete environment system.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock

from bot.rl_bot_system.environment.game_environment import GameEnvironment, TrainingMode
from bot.rl_bot_system.environment.state_processors import StateProcessorFactory, StateRepresentationType
from bot.rl_bot_system.environment.action_spaces import ActionSpaceFactory, ActionSpaceType
from bot.rl_bot_system.environment.reward_functions import RewardFunctionFactory, RewardType


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


class TestEnvironmentIntegration:
    """Integration tests for the complete environment system."""
    
    @pytest.mark.asyncio
    async def test_environment_with_different_configurations(self, mock_client):
        """Test environment with different processor/action/reward combinations."""
        # Test different state representations
        state_types = [
            StateRepresentationType.RAW_COORDINATE,
            StateRepresentationType.GRID_BASED,
            StateRepresentationType.FEATURE_VECTOR
        ]
        
        # Test different action spaces
        action_types = [
            ActionSpaceType.DISCRETE,
            ActionSpaceType.CONTINUOUS,
            ActionSpaceType.HYBRID,
            ActionSpaceType.MULTI_DISCRETE
        ]
        
        # Test different reward functions
        reward_types = [
            RewardType.SPARSE,
            RewardType.DENSE,
            RewardType.SHAPED,
            RewardType.MULTI_OBJECTIVE
        ]
        
        # Test a few combinations
        test_combinations = [
            (StateRepresentationType.RAW_COORDINATE, ActionSpaceType.DISCRETE, RewardType.SPARSE),
            (StateRepresentationType.GRID_BASED, ActionSpaceType.CONTINUOUS, RewardType.DENSE),
            (StateRepresentationType.FEATURE_VECTOR, ActionSpaceType.HYBRID, RewardType.SHAPED)
        ]
        
        for state_type, action_type, reward_type in test_combinations:
            # Create components
            state_processor = StateProcessorFactory.create_processor(state_type)
            action_space = ActionSpaceFactory.create_action_space(action_type)
            reward_function = RewardFunctionFactory.create_reward_function(reward_type)
            
            # Create environment
            env = GameEnvironment(
                game_client=mock_client,
                state_processor=state_processor,
                action_space_config=action_space,
                reward_function=reward_function
            )
            
            # Test basic functionality
            obs, info = env.reset()
            assert isinstance(obs, np.ndarray)
            assert isinstance(info, dict)
            
            # Test step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            
            # Verify observation fits space
            assert env.observation_space.contains(obs)
            
            # Verify action was valid
            assert env.action_space.contains(action)
    
    @pytest.mark.asyncio
    async def test_environment_episode_flow(self, mock_client):
        """Test complete episode flow."""
        # Create environment with default components
        env = GameEnvironment(
            game_client=mock_client,
            max_episode_steps=10
        )
        
        # Reset environment
        obs, info = env.reset()
        assert env.current_step == 0
        assert not env.episode_done
        
        # Run episode
        total_reward = 0
        for step in range(15):  # More than max_episode_steps
            if env.episode_done:
                break
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            assert env.current_step == step + 1
            
            if step >= env.max_episode_steps - 1:
                assert truncated  # Should be truncated at max steps
                assert env.episode_done
        
        # Verify episode statistics
        stats = env.get_episode_stats()
        assert stats['total_reward'] == total_reward
        assert stats['steps'] == env.current_step
    
    def test_training_mode_switching(self, mock_client):
        """Test switching between training modes."""
        env = GameEnvironment(
            game_client=mock_client,
            training_mode=TrainingMode.TRAINING
        )
        
        assert env.training_mode == TrainingMode.TRAINING
        
        # Switch to evaluation mode
        env.set_training_mode(TrainingMode.EVALUATION)
        assert env.training_mode == TrainingMode.EVALUATION
    
    @pytest.mark.asyncio
    async def test_reward_function_integration(self, mock_client):
        """Test reward function integration with environment."""
        # Create environment with multi-objective reward
        reward_config = {
            'reward_functions': [
                {'type': 'sparse', 'weight': 0.6},
                {'type': 'dense', 'weight': 0.4}
            ]
        }
        reward_function = RewardFunctionFactory.create_reward_function(
            RewardType.MULTI_OBJECTIVE, reward_config
        )
        
        env = GameEnvironment(
            game_client=mock_client,
            reward_function=reward_function
        )
        
        # Run a few steps
        env.reset()
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Verify reward is calculated
            assert isinstance(reward, float)
            assert np.isfinite(reward)
        
        # Check reward statistics
        reward_stats = reward_function.get_episode_statistics()
        assert 'total_reward' in reward_stats
        assert 'mean_reward' in reward_stats
    
    def test_state_processor_integration(self, mock_client):
        """Test state processor integration with environment."""
        # Test grid-based processor
        grid_config = {
            'grid_width': 16,
            'grid_height': 12,
            'multi_channel': True
        }
        state_processor = StateProcessorFactory.create_processor(
            StateRepresentationType.GRID_BASED, grid_config
        )
        
        env = GameEnvironment(
            game_client=mock_client,
            state_processor=state_processor
        )
        
        obs, info = env.reset()
        
        # Should have grid shape
        expected_shape = (len(state_processor.channels), 12, 16)
        assert obs.shape == expected_shape
        
        # Should be within valid range
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    
    @pytest.mark.asyncio
    async def test_action_space_integration(self, mock_client):
        """Test action space integration with environment."""
        # Test hybrid action space
        hybrid_config = {
            'discrete_actions': ['no_action', 'move_left', 'move_right'],
            'continuous_dims': 2
        }
        action_space = ActionSpaceFactory.create_action_space(
            ActionSpaceType.HYBRID, hybrid_config
        )
        
        env = GameEnvironment(
            game_client=mock_client,
            action_space_config=action_space
        )
        
        obs, info = env.reset()
        
        # Test hybrid action
        action = {
            'discrete': 1,  # move_left
            'continuous': np.array([0.5, -0.3])
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should execute without error
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_client):
        """Test error handling in environment."""
        env = GameEnvironment(game_client=mock_client)
        
        # Reset and test invalid action
        env.reset()
        
        # Invalid action should be handled gracefully
        invalid_action = -1  # Invalid for discrete action space
        obs, reward, terminated, truncated, info = env.step(invalid_action)
        
        # Should still return valid values
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
    
    @pytest.mark.asyncio
    async def test_environment_cleanup(self, mock_client):
        """Test environment cleanup."""
        env = GameEnvironment(game_client=mock_client)
        
        # Close environment
        env.close()
        # Give a moment for the async task to be scheduled
        await asyncio.sleep(0.01)
        
        # Should not raise errors