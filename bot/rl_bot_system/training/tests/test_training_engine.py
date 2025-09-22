"""
Tests for the RL training engine.

This module tests the core training functionality including model initialization,
training loops, behavior cloning, and evaluation.
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from bot.rl_bot_system.training.training_engine import (
    TrainingEngine, TrainingConfig, TrainingMetrics, EvaluationResult
)
from bot.rl_bot_system.training.model_manager import ModelManager, RLModel
from bot.rl_bot_system.training.cohort_training import CohortTrainingSystem
from bot.rl_bot_system.models import DQNAgent, PPOAgent, A2CAgent
from bot.rl_bot_system.environment import GameEnvironment, TrainingMode


class MockGameEnvironment:
    """Mock game environment for testing."""
    
    def __init__(self):
        self.observation_space = Mock()
        self.observation_space.shape = (10,)  # 10-dimensional state
        self.action_space = Mock()
        self.action_space.n = 5  # 5 discrete actions
        self.action_space.sample = Mock(return_value=0)
        
        self.training_mode = TrainingMode.TRAINING
        self.reset_count = 0
        self.step_count = 0
        
    def reset(self):
        """Reset environment."""
        self.reset_count += 1
        state = np.random.random(10)
        info = {}
        return state, info
    
    def step(self, action):
        """Take a step in the environment."""
        self.step_count += 1
        next_state = np.random.random(10)
        reward = np.random.uniform(-1, 1)
        terminated = self.step_count % 20 == 0  # Episode ends every 20 steps
        truncated = False
        info = {}
        return next_state, reward, terminated, truncated, info
    
    def set_training_mode(self, mode):
        """Set training mode."""
        self.training_mode = mode


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = TrainingConfig(algorithm='DQN')
        
        assert config.algorithm == 'DQN'
        assert config.max_episodes == 1000
        assert config.max_steps_per_episode == 1000
        assert config.evaluation_frequency == 100
        assert config.use_behavior_cloning is True
        assert config.hyperparameters is not None
        
    def test_dqn_hyperparameters(self):
        """Test DQN default hyperparameters."""
        config = TrainingConfig(algorithm='DQN')
        
        assert 'learning_rate' in config.hyperparameters
        assert 'gamma' in config.hyperparameters
        assert 'epsilon' in config.hyperparameters
        assert config.hyperparameters['double_dqn'] is True
        
    def test_ppo_hyperparameters(self):
        """Test PPO default hyperparameters."""
        config = TrainingConfig(algorithm='PPO')
        
        assert 'learning_rate' in config.hyperparameters
        assert 'gamma' in config.hyperparameters
        assert 'clip_epsilon' in config.hyperparameters
        assert 'ppo_epochs' in config.hyperparameters
        
    def test_a2c_hyperparameters(self):
        """Test A2C default hyperparameters."""
        config = TrainingConfig(algorithm='A2C')
        
        assert 'learning_rate' in config.hyperparameters
        assert 'gamma' in config.hyperparameters
        assert 'n_steps' in config.hyperparameters
        
    def test_custom_hyperparameters(self):
        """Test custom hyperparameter override."""
        custom_params = {'learning_rate': 0.001, 'gamma': 0.95}
        config = TrainingConfig(algorithm='DQN', hyperparameters=custom_params)
        
        assert config.hyperparameters == custom_params


class TestTrainingEngine:
    """Test TrainingEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = ModelManager(models_dir=self.temp_dir)
        self.cohort_system = Mock(spec=CohortTrainingSystem)
        self.engine = TrainingEngine(
            model_manager=self.model_manager,
            cohort_system=self.cohort_system,
            device='cpu'
        )
        self.mock_env = MockGameEnvironment()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_create_dqn_agent(self):
        """Test DQN agent creation."""
        config = TrainingConfig(algorithm='DQN', max_episodes=10)
        agent = self.engine._create_agent(config, self.mock_env)
        
        assert isinstance(agent, DQNAgent)
        assert agent.state_size == 10
        assert agent.action_size == 5
        
    def test_create_ppo_agent(self):
        """Test PPO agent creation."""
        config = TrainingConfig(algorithm='PPO', max_episodes=10)
        agent = self.engine._create_agent(config, self.mock_env)
        
        assert isinstance(agent, PPOAgent)
        assert agent.state_size == 10
        assert agent.action_size == 5
        
    def test_create_a2c_agent(self):
        """Test A2C agent creation."""
        config = TrainingConfig(algorithm='A2C', max_episodes=10)
        agent = self.engine._create_agent(config, self.mock_env)
        
        assert isinstance(agent, A2CAgent)
        assert agent.state_size == 10
        assert agent.action_size == 5
        
    def test_unsupported_algorithm(self):
        """Test error handling for unsupported algorithm."""
        config = TrainingConfig(algorithm='UNKNOWN')
        
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            self.engine._create_agent(config, self.mock_env)
    
    @patch('bot.rl_bot_system.training.training_engine.RulesBasedBot')
    def test_behavior_cloning_initialization(self, mock_rules_bot_class):
        """Test behavior cloning initialization."""
        # Mock rules-based bot
        mock_rules_bot = Mock()
        mock_rules_bot.select_action.return_value = 0
        mock_rules_bot_class.return_value = mock_rules_bot
        
        config = TrainingConfig(algorithm='DQN', bc_episodes=5, max_episodes=10)
        self.engine.current_agent = self.engine._create_agent(config, self.mock_env)
        
        # Mock the behavior cloning training method
        with patch.object(self.engine, '_train_behavior_cloning') as mock_bc_train:
            self.engine._behavior_cloning_initialization(config, self.mock_env)
            
            # Verify behavior cloning was called
            mock_bc_train.assert_called_once()
            args, kwargs = mock_bc_train.call_args
            states, actions, lr = args
            
            assert len(states) > 0
            assert len(actions) > 0
            assert lr == config.bc_learning_rate
    
    def test_train_behavior_cloning_dqn(self):
        """Test behavior cloning training for DQN."""
        config = TrainingConfig(algorithm='DQN', max_episodes=10)
        self.engine.current_agent = self.engine._create_agent(config, self.mock_env)
        
        # Create mock demonstration data
        states = [np.random.random(10) for _ in range(10)]
        actions = [np.random.randint(0, 5) for _ in range(10)]
        
        # This should not raise an error
        self.engine._train_behavior_cloning(states, actions, 0.001)
    
    def test_train_behavior_cloning_ppo(self):
        """Test behavior cloning training for PPO."""
        config = TrainingConfig(algorithm='PPO', max_episodes=10)
        self.engine.current_agent = self.engine._create_agent(config, self.mock_env)
        
        # Create mock demonstration data
        states = [np.random.random(10) for _ in range(10)]
        actions = [np.random.randint(0, 5) for _ in range(10)]
        
        # This should not raise an error
        self.engine._train_behavior_cloning(states, actions, 0.001)
    
    def test_evaluate_model_dqn(self):
        """Test model evaluation for DQN."""
        config = TrainingConfig(algorithm='DQN', max_episodes=10)
        self.engine.current_config = config
        self.engine.current_agent = self.engine._create_agent(config, self.mock_env)
        
        result = self.engine._evaluate_model(self.mock_env, num_episodes=5)
        
        assert isinstance(result, EvaluationResult)
        assert result.total_episodes == 5
        assert result.wins + result.losses + result.draws == 5
        assert 0 <= result.win_rate <= 1
        assert result.average_episode_length > 0
    
    def test_evaluate_model_ppo(self):
        """Test model evaluation for PPO."""
        config = TrainingConfig(algorithm='PPO', max_episodes=10)
        self.engine.current_config = config
        self.engine.current_agent = self.engine._create_agent(config, self.mock_env)
        
        result = self.engine._evaluate_model(self.mock_env, num_episodes=3)
        
        assert isinstance(result, EvaluationResult)
        assert result.total_episodes == 3
        assert result.wins + result.losses + result.draws == 3
    
    def test_training_loop_dqn(self):
        """Test training loop for DQN."""
        config = TrainingConfig(algorithm='DQN', max_episodes=5, max_steps_per_episode=10)
        self.engine.current_config = config
        self.engine.current_agent = self.engine._create_agent(config, self.mock_env)
        
        self.engine._training_loop(config, self.mock_env)
        
        assert self.engine.episodes_completed == 5
        assert len(self.engine.training_metrics) == 5
        
        # Check that metrics are properly recorded
        for metric in self.engine.training_metrics:
            assert isinstance(metric, TrainingMetrics)
            assert metric.episode >= 0
            assert metric.episode_length > 0
    
    def test_training_loop_ppo(self):
        """Test training loop for PPO."""
        config = TrainingConfig(algorithm='PPO', max_episodes=3, max_steps_per_episode=10)
        self.engine.current_config = config
        self.engine.current_agent = self.engine._create_agent(config, self.mock_env)
        
        self.engine._training_loop(config, self.mock_env)
        
        assert self.engine.episodes_completed == 3
        assert len(self.engine.training_metrics) == 3
    
    def test_knowledge_transfer_same_algorithm(self):
        """Test knowledge transfer between same algorithm models."""
        # Create a previous model
        config = TrainingConfig(algorithm='DQN', max_episodes=10)
        agent = self.engine._create_agent(config, self.mock_env)
        
        # Save the model
        metadata = {
            'algorithm': 'DQN',
            'network_architecture': agent.get_network_architecture(),
            'hyperparameters': agent.get_hyperparameters(),
            'training_episodes': 100,
            'performance_metrics': {'win_rate': 0.7}
        }
        previous_model = self.model_manager.save_model(agent.q_network, 1, metadata)
        
        # Create new agent and transfer knowledge
        self.engine.current_agent = self.engine._create_agent(config, self.mock_env)
        
        # This should not raise an error
        self.engine._transfer_knowledge(previous_model)
    
    def test_knowledge_transfer_different_algorithm(self):
        """Test knowledge transfer between different algorithms."""
        # Create a DQN model
        dqn_config = TrainingConfig(algorithm='DQN', max_episodes=10)
        dqn_agent = self.engine._create_agent(dqn_config, self.mock_env)
        
        metadata = {
            'algorithm': 'DQN',
            'network_architecture': dqn_agent.get_network_architecture(),
            'hyperparameters': dqn_agent.get_hyperparameters(),
            'training_episodes': 100,
            'performance_metrics': {'win_rate': 0.7}
        }
        previous_model = self.model_manager.save_model(dqn_agent.q_network, 1, metadata)
        
        # Try to transfer to PPO (should handle gracefully)
        ppo_config = TrainingConfig(algorithm='PPO', max_episodes=10)
        self.engine.current_agent = self.engine._create_agent(ppo_config, self.mock_env)
        
        # This should not raise an error (should log warning and continue)
        self.engine._transfer_knowledge(previous_model)
    
    def test_get_training_progress(self):
        """Test training progress retrieval."""
        # Add some mock metrics
        self.engine.episodes_completed = 50
        self.engine.best_performance = 10.5
        self.engine.training_metrics = [
            TrainingMetrics(episode=i, total_reward=i*0.1, episode_length=20)
            for i in range(15)
        ]
        
        progress = self.engine.get_training_progress()
        
        assert progress['episodes_completed'] == 50
        assert progress['best_performance'] == 10.5
        assert len(progress['recent_metrics']) == 10  # Last 10 metrics
        assert len(progress['evaluation_results']) == 0
    
    def test_save_and_load_training_progress(self):
        """Test saving and loading training progress."""
        # Add some mock data
        self.engine.episodes_completed = 25
        self.engine.best_performance = 5.5
        self.engine.training_metrics = [
            TrainingMetrics(episode=i, total_reward=i*0.2, episode_length=15)
            for i in range(5)
        ]
        
        # Save progress
        progress_file = Path(self.temp_dir) / "progress.json"
        self.engine.save_training_progress(str(progress_file))
        
        # Create new engine and load progress
        new_engine = TrainingEngine(self.model_manager, device='cpu')
        new_engine.load_training_progress(str(progress_file))
        
        assert new_engine.episodes_completed == 25
        assert new_engine.best_performance == 5.5
        assert len(new_engine.training_metrics) == 5
    
    def test_train_next_generation_complete_flow(self):
        """Test complete training flow for next generation."""
        config = TrainingConfig(
            algorithm='DQN', 
            max_episodes=3, 
            max_steps_per_episode=5,
            evaluation_episodes=2,
            use_behavior_cloning=False  # Skip BC for faster test
        )
        
        with patch.object(self.engine, '_behavior_cloning_initialization'):
            model = self.engine.train_next_generation(
                generation=1,
                config=config,
                environment=self.mock_env
            )
        
        assert isinstance(model, RLModel)
        assert model.generation == 1
        assert model.algorithm == 'DQN'
        assert model.training_episodes == 3
        assert 'final_evaluation' in model.performance_metrics
    
    def test_train_next_generation_with_previous_model(self):
        """Test training with knowledge transfer from previous model."""
        # Create previous model
        config = TrainingConfig(algorithm='DQN', max_episodes=10)
        agent = self.engine._create_agent(config, self.mock_env)
        
        metadata = {
            'algorithm': 'DQN',
            'network_architecture': agent.get_network_architecture(),
            'hyperparameters': agent.get_hyperparameters(),
            'training_episodes': 100,
            'performance_metrics': {'win_rate': 0.6}
        }
        previous_model = self.model_manager.save_model(agent.q_network, 1, metadata)
        
        # Train next generation
        new_config = TrainingConfig(
            algorithm='DQN', 
            max_episodes=2, 
            max_steps_per_episode=5,
            evaluation_episodes=1,
            use_behavior_cloning=False
        )
        
        with patch.object(self.engine, '_transfer_knowledge') as mock_transfer:
            model = self.engine.train_next_generation(
                generation=2,
                config=new_config,
                environment=self.mock_env,
                previous_model=previous_model
            )
        
        # Verify knowledge transfer was called
        mock_transfer.assert_called_once_with(previous_model)
        assert model.generation == 2
        assert model.parent_generation == 1


class TestTrainingMetrics:
    """Test TrainingMetrics class."""
    
    def test_metrics_creation(self):
        """Test metrics creation with default timestamp."""
        metrics = TrainingMetrics(
            episode=10,
            total_reward=5.5,
            episode_length=100,
            loss=0.1
        )
        
        assert metrics.episode == 10
        assert metrics.total_reward == 5.5
        assert metrics.episode_length == 100
        assert metrics.loss == 0.1
        assert metrics.timestamp is not None


class TestEvaluationResult:
    """Test EvaluationResult class."""
    
    def test_evaluation_result_creation(self):
        """Test evaluation result creation."""
        result = EvaluationResult(
            generation=1,
            total_episodes=10,
            wins=7,
            losses=2,
            draws=1,
            average_reward=3.5,
            win_rate=0.7,
            average_episode_length=50.0
        )
        
        assert result.generation == 1
        assert result.total_episodes == 10
        assert result.wins == 7
        assert result.win_rate == 0.7
        assert result.evaluation_time is not None


if __name__ == '__main__':
    pytest.main([__file__])