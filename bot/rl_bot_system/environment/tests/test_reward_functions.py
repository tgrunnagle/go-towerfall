"""
Tests for reward function implementations.
"""

import pytest
import numpy as np

from rl_bot_system.environment.reward_functions import (
    RewardType,
    SparseRewardFunction,
    DenseRewardFunction,
    ShapedRewardFunction,
    MultiObjectiveRewardFunction,
    HorizonBasedRewardFunction,
    RewardFunctionFactory,
    RewardTuningFramework
)


@pytest.fixture
def sample_state():
    """Sample game state fixture."""
    return {
        'player': {
            'position': {'x': 100, 'y': 50},
            'health': 80,
            'is_alive': True
        },
        'enemies': [
            {
                'position': {'x': 200, 'y': 100},
                'health': 60,
                'is_alive': True
            }
        ],
        'round_active': True
    }


@pytest.fixture
def previous_state():
    """Previous game state fixture."""
    return {
        'player': {
            'position': {'x': 90, 'y': 45},
            'health': 85,
            'is_alive': True
        },
        'enemies': [
            {
                'position': {'x': 195, 'y': 95},
                'health': 60,
                'is_alive': True
            }
        ],
        'round_active': True
    }


@pytest.fixture
def episode_stats():
    """Episode statistics fixture."""
    return {
        'kills': 0,
        'deaths': 0,
        'shots_fired': 0,
        'shots_hit': 0,
        'damage_dealt': 0,
        'damage_taken': 0
    }


@pytest.fixture
def sparse_reward_function():
    """Sparse reward function fixture."""
    return SparseRewardFunction()


@pytest.fixture
def dense_reward_function():
    """Dense reward function fixture."""
    return DenseRewardFunction()


@pytest.fixture
def shaped_reward_function():
    """Shaped reward function fixture."""
    return ShapedRewardFunction()


class TestSparseRewardFunction:
    """Test cases for SparseRewardFunction."""
    
    def test_initialization(self, sparse_reward_function):
        """Test reward function initialization."""
        assert sparse_reward_function.reward_type == RewardType.SPARSE
        assert sparse_reward_function.config['win_reward'] == 10.0
        assert sparse_reward_function.config['survival_reward'] == 0.01
    
    def test_survival_reward(self, sparse_reward_function, sample_state, previous_state, episode_stats):
        """Test basic survival reward."""
        reward = sparse_reward_function.calculate_reward(
            sample_state, 0, previous_state, episode_stats
        )
        
        # Should get survival reward
        assert reward == sparse_reward_function.config['survival_reward']
    
    def test_death_penalty(self, sparse_reward_function, sample_state, previous_state, episode_stats):
        """Test death penalty."""
        # Player dies
        dead_state = sample_state.copy()
        dead_state['player']['is_alive'] = False
        
        reward = sparse_reward_function.calculate_reward(
            dead_state, 0, previous_state, episode_stats
        )
        
        # Should get death penalty
        assert reward == sparse_reward_function.config['death_penalty']
    
    def test_kill_reward(self, sparse_reward_function, sample_state, previous_state, episode_stats):
        """Test kill reward."""
        # First call with 0 kills to establish baseline
        sparse_reward_function.calculate_reward(
            sample_state, 0, previous_state, episode_stats
        )
        
        # Second call with increased kill count
        kill_stats = episode_stats.copy()
        kill_stats['kills'] = 1
        
        reward = sparse_reward_function.calculate_reward(
            sample_state, 0, previous_state, kill_stats
        )
        
        # Should get kill reward + survival reward
        expected = sparse_reward_function.config['kill_reward'] + sparse_reward_function.config['survival_reward']
        assert reward == expected
    
    def test_win_reward(self, sparse_reward_function, sample_state, previous_state, episode_stats):
        """Test win reward."""
        # Game ends with player alive
        win_state = sample_state.copy()
        win_state['round_active'] = False
        
        reward = sparse_reward_function.calculate_reward(
            win_state, 0, previous_state, episode_stats
        )
        
        # Should get win reward + survival reward
        expected = sparse_reward_function.config['win_reward'] + sparse_reward_function.config['survival_reward']
        assert reward == expected
    
    def test_loss_penalty(self, sparse_reward_function, sample_state, previous_state, episode_stats):
        """Test loss penalty."""
        # Game ends with player dead
        loss_state = sample_state.copy()
        loss_state['round_active'] = False
        loss_state['player']['is_alive'] = False
        
        reward = sparse_reward_function.calculate_reward(
            loss_state, 0, previous_state, episode_stats
        )
        
        # Should get loss penalty + death penalty
        expected = sparse_reward_function.config['loss_penalty'] + sparse_reward_function.config['death_penalty']
        assert reward == expected
    
    def test_episode_statistics(self, sparse_reward_function, sample_state, previous_state, episode_stats):
        """Test episode statistics tracking."""
        # Calculate some rewards
        for _ in range(5):
            sparse_reward_function.calculate_reward(
                sample_state, 0, previous_state, episode_stats
            )
        
        stats = sparse_reward_function.get_episode_statistics()
        
        assert stats['total_reward'] > 0
        assert stats['mean_reward'] > 0
        assert len(sparse_reward_function.reward_history) == 5


class TestDenseRewardFunction:
    """Test cases for DenseRewardFunction."""
    
    @pytest.fixture
    def dense_sample_state(self):
        """Dense reward function sample state."""
        return {
            'player': {
                'position': {'x': 100, 'y': 50},
                'health': 80,
                'is_alive': True
            },
            'enemies': [
                {
                    'position': {'x': 200, 'y': 100},
                    'health': 60,
                    'is_alive': True
                }
            ],
            'boundaries': {
                'left': -400, 'right': 400,
                'top': -300, 'bottom': 300
            }
        }
    
    @pytest.fixture
    def dense_previous_state(self):
        """Dense reward function previous state."""
        return {
            'player': {
                'position': {'x': 90, 'y': 45},
                'health': 85,
                'is_alive': True
            },
            'enemies': [
                {
                    'position': {'x': 195, 'y': 95},
                    'health': 60,
                    'is_alive': True
                }
            ]
        }
    
    def test_initialization(self, dense_reward_function):
        """Test reward function initialization."""
        assert dense_reward_function.reward_type == RewardType.DENSE
        assert dense_reward_function.config['damage_dealt_scale'] == 0.1
        assert dense_reward_function.config['health_differential_scale'] == 0.05
    
    def test_damage_dealt_reward(self, dense_reward_function, dense_sample_state, dense_previous_state):
        """Test damage dealt reward."""
        # Increase damage dealt
        damage_stats = {'damage_dealt': 10, 'damage_taken': 0}
        
        reward = dense_reward_function.calculate_reward(
            dense_sample_state, 0, dense_previous_state, damage_stats
        )
        
        # Should include damage dealt reward
        assert reward != 0.0
        
        # Second call should not give additional reward for same damage
        reward2 = dense_reward_function.calculate_reward(
            dense_sample_state, 0, dense_previous_state, damage_stats
        )
        
        # Should be different from first call (no additional damage dealt)
        assert reward2 != reward
    
    def test_health_differential_reward(self, dense_reward_function, dense_previous_state):
        """Test health differential reward."""
        # Player has higher health than enemy
        high_health_state = {
            'player': {
                'position': {'x': 100, 'y': 50},
                'health': 100,
                'is_alive': True
            },
            'enemies': [
                {
                    'position': {'x': 200, 'y': 100},
                    'health': 30,
                    'is_alive': True
                }
            ],
            'boundaries': {
                'left': -400, 'right': 400,
                'top': -300, 'bottom': 300
            }
        }
        
        episode_stats = {'damage_dealt': 0, 'damage_taken': 0}
        
        reward = dense_reward_function.calculate_reward(
            high_health_state, 0, dense_previous_state, episode_stats
        )
        
        # Should get positive health differential reward
        assert reward > 0
    
    def test_positioning_reward(self, dense_reward_function, dense_sample_state, dense_previous_state):
        """Test positioning reward calculation."""
        episode_stats = {'damage_dealt': 0, 'damage_taken': 0}
        
        reward = dense_reward_function.calculate_reward(
            dense_sample_state, 0, dense_previous_state, episode_stats
        )
        
        # Should calculate some positioning reward
        assert isinstance(reward, float)
        assert np.isfinite(reward)


class TestShapedRewardFunction:
    """Test cases for ShapedRewardFunction."""
    
    @pytest.fixture
    def shaped_sample_state(self):
        """Shaped reward function sample state."""
        return {
            'player': {
                'position': {'x': 100, 'y': 50},
                'health': 80,
                'is_alive': True
            },
            'enemies': [
                {
                    'position': {'x': 200, 'y': 100},
                    'health': 60,
                    'is_alive': True,
                    'velocity': {'x': -5, 'y': 0}
                }
            ],
            'projectiles': [
                {
                    'position': {'x': 150, 'y': 75},
                    'velocity': {'x': 20, 'y': 0}
                }
            ],
            'power_ups': [
                {
                    'position': {'x': 120, 'y': 60},
                    'type': 'health'
                }
            ]
        }
    
    @pytest.fixture
    def shaped_previous_state(self):
        """Shaped reward function previous state."""
        return {
            'player': {
                'position': {'x': 90, 'y': 45},
                'health': 85,
                'is_alive': True
            }
        }
    
    def test_initialization(self, shaped_reward_function):
        """Test reward function initialization."""
        assert shaped_reward_function.reward_type == RewardType.SHAPED
        assert shaped_reward_function.config['aim_accuracy_scale'] == 0.2
        assert len(shaped_reward_function.visited_positions) == 0
    
    def test_aim_accuracy_reward(self, shaped_reward_function, shaped_sample_state, shaped_previous_state):
        """Test aim accuracy reward."""
        episode_stats = {'shots_fired': 10, 'shots_hit': 7}
        
        reward = shaped_reward_function.calculate_reward(
            shaped_sample_state, 0, shaped_previous_state, episode_stats
        )
        
        # Should calculate some reward based on accuracy
        assert isinstance(reward, float)
        assert np.isfinite(reward)
    
    def test_exploration_bonus(self, shaped_reward_function, shaped_sample_state, shaped_previous_state):
        """Test exploration bonus."""
        episode_stats = {'shots_fired': 10, 'shots_hit': 7}
        
        # First visit to a position
        reward1 = shaped_reward_function.calculate_reward(
            shaped_sample_state, 0, shaped_previous_state, episode_stats
        )
        
        # Second visit to same position
        reward2 = shaped_reward_function.calculate_reward(
            shaped_sample_state, 0, shaped_previous_state, episode_stats
        )
        
        # Should have exploration bonus in first visit
        assert len(shaped_reward_function.visited_positions) > 0
    
    def test_tactical_positioning(self, shaped_reward_function, shaped_sample_state, shaped_previous_state):
        """Test tactical positioning reward."""
        episode_stats = {'shots_fired': 10, 'shots_hit': 7}
        
        reward = shaped_reward_function.calculate_reward(
            shaped_sample_state, 0, shaped_previous_state, episode_stats
        )
        
        # Should calculate tactical positioning
        assert isinstance(reward, float)
        assert np.isfinite(reward)


class TestMultiObjectiveRewardFunction:
    """Test cases for MultiObjectiveRewardFunction."""
    
    @pytest.fixture
    def multi_objective_reward_function(self):
        """Multi-objective reward function fixture."""
        config = {
            'reward_functions': [
                {'type': 'sparse', 'weight': 0.5},
                {'type': 'dense', 'weight': 0.3},
                {'type': 'shaped', 'weight': 0.2}
            ]
        }
        return MultiObjectiveRewardFunction(config)
    
    def test_initialization(self, multi_objective_reward_function):
        """Test reward function initialization."""
        assert multi_objective_reward_function.reward_type == RewardType.MULTI_OBJECTIVE
        assert len(multi_objective_reward_function.component_functions) == 3
        assert len(multi_objective_reward_function.component_weights) == 3
    
    def test_multi_objective_calculation(self, multi_objective_reward_function, sample_state, previous_state, episode_stats):
        """Test multi-objective reward calculation."""
        reward = multi_objective_reward_function.calculate_reward(
            sample_state, 0, previous_state, episode_stats
        )
        
        # Should combine multiple reward signals
        assert isinstance(reward, float)
        assert np.isfinite(reward)
    
    def test_component_statistics(self, multi_objective_reward_function, sample_state, previous_state, episode_stats):
        """Test component statistics."""
        # Calculate some rewards
        for _ in range(3):
            multi_objective_reward_function.calculate_reward(
                sample_state, 0, previous_state, episode_stats
            )
        
        stats = multi_objective_reward_function.get_component_statistics()
        
        # Should have statistics for each component
        assert len(stats) == 3
        for component_stats in stats.values():
            assert 'total_reward' in component_stats


class TestHorizonBasedRewardFunction:
    """Test cases for HorizonBasedRewardFunction."""
    
    @pytest.fixture
    def horizon_reward_function(self):
        """Horizon-based reward function fixture."""
        return HorizonBasedRewardFunction()
    
    def test_initialization(self, horizon_reward_function):
        """Test reward function initialization."""
        assert horizon_reward_function.config['short_term_weight'] == 0.5
        assert len(horizon_reward_function.config['horizon_lengths']) == 3
        assert horizon_reward_function.step_count == 0
    
    def test_horizon_reward_calculation(self, horizon_reward_function, sample_state, previous_state, episode_stats):
        """Test horizon-based reward calculation."""
        # Calculate rewards for multiple steps
        rewards = []
        for _ in range(10):
            reward = horizon_reward_function.calculate_reward(
                sample_state, 0, previous_state, episode_stats
            )
            rewards.append(reward)
        
        # Should have calculated rewards for all steps
        assert len(rewards) == 10
        assert all(isinstance(r, float) and np.isfinite(r) for r in rewards)
        assert horizon_reward_function.step_count == 10
    
    def test_episode_reset(self, horizon_reward_function, sample_state, previous_state, episode_stats):
        """Test episode reset."""
        # Calculate some rewards
        for _ in range(5):
            horizon_reward_function.calculate_reward(
                sample_state, 0, previous_state, episode_stats
            )
        
        assert horizon_reward_function.step_count == 5
        assert len(horizon_reward_function.short_term_history) == 5
        
        # Reset episode
        horizon_reward_function.reset_episode()
        
        assert horizon_reward_function.step_count == 0
        assert len(horizon_reward_function.short_term_history) == 0


class TestRewardFunctionFactory:
    """Test cases for RewardFunctionFactory."""
    
    def test_create_sparse_reward_function(self):
        """Test creating sparse reward function."""
        rf = RewardFunctionFactory.create_reward_function(RewardType.SPARSE)
        
        assert isinstance(rf, SparseRewardFunction)
        assert rf.reward_type == RewardType.SPARSE
    
    def test_create_dense_reward_function(self):
        """Test creating dense reward function."""
        config = {'damage_dealt_scale': 0.2}
        rf = RewardFunctionFactory.create_reward_function(RewardType.DENSE, config)
        
        assert isinstance(rf, DenseRewardFunction)
        assert rf.reward_type == RewardType.DENSE
        assert rf.config['damage_dealt_scale'] == 0.2
    
    def test_create_shaped_reward_function(self):
        """Test creating shaped reward function."""
        rf = RewardFunctionFactory.create_reward_function(RewardType.SHAPED)
        
        assert isinstance(rf, ShapedRewardFunction)
        assert rf.reward_type == RewardType.SHAPED
    
    def test_create_multi_objective_reward_function(self):
        """Test creating multi-objective reward function."""
        rf = RewardFunctionFactory.create_reward_function(RewardType.MULTI_OBJECTIVE)
        
        assert isinstance(rf, MultiObjectiveRewardFunction)
        assert rf.reward_type == RewardType.MULTI_OBJECTIVE
    
    def test_invalid_reward_type(self):
        """Test error handling for invalid reward type."""
        with pytest.raises(ValueError):
            RewardFunctionFactory.create_reward_function("invalid_type")
    
    def test_get_available_types(self):
        """Test getting available reward types."""
        types = RewardFunctionFactory.get_available_types()
        
        assert RewardType.SPARSE in types
        assert RewardType.DENSE in types
        assert RewardType.SHAPED in types
        assert RewardType.MULTI_OBJECTIVE in types


class TestRewardTuningFramework:
    """Test cases for RewardTuningFramework."""
    
    @pytest.fixture
    def tuning_framework(self):
        """Reward tuning framework fixture."""
        return RewardTuningFramework()
    
    def test_create_experiment(self, tuning_framework):
        """Test creating a tuning experiment."""
        parameter_ranges = {
            'win_reward': (5.0, 15.0),
            'survival_reward': (0.005, 0.02)
        }
        
        configs = tuning_framework.create_experiment(
            'test_experiment',
            RewardType.SPARSE,
            parameter_ranges,
            num_trials=5
        )
        
        assert len(configs) == 5
        assert 'test_experiment' in tuning_framework.current_experiments
        
        # Check that parameters are within ranges
        for config in configs:
            assert 5.0 <= config['win_reward'] <= 15.0
            assert 0.005 <= config['survival_reward'] <= 0.02
    
    def test_record_experiment_result(self, tuning_framework):
        """Test recording experiment results."""
        parameter_ranges = {
            'win_reward': (5.0, 15.0)
        }
        
        configs = tuning_framework.create_experiment(
            'test_experiment',
            RewardType.SPARSE,
            parameter_ranges,
            num_trials=3
        )
        
        # Record results
        metrics = {'total_reward': 100.0, 'win_rate': 0.8}
        tuning_framework.record_experiment_result('test_experiment', 0, metrics)
        
        experiment = tuning_framework.current_experiments['test_experiment']
        assert len(experiment['results']) == 1
        assert experiment['results'][0]['metrics']['total_reward'] == 100.0
    
    def test_get_best_configuration(self, tuning_framework):
        """Test getting best configuration."""
        parameter_ranges = {
            'win_reward': (5.0, 15.0)
        }
        
        configs = tuning_framework.create_experiment(
            'test_experiment',
            RewardType.SPARSE,
            parameter_ranges,
            num_trials=3
        )
        
        # Record results with different performance
        tuning_framework.record_experiment_result(
            'test_experiment', 0, {'total_reward': 80.0}
        )
        tuning_framework.record_experiment_result(
            'test_experiment', 1, {'total_reward': 120.0}
        )
        tuning_framework.record_experiment_result(
            'test_experiment', 2, {'total_reward': 100.0}
        )
        
        best_config = tuning_framework.get_best_configuration('test_experiment')
        
        # Should return config from index 1 (highest total_reward)
        assert best_config == configs[1]
    
    def test_no_results(self, tuning_framework):
        """Test handling when no results are available."""
        parameter_ranges = {
            'win_reward': (5.0, 15.0)
        }
        
        tuning_framework.create_experiment(
            'test_experiment',
            RewardType.SPARSE,
            parameter_ranges,
            num_trials=3
        )
        
        # No results recorded
        best_config = tuning_framework.get_best_configuration('test_experiment')
        assert best_config is None