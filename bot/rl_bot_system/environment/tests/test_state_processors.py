"""
Tests for state processor implementations.
"""

import pytest
import numpy as np

from bot.rl_bot_system.environment.state_processors import (
    StateRepresentationType,
    RawCoordinateProcessor,
    GridBasedProcessor,
    FeatureVectorProcessor,
    StateProcessorFactory,
    ABTestingFramework
)


@pytest.fixture
def sample_state():
    """Sample game state fixture."""
    return {
        'player': {
            'position': {'x': 100, 'y': 50},
            'velocity': {'x': 10, 'y': -5},
            'health': 80,
            'ammunition': 8
        },
        'enemies': [
            {
                'position': {'x': 200, 'y': 100},
                'velocity': {'x': -5, 'y': 0},
                'health': 60
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
                'position': {'x': 300, 'y': 200},
                'type': 'health'
            }
        ],
        'boundaries': {
            'left': -400, 'right': 400,
            'top': -300, 'bottom': 300
        },
        'game_time': 60,
        'round_active': True
    }


@pytest.fixture
def raw_processor():
    """Raw coordinate processor fixture."""
    return RawCoordinateProcessor()


@pytest.fixture
def grid_processor():
    """Grid-based processor fixture."""
    return GridBasedProcessor({
        'grid_width': 16,
        'grid_height': 12,
        'multi_channel': True
    })


@pytest.fixture
def feature_processor():
    """Feature vector processor fixture."""
    return FeatureVectorProcessor()


class TestRawCoordinateProcessor:
    """Test cases for RawCoordinateProcessor."""
    
    def test_initialization(self, raw_processor):
        """Test processor initialization."""
        assert raw_processor.representation_type == StateRepresentationType.RAW_COORDINATE
        assert raw_processor.config['max_enemies'] == 5
        assert raw_processor.config['normalize_coordinates'] == True
        assert raw_processor.total_features > 0
    
    def test_process_state(self, raw_processor, sample_state):
        """Test state processing."""
        features = raw_processor.process_state(sample_state)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) == raw_processor.total_features
        
        # Check that features are within reasonable bounds
        assert np.all(np.isfinite(features))
    
    def test_observation_space(self, raw_processor, sample_state):
        """Test observation space."""
        obs_space = raw_processor.get_observation_space()
        
        assert obs_space.shape == (raw_processor.total_features,)
        assert obs_space.dtype == np.float32
        
        # Test that processed state fits in observation space
        features = raw_processor.process_state(sample_state)
        assert obs_space.contains(features)
    
    def test_normalization(self, sample_state):
        """Test coordinate normalization."""
        config = {'normalize_coordinates': True}
        processor = RawCoordinateProcessor(config)
        
        features = processor.process_state(sample_state)
        
        # Check that normalized coordinates are in reasonable range
        assert -2.0 <= features[0] <= 2.0  # x position
        assert -2.0 <= features[1] <= 2.0  # y position
    
    def test_empty_state(self, raw_processor):
        """Test processing empty state."""
        empty_state = {}
        features = raw_processor.process_state(empty_state)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == raw_processor.total_features
        assert np.all(np.isfinite(features))


class TestGridBasedProcessor:
    """Test cases for GridBasedProcessor."""
    
    def test_initialization(self, grid_processor):
        """Test processor initialization."""
        assert grid_processor.representation_type == StateRepresentationType.GRID_BASED
        assert grid_processor.grid_width == 16
        assert grid_processor.grid_height == 12
        assert grid_processor.num_channels == len(grid_processor.channels)
    
    def test_process_state_multi_channel(self, grid_processor, sample_state):
        """Test multi-channel grid processing."""
        grid = grid_processor.process_state(sample_state)
        
        assert isinstance(grid, np.ndarray)
        assert grid.shape == (grid_processor.num_channels, 12, 16)
        assert grid.dtype == np.float32
        assert np.all(grid >= 0.0) and np.all(grid <= 1.0)
    
    def test_process_state_single_channel(self, sample_state):
        """Test single-channel grid processing."""
        processor = GridBasedProcessor({
            'grid_width': 16,
            'grid_height': 12,
            'multi_channel': False
        })
        
        grid = processor.process_state(sample_state)
        
        assert isinstance(grid, np.ndarray)
        assert grid.shape == (12, 16)
        assert grid.dtype == np.float32
    
    def test_world_to_grid_conversion(self, grid_processor):
        """Test world to grid coordinate conversion."""
        # Test center of world (0, 0) -> center of grid
        grid_x, grid_y = grid_processor._world_to_grid(0, 0)
        assert grid_x == 8  # Center of 16-wide grid
        assert grid_y == 6  # Center of 12-high grid
        
        # Test boundaries
        grid_x, grid_y = grid_processor._world_to_grid(-400, -300)
        assert grid_x == 0
        assert grid_y == 0
        
        grid_x, grid_y = grid_processor._world_to_grid(400, 300)
        assert grid_x == 15
        assert grid_y == 11
    
    def test_observation_space(self, grid_processor, sample_state):
        """Test observation space."""
        obs_space = grid_processor.get_observation_space()
        
        expected_shape = (grid_processor.num_channels, 12, 16)
        assert obs_space.shape == expected_shape
        assert obs_space.dtype == np.float32
        
        # Test that processed state fits in observation space
        grid = grid_processor.process_state(sample_state)
        assert obs_space.contains(grid)


class TestFeatureVectorProcessor:
    """Test cases for FeatureVectorProcessor."""
    
    def test_initialization(self, feature_processor):
        """Test processor initialization."""
        assert feature_processor.representation_type == StateRepresentationType.FEATURE_VECTOR
        assert feature_processor.total_features > 0
        assert feature_processor.config['include_distances'] == True
        assert feature_processor.config['include_angles'] == True
    
    def test_process_state(self, feature_processor, sample_state):
        """Test feature vector processing."""
        features = feature_processor.process_state(sample_state)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) == feature_processor.total_features
        assert np.all(np.isfinite(features))
    
    def test_distance_calculations(self, feature_processor):
        """Test distance calculation methods."""
        pos1 = {'x': 0, 'y': 0}
        pos2 = {'x': 3, 'y': 4}
        
        distance = feature_processor._calculate_distance(pos1, pos2)
        assert abs(distance - 5.0) < 1e-6  # 3-4-5 triangle
    
    def test_angle_calculations(self, feature_processor):
        """Test angle calculation methods."""
        from_pos = {'x': 0, 'y': 0}
        to_pos = {'x': 1, 'y': 0}
        
        angle = feature_processor._calculate_angle(from_pos, to_pos)
        assert abs(angle - 0.0) < 1e-6  # 0 radians for positive x direction
        
        to_pos = {'x': 0, 'y': 1}
        angle = feature_processor._calculate_angle(from_pos, to_pos)
        assert abs(angle - np.pi/2) < 1e-6  # π/2 radians for positive y direction
    
    def test_observation_space(self, feature_processor, sample_state):
        """Test observation space."""
        obs_space = feature_processor.get_observation_space()
        
        assert obs_space.shape == (feature_processor.total_features,)
        assert obs_space.dtype == np.float32
        
        # Test that processed state fits in observation space
        features = feature_processor.process_state(sample_state)
        assert obs_space.contains(features)
    
    def test_configurable_features(self, sample_state):
        """Test configurable feature inclusion."""
        # Test with minimal features
        config = {
            'include_distances': False,
            'include_angles': False,
            'include_tactical': False,
            'include_health_ratios': False,
            'max_enemies_for_features': 1
        }
        
        processor = FeatureVectorProcessor(config)
        features = processor.process_state(sample_state)
        
        # Should have fewer features
        default_processor = FeatureVectorProcessor()
        assert len(features) < default_processor.total_features
        assert len(features) == processor.total_features


class TestStateProcessorFactory:
    """Test cases for StateProcessorFactory."""
    
    def test_create_raw_coordinate_processor(self):
        """Test creating raw coordinate processor."""
        processor = StateProcessorFactory.create_processor(
            StateRepresentationType.RAW_COORDINATE
        )
        
        assert isinstance(processor, RawCoordinateProcessor)
        assert processor.representation_type == StateRepresentationType.RAW_COORDINATE
    
    def test_create_grid_based_processor(self):
        """Test creating grid-based processor."""
        processor = StateProcessorFactory.create_processor(
            StateRepresentationType.GRID_BASED,
            {'grid_width': 32, 'grid_height': 24}
        )
        
        assert isinstance(processor, GridBasedProcessor)
        assert processor.representation_type == StateRepresentationType.GRID_BASED
        assert processor.grid_width == 32
        assert processor.grid_height == 24
    
    def test_create_feature_vector_processor(self):
        """Test creating feature vector processor."""
        processor = StateProcessorFactory.create_processor(
            StateRepresentationType.FEATURE_VECTOR
        )
        
        assert isinstance(processor, FeatureVectorProcessor)
        assert processor.representation_type == StateRepresentationType.FEATURE_VECTOR
    
    def test_invalid_representation_type(self):
        """Test error handling for invalid representation type."""
        with pytest.raises(ValueError):
            StateProcessorFactory.create_processor("invalid_type")
    
    def test_get_available_types(self):
        """Test getting available representation types."""
        types = StateProcessorFactory.get_available_types()
        
        assert StateRepresentationType.RAW_COORDINATE in types
        assert StateRepresentationType.GRID_BASED in types
        assert StateRepresentationType.FEATURE_VECTOR in types


class TestABTestingFramework:
    """Test cases for ABTestingFramework."""
    
    @pytest.fixture
    def framework(self):
        """A/B testing framework fixture."""
        return ABTestingFramework()
    
    def test_create_test(self, framework):
        """Test creating an A/B test."""
        processors = framework.create_test(
            'test1',
            StateRepresentationType.RAW_COORDINATE,
            StateRepresentationType.FEATURE_VECTOR
        )
        
        assert 'A' in processors
        assert 'B' in processors
        assert isinstance(processors['A'], RawCoordinateProcessor)
        assert isinstance(processors['B'], FeatureVectorProcessor)
        assert 'test1' in framework.current_tests
    
    def test_record_and_summarize_results(self, framework):
        """Test recording and summarizing test results."""
        framework.create_test(
            'test1',
            StateRepresentationType.RAW_COORDINATE,
            StateRepresentationType.FEATURE_VECTOR
        )
        
        # Record some results
        framework.record_result('test1', 'A', {
            'total_reward': 100,
            'win_rate': 0.8,
            'episode_length': 200
        })
        
        framework.record_result('test1', 'B', {
            'total_reward': 120,
            'win_rate': 0.9,
            'episode_length': 180
        })
        
        summary = framework.get_test_summary('test1')
        
        assert 'A' in summary
        assert 'B' in summary
        assert summary['A']['count'] == 1
        assert summary['B']['count'] == 1
        assert summary['A']['avg_reward'] == 100
        assert summary['B']['avg_reward'] == 120