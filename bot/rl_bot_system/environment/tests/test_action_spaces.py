"""
Tests for action space implementations.
"""

import pytest
import numpy as np
from gymnasium import spaces

from bot.rl_bot_system.environment.action_spaces import (
    ActionSpaceType,
    DiscreteActionSpace,
    ContinuousActionSpace,
    HybridActionSpace,
    MultiDiscreteActionSpace,
    ActionSpaceFactory,
    ActionTimingController
)


@pytest.fixture
def discrete_action_space():
    """Discrete action space fixture."""
    return DiscreteActionSpace()


@pytest.fixture
def continuous_action_space():
    """Continuous action space fixture."""
    return ContinuousActionSpace()


@pytest.fixture
def hybrid_action_space():
    """Hybrid action space fixture."""
    return HybridActionSpace()


@pytest.fixture
def multi_discrete_action_space():
    """Multi-discrete action space fixture."""
    return MultiDiscreteActionSpace()


@pytest.fixture
def action_timing_controller():
    """Action timing controller fixture."""
    return ActionTimingController()


class TestDiscreteActionSpace:
    """Test cases for DiscreteActionSpace."""
    
    def test_initialization(self, discrete_action_space):
        """Test action space initialization."""
        assert discrete_action_space.action_space_type == ActionSpaceType.DISCRETE
        assert len(discrete_action_space.actions) > 0
        assert 'no_action' in discrete_action_space.actions
        assert discrete_action_space.action_count == len(discrete_action_space.actions)
    
    def test_get_action_space(self, discrete_action_space):
        """Test getting gymnasium action space."""
        gym_space = discrete_action_space.get_action_space()
        
        assert isinstance(gym_space, spaces.Discrete)
        assert gym_space.n == discrete_action_space.action_count
    
    def test_action_to_inputs_basic(self, discrete_action_space):
        """Test basic action conversion."""
        # Test no action
        inputs = discrete_action_space.action_to_inputs(0)  # no_action
        assert inputs == []
        
        # Test movement actions
        if 'move_left' in discrete_action_space.actions:
            action_idx = discrete_action_space.actions.index('move_left')
            inputs = discrete_action_space.action_to_inputs(action_idx)
            assert len(inputs) > 0
            assert inputs[0]['type'] == 'keyboard'
            assert inputs[0]['key'] == 'A'
            assert inputs[0]['pressed'] == True
    
    def test_action_to_inputs_shooting(self, discrete_action_space):
        """Test shooting action conversion."""
        if 'shoot_center' in discrete_action_space.actions:
            action_idx = discrete_action_space.actions.index('shoot_center')
            inputs = discrete_action_space.action_to_inputs(action_idx)
            assert len(inputs) > 0
            assert inputs[0]['type'] == 'mouse'
            assert inputs[0]['button'] == 'left'
            assert inputs[0]['pressed'] == True
    
    def test_invalid_action(self, discrete_action_space):
        """Test handling of invalid actions."""
        inputs = discrete_action_space.action_to_inputs(-1)
        assert inputs == []
        
        inputs = discrete_action_space.action_to_inputs(discrete_action_space.action_count)
        assert inputs == []
    
    def test_complexity_levels(self):
        """Test different complexity levels."""
        basic_config = {'complexity': 'basic'}
        basic_space = DiscreteActionSpace(basic_config)
        
        advanced_config = {'complexity': 'advanced'}
        advanced_space = DiscreteActionSpace(advanced_config)
        
        assert len(basic_space.actions) < len(advanced_space.actions)
    
    def test_get_action_names(self, discrete_action_space):
        """Test getting action names."""
        names = discrete_action_space.get_action_names()
        assert isinstance(names, list)
        assert len(names) == discrete_action_space.action_count
        assert names == discrete_action_space.actions


class TestContinuousActionSpace:
    """Test cases for ContinuousActionSpace."""
    
    def test_initialization(self, continuous_action_space):
        """Test action space initialization."""
        assert continuous_action_space.action_space_type == ActionSpaceType.CONTINUOUS
        assert continuous_action_space.total_dims > 0
    
    def test_get_action_space(self, continuous_action_space):
        """Test getting gymnasium action space."""
        gym_space = continuous_action_space.get_action_space()
        
        assert isinstance(gym_space, spaces.Box)
        assert gym_space.shape == (continuous_action_space.total_dims,)
        assert gym_space.dtype == np.float32
        assert np.all(gym_space.low == -1.0)
        assert np.all(gym_space.high == 1.0)
    
    def test_action_to_inputs(self, continuous_action_space):
        """Test continuous action conversion."""
        # Test zero action (should produce minimal inputs)
        zero_action = np.zeros(continuous_action_space.total_dims)
        inputs = continuous_action_space.action_to_inputs(zero_action)
        # Should have few or no inputs due to deadzone
        
        # Test movement action
        movement_action = np.zeros(continuous_action_space.total_dims)
        movement_action[0] = 0.8  # Strong right movement
        inputs = continuous_action_space.action_to_inputs(movement_action)
        
        # Should contain movement input
        movement_inputs = [inp for inp in inputs if inp['type'] == 'keyboard' and inp['key'] == 'D']
        assert len(movement_inputs) > 0
    
    def test_shooting_action(self, continuous_action_space):
        """Test shooting with continuous action."""
        shoot_action = np.zeros(continuous_action_space.total_dims)
        shoot_action[4] = 0.8  # Strong shoot action
        inputs = continuous_action_space.action_to_inputs(shoot_action)
        
        # Should contain mouse input
        mouse_inputs = [inp for inp in inputs if inp['type'] == 'mouse']
        assert len(mouse_inputs) > 0
    
    def test_invalid_action_size(self, continuous_action_space):
        """Test handling of invalid action size."""
        wrong_size_action = np.array([0.5, 0.5])  # Wrong size
        inputs = continuous_action_space.action_to_inputs(wrong_size_action)
        assert inputs == []


class TestHybridActionSpace:
    """Test cases for HybridActionSpace."""
    
    def test_initialization(self, hybrid_action_space):
        """Test action space initialization."""
        assert hybrid_action_space.action_space_type == ActionSpaceType.HYBRID
        assert len(hybrid_action_space.discrete_actions) > 0
        assert hybrid_action_space.continuous_dims > 0
    
    def test_get_action_space(self, hybrid_action_space):
        """Test getting gymnasium action space."""
        gym_space = hybrid_action_space.get_action_space()
        
        assert isinstance(gym_space, spaces.Dict)
        assert 'discrete' in gym_space.spaces
        assert 'continuous' in gym_space.spaces
        
        discrete_space = gym_space.spaces['discrete']
        continuous_space = gym_space.spaces['continuous']
        
        assert isinstance(discrete_space, spaces.Discrete)
        assert isinstance(continuous_space, spaces.Box)
        assert discrete_space.n == len(hybrid_action_space.discrete_actions)
        assert continuous_space.shape == (hybrid_action_space.continuous_dims,)
    
    def test_action_to_inputs(self, hybrid_action_space):
        """Test hybrid action conversion."""
        action = {
            'discrete': 1,  # Some movement action
            'continuous': np.array([0.5, -0.3, 0.8])  # Mouse and shoot
        }
        
        inputs = hybrid_action_space.action_to_inputs(action)
        assert isinstance(inputs, list)
        
        # Should have both keyboard and mouse inputs
        keyboard_inputs = [inp for inp in inputs if inp['type'] == 'keyboard']
        mouse_inputs = [inp for inp in inputs if inp['type'] == 'mouse']
        
        # Depending on the discrete action, should have keyboard input
        if action['discrete'] > 0:  # Not no_action
            assert len(keyboard_inputs) > 0
        
        # Should have mouse input due to shoot intensity > 0.1
        assert len(mouse_inputs) > 0
    
    def test_missing_action_components(self, hybrid_action_space):
        """Test handling of missing action components."""
        # Missing continuous component
        action = {'discrete': 1}
        inputs = hybrid_action_space.action_to_inputs(action)
        assert isinstance(inputs, list)
        
        # Missing discrete component
        action = {'continuous': np.array([0.5, -0.3, 0.8])}
        inputs = hybrid_action_space.action_to_inputs(action)
        assert isinstance(inputs, list)


class TestMultiDiscreteActionSpace:
    """Test cases for MultiDiscreteActionSpace."""
    
    def test_initialization(self, multi_discrete_action_space):
        """Test action space initialization."""
        assert multi_discrete_action_space.action_space_type == ActionSpaceType.MULTI_DISCRETE
        assert len(multi_discrete_action_space.movement_actions) > 0
        assert len(multi_discrete_action_space.shooting_actions) > 0
        assert len(multi_discrete_action_space.special_actions) > 0
    
    def test_get_action_space(self, multi_discrete_action_space):
        """Test getting gymnasium action space."""
        gym_space = multi_discrete_action_space.get_action_space()
        
        assert isinstance(gym_space, spaces.MultiDiscrete)
        assert len(gym_space.nvec) == 3  # movement, shooting, special
        assert gym_space.nvec[0] == multi_discrete_action_space.config['movement_actions']
        assert gym_space.nvec[1] == multi_discrete_action_space.config['shooting_actions']
        assert gym_space.nvec[2] == multi_discrete_action_space.config['special_actions']
    
    def test_action_to_inputs(self, multi_discrete_action_space):
        """Test multi-discrete action conversion."""
        action = np.array([1, 1, 0])  # move_left, shoot_center, no_special
        inputs = multi_discrete_action_space.action_to_inputs(action)
        
        assert isinstance(inputs, list)
        
        # Should have keyboard input for movement
        keyboard_inputs = [inp for inp in inputs if inp['type'] == 'keyboard']
        assert len(keyboard_inputs) > 0
        
        # Should have mouse input for shooting
        mouse_inputs = [inp for inp in inputs if inp['type'] == 'mouse']
        assert len(mouse_inputs) > 0
    
    def test_no_actions(self, multi_discrete_action_space):
        """Test all no-action case."""
        action = np.array([0, 0, 0])  # All no-action
        inputs = multi_discrete_action_space.action_to_inputs(action)
        
        # Should have no inputs or very few
        assert len(inputs) == 0
    
    def test_invalid_action_size(self, multi_discrete_action_space):
        """Test handling of invalid action size."""
        wrong_size_action = np.array([1, 1])  # Missing special action
        inputs = multi_discrete_action_space.action_to_inputs(wrong_size_action)
        assert inputs == []


class TestActionSpaceFactory:
    """Test cases for ActionSpaceFactory."""
    
    def test_create_discrete_action_space(self):
        """Test creating discrete action space."""
        action_space = ActionSpaceFactory.create_action_space(ActionSpaceType.DISCRETE)
        
        assert isinstance(action_space, DiscreteActionSpace)
        assert action_space.action_space_type == ActionSpaceType.DISCRETE
    
    def test_create_continuous_action_space(self):
        """Test creating continuous action space."""
        config = {'movement_range': [-2.0, 2.0]}
        action_space = ActionSpaceFactory.create_action_space(
            ActionSpaceType.CONTINUOUS, config
        )
        
        assert isinstance(action_space, ContinuousActionSpace)
        assert action_space.action_space_type == ActionSpaceType.CONTINUOUS
        assert action_space.config['movement_range'] == [-2.0, 2.0]
    
    def test_create_hybrid_action_space(self):
        """Test creating hybrid action space."""
        action_space = ActionSpaceFactory.create_action_space(ActionSpaceType.HYBRID)
        
        assert isinstance(action_space, HybridActionSpace)
        assert action_space.action_space_type == ActionSpaceType.HYBRID
    
    def test_create_multi_discrete_action_space(self):
        """Test creating multi-discrete action space."""
        action_space = ActionSpaceFactory.create_action_space(ActionSpaceType.MULTI_DISCRETE)
        
        assert isinstance(action_space, MultiDiscreteActionSpace)
        assert action_space.action_space_type == ActionSpaceType.MULTI_DISCRETE
    
    def test_invalid_action_space_type(self):
        """Test error handling for invalid action space type."""
        with pytest.raises(ValueError):
            ActionSpaceFactory.create_action_space("invalid_type")
    
    def test_get_available_types(self):
        """Test getting available action space types."""
        types = ActionSpaceFactory.get_available_types()
        
        assert ActionSpaceType.DISCRETE in types
        assert ActionSpaceType.CONTINUOUS in types
        assert ActionSpaceType.HYBRID in types
        assert ActionSpaceType.MULTI_DISCRETE in types


class TestActionTimingController:
    """Test cases for ActionTimingController."""
    
    def test_schedule_action(self, action_timing_controller):
        """Test scheduling an action."""
        inputs = [{'type': 'keyboard', 'key': 'A', 'pressed': True}]
        action_timing_controller.schedule_action('test_action', inputs, 1.0, 0.5)
        
        assert len(action_timing_controller.action_queue) == 1
        assert action_timing_controller.action_queue[0]['id'] == 'test_action'
    
    def test_get_current_inputs(self, action_timing_controller):
        """Test getting current inputs."""
        inputs = [{'type': 'keyboard', 'key': 'A', 'pressed': True}]
        action_timing_controller.schedule_action('test_action', inputs, 1.0, 0.5)
        
        # Before start time
        current_inputs = action_timing_controller.get_current_inputs(0.5)
        assert len(current_inputs) == 0
        
        # At start time
        current_inputs = action_timing_controller.get_current_inputs(1.0)
        assert len(current_inputs) == 1
        assert current_inputs[0]['key'] == 'A'
        
        # During action
        current_inputs = action_timing_controller.get_current_inputs(1.2)
        assert len(current_inputs) == 1
        
        # After action expires
        current_inputs = action_timing_controller.get_current_inputs(2.0)
        assert len(current_inputs) == 0
    
    def test_cancel_action(self, action_timing_controller):
        """Test canceling an action."""
        inputs = [{'type': 'keyboard', 'key': 'A', 'pressed': True}]
        action_timing_controller.schedule_action('test_action', inputs, 1.0, 0.5)
        
        assert len(action_timing_controller.action_queue) == 1
        
        action_timing_controller.cancel_action('test_action')
        assert len(action_timing_controller.action_queue) == 0
    
    def test_clear_all_actions(self, action_timing_controller):
        """Test clearing all actions."""
        inputs1 = [{'type': 'keyboard', 'key': 'A', 'pressed': True}]
        inputs2 = [{'type': 'keyboard', 'key': 'D', 'pressed': True}]
        
        action_timing_controller.schedule_action('action1', inputs1, 1.0, 0.5)
        action_timing_controller.schedule_action('action2', inputs2, 2.0, 0.5)
        
        assert len(action_timing_controller.action_queue) == 2
        
        action_timing_controller.clear_all_actions()
        assert len(action_timing_controller.action_queue) == 0
        assert len(action_timing_controller.active_actions) == 0