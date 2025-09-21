"""
Action space implementations for different action representations.

This module provides configurable action space systems that support discrete,
continuous, and hybrid action spaces with mapping to GameClient inputs.
"""

import numpy as np
from gymnasium import spaces
from typing import Dict, Any, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum
import math


class ActionSpaceType(Enum):
    """Types of action spaces available."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"
    MULTI_DISCRETE = "multi_discrete"


class ActionSpaceConfig(ABC):
    """Base class for action space configurations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize action space configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.action_space_type = self._get_action_space_type()
    
    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """Get the action space."""
        pass
    
    @abstractmethod
    def action_to_inputs(self, action) -> List[Dict[str, Any]]:
        """Convert action to game client inputs."""
        pass
    
    @abstractmethod
    def _get_action_space_type(self) -> ActionSpaceType:
        """Get the action space type."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get action space configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update action space configuration."""
        self.config.update(new_config)


class DiscreteActionSpace(ActionSpaceConfig):
    """
    Discrete action space with configurable actions.
    
    Supports different complexity levels from basic movement to complex combinations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize discrete action space.
        
        Config options:
            - complexity: 'basic', 'intermediate', 'advanced' (default: 'intermediate')
            - include_combinations: Include combined actions (default: True)
            - action_duration: Default duration for actions in seconds (default: 0.1)
            - mouse_target_mode: 'fixed', 'relative', 'smart' (default: 'smart')
        """
        default_config = {
            'complexity': 'intermediate',
            'include_combinations': True,
            'action_duration': 0.1,
            'mouse_target_mode': 'smart'
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Build action list based on complexity
        self.actions = self._build_action_list()
        self.action_count = len(self.actions)
    
    def _get_action_space_type(self) -> ActionSpaceType:
        """Get action space type."""
        return ActionSpaceType.DISCRETE
    
    def get_action_space(self) -> spaces.Space:
        """Get discrete action space."""
        return spaces.Discrete(self.action_count)
    
    def action_to_inputs(self, action: int) -> List[Dict[str, Any]]:
        """Convert discrete action to game client inputs."""
        if not (0 <= action < self.action_count):
            return []
        
        action_name = self.actions[action]
        return self._action_name_to_inputs(action_name)
    
    def _build_action_list(self) -> List[str]:
        """Build list of available actions based on configuration."""
        actions = ['no_action']
        
        complexity = self.config['complexity']
        
        # Basic actions (available in all complexity levels)
        basic_actions = [
            'move_left', 'move_right', 'jump', 'crouch',
            'shoot_center', 'stop_movement'
        ]
        actions.extend(basic_actions)
        
        if complexity in ['intermediate', 'advanced']:
            # Intermediate actions
            intermediate_actions = [
                'move_left_jump', 'move_right_jump',
                'shoot_left', 'shoot_right', 'shoot_up', 'shoot_down',
                'strafe_left_shoot', 'strafe_right_shoot'
            ]
            actions.extend(intermediate_actions)
        
        if complexity == 'advanced':
            # Advanced actions
            advanced_actions = [
                'dodge_left', 'dodge_right',
                'shoot_nearest_enemy', 'shoot_weakest_enemy',
                'retreat_and_shoot', 'advance_and_shoot',
                'circle_strafe_left', 'circle_strafe_right',
                'quick_peek_left', 'quick_peek_right'
            ]
            actions.extend(advanced_actions)
        
        # Add combinations if enabled
        if self.config['include_combinations'] and complexity != 'basic':
            combination_actions = [
                'jump_shoot_center', 'crouch_shoot_center',
                'move_left_shoot_right', 'move_right_shoot_left'
            ]
            actions.extend(combination_actions)
        
        return actions
    
    def _action_name_to_inputs(self, action_name: str) -> List[Dict[str, Any]]:
        """Convert action name to input commands."""
        duration = self.config['action_duration']
        
        if action_name == 'no_action':
            return []
        
        # Basic movement actions
        elif action_name == 'move_left':
            return [self._keyboard_input('A', True, duration)]
        
        elif action_name == 'move_right':
            return [self._keyboard_input('D', True, duration)]
        
        elif action_name == 'jump':
            return [self._keyboard_input('W', True, duration)]
        
        elif action_name == 'crouch':
            return [self._keyboard_input('S', True, duration)]
        
        elif action_name == 'stop_movement':
            return [
                self._keyboard_input('A', False),
                self._keyboard_input('D', False),
                self._keyboard_input('W', False),
                self._keyboard_input('S', False)
            ]
        
        # Shooting actions
        elif action_name == 'shoot_center':
            return [self._mouse_input('left', True, 0, 0, duration)]
        
        elif action_name == 'shoot_left':
            return [self._mouse_input('left', True, -100, 0, duration)]
        
        elif action_name == 'shoot_right':
            return [self._mouse_input('left', True, 100, 0, duration)]
        
        elif action_name == 'shoot_up':
            return [self._mouse_input('left', True, 0, -100, duration)]
        
        elif action_name == 'shoot_down':
            return [self._mouse_input('left', True, 0, 100, duration)]
        
        # Combined movement and jump actions
        elif action_name == 'move_left_jump':
            return [
                self._keyboard_input('A', True, duration),
                self._keyboard_input('W', True, duration * 0.5)
            ]
        
        elif action_name == 'move_right_jump':
            return [
                self._keyboard_input('D', True, duration),
                self._keyboard_input('W', True, duration * 0.5)
            ]
        
        # Strafe and shoot combinations
        elif action_name == 'strafe_left_shoot':
            return [
                self._keyboard_input('A', True, duration),
                self._mouse_input('left', True, 0, 0, duration * 0.8)
            ]
        
        elif action_name == 'strafe_right_shoot':
            return [
                self._keyboard_input('D', True, duration),
                self._mouse_input('left', True, 0, 0, duration * 0.8)
            ]
        
        # Advanced dodge actions
        elif action_name == 'dodge_left':
            return [self._keyboard_input('A', True, duration * 0.3)]
        
        elif action_name == 'dodge_right':
            return [self._keyboard_input('D', True, duration * 0.3)]
        
        # Smart shooting actions (require game state context)
        elif action_name in ['shoot_nearest_enemy', 'shoot_weakest_enemy']:
            # These would need game state context to determine target
            # For now, default to center shot
            return [self._mouse_input('left', True, 0, 0, duration)]
        
        # Tactical movement actions
        elif action_name == 'retreat_and_shoot':
            return [
                self._keyboard_input('S', True, duration),
                self._mouse_input('left', True, 0, 0, duration * 0.7)
            ]
        
        elif action_name == 'advance_and_shoot':
            return [
                self._keyboard_input('W', True, duration),
                self._mouse_input('left', True, 0, 0, duration * 0.7)
            ]
        
        # Circle strafing
        elif action_name == 'circle_strafe_left':
            return [
                self._keyboard_input('A', True, duration),
                self._mouse_input('left', True, 50, 0, duration * 0.8)
            ]
        
        elif action_name == 'circle_strafe_right':
            return [
                self._keyboard_input('D', True, duration),
                self._mouse_input('left', True, -50, 0, duration * 0.8)
            ]
        
        # Quick peek actions
        elif action_name == 'quick_peek_left':
            return [
                self._keyboard_input('A', True, duration * 0.2),
                self._mouse_input('left', True, -75, 0, duration * 0.1)
            ]
        
        elif action_name == 'quick_peek_right':
            return [
                self._keyboard_input('D', True, duration * 0.2),
                self._mouse_input('left', True, 75, 0, duration * 0.1)
            ]
        
        # Complex combinations
        elif action_name == 'jump_shoot_center':
            return [
                self._keyboard_input('W', True, duration * 0.5),
                self._mouse_input('left', True, 0, 0, duration * 0.3)
            ]
        
        elif action_name == 'crouch_shoot_center':
            return [
                self._keyboard_input('S', True, duration),
                self._mouse_input('left', True, 0, 0, duration * 0.8)
            ]
        
        elif action_name == 'move_left_shoot_right':
            return [
                self._keyboard_input('A', True, duration),
                self._mouse_input('left', True, 100, 0, duration * 0.8)
            ]
        
        elif action_name == 'move_right_shoot_left':
            return [
                self._keyboard_input('D', True, duration),
                self._mouse_input('left', True, -100, 0, duration * 0.8)
            ]
        
        return []
    
    def _keyboard_input(self, key: str, pressed: bool, duration: Optional[float] = None) -> Dict[str, Any]:
        """Create keyboard input command."""
        cmd = {
            'type': 'keyboard',
            'key': key,
            'pressed': pressed
        }
        if duration is not None:
            cmd['duration'] = duration
        return cmd
    
    def _mouse_input(
        self, 
        button: str, 
        pressed: bool, 
        x: float, 
        y: float, 
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create mouse input command."""
        cmd = {
            'type': 'mouse',
            'button': button,
            'pressed': pressed,
            'x': x,
            'y': y
        }
        if duration is not None:
            cmd['duration'] = duration
        return cmd
    
    def get_action_names(self) -> List[str]:
        """Get list of action names."""
        return self.actions.copy()


class ContinuousActionSpace(ActionSpaceConfig):
    """
    Continuous action space for fine-grained control.
    
    Provides continuous control over movement and aiming.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize continuous action space.
        
        Config options:
            - movement_range: Range for movement actions (default: [-1.0, 1.0])
            - mouse_range: Range for mouse coordinates (default: [-200, 200])
            - include_discrete_actions: Include discrete actions like jump (default: True)
            - action_duration: Duration for actions (default: 0.1)
        """
        default_config = {
            'movement_range': [-1.0, 1.0],
            'mouse_range': [-200, 200],
            'include_discrete_actions': True,
            'action_duration': 0.1
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Calculate action space dimensions
        self.movement_dims = 2  # x, y movement
        self.mouse_dims = 2     # x, y mouse position
        self.shoot_dims = 1     # shoot intensity
        
        self.discrete_dims = 0
        if self.config['include_discrete_actions']:
            self.discrete_dims = 2  # jump, crouch
        
        self.total_dims = self.movement_dims + self.mouse_dims + self.shoot_dims + self.discrete_dims
    
    def _get_action_space_type(self) -> ActionSpaceType:
        """Get action space type."""
        return ActionSpaceType.CONTINUOUS
    
    def get_action_space(self) -> spaces.Space:
        """Get continuous action space."""
        low = np.array([-1.0] * self.total_dims, dtype=np.float32)
        high = np.array([1.0] * self.total_dims, dtype=np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def action_to_inputs(self, action: np.ndarray) -> List[Dict[str, Any]]:
        """Convert continuous action to game client inputs."""
        if len(action) != self.total_dims:
            return []
        
        inputs = []
        duration = self.config['action_duration']
        
        # Movement actions (dimensions 0-1)
        move_x = action[0]  # -1 = left, +1 = right
        move_y = action[1]  # -1 = down/crouch, +1 = up/jump
        
        # Convert movement to keyboard inputs
        if abs(move_x) > 0.1:  # Deadzone
            if move_x < 0:
                inputs.append(self._keyboard_input('A', True, duration))
            else:
                inputs.append(self._keyboard_input('D', True, duration))
        
        if abs(move_y) > 0.1:  # Deadzone
            if move_y > 0:
                inputs.append(self._keyboard_input('W', True, duration))
            elif move_y < -0.5:  # Require stronger input for crouch
                inputs.append(self._keyboard_input('S', True, duration))
        
        # Mouse actions (dimensions 2-3)
        mouse_x = action[2] * self.config['mouse_range'][1]  # Scale to mouse range
        mouse_y = action[3] * self.config['mouse_range'][1]
        
        # Shooting action (dimension 4)
        shoot_intensity = action[4]
        if shoot_intensity > 0.1:  # Threshold for shooting
            shoot_duration = duration * shoot_intensity
            inputs.append(self._mouse_input('left', True, mouse_x, mouse_y, shoot_duration))
        
        # Discrete actions (dimensions 5-6) if enabled
        if self.config['include_discrete_actions'] and len(action) > 5:
            jump_action = action[5]
            if jump_action > 0.5:
                inputs.append(self._keyboard_input('W', True, duration * 0.5))
            
            if len(action) > 6:
                crouch_action = action[6]
                if crouch_action > 0.5:
                    inputs.append(self._keyboard_input('S', True, duration))
        
        return inputs
    
    def _keyboard_input(self, key: str, pressed: bool, duration: Optional[float] = None) -> Dict[str, Any]:
        """Create keyboard input command."""
        cmd = {
            'type': 'keyboard',
            'key': key,
            'pressed': pressed
        }
        if duration is not None:
            cmd['duration'] = duration
        return cmd
    
    def _mouse_input(
        self, 
        button: str, 
        pressed: bool, 
        x: float, 
        y: float, 
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create mouse input command."""
        cmd = {
            'type': 'mouse',
            'button': button,
            'pressed': pressed,
            'x': x,
            'y': y
        }
        if duration is not None:
            cmd['duration'] = duration
        return cmd


class HybridActionSpace(ActionSpaceConfig):
    """
    Hybrid action space combining discrete and continuous actions.
    
    Uses discrete actions for high-level decisions and continuous for fine control.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hybrid action space.
        
        Config options:
            - discrete_actions: List of discrete action names (default: basic set)
            - continuous_dims: Number of continuous dimensions (default: 3)
            - action_duration: Duration for actions (default: 0.1)
        """
        default_config = {
            'discrete_actions': [
                'no_action', 'move_left', 'move_right', 'jump', 'crouch', 'stop'
            ],
            'continuous_dims': 3,  # mouse_x, mouse_y, shoot_intensity
            'action_duration': 0.1
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        self.discrete_actions = self.config['discrete_actions']
        self.continuous_dims = self.config['continuous_dims']
    
    def _get_action_space_type(self) -> ActionSpaceType:
        """Get action space type."""
        return ActionSpaceType.HYBRID
    
    def get_action_space(self) -> spaces.Space:
        """Get hybrid action space."""
        discrete_space = spaces.Discrete(len(self.discrete_actions))
        continuous_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.continuous_dims,), 
            dtype=np.float32
        )
        
        return spaces.Dict({
            'discrete': discrete_space,
            'continuous': continuous_space
        })
    
    def action_to_inputs(self, action: Dict[str, Union[int, np.ndarray]]) -> List[Dict[str, Any]]:
        """Convert hybrid action to game client inputs."""
        inputs = []
        duration = self.config['action_duration']
        
        # Process discrete action
        discrete_action = action.get('discrete', 0)
        if 0 <= discrete_action < len(self.discrete_actions):
            action_name = self.discrete_actions[discrete_action]
            inputs.extend(self._discrete_action_to_inputs(action_name, duration))
        
        # Process continuous action
        continuous_action = action.get('continuous', np.zeros(self.continuous_dims))
        if len(continuous_action) >= 3:
            mouse_x = continuous_action[0] * 200  # Scale to mouse range
            mouse_y = continuous_action[1] * 200
            shoot_intensity = continuous_action[2]
            
            if shoot_intensity > 0.1:
                shoot_duration = duration * shoot_intensity
                inputs.append(self._mouse_input('left', True, mouse_x, mouse_y, shoot_duration))
        
        return inputs
    
    def _discrete_action_to_inputs(self, action_name: str, duration: float) -> List[Dict[str, Any]]:
        """Convert discrete action name to inputs."""
        if action_name == 'no_action':
            return []
        elif action_name == 'move_left':
            return [self._keyboard_input('A', True, duration)]
        elif action_name == 'move_right':
            return [self._keyboard_input('D', True, duration)]
        elif action_name == 'jump':
            return [self._keyboard_input('W', True, duration)]
        elif action_name == 'crouch':
            return [self._keyboard_input('S', True, duration)]
        elif action_name == 'stop':
            return [
                self._keyboard_input('A', False),
                self._keyboard_input('D', False),
                self._keyboard_input('W', False),
                self._keyboard_input('S', False)
            ]
        return []
    
    def _keyboard_input(self, key: str, pressed: bool, duration: Optional[float] = None) -> Dict[str, Any]:
        """Create keyboard input command."""
        cmd = {
            'type': 'keyboard',
            'key': key,
            'pressed': pressed
        }
        if duration is not None:
            cmd['duration'] = duration
        return cmd
    
    def _mouse_input(
        self, 
        button: str, 
        pressed: bool, 
        x: float, 
        y: float, 
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create mouse input command."""
        cmd = {
            'type': 'mouse',
            'button': button,
            'pressed': pressed,
            'x': x,
            'y': y
        }
        if duration is not None:
            cmd['duration'] = duration
        return cmd


class MultiDiscreteActionSpace(ActionSpaceConfig):
    """
    Multi-discrete action space for independent action dimensions.
    
    Allows independent control of different action categories.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-discrete action space.
        
        Config options:
            - movement_actions: Number of movement actions (default: 5)
            - shooting_actions: Number of shooting actions (default: 9)
            - special_actions: Number of special actions (default: 3)
            - action_duration: Duration for actions (default: 0.1)
        """
        default_config = {
            'movement_actions': 5,  # no_move, left, right, jump, crouch
            'shooting_actions': 9,  # no_shoot, center, 8 directions
            'special_actions': 3,   # no_special, reload, use_item
            'action_duration': 0.1
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        self.movement_actions = [
            'no_move', 'move_left', 'move_right', 'jump', 'crouch'
        ]
        
        self.shooting_actions = [
            'no_shoot', 'shoot_center', 'shoot_up', 'shoot_down',
            'shoot_left', 'shoot_right', 'shoot_up_left', 'shoot_up_right',
            'shoot_down_left', 'shoot_down_right'
        ]
        
        self.special_actions = [
            'no_special', 'reload', 'use_item'
        ]
    
    def _get_action_space_type(self) -> ActionSpaceType:
        """Get action space type."""
        return ActionSpaceType.MULTI_DISCRETE
    
    def get_action_space(self) -> spaces.Space:
        """Get multi-discrete action space."""
        nvec = [
            self.config['movement_actions'],
            self.config['shooting_actions'],
            self.config['special_actions']
        ]
        return spaces.MultiDiscrete(nvec)
    
    def action_to_inputs(self, action: np.ndarray) -> List[Dict[str, Any]]:
        """Convert multi-discrete action to game client inputs."""
        if len(action) != 3:
            return []
        
        inputs = []
        duration = self.config['action_duration']
        
        # Movement action
        movement_idx = action[0]
        if 0 <= movement_idx < len(self.movement_actions):
            movement_action = self.movement_actions[movement_idx]
            inputs.extend(self._movement_action_to_inputs(movement_action, duration))
        
        # Shooting action
        shooting_idx = action[1]
        if 0 <= shooting_idx < len(self.shooting_actions):
            shooting_action = self.shooting_actions[shooting_idx]
            inputs.extend(self._shooting_action_to_inputs(shooting_action, duration))
        
        # Special action
        special_idx = action[2]
        if 0 <= special_idx < len(self.special_actions):
            special_action = self.special_actions[special_idx]
            inputs.extend(self._special_action_to_inputs(special_action, duration))
        
        return inputs
    
    def _movement_action_to_inputs(self, action_name: str, duration: float) -> List[Dict[str, Any]]:
        """Convert movement action to inputs."""
        if action_name == 'no_move':
            return []
        elif action_name == 'move_left':
            return [self._keyboard_input('A', True, duration)]
        elif action_name == 'move_right':
            return [self._keyboard_input('D', True, duration)]
        elif action_name == 'jump':
            return [self._keyboard_input('W', True, duration)]
        elif action_name == 'crouch':
            return [self._keyboard_input('S', True, duration)]
        return []
    
    def _shooting_action_to_inputs(self, action_name: str, duration: float) -> List[Dict[str, Any]]:
        """Convert shooting action to inputs."""
        if action_name == 'no_shoot':
            return []
        
        # Define shooting directions
        directions = {
            'shoot_center': (0, 0),
            'shoot_up': (0, -100),
            'shoot_down': (0, 100),
            'shoot_left': (-100, 0),
            'shoot_right': (100, 0),
            'shoot_up_left': (-70, -70),
            'shoot_up_right': (70, -70),
            'shoot_down_left': (-70, 70),
            'shoot_down_right': (70, 70)
        }
        
        if action_name in directions:
            x, y = directions[action_name]
            return [self._mouse_input('left', True, x, y, duration)]
        
        return []
    
    def _special_action_to_inputs(self, action_name: str, duration: float) -> List[Dict[str, Any]]:
        """Convert special action to inputs."""
        if action_name == 'no_special':
            return []
        elif action_name == 'reload':
            # Reload could be mapped to a specific key if the game supports it
            return []
        elif action_name == 'use_item':
            # Use item could be mapped to a specific key
            return []
        return []
    
    def _keyboard_input(self, key: str, pressed: bool, duration: Optional[float] = None) -> Dict[str, Any]:
        """Create keyboard input command."""
        cmd = {
            'type': 'keyboard',
            'key': key,
            'pressed': pressed
        }
        if duration is not None:
            cmd['duration'] = duration
        return cmd
    
    def _mouse_input(
        self, 
        button: str, 
        pressed: bool, 
        x: float, 
        y: float, 
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create mouse input command."""
        cmd = {
            'type': 'mouse',
            'button': button,
            'pressed': pressed,
            'x': x,
            'y': y
        }
        if duration is not None:
            cmd['duration'] = duration
        return cmd


class ActionSpaceFactory:
    """Factory for creating action space configurations."""
    
    @staticmethod
    def create_action_space(
        action_space_type: ActionSpaceType,
        config: Optional[Dict[str, Any]] = None
    ) -> ActionSpaceConfig:
        """
        Create an action space configuration of the specified type.
        
        Args:
            action_space_type: Type of action space
            config: Configuration for the action space
            
        Returns:
            ActionSpaceConfig instance
        """
        if action_space_type == ActionSpaceType.DISCRETE:
            return DiscreteActionSpace(config)
        elif action_space_type == ActionSpaceType.CONTINUOUS:
            return ContinuousActionSpace(config)
        elif action_space_type == ActionSpaceType.HYBRID:
            return HybridActionSpace(config)
        elif action_space_type == ActionSpaceType.MULTI_DISCRETE:
            return MultiDiscreteActionSpace(config)
        else:
            raise ValueError(f"Unsupported action space type: {action_space_type}")
    
    @staticmethod
    def get_available_types() -> List[ActionSpaceType]:
        """Get list of available action space types."""
        return list(ActionSpaceType)


class ActionTimingController:
    """Controller for managing action timing and duration."""
    
    def __init__(self):
        self.active_actions = {}
        self.action_queue = []
    
    def schedule_action(
        self, 
        action_id: str, 
        inputs: List[Dict[str, Any]], 
        start_time: float,
        duration: Optional[float] = None
    ) -> None:
        """
        Schedule an action to be executed.
        
        Args:
            action_id: Unique identifier for the action
            inputs: List of input commands
            start_time: When to start the action
            duration: How long the action should last
        """
        self.action_queue.append({
            'id': action_id,
            'inputs': inputs,
            'start_time': start_time,
            'duration': duration
        })
    
    def get_current_inputs(self, current_time: float) -> List[Dict[str, Any]]:
        """
        Get inputs that should be active at the current time.
        
        Args:
            current_time: Current time
            
        Returns:
            List of input commands to execute
        """
        inputs = []
        
        # Process queued actions
        for action in self.action_queue[:]:
            if current_time >= action['start_time']:
                self.active_actions[action['id']] = action
                self.action_queue.remove(action)
        
        # Get inputs from active actions
        for action_id, action in list(self.active_actions.items()):
            if action['duration'] is None or current_time <= action['start_time'] + action['duration']:
                inputs.extend(action['inputs'])
            else:
                # Action has expired
                del self.active_actions[action_id]
        
        return inputs
    
    def cancel_action(self, action_id: str) -> None:
        """Cancel a scheduled or active action."""
        if action_id in self.active_actions:
            del self.active_actions[action_id]
        
        self.action_queue = [a for a in self.action_queue if a['id'] != action_id]
    
    def clear_all_actions(self) -> None:
        """Clear all scheduled and active actions."""
        self.active_actions.clear()
        self.action_queue.clear()