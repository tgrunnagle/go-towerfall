"""
Base game environment wrapper for RL training.

This module implements the gymnasium.Env interface to wrap the existing GameClient
for reinforcement learning training. It provides state extraction, action execution,
and reward calculation functionality.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum

import gymnasium as gym
from gymnasium import spaces

from rl_bot_system.utils.async_env import AsyncEnvWrapper
from core.game_client import GameClient


class TrainingMode(Enum):
    """Training mode enumeration"""
    TRAINING = "training"  # Accelerated mode for training
    EVALUATION = "evaluation"  # Real-time mode for evaluation


class GameEnvironment(gym.Env):
    """
    Base game environment that implements gymnasium.Env interface.
    
    This class wraps the GameClient to provide a standard RL environment interface
    with configurable state representations, action spaces, and reward functions.
    """
    
    def __init__(
        self,
        game_client: GameClient,
        state_processor=None,
        action_space_config=None,
        reward_function=None,
        training_mode: TrainingMode = TrainingMode.TRAINING,
        max_episode_steps: int = 1000,
        **kwargs
    ):
        """
        Initialize the game environment.
        
        Args:
            game_client: GameClient instance for game interaction
            state_processor: StateProcessor for state representation
            action_space_config: Configuration for action space
            reward_function: RewardFunction for reward calculation
            training_mode: Training or evaluation mode
            max_episode_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.game_client = game_client
        self.training_mode = training_mode
        self.max_episode_steps = max_episode_steps
        self._logger = logging.getLogger(__name__)
        
        # Environment state
        self.current_step = 0
        self.episode_reward = 0.0
        self.game_state = {}
        self.last_action = None
        self.episode_done = False
        
        # Initialize components with defaults if not provided
        self.state_processor = state_processor or self._create_default_state_processor()
        self.action_space_config = action_space_config or self._create_default_action_space()
        self.reward_function = reward_function or self._create_default_reward_function()
        
        # Set up observation and action spaces
        self.observation_space = self.state_processor.get_observation_space()
        self.action_space = self.action_space_config.get_action_space()
        
        # Message handling
        self.game_client.register_message_handler(self._handle_game_message)
        
        # Episode tracking
        self.episode_stats = {
            'total_reward': 0.0,
            'steps': 0,
            'kills': 0,
            'deaths': 0,
            'shots_fired': 0,
            'shots_hit': 0,
            'damage_dealt': 0,
            'damage_taken': 0
        }
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_done = False
        self.last_action = None
        
        # Reset episode statistics
        self.episode_stats = {
            'total_reward': 0.0,
            'steps': 0,
            'kills': 0,
            'deaths': 0,
            'shots_fired': 0,
            'shots_hit': 0,
            'damage_dealt': 0,
            'damage_taken': 0
        }
        
        # Configure training mode
        self._configure_training_mode()
        
        # Wait for initial game state
        initial_state = self._wait_for_game_state()
        
        # Process initial observation
        observation = self.state_processor.process_state(initial_state)
        
        info = {
            'episode_step': self.current_step,
            'training_mode': self.training_mode.value,
            'raw_state': initial_state
        }
        
        self._logger.debug(f"Environment reset - Mode: {self.training_mode.value}")
        
        return observation, info
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        self.current_step += 1
        self.last_action = action
        
        # Execute action through game client
        self._execute_action(action)
        
        # Wait for game state update
        new_state = self._wait_for_game_state()
        
        # Process observation
        observation = self.state_processor.process_state(new_state)
        
        # Calculate reward
        reward = self.reward_function.calculate_reward(
            state=new_state,
            action=action,
            previous_state=self.game_state,
            episode_stats=self.episode_stats
        )
        
        self.episode_reward += reward
        self.episode_stats['total_reward'] = self.episode_reward
        self.episode_stats['steps'] = self.current_step
        
        # Check if episode is done
        terminated = self._check_terminated(new_state)
        truncated = self.current_step >= self.max_episode_steps
        
        self.episode_done = terminated or truncated
        
        # Update game state
        self.game_state = new_state
        
        info = {
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'raw_state': new_state,
            'episode_stats': self.episode_stats.copy(),
            'action_executed': action
        }
        
        if self.episode_done:
            info['episode_length'] = self.current_step
            info['final_reward'] = self.episode_reward
            self._logger.info(f"Episode finished - Steps: {self.current_step}, Reward: {self.episode_reward:.2f}")
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered frame if mode is "rgb_array", None otherwise
        """
        if mode == "human":
            # In human mode, the game client handles rendering
            return None
        elif mode == "rgb_array":
            # For rgb_array mode, we would need to capture the game screen
            # This is not implemented in the base version
            self._logger.warning("RGB array rendering not implemented")
            return None
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        asyncio.create_task(self.game_client.close())
        self._logger.info("Environment closed")
    
    def set_training_mode(self, mode: TrainingMode) -> None:
        """
        Set the training mode.
        
        Args:
            mode: Training mode to set
        """
        self.training_mode = mode
        self._configure_training_mode()
        self._logger.info(f"Training mode set to: {mode.value}")
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get current episode statistics."""
        return self.episode_stats.copy()
    
    def _configure_training_mode(self) -> None:
        """Configure environment based on training mode."""
        if self.training_mode == TrainingMode.TRAINING:
            # Configure for accelerated training
            # This would integrate with the game speed controller
            pass
        elif self.training_mode == TrainingMode.EVALUATION:
            # Configure for real-time evaluation
            pass
    
    def _execute_action(self, action: Union[int, np.ndarray]) -> None:
        """
        Execute action through the game client.
        
        Args:
            action: Action to execute
        """
        # Convert action to game client inputs
        game_inputs = self.action_space_config.action_to_inputs(action)
        
        # Execute inputs through game client
        for input_command in game_inputs:
            asyncio.create_task(self._send_input(input_command))
    
    async def _send_input(self, input_command: Dict[str, Any]) -> None:
        """
        Send input command to game client.
        
        Args:
            input_command: Input command to send
        """
        input_type = input_command.get('type')
        
        if input_type == 'keyboard':
            await self.game_client.send_keyboard_input(
                key=input_command['key'],
                pressed=input_command['pressed']
            )
        elif input_type == 'mouse':
            await self.game_client.send_mouse_input(
                button=input_command['button'],
                pressed=input_command['pressed'],
                x=input_command['x'],
                y=input_command['y']
            )
    
    def _wait_for_game_state(self) -> Dict[str, Any]:
        """
        Wait for and return the current game state.
        
        Returns:
            Current game state
        """
        # This is a simplified implementation
        # In a real scenario, this would wait for actual game state updates
        return self._get_mock_game_state()
    
    def _get_mock_game_state(self) -> Dict[str, Any]:
        """
        Get mock game state for testing.
        
        Returns:
            Mock game state
        """
        return {
            'player': {
                'id': 'rl_bot',
                'position': {'x': 100.0, 'y': 100.0},
                'velocity': {'x': 0.0, 'y': 0.0},
                'health': 100,
                'max_health': 100,
                'ammunition': 10,
                'is_alive': True
            },
            'enemies': [
                {
                    'id': 'enemy_1',
                    'position': {'x': 200.0, 'y': 150.0},
                    'velocity': {'x': 5.0, 'y': 0.0},
                    'health': 80,
                    'is_alive': True,
                    'has_line_of_sight': True
                }
            ],
            'projectiles': [],
            'power_ups': [
                {
                    'type': 'health',
                    'position': {'x': 150.0, 'y': 200.0}
                }
            ],
            'boundaries': {
                'left': -400,
                'right': 400,
                'top': -300,
                'bottom': 300
            },
            'game_time': self.current_step * 0.1,  # Mock time progression
            'round_active': True
        }
    
    def _check_terminated(self, state: Dict[str, Any]) -> bool:
        """
        Check if episode should be terminated.
        
        Args:
            state: Current game state
            
        Returns:
            True if episode should be terminated
        """
        player = state.get('player', {})
        
        # Episode ends if player dies
        if not player.get('is_alive', True):
            return True
        
        # Episode ends if round is not active
        if not state.get('round_active', True):
            return True
        
        return False
    
    async def _handle_game_message(self, message: Dict[str, Any]) -> None:
        """
        Handle incoming game messages.
        
        Args:
            message: Game message
        """
        message_type = message.get('type', '')
        
        if message_type == 'game_state_update':
            self.game_state = message.get('state', {})
        
        elif message_type == 'player_killed':
            victim = message.get('victim')
            killer = message.get('killer')
            
            if victim == self.game_client.player_id:
                self.episode_stats['deaths'] += 1
            elif killer == self.game_client.player_id:
                self.episode_stats['kills'] += 1
        
        elif message_type == 'shot_fired':
            shooter = message.get('shooter')
            if shooter == self.game_client.player_id:
                self.episode_stats['shots_fired'] += 1
        
        elif message_type == 'shot_hit':
            shooter = message.get('shooter')
            damage = message.get('damage', 0)
            if shooter == self.game_client.player_id:
                self.episode_stats['shots_hit'] += 1
                self.episode_stats['damage_dealt'] += damage
        
        elif message_type == 'damage_taken':
            victim = message.get('victim')
            damage = message.get('damage', 0)
            if victim == self.game_client.player_id:
                self.episode_stats['damage_taken'] += damage
    
    def _create_default_state_processor(self):
        """Create default state processor."""
        from rl_bot_system.environment.state_processors import RawCoordinateProcessor
        return RawCoordinateProcessor()
    
    def _create_default_action_space(self):
        """Create default action space configuration."""
        from rl_bot_system.environment.action_spaces import DiscreteActionSpace
        return DiscreteActionSpace()
    
    def _create_default_reward_function(self):
        """Create default reward function."""
        from rl_bot_system.environment.reward_functions import SparseRewardFunction
        return SparseRewardFunction()