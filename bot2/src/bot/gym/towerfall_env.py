"""Gymnasium environment for TowerFall RL training.

This module provides a gymnasium.Env subclass that wraps the GameClient
to provide a standard RL environment interface for training agents.
"""

from __future__ import annotations

import asyncio
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from bot.actions import ACTION_SPACE_SIZE, Action, execute_action
from bot.client import ClientMode, GameClient
from bot.models import GameState
from bot.observation import ObservationBuilder, ObservationConfig


class TowerfallEnv(gym.Env[NDArray[np.float32], int]):
    """Gymnasium environment for TowerFall RL training.

    This environment wraps a go-towerfall game instance, providing:
    - Discrete action space (27 actions: movement, aim, shoot, no-op)
    - Box observation space (normalized values in [-1, 1])
    - Synchronous step/reset interface via REST API

    The environment bridges the async GameClient API with gymnasium's
    synchronous interface using an internal event loop.

    Attributes:
        metadata: Gymnasium metadata dict with render modes
        action_space: Discrete(27) action space
        observation_space: Box observation space with normalized values
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        http_url: str = "http://localhost:4000",
        player_name: str = "MLBot",
        room_name: str = "Training",
        map_type: str = "arena1",
        opponent_type: str = "rule_based",
        tick_rate_multiplier: float = 1.0,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        observation_config: ObservationConfig | None = None,
    ):
        """Initialize the TowerFall environment.

        Args:
            http_url: Base URL for game server REST API.
            player_name: Name for the RL agent player.
            room_name: Name for the training room.
            map_type: Map to use for training (e.g., "arena1").
            opponent_type: Type of opponent ("rule_based" or "none").
            tick_rate_multiplier: Game speed multiplier (requires server support).
            max_episode_steps: Maximum steps per episode before truncation.
            render_mode: Gymnasium render mode ("human", "rgb_array", or None).
            observation_config: Optional custom observation configuration.
        """
        super().__init__()

        # Configuration
        self.http_url = http_url
        self.player_name = player_name
        self.room_name = room_name
        self.map_type = map_type
        self.opponent_type = opponent_type
        self.tick_rate_multiplier = tick_rate_multiplier
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Observation configuration
        self._obs_config = observation_config or ObservationConfig()
        self._obs_builder = ObservationBuilder(self._obs_config)

        # Define spaces
        self.action_space: spaces.Discrete = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space: spaces.Box = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._obs_config.total_size,),
            dtype=np.float32,
        )

        # Runtime state
        self._client: GameClient | None = None
        self._episode_step = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owns_loop = False

        # Stats tracking for reward calculation
        self._prev_kills: int = 0
        self._prev_deaths: int = 0

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for async operations.

        Returns:
            An asyncio event loop for running async operations.
        """
        if self._loop is not None and not self._loop.is_closed():
            return self._loop

        try:
            self._loop = asyncio.get_running_loop()
            self._owns_loop = False
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._owns_loop = True

        return self._loop

    def _run_async(self, coro: Any) -> Any:
        """Run async coroutine synchronously.

        Args:
            coro: Async coroutine to run.

        Returns:
            Result of the coroutine.
        """
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment for new episode.

        Creates a new game room or resets the existing one, returning
        the initial observation.

        Args:
            seed: Random seed for reproducibility (passed to parent).
            options: Optional reset configuration. Supported keys:
                - map_type: Override the map type for this episode.

        Returns:
            observation: Initial normalized observation vector.
            info: Dictionary with additional information.
        """
        super().reset(seed=seed)
        self._episode_step = 0
        self._prev_kills = 0
        self._prev_deaths = 0

        # Handle options
        reset_map_type = self.map_type
        if options and "map_type" in options:
            reset_map_type = options["map_type"]

        async def _async_reset() -> tuple[GameState, dict[str, Any]]:
            # If we already have a client with a room, reset the game
            if self._client is not None and self._client.room_id is not None:
                await self._client.reset_game(map_type=reset_map_type)
                game_state = await self._client.get_game_state()
                info = {
                    "room_id": self._client.room_id,
                    "room_code": self._client.room_code,
                    "player_id": self._client.player_id,
                }
                return game_state, info

            # Close existing client if any
            if self._client is not None:
                await self._client.close()

            # Create new client in REST mode
            self._client = GameClient(
                http_url=self.http_url,
                mode=ClientMode.REST,
            )
            await self._client._http_client.connect()

            # Create training game
            await self._client.create_game(
                player_name=self.player_name,
                room_name=self.room_name,
                map_type=reset_map_type,
                training_mode=True,
                tick_rate_multiplier=self.tick_rate_multiplier,
            )

            # Get initial state
            game_state = await self._client.get_game_state()

            info = {
                "room_id": self._client.room_id,
                "room_code": self._client.room_code,
                "player_id": self._client.player_id,
            }
            return game_state, info

        game_state, info = self._run_async(_async_reset())

        # Build observation
        if self._client is None or self._client.player_id is None:
            raise RuntimeError("Client not properly initialized after reset")

        observation = self._obs_builder.build(
            game_state=game_state,
            own_player_id=self._client.player_id,
        )

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Execute action and return results.

        Args:
            action: Discrete action index (0-26).

        Returns:
            observation: Normalized observation after action.
            reward: Reward for this step.
            terminated: Whether episode ended (death, win condition).
            truncated: Whether episode was cut short (max steps).
            info: Additional information dictionary.

        Raises:
            RuntimeError: If environment not initialized (call reset() first).
        """
        if self._client is None or self._client.player_id is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._episode_step += 1

        async def _async_step() -> tuple[GameState, dict[str, Any]]:
            assert self._client is not None

            # Execute the action
            action_enum = Action(action)
            await execute_action(self._client, action_enum)

            # Wait for next game tick
            # Server processes at 50 ticks/sec = 20ms base
            # With tick_rate_multiplier, this is faster
            await asyncio.sleep(0.02 / self.tick_rate_multiplier)

            # Get new state
            game_state = await self._client.get_game_state()
            stats = await self._client.get_stats()

            return game_state, stats

        game_state, stats = self._run_async(_async_step())

        # Build observation
        observation = self._obs_builder.build(
            game_state=game_state,
            own_player_id=self._client.player_id,
        )

        # Calculate reward
        reward = self._calculate_reward(game_state, stats)

        # Check termination
        terminated = self._check_terminated(game_state)
        truncated = self._episode_step >= self.max_episode_steps

        info: dict[str, Any] = {
            "episode_step": self._episode_step,
            "stats": stats,
        }

        return observation, reward, terminated, truncated, info

    def _calculate_reward(
        self,
        game_state: GameState,
        stats: dict[str, Any],
    ) -> float:
        """Calculate reward for current step.

        Basic reward shaping:
        - +1.0 for each kill
        - -1.0 for each death
        - -0.001 per step (encourages efficiency)

        Full reward shaping will be implemented in TASK-015.

        Args:
            game_state: Current game state.
            stats: Player statistics from server.

        Returns:
            Reward value for this step.
        """
        reward = 0.0

        # Get current player stats
        if self._client and self._client.player_id and self._client.player_id in stats:
            player_stats = stats[self._client.player_id]
            current_kills = player_stats.kills
            current_deaths = player_stats.deaths

            # Reward for kills
            kill_diff = current_kills - self._prev_kills
            reward += kill_diff * 1.0

            # Penalty for deaths
            death_diff = current_deaths - self._prev_deaths
            reward -= death_diff * 1.0

            # Update tracking
            self._prev_kills = current_kills
            self._prev_deaths = current_deaths

        # Small negative reward per step to encourage efficiency
        reward -= 0.001

        return reward

    def _check_terminated(self, game_state: GameState) -> bool:
        """Check if episode should terminate.

        Episode terminates when:
        - Own player is dead
        - Player not found in game state (disconnected)

        Full termination logic will be implemented in TASK-016.

        Args:
            game_state: Current game state.

        Returns:
            True if episode should terminate.
        """
        if self._client is None or self._client.player_id is None:
            return True

        # Check if own player exists and is dead
        own_player = game_state.players.get(self._client.player_id)

        if own_player is None:
            return True  # Player not found, terminate

        return own_player.dead

    def render(self) -> NDArray[np.uint8] | None:
        """Render the environment.

        In 'human' mode: Logs game state to console.
        In 'rgb_array' mode: Returns placeholder pixel array.

        Returns:
            RGB array if render_mode is 'rgb_array', else None.
        """
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            if self._client is not None:
                print(f"Room: {self._client.room_code}, Step: {self._episode_step}")
            return None

        if self.render_mode == "rgb_array":
            # Placeholder - full rendering requires screenshot endpoint
            return np.zeros((600, 800, 3), dtype=np.uint8)

        return None

    def close(self) -> None:
        """Clean up environment resources."""
        if self._client is not None:
            self._run_async(self._client.close())
            self._client = None

        if self._loop is not None and self._owns_loop and not self._loop.is_running():
            self._loop.close()
            self._loop = None
