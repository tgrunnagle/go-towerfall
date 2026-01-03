"""Vectorized environment for parallel TowerFall RL training.

This module provides a vectorized wrapper that manages multiple TowerfallEnv
instances for faster PPO training through parallel environment execution.

The implementation uses asyncio for concurrent execution of environment
operations (reset, step) across all parallel environments.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import gymnasium as gym
import gymnasium.utils.seeding
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from numpy.typing import NDArray

from bot.actions import ACTION_SPACE_SIZE, Action, execute_action
from bot.client import ClientMode, GameClient
from bot.gym.reward import RewardConfig, StandardRewardFunction
from bot.gym.termination import TerminationConfig, TerminationTracker
from bot.models import PlayerStatsDTO
from bot.models.constants import GAME_CONSTANTS
from bot.observation import ObservationBuilder, ObservationConfig


class VectorizedTowerfallEnv(gym.vector.VectorEnv):
    """Vectorized wrapper for running multiple Towerfall environments in parallel.

    This class implements the gymnasium.vector.VectorEnv interface, managing
    multiple game rooms on the server and executing operations concurrently
    via asyncio.

    Features:
    - Configurable number of parallel environments (2, 4, 8, 16+)
    - Concurrent reset and step operations via asyncio
    - Automatic reset of individual environments when they terminate
    - Each environment has its own isolated game room
    - Batched observations with shape (num_envs, obs_dim)

    Example:
        >>> env = VectorizedTowerfallEnv(
        ...     num_envs=4,
        ...     http_url="http://localhost:4000",
        ...     tick_rate_multiplier=10.0,
        ... )
        >>> observations, infos = env.reset()
        >>> actions = np.array([env.single_action_space.sample() for _ in range(4)])
        >>> observations, rewards, terminated, truncated, infos = env.step(actions)
        >>> env.close()

    Attributes:
        num_envs: Number of parallel environments
        single_observation_space: Observation space for a single environment
        single_action_space: Action space for a single environment
        observation_space: Batched observation space
        action_space: Batched action space
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        num_envs: int,
        http_url: str = "http://localhost:4000",
        player_name: str = "MLBot",
        room_name_prefix: str = "Training",
        map_type: str = "default",
        opponent_type: str = "rule_based",
        tick_rate_multiplier: float = 1.0,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        observation_config: ObservationConfig | None = None,
        reward_config: RewardConfig | None = None,
        termination_config: TerminationConfig | None = None,
    ):
        """Initialize the vectorized TowerFall environment.

        Args:
            num_envs: Number of parallel environments to create.
            http_url: Base URL for game server REST API.
            player_name: Name prefix for the RL agent players.
            room_name_prefix: Prefix for training room names.
            map_type: Map to use for training (e.g., "arena1").
            opponent_type: Type of opponent ("rule_based" or "none").
            tick_rate_multiplier: Game speed multiplier (requires server support).
            max_episode_steps: Maximum steps per episode before truncation.
            render_mode: Gymnasium render mode ("human", "rgb_array", or None).
            observation_config: Optional custom observation configuration.
            reward_config: Optional custom reward function configuration.
            termination_config: Optional episode termination configuration.
        """
        self.num_envs = num_envs
        self.http_url = http_url
        self.player_name = player_name
        self.room_name_prefix = room_name_prefix
        self.map_type = map_type
        self.opponent_type = opponent_type
        self.tick_rate_multiplier = tick_rate_multiplier
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Generate unique session ID for this vectorized env instance
        self._session_id = uuid.uuid4().hex[:8]

        # Observation configuration
        self._obs_config = observation_config or ObservationConfig()
        self._obs_builders: list[ObservationBuilder] = [
            ObservationBuilder(self._obs_config) for _ in range(num_envs)
        ]

        # Reward configuration
        self._reward_config = reward_config
        self._reward_fns: list[StandardRewardFunction] = [
            StandardRewardFunction(reward_config) for _ in range(num_envs)
        ]

        # Termination configuration
        if termination_config is None:
            self._termination_config = TerminationConfig(
                max_timesteps=max_episode_steps
            )
        else:
            self._termination_config = termination_config
        self._termination_trackers: list[TerminationTracker] = [
            TerminationTracker(self._termination_config) for _ in range(num_envs)
        ]

        # Define single-environment spaces
        self._single_observation_space: spaces.Box = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._obs_config.total_size,),
            dtype=np.float32,
        )
        self._single_action_space: spaces.Discrete = spaces.Discrete(ACTION_SPACE_SIZE)

        # Set up batched spaces for VectorEnv interface
        # VectorEnv is a Generic class that doesn't have an __init__ with parameters,
        # so we set the required attributes directly
        self.observation_space = batch_space(self._single_observation_space, num_envs)
        self.action_space = batch_space(self._single_action_space, num_envs)
        self.closed = False

        # Runtime state for each environment
        self._clients: list[GameClient | None] = [None] * num_envs
        self._episode_steps: list[int] = [0] * num_envs
        self._prev_player_stats: list[PlayerStatsDTO | None] = [None] * num_envs
        self._prev_opponent_deaths: list[int] = [0] * num_envs

        # Event loop management
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owns_loop = False

    @property
    def single_observation_space(self) -> spaces.Box:
        """Observation space for a single environment."""
        return self._single_observation_space

    @property
    def single_action_space(self) -> spaces.Discrete:
        """Action space for a single environment."""
        return self._single_action_space

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for async operations.

        Note: This method is not thread-safe. The environment should only be
        accessed from a single thread, which is the standard usage pattern for
        gymnasium environments.

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

    def _get_room_name(self, env_idx: int) -> str:
        """Generate unique room name for an environment.

        Args:
            env_idx: Index of the environment.

        Returns:
            Unique room name string.
        """
        return f"{self.room_name_prefix}_{self._session_id}_{env_idx}"

    def _get_player_name(self, env_idx: int) -> str:
        """Generate unique player name for an environment.

        Args:
            env_idx: Index of the environment.

        Returns:
            Unique player name string.
        """
        return f"{self.player_name}_{env_idx}"

    async def _reset_single_env(
        self,
        env_idx: int,
        map_type: str,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset a single environment asynchronously.

        Args:
            env_idx: Index of the environment to reset.
            map_type: Map type to use for this reset.

        Returns:
            Tuple of (observation, info).
        """
        client = self._clients[env_idx]

        # If we already have a client with a room, reset the game
        if client is not None and client.room_id is not None:
            await client.reset_game(map_type=map_type)
            # Wait for game state to be available after reset
            game_state = await client.wait_for_game_state()
        else:
            # Close existing client if any
            if client is not None:
                await client.close()

            # Create new client in REST mode
            client = GameClient(
                http_url=self.http_url,
                mode=ClientMode.REST,
            )
            await client.connect()

            # Create training game with unique room name
            await client.create_game(
                player_name=self._get_player_name(env_idx),
                room_name=self._get_room_name(env_idx),
                map_type=map_type,
                training_mode=True,
                tick_rate_multiplier=self.tick_rate_multiplier,
            )

            self._clients[env_idx] = client
            # Wait for initial state to be available
            game_state = await client.wait_for_game_state()

        # Reset episode tracking
        self._episode_steps[env_idx] = 0
        self._reward_fns[env_idx].reset(None)
        self._termination_trackers[env_idx].reset()
        self._prev_player_stats[env_idx] = None
        self._prev_opponent_deaths[env_idx] = 0

        # Build observation
        if client.player_id is None:
            raise RuntimeError(f"Client {env_idx} not properly initialized after reset")

        observation = self._obs_builders[env_idx].build(
            game_state=game_state,
            own_player_id=client.player_id,
        )

        info: dict[str, Any] = {
            "room_id": client.room_id,
            "room_code": client.room_code,
            "player_id": client.player_id,
            "env_idx": env_idx,
        }

        return observation, info

    async def _step_single_env(
        self,
        env_idx: int,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Step a single environment asynchronously.

        Handles automatic reset when the environment terminates.

        Args:
            env_idx: Index of the environment.
            action: Action to take.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        client = self._clients[env_idx]
        if client is None or client.player_id is None:
            raise RuntimeError(
                f"Environment {env_idx} not initialized. Call reset() first."
            )

        self._episode_steps[env_idx] += 1

        # Execute the action
        action_enum = Action(action)
        await execute_action(client, action_enum)

        # Wait for next game tick (adjusted by tick rate multiplier)
        await asyncio.sleep(
            GAME_CONSTANTS.BASE_TICK_DURATION_SEC / self.tick_rate_multiplier
        )

        # Get new state
        game_state = await client.get_game_state()
        stats = await client.get_stats()

        # Build observation
        observation = self._obs_builders[env_idx].build(
            game_state=game_state,
            own_player_id=client.player_id,
        )

        # Calculate reward
        player_stats = None
        if client.player_id in stats:
            player_stats = stats[client.player_id]
        reward = self._reward_fns[env_idx].calculate(player_stats)

        # Update termination tracking
        tracker = self._termination_trackers[env_idx]
        tracker.increment_timestep()
        self._update_episode_stats(env_idx, stats, client.player_id)

        # Check termination
        is_game_over = game_state.is_game_over
        terminated, truncated = tracker.check_termination(is_game_over)
        termination_reason = tracker.get_termination_reason(terminated, truncated)

        # Build info dict
        episode_stats = tracker.get_episode_stats()
        info: dict[str, Any] = {
            "episode_step": self._episode_steps[env_idx],
            "stats": stats,
            "episode_timesteps": episode_stats["episode_timesteps"],
            "episode_deaths": episode_stats["episode_deaths"],
            "episode_kills": episode_stats["episode_kills"],
            "episode_opponent_deaths": episode_stats["episode_opponent_deaths"],
            "termination_reason": termination_reason,
            "env_idx": env_idx,
        }

        # Handle auto-reset for terminated/truncated environments
        if terminated or truncated:
            # Store terminal observation in info
            info["terminal_observation"] = observation
            # Store final episode info
            info["episode"] = {
                "r": episode_stats["episode_kills"] - episode_stats["episode_deaths"],
                "l": episode_stats["episode_timesteps"],
                "t": self._episode_steps[env_idx],
            }

            # Auto-reset and get new initial observation
            observation, reset_info = await self._reset_single_env(
                env_idx, self.map_type
            )
            # Merge reset info
            info.update({f"reset_{k}": v for k, v in reset_info.items()})

        return observation, reward, terminated, truncated, info

    def _update_episode_stats(
        self,
        env_idx: int,
        stats: dict[str, Any],
        player_id: str,
    ) -> None:
        """Update episode statistics for a single environment.

        Args:
            env_idx: Index of the environment.
            stats: Player statistics dictionary from server.
            player_id: ID of the player.
        """
        current_stats: PlayerStatsDTO | None = None
        if player_id in stats:
            current_stats = stats[player_id]

        if current_stats is None:
            return

        # Get previous values
        prev_stats = self._prev_player_stats[env_idx]
        prev_kills = prev_stats.kills if prev_stats else 0
        prev_deaths = prev_stats.deaths if prev_stats else 0

        # Calculate opponent deaths
        current_opponent_deaths = 0
        for pid, player_stats in stats.items():
            if pid != player_id:
                current_opponent_deaths += player_stats.deaths

        prev_opponent_deaths = self._prev_opponent_deaths[env_idx]

        # Update termination tracker
        self._termination_trackers[env_idx].update_from_stats(
            current_kills=current_stats.kills,
            current_deaths=current_stats.deaths,
            prev_kills=prev_kills,
            prev_deaths=prev_deaths,
            current_opponent_deaths=current_opponent_deaths,
            prev_opponent_deaths=prev_opponent_deaths,
        )

        # Store for next step
        self._prev_player_stats[env_idx] = current_stats
        self._prev_opponent_deaths[env_idx] = current_opponent_deaths

    async def _async_reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Async implementation of reset for all environments.

        Args:
            seed: Random seed (unused, for interface compatibility).
            options: Optional reset configuration.

        Returns:
            Tuple of (batched_observations, infos_dict).
        """
        map_type = self.map_type
        if options and "map_type" in options:
            map_type = options["map_type"]

        # Reset all environments concurrently
        # Use return_exceptions=True so one environment failure doesn't stop others
        tasks = [
            self._reset_single_env(env_idx, map_type)
            for env_idx in range(self.num_envs)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling any exceptions
        processed_results: list[tuple[NDArray[np.float32], dict[str, Any]]] = []
        for env_idx, result in enumerate(results):
            if isinstance(result, BaseException):
                # Environment failed - return zero observation and include error
                obs_size = self._obs_builders[env_idx].config.total_size
                error_obs = np.zeros(obs_size, dtype=np.float32)
                error_info: dict[str, Any] = {
                    "env_idx": env_idx,
                    "error": str(result),
                    "error_type": type(result).__name__,
                }
                processed_results.append((error_obs, error_info))
            else:
                processed_results.append(result)

        # Stack results
        observations = np.stack([obs for obs, _ in processed_results], axis=0)
        infos: dict[str, Any] = {
            "env_infos": [info for _, info in processed_results],
        }

        return observations, infos

    async def _async_step(
        self,
        actions: NDArray[np.int64],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict[str, Any],
    ]:
        """Async implementation of step for all environments.

        Args:
            actions: Array of actions, one per environment.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos).
        """
        # Step all environments concurrently
        # Use return_exceptions=True so one environment failure doesn't stop others
        tasks = [
            self._step_single_env(env_idx, int(actions[env_idx]))
            for env_idx in range(self.num_envs)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling any exceptions
        processed_results: list[
            tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]
        ] = []
        for env_idx, result in enumerate(results):
            if isinstance(result, BaseException):
                # Environment failed - return zero observation, truncate, and include error
                obs_size = self._obs_builders[env_idx].config.total_size
                error_obs = np.zeros(obs_size, dtype=np.float32)
                error_info: dict[str, Any] = {
                    "env_idx": env_idx,
                    "error": str(result),
                    "error_type": type(result).__name__,
                }
                processed_results.append((error_obs, 0.0, False, True, error_info))
            else:
                processed_results.append(result)

        # Stack results
        observations = np.stack([obs for obs, _, _, _, _ in processed_results], axis=0)
        rewards = np.array(
            [reward for _, reward, _, _, _ in processed_results], dtype=np.float32
        )
        terminated = np.array(
            [term for _, _, term, _, _ in processed_results], dtype=np.bool_
        )
        truncated = np.array(
            [trunc for _, _, _, trunc, _ in processed_results], dtype=np.bool_
        )
        infos: dict[str, Any] = {
            "env_infos": [info for _, _, _, _, info in processed_results],
            # Extract terminal observations and episode info for convenience
            "_terminal_observation": [
                info.get("terminal_observation")
                for _, _, _, _, info in processed_results
            ],
            "_episode": [info.get("episode") for _, _, _, _, info in processed_results],
        }

        return observations, rewards, terminated, truncated, infos

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset all environments for new episodes.

        Creates new game rooms or resets existing ones for all environments,
        returning batched initial observations.

        Args:
            seed: Random seed for the environment's numpy random generator.
                This seeds `self.np_random` used for action space sampling and
                any client-side randomization. Game-level randomization (spawns,
                pickups, AI behavior) is handled by the game server independently.
            options: Optional reset configuration. Supported keys:
                - map_type: Override the map type for all environments.

        Returns:
            observations: Batched initial observations of shape (num_envs, obs_dim).
            infos: Dictionary with additional information for each environment.
        """
        # Handle seeding if provided
        if seed is not None:
            self._np_random, self._np_random_seed = gym.utils.seeding.np_random(seed)

        return self._run_async(self._async_reset(seed, options))

    def step(
        self,
        actions: NDArray[np.int64],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict[str, Any],
    ]:
        """Execute actions in all environments and return results.

        Args:
            actions: Array of discrete actions of shape (num_envs,).

        Returns:
            observations: Batched observations of shape (num_envs, obs_dim).
            rewards: Array of rewards of shape (num_envs,).
            terminated: Array of termination flags of shape (num_envs,).
            truncated: Array of truncation flags of shape (num_envs,).
            infos: Dictionary with additional information for each environment.

        Raises:
            ValueError: If actions array has incorrect shape.
        """
        # Validate actions shape
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,):
            raise ValueError(
                f"Expected actions with shape ({self.num_envs},), "
                f"got shape {actions.shape}"
            )

        return self._run_async(self._async_step(actions))

    def render(self) -> tuple[NDArray[np.uint8], ...] | None:
        """Render the environments.

        In 'human' mode: Logs game state to console for first environment.
        In 'rgb_array' mode: Returns placeholder pixel arrays for each env.

        Returns:
            Tuple of render frames if render_mode is 'rgb_array', else None.
        """
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            client = self._clients[0]
            if client is not None:
                print(
                    f"Env 0 - Room: {client.room_code}, Step: {self._episode_steps[0]}"
                )
            return None

        if self.render_mode == "rgb_array":
            # Placeholder - full rendering requires screenshot endpoint
            # Return a tuple of render frames (one per environment)
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            return tuple(frame.copy() for _ in range(self.num_envs))

        return None

    async def _async_close(self) -> None:
        """Async implementation of close for all environments."""
        close_tasks = []
        for client in self._clients:
            if client is not None:
                close_tasks.append(client.close())

        if close_tasks:
            await asyncio.gather(*close_tasks)

    def close_extras(self, **kwargs: Any) -> None:
        """Clean up all environment resources.

        This is called by the parent VectorEnv.close() method.
        """
        if self._loop is None:
            # No event loop, nothing to clean up
            return

        try:
            self._run_async(self._async_close())
        except RuntimeError:
            # Event loop may already be closed
            pass

        # Clear client references
        self._clients = [None] * self.num_envs

        # Clean up event loop if we own it
        if self._loop is not None and self._owns_loop and not self._loop.is_running():
            self._loop.close()
            self._loop = None
