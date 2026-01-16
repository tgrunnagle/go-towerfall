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
from bot.gym.opponent_manager import OpponentProtocol, create_opponent
from bot.gym.reward import RewardConfig, StandardRewardFunction
from bot.gym.termination import TerminationConfig, TerminationTracker
from bot.models import GameState, PlayerStatsDTO
from bot.models.constants import GAME_CONSTANTS
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
        ws_url: str = "ws://localhost:4000/ws",
        player_name: str = "MLBot",
        room_name: str = "Training",
        map_type: str = "default",
        opponent_type: str = "rule_based",
        opponent_name: str = "RuleBot",
        tick_rate_multiplier: float = 1.0,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        observation_config: ObservationConfig | None = None,
        reward_config: RewardConfig | None = None,
        termination_config: TerminationConfig | None = None,
    ):
        """Initialize the TowerFall environment.

        Args:
            http_url: Base URL for game server REST API.
            ws_url: WebSocket URL for game server.
            player_name: Name for the RL agent player.
            room_name: Name for the training room.
            map_type: Map to use for training (e.g., "arena1").
            opponent_type: Type of opponent ("rule_based" or "none").
            opponent_name: Name for the opponent player.
            tick_rate_multiplier: Game speed multiplier (requires server support).
            max_episode_steps: Maximum steps per episode before truncation.
                Note: This is deprecated in favor of termination_config.max_timesteps.
                If termination_config is provided, its max_timesteps takes precedence.
            render_mode: Gymnasium render mode ("human", "rgb_array", or None).
            observation_config: Optional custom observation configuration.
            reward_config: Optional custom reward function configuration.
            termination_config: Optional episode termination configuration.
                Controls when episodes end based on timesteps, deaths, kills, etc.
        """
        super().__init__()

        # Configuration
        self.http_url = http_url
        self.ws_url = ws_url
        self.player_name = player_name
        self.room_name = room_name
        self.map_type = map_type
        self.opponent_type = opponent_type
        self.opponent_name = opponent_name
        self.tick_rate_multiplier = tick_rate_multiplier
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Observation configuration
        self._obs_config = observation_config or ObservationConfig()
        self._obs_builder = ObservationBuilder(self._obs_config)

        # Reward configuration
        self._reward_fn = StandardRewardFunction(reward_config)

        # Termination configuration
        # If no termination config provided, create one using max_episode_steps for backward compatibility
        if termination_config is None:
            self._termination_config = TerminationConfig(
                max_timesteps=max_episode_steps
            )
        else:
            self._termination_config = termination_config
        self._termination_tracker = TerminationTracker(self._termination_config)

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
        self._opponent: OpponentProtocol | None = None
        self._episode_step = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owns_loop = False

        # Stats tracking for termination logic
        self._prev_player_stats: PlayerStatsDTO | None = None
        self._prev_opponent_deaths: int = 0

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

        Handles the case when called from within an already-running event loop
        (e.g., during pytest-asyncio tests) by running the coroutine in a
        separate thread with its own event loop.

        Args:
            coro: Async coroutine to run.

        Returns:
            Result of the coroutine.
        """
        # Check if there's already a running event loop
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None:
            # We're inside an async context - run in a separate thread
            import concurrent.futures

            def run_in_thread() -> Any:
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                return future.result()

        # No running loop - use the normal approach
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    async def _async_reset(
        self, reset_map_type: str
    ) -> tuple[GameState, dict[str, Any]]:
        """Async implementation of reset logic.

        Args:
            reset_map_type: Map type to use for this reset.

        Returns:
            Tuple of (game_state, info_dict).
        """
        # If we already have a client with a room, reset the game
        if self._client is not None and self._client.room_id is not None:
            # Reset opponent state for new episode
            if self._opponent is not None:
                self._opponent.reset()

            await self._client.reset_game(map_type=reset_map_type)
            # Wait for game state to be available after reset
            game_state = await self._client.wait_for_game_state()
            info = {
                "room_id": self._client.room_id,
                "room_code": self._client.room_code,
                "player_id": self._client.player_id,
                "opponent_type": self.opponent_type,
            }
            return game_state, info

        # Stop existing opponent if any
        if self._opponent is not None:
            await self._opponent.stop()
            self._opponent = None

        # Close existing client if any
        if self._client is not None:
            await self._client.close()

        # Create new client in REST mode
        self._client = GameClient(
            http_url=self.http_url,
            ws_url=self.ws_url,
            mode=ClientMode.REST,
        )
        await self._client.connect()

        # Create training game
        await self._client.create_game(
            player_name=self.player_name,
            room_name=self.room_name,
            map_type=reset_map_type,
            training_mode=True,
            tick_rate_multiplier=self.tick_rate_multiplier,
        )

        # Create and start opponent
        self._opponent = create_opponent(
            opponent_type=self.opponent_type,
            http_url=self.http_url,
            ws_url=self.ws_url,
            player_name=self.opponent_name,
        )

        # For real opponents, get player count before starting, then wait for +1
        if self.opponent_type != "none":
            initial_state = await self._client.wait_for_game_state()
            initial_player_count = len(initial_state.players)

            await self._opponent.start(
                room_code=self._client.room_code or "",
                room_password=self._client.room_password or "",
            )

            # Wait for opponent to connect (player count increases by 1)
            game_state = await self._wait_for_player_count(initial_player_count + 1)
        else:
            # No opponent - just wait for initial state
            await self._opponent.start(
                room_code=self._client.room_code or "",
                room_password=self._client.room_password or "",
            )
            game_state = await self._client.wait_for_game_state()

        info = {
            "room_id": self._client.room_id,
            "room_code": self._client.room_code,
            "player_id": self._client.player_id,
            "opponent_type": self.opponent_type,
        }
        return game_state, info

    async def _wait_for_player_count(
        self,
        expected_count: int,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> GameState:
        """Wait for expected number of players to be in game state.

        Polls the game state until the player count reaches the expected value.
        For no-opponent mode (expected_count equals current), returns immediately.

        Args:
            expected_count: Number of players to wait for.
            timeout: Maximum time to wait in seconds.
            poll_interval: Time between polls in seconds.

        Returns:
            GameState with expected player count.

        Raises:
            TimeoutError: If player count doesn't reach expected within timeout.
        """
        assert self._client is not None

        elapsed = 0.0
        game_state: GameState | None = None
        while elapsed < timeout:
            remaining = max(0.1, timeout - elapsed)
            game_state = await self._client.wait_for_game_state(timeout=remaining)
            if len(game_state.players) >= expected_count:
                return game_state
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        player_count = len(game_state.players) if game_state else 0
        raise TimeoutError(
            f"Expected {expected_count} players but only {player_count} connected "
            f"within {timeout}s."
        )

    async def _async_step(self, action: int) -> tuple[GameState, dict[str, Any]]:
        """Async implementation of step logic.

        Args:
            action: Discrete action index (0-26).

        Returns:
            Tuple of (game_state, stats_dict).
        """
        assert self._client is not None

        # Execute the action
        action_enum = Action(action)
        await execute_action(self._client, action_enum)

        # Wait for next game tick (adjusted by tick rate multiplier)
        await asyncio.sleep(
            GAME_CONSTANTS.BASE_TICK_DURATION_SEC / self.tick_rate_multiplier
        )

        # Get new state
        game_state = await self._client.get_game_state()
        stats = await self._client.get_stats()

        # Update opponent with current state (for synchronization)
        if self._opponent is not None:
            await self._opponent.on_game_state(game_state)

        return game_state, stats

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
            seed: Random seed for the environment's numpy random generator.
                This seeds `self.np_random` used for action space sampling and
                any client-side randomization. Game-level randomization (spawns,
                pickups, AI behavior) is handled by the game server independently.
            options: Optional reset configuration. Supported keys:
                - map_type: Override the map type for this episode.

        Returns:
            observation: Initial normalized observation vector.
            info: Dictionary with additional information.
        """
        super().reset(seed=seed)
        self._episode_step = 0

        # Handle options
        reset_map_type = self.map_type
        if options and "map_type" in options:
            reset_map_type = options["map_type"]

        game_state, info = self._run_async(self._async_reset(reset_map_type))

        # Reset reward function tracking
        self._reward_fn.reset(None)

        # Reset termination tracking
        self._termination_tracker.reset()
        self._prev_player_stats = None
        self._prev_opponent_deaths = 0

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

        game_state, stats = self._run_async(self._async_step(action))

        # Build observation
        observation = self._obs_builder.build(
            game_state=game_state,
            own_player_id=self._client.player_id,
        )

        # Calculate reward
        reward = self._calculate_reward(game_state, stats)

        # Update termination tracking
        self._termination_tracker.increment_timestep()
        self._update_episode_stats(stats)

        # Check termination using the tracker
        is_game_over = self._check_game_over(game_state)
        terminated, truncated = self._termination_tracker.check_termination(
            is_game_over
        )
        termination_reason = self._termination_tracker.get_termination_reason(
            terminated, truncated
        )

        # Build info dict with episode stats
        episode_stats = self._termination_tracker.get_episode_stats()
        info: dict[str, Any] = {
            "episode_step": self._episode_step,
            "stats": stats,
            "episode_timesteps": episode_stats["episode_timesteps"],
            "episode_deaths": episode_stats["episode_deaths"],
            "episode_kills": episode_stats["episode_kills"],
            "episode_opponent_deaths": episode_stats["episode_opponent_deaths"],
            "termination_reason": termination_reason,
        }

        return observation, reward, terminated, truncated, info

    def _calculate_reward(
        self,
        game_state: GameState,
        stats: dict[str, Any],
    ) -> float:
        """Calculate reward for current step using the reward function.

        Uses StandardRewardFunction with configurable reward shaping:
        - +kill_reward for each kill (default: +1.0)
        - +death_penalty for each death (default: -1.0)
        - +timestep_penalty per step (default: -0.001)

        Args:
            game_state: Current game state.
            stats: Player statistics from server.

        Returns:
            Reward value for this step.
        """
        # Get current player stats
        player_stats = None
        if self._client and self._client.player_id and self._client.player_id in stats:
            player_stats = stats[self._client.player_id]

        return self._reward_fn.calculate(player_stats)

    def _update_episode_stats(self, stats: dict[str, Any]) -> None:
        """Update episode statistics from game stats.

        Tracks kill/death deltas for termination logic by comparing
        current stats with previous stats.

        Args:
            stats: Player statistics dictionary from server.
        """
        if self._client is None or self._client.player_id is None:
            return

        # Get current player stats
        current_stats: PlayerStatsDTO | None = None
        if self._client.player_id in stats:
            current_stats = stats[self._client.player_id]

        if current_stats is None:
            return

        # Get previous values (default to 0 if first step)
        prev_kills = self._prev_player_stats.kills if self._prev_player_stats else 0
        prev_deaths = self._prev_player_stats.deaths if self._prev_player_stats else 0

        # Calculate opponent deaths
        current_opponent_deaths = 0
        prev_opponent_deaths_total = self._prev_opponent_deaths
        for player_id, player_stats in stats.items():
            if player_id != self._client.player_id:
                current_opponent_deaths += player_stats.deaths

        # Update termination tracker
        self._termination_tracker.update_from_stats(
            current_kills=current_stats.kills,
            current_deaths=current_stats.deaths,
            prev_kills=prev_kills,
            prev_deaths=prev_deaths,
            current_opponent_deaths=current_opponent_deaths,
            prev_opponent_deaths=prev_opponent_deaths_total,
        )

        # Store for next step
        self._prev_player_stats = current_stats
        self._prev_opponent_deaths = current_opponent_deaths

    def _check_game_over(self, game_state: GameState) -> bool:
        """Check if game signals episode should end.

        This checks game-level termination signals that indicate
        the match or round is over, independent of death/kill counts.

        Args:
            game_state: Current game state.

        Returns:
            True if game signals it's over.
        """
        # Use the is_game_over signal from the server (training_complete)
        return game_state.is_game_over

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

    async def _async_close(self) -> None:
        """Async implementation of close logic."""
        if self._opponent is not None:
            await self._opponent.stop()
            self._opponent = None

        if self._client is not None:
            await self._client.close()
            self._client = None

    def close(self) -> None:
        """Clean up environment resources."""
        self._run_async(self._async_close())

        if self._loop is not None and self._owns_loop and not self._loop.is_running():
            self._loop.close()
            self._loop = None
