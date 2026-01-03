"""Episode termination logic for TowerFall RL training.

This module provides configurable episode termination conditions for training
reinforcement learning agents. Episodes can terminate based on:
- Time limits (max timesteps)
- Death thresholds
- Kill targets
- Score differences
- Game-level signals

The termination logic follows Gymnasium's terminated/truncated convention:
- terminated: Episode ended naturally (death threshold, game over, target achieved)
- truncated: Episode cut short artificially (max timesteps reached)

Usage:
    from bot.gym.termination import TerminationConfig

    # Default configuration (max 10,000 timesteps, 5 deaths to terminate)
    config = TerminationConfig()

    # Custom: shorter episodes, end on first kill
    config = TerminationConfig(
        max_timesteps=1000,
        max_deaths=None,  # Disable death termination
        target_kills=1,   # End when agent gets a kill
    )

    env = TowerfallEnv(termination_config=config)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TerminationConfig:
    """Configuration for episode termination conditions.

    All conditions are optional. When multiple conditions are enabled,
    the episode terminates when ANY condition is met (OR logic).

    Attributes:
        max_timesteps: Maximum timesteps per episode before truncation.
            At 50 ticks/sec, 10,000 ticks = 200 seconds.
            Set to None to disable (not recommended for training).
        max_deaths: End episode after N agent deaths. Useful for limiting
            negative experiences. Set to None to disable.
        max_opponent_deaths: End episode after opponent dies N times.
            Useful for target practice scenarios. Set to None to disable.
        target_kills: End episode when agent achieves N kills.
            Useful for goal-conditioned training. Set to None to disable.
        target_score_diff: End episode when |kills - deaths| reaches threshold.
            Useful for competitive scenarios. Set to None to disable.
        use_game_over_signal: Honor game's own termination signal (round/match over).
            This catches server-side game endings.
    """

    # Time-based termination
    max_timesteps: int = 10_000  # ~200 seconds at 50 ticks/sec

    # Death-based termination
    max_deaths: int | None = 5  # End episode after N agent deaths
    max_opponent_deaths: int | None = None  # End if opponent dies N times

    # Kill-based termination
    target_kills: int | None = None  # End when agent achieves N kills

    # Score-based termination
    target_score_diff: int | None = None  # End when |kills - deaths| reaches threshold

    # Game-level signals
    use_game_over_signal: bool = True  # Honor game's own termination signal

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_timesteps is not None and self.max_timesteps <= 0:
            raise ValueError("max_timesteps must be positive or None")
        if self.max_deaths is not None and self.max_deaths <= 0:
            raise ValueError("max_deaths must be positive or None")
        if self.max_opponent_deaths is not None and self.max_opponent_deaths <= 0:
            raise ValueError("max_opponent_deaths must be positive or None")
        if self.target_kills is not None and self.target_kills <= 0:
            raise ValueError("target_kills must be positive or None")
        if self.target_score_diff is not None and self.target_score_diff <= 0:
            raise ValueError("target_score_diff must be positive or None")


class TerminationTracker:
    """Tracks episode statistics for termination decisions.

    This class maintains running counts of deaths, kills, and timesteps
    for the current episode, and provides methods to check termination
    conditions and report termination reasons.

    Usage:
        tracker = TerminationTracker(config)
        tracker.reset()

        # In step loop:
        tracker.update(current_stats, previous_stats, opponent_ids)
        terminated, truncated = tracker.check_termination(game_state)
        reason = tracker.get_termination_reason(terminated, truncated)
    """

    def __init__(self, config: TerminationConfig | None = None):
        """Initialize the termination tracker.

        Args:
            config: Termination configuration. Uses defaults if not provided.
        """
        self.config = config or TerminationConfig()
        self._episode_timesteps: int = 0
        self._episode_deaths: int = 0
        self._episode_kills: int = 0
        self._episode_opponent_deaths: int = 0

    def reset(self) -> None:
        """Reset all counters for a new episode."""
        self._episode_timesteps = 0
        self._episode_deaths = 0
        self._episode_kills = 0
        self._episode_opponent_deaths = 0

    def increment_timestep(self) -> None:
        """Increment the timestep counter."""
        self._episode_timesteps += 1

    def update_from_stats(
        self,
        current_kills: int,
        current_deaths: int,
        prev_kills: int,
        prev_deaths: int,
        current_opponent_deaths: int = 0,
        prev_opponent_deaths: int = 0,
    ) -> None:
        """Update episode statistics from kill/death counts.

        Args:
            current_kills: Agent's current total kills.
            current_deaths: Agent's current total deaths.
            prev_kills: Agent's kills at previous step.
            prev_deaths: Agent's deaths at previous step.
            current_opponent_deaths: Sum of opponent deaths (current).
            prev_opponent_deaths: Sum of opponent deaths (previous).
        """
        # Calculate deltas
        new_kills = current_kills - prev_kills
        new_deaths = current_deaths - prev_deaths
        new_opponent_deaths = current_opponent_deaths - prev_opponent_deaths

        # Update episode totals
        self._episode_kills += new_kills
        self._episode_deaths += new_deaths
        self._episode_opponent_deaths += new_opponent_deaths

    def check_termination(self, is_game_over: bool = False) -> tuple[bool, bool]:
        """Check if episode should terminate.

        Returns:
            Tuple of (terminated, truncated):
            - terminated: True if episode ended naturally
            - truncated: True if episode was cut short (max timesteps)
        """
        config = self.config

        # Check truncation (time limit) - this takes precedence
        if config.max_timesteps is not None:
            if self._episode_timesteps >= config.max_timesteps:
                return False, True

        # Check natural termination conditions
        terminated = False

        # Death threshold
        if config.max_deaths is not None:
            if self._episode_deaths >= config.max_deaths:
                terminated = True

        # Opponent death threshold
        if config.max_opponent_deaths is not None:
            if self._episode_opponent_deaths >= config.max_opponent_deaths:
                terminated = True

        # Kill target
        if config.target_kills is not None:
            if self._episode_kills >= config.target_kills:
                terminated = True

        # Score difference target
        if config.target_score_diff is not None:
            score_diff = self._episode_kills - self._episode_deaths
            if abs(score_diff) >= config.target_score_diff:
                terminated = True

        # Game-level signal
        if config.use_game_over_signal and is_game_over:
            terminated = True

        return terminated, False

    def get_termination_reason(self, terminated: bool, truncated: bool) -> str | None:
        """Get human-readable termination reason.

        Args:
            terminated: Whether episode terminated naturally.
            truncated: Whether episode was truncated.

        Returns:
            String describing why episode ended, or None if still running.
        """
        if truncated:
            return "max_timesteps"

        if not terminated:
            return None

        config = self.config

        # Check each condition in priority order
        if config.max_deaths is not None:
            if self._episode_deaths >= config.max_deaths:
                return "max_deaths"

        if config.max_opponent_deaths is not None:
            if self._episode_opponent_deaths >= config.max_opponent_deaths:
                return "opponent_deaths"

        if config.target_kills is not None:
            if self._episode_kills >= config.target_kills:
                return "target_kills"

        if config.target_score_diff is not None:
            score_diff = self._episode_kills - self._episode_deaths
            if abs(score_diff) >= config.target_score_diff:
                return "score_diff"

        # If we get here, it must be game_over signal
        return "game_over"

    @property
    def episode_timesteps(self) -> int:
        """Get current episode timestep count."""
        return self._episode_timesteps

    @property
    def episode_deaths(self) -> int:
        """Get current episode death count."""
        return self._episode_deaths

    @property
    def episode_kills(self) -> int:
        """Get current episode kill count."""
        return self._episode_kills

    @property
    def episode_opponent_deaths(self) -> int:
        """Get current episode opponent death count."""
        return self._episode_opponent_deaths

    def get_episode_stats(self) -> dict[str, int]:
        """Get all episode statistics as a dictionary.

        Returns:
            Dictionary with timesteps, deaths, kills, and opponent_deaths.
        """
        return {
            "episode_timesteps": self._episode_timesteps,
            "episode_deaths": self._episode_deaths,
            "episode_kills": self._episode_kills,
            "episode_opponent_deaths": self._episode_opponent_deaths,
        }
