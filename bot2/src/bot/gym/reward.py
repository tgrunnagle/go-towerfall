"""Reward function implementation for TowerFall RL training.

This module provides configurable reward functions for training RL agents.
The reward function incentivizes aggressive play (kills) while penalizing
deaths and encouraging efficient gameplay through time penalties.

Design rationale:
- Kill reward (+1.0): Positive reinforcement for eliminating opponents
- Death penalty (-1.0): Symmetric penalty to optimize kill/death ratio
- Timestep penalty (-0.001): Small negative to discourage passive play

Usage:
    from bot.gym.reward import RewardConfig, StandardRewardFunction

    # Default configuration
    reward_fn = StandardRewardFunction()

    # Custom configuration
    config = RewardConfig(kill_reward=2.0, timestep_penalty=-0.002)
    reward_fn = StandardRewardFunction(config)

    # In training loop
    reward_fn.reset(initial_stats)
    reward = reward_fn.calculate(current_stats)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from bot.models import PlayerStatsDTO


@dataclass
class RewardConfig:
    """Configuration for reward function parameters.

    Attributes:
        kill_reward: Reward for each kill. Default: +1.0
        death_penalty: Penalty for each death (negative value). Default: -1.0
        timestep_penalty: Small penalty per step to encourage efficiency. Default: -0.001
    """

    kill_reward: float = 1.0
    death_penalty: float = -1.0
    timestep_penalty: float = -0.001


class RewardFunction(Protocol):
    """Protocol for reward function implementations.

    This protocol allows for different reward function strategies to be
    used interchangeably in the TowerfallEnv.
    """

    def reset(self, initial_stats: PlayerStatsDTO | None) -> None:
        """Reset tracking for new episode.

        Args:
            initial_stats: Initial player statistics, or None if unavailable.
        """
        ...

    def calculate(self, current_stats: PlayerStatsDTO | None) -> float:
        """Calculate reward based on current statistics.

        Args:
            current_stats: Current player statistics, or None if unavailable.

        Returns:
            Reward value for this step.
        """
        ...


class StandardRewardFunction:
    """Standard reward function for TowerFall training.

    Calculates rewards based on:
    - Kill deltas: +kill_reward for each new kill
    - Death deltas: +death_penalty (negative) for each new death
    - Timestep penalty: Small negative reward each step

    The function tracks previous kill/death counts to calculate deltas,
    ensuring rewards reflect changes rather than absolute values.

    Example:
        >>> config = RewardConfig(kill_reward=1.0, death_penalty=-1.0)
        >>> reward_fn = StandardRewardFunction(config)
        >>> reward_fn.reset(PlayerStatsDTO(player_id="p1", player_name="Bot", kills=0, deaths=0))
        >>> # After getting a kill
        >>> reward = reward_fn.calculate(PlayerStatsDTO(player_id="p1", player_name="Bot", kills=1, deaths=0))
        >>> # reward ≈ 0.999 (1.0 for kill - 0.001 timestep penalty)
    """

    def __init__(self, config: RewardConfig | None = None):
        """Initialize the reward function.

        Args:
            config: Optional reward configuration. Uses defaults if not provided.
        """
        self.config = config or RewardConfig()
        self._prev_kills: int = 0
        self._prev_deaths: int = 0

    def reset(self, initial_stats: PlayerStatsDTO | None) -> None:
        """Reset tracking for new episode.

        Args:
            initial_stats: Initial player statistics, or None if unavailable.
                          If None, previous counts are reset to 0.
        """
        if initial_stats is not None:
            self._prev_kills = initial_stats.kills
            self._prev_deaths = initial_stats.deaths
        else:
            self._prev_kills = 0
            self._prev_deaths = 0

    def calculate(self, current_stats: PlayerStatsDTO | None) -> float:
        """Calculate reward based on current statistics.

        The reward is computed as:
            reward = (kill_delta * kill_reward) +
                     (death_delta * death_penalty) +
                     timestep_penalty

        Args:
            current_stats: Current player statistics, or None if unavailable.

        Returns:
            Reward value for this step. If stats are unavailable,
            returns only the timestep penalty.
        """
        reward = 0.0

        if current_stats is not None:
            current_kills = current_stats.kills
            current_deaths = current_stats.deaths

            # Reward for kills (delta-based)
            kill_diff = current_kills - self._prev_kills
            reward += kill_diff * self.config.kill_reward

            # Penalty for deaths (delta-based)
            death_diff = current_deaths - self._prev_deaths
            reward += death_diff * self.config.death_penalty

            # Update tracking for next step
            self._prev_kills = current_kills
            self._prev_deaths = current_deaths

        # Always apply timestep penalty
        reward += self.config.timestep_penalty

        return reward

    def get_config(self) -> RewardConfig:
        """Get the current reward configuration.

        Returns:
            The RewardConfig instance used by this function.
        """
        return self.config

    def get_episode_stats(self) -> dict[str, Any]:
        """Get current episode tracking statistics.

        Useful for debugging and logging reward calculations.

        Returns:
            Dictionary with current tracked kills and deaths.
        """
        return {
            "prev_kills": self._prev_kills,
            "prev_deaths": self._prev_deaths,
        }
