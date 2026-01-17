"""Rolling statistics aggregation for training metrics.

This module provides classes for computing rolling statistics over
a window of episodes, useful for monitoring training progress.
"""

import math
from collections import deque
from datetime import datetime

from bot.training.metrics.models import AggregateMetrics, EpisodeMetrics


class RollingAggregator:
    """Computes rolling statistics over a window of episodes.

    Maintains a circular buffer of episode metrics and computes
    aggregate statistics on demand.

    Attributes:
        window_size: Maximum number of episodes to keep in the buffer
    """

    def __init__(self, window_size: int = 100) -> None:
        """Initialize the rolling aggregator.

        Args:
            window_size: Maximum episodes in the rolling window (default 100)
        """
        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        self.window_size = window_size
        self._episodes: deque[EpisodeMetrics] = deque(maxlen=window_size)

    def add_episode(self, episode: EpisodeMetrics) -> None:
        """Add an episode to the rolling window.

        Args:
            episode: Episode metrics to add
        """
        self._episodes.append(episode)

    def get_aggregate(self) -> AggregateMetrics | None:
        """Compute aggregate statistics over the current window.

        Returns:
            AggregateMetrics if there are episodes, None otherwise
        """
        if not self._episodes:
            return None

        rewards = [e.total_reward for e in self._episodes]
        lengths = [e.length for e in self._episodes]
        kills = [e.kills for e in self._episodes]
        deaths = [e.deaths for e in self._episodes]
        wins = [1 if e.win else 0 for e in self._episodes]

        mean_reward = sum(rewards) / len(rewards)
        mean_kills = sum(kills) / len(kills)
        mean_deaths = sum(deaths) / len(deaths)

        # Compute standard deviation
        std_reward = self._compute_std(rewards, mean_reward)

        # Compute aggregate K/D ratio (total kills / total deaths)
        total_deaths = sum(deaths)
        if total_deaths > 0:
            kd_ratio = sum(kills) / total_deaths
        else:
            kd_ratio = float(sum(kills)) if sum(kills) > 0 else 0.0

        return AggregateMetrics(
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_length=sum(lengths) / len(lengths),
            win_rate=sum(wins) / len(wins),
            mean_kills=mean_kills,
            mean_deaths=mean_deaths,
            kd_ratio=kd_ratio,
            episodes_count=len(self._episodes),
            timestamp=datetime.now(),
        )

    def _compute_std(self, values: list[float], mean: float) -> float:
        """Compute standard deviation.

        Args:
            values: List of values
            mean: Pre-computed mean

        Returns:
            Standard deviation of values
        """
        if len(values) < 2:
            return 0.0

        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def clear(self) -> None:
        """Clear all episodes from the buffer."""
        self._episodes.clear()

    def __len__(self) -> int:
        """Return the number of episodes in the buffer."""
        return len(self._episodes)
