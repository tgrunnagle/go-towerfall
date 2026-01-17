"""Unit tests for rolling aggregators."""

import math

import pytest

from bot.training.metrics.aggregators import RollingAggregator
from bot.training.metrics.models import EpisodeMetrics


class TestRollingAggregator:
    """Tests for RollingAggregator class."""

    def test_init_default_window_size(self) -> None:
        """Test default window size initialization."""
        aggregator = RollingAggregator()
        assert aggregator.window_size == 100
        assert len(aggregator) == 0

    def test_init_custom_window_size(self) -> None:
        """Test custom window size initialization."""
        aggregator = RollingAggregator(window_size=50)
        assert aggregator.window_size == 50

    def test_init_invalid_window_size(self) -> None:
        """Test that window_size < 1 raises error."""
        with pytest.raises(ValueError, match="window_size must be at least 1"):
            RollingAggregator(window_size=0)

        with pytest.raises(ValueError, match="window_size must be at least 1"):
            RollingAggregator(window_size=-5)

    def test_add_episode(self) -> None:
        """Test adding episodes to the aggregator."""
        aggregator = RollingAggregator(window_size=10)
        episode = EpisodeMetrics(
            episode_id=1,
            total_reward=10.0,
            length=100,
            kills=2,
            deaths=1,
            win=True,
        )
        aggregator.add_episode(episode)
        assert len(aggregator) == 1

    def test_window_overflow(self) -> None:
        """Test that old episodes are removed when window overflows."""
        aggregator = RollingAggregator(window_size=3)

        for i in range(5):
            aggregator.add_episode(
                EpisodeMetrics(episode_id=i, total_reward=float(i), length=100)
            )

        assert len(aggregator) == 3
        # Window should contain episodes 2, 3, 4 (rewards 2.0, 3.0, 4.0)
        aggregate = aggregator.get_aggregate()
        assert aggregate is not None
        assert aggregate.mean_reward == 3.0  # (2 + 3 + 4) / 3

    def test_get_aggregate_empty(self) -> None:
        """Test that get_aggregate returns None when empty."""
        aggregator = RollingAggregator()
        assert aggregator.get_aggregate() is None

    def test_get_aggregate_single_episode(self) -> None:
        """Test aggregate with single episode."""
        aggregator = RollingAggregator()
        aggregator.add_episode(
            EpisodeMetrics(
                episode_id=1,
                total_reward=10.0,
                length=100,
                kills=3,
                deaths=1,
                win=True,
            )
        )

        aggregate = aggregator.get_aggregate()
        assert aggregate is not None
        assert aggregate.mean_reward == 10.0
        assert aggregate.std_reward == 0.0  # Single value has 0 std
        assert aggregate.mean_length == 100.0
        assert aggregate.win_rate == 1.0
        assert aggregate.mean_kills == 3.0
        assert aggregate.mean_deaths == 1.0
        assert aggregate.kd_ratio == 3.0  # 3 kills / 1 death
        assert aggregate.episodes_count == 1

    def test_get_aggregate_multiple_episodes(self) -> None:
        """Test aggregate with multiple episodes."""
        aggregator = RollingAggregator()

        episodes = [
            EpisodeMetrics(
                episode_id=1, total_reward=10.0, length=100, kills=3, deaths=1, win=True
            ),
            EpisodeMetrics(
                episode_id=2, total_reward=20.0, length=200, kills=5, deaths=3, win=True
            ),
            EpisodeMetrics(
                episode_id=3,
                total_reward=15.0,
                length=150,
                kills=2,
                deaths=2,
                win=False,
            ),
        ]

        for ep in episodes:
            aggregator.add_episode(ep)

        aggregate = aggregator.get_aggregate()
        assert aggregate is not None

        # Mean reward: (10 + 20 + 15) / 3 = 15.0
        assert aggregate.mean_reward == 15.0

        # Mean length: (100 + 200 + 150) / 3 = 150.0
        assert aggregate.mean_length == 150.0

        # Win rate: 2/3 ≈ 0.666...
        assert abs(aggregate.win_rate - 2 / 3) < 0.001

        # Mean kills: (3 + 5 + 2) / 3 ≈ 3.33
        assert abs(aggregate.mean_kills - 10 / 3) < 0.001

        # Mean deaths: (1 + 3 + 2) / 3 = 2.0
        assert aggregate.mean_deaths == 2.0

        # K/D ratio: total kills / total deaths = 10 / 6 ≈ 1.67
        assert abs(aggregate.kd_ratio - 10 / 6) < 0.001

        assert aggregate.episodes_count == 3

    def test_get_aggregate_zero_deaths(self) -> None:
        """Test K/D ratio when there are no deaths."""
        aggregator = RollingAggregator()
        aggregator.add_episode(
            EpisodeMetrics(
                episode_id=1,
                total_reward=10.0,
                length=100,
                kills=5,
                deaths=0,
                win=True,
            )
        )

        aggregate = aggregator.get_aggregate()
        assert aggregate is not None
        # When deaths=0, K/D ratio should be the total kills
        assert aggregate.kd_ratio == 5.0

    def test_get_aggregate_zero_kills_zero_deaths(self) -> None:
        """Test K/D ratio when there are no kills or deaths."""
        aggregator = RollingAggregator()
        aggregator.add_episode(
            EpisodeMetrics(
                episode_id=1,
                total_reward=10.0,
                length=100,
                kills=0,
                deaths=0,
                win=False,
            )
        )

        aggregate = aggregator.get_aggregate()
        assert aggregate is not None
        assert aggregate.kd_ratio == 0.0

    def test_std_reward_calculation(self) -> None:
        """Test standard deviation calculation."""
        aggregator = RollingAggregator()

        # Add episodes with known std: [10, 20, 30]
        # Mean = 20, variance = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 3 = 200/3
        # std = sqrt(200/3) ≈ 8.165
        for i, reward in enumerate([10.0, 20.0, 30.0]):
            aggregator.add_episode(
                EpisodeMetrics(episode_id=i, total_reward=reward, length=100)
            )

        aggregate = aggregator.get_aggregate()
        assert aggregate is not None
        expected_std = math.sqrt(200 / 3)
        assert abs(aggregate.std_reward - expected_std) < 0.001

    def test_clear(self) -> None:
        """Test clearing the aggregator."""
        aggregator = RollingAggregator()

        for i in range(5):
            aggregator.add_episode(
                EpisodeMetrics(episode_id=i, total_reward=float(i), length=100)
            )

        assert len(aggregator) == 5
        aggregator.clear()
        assert len(aggregator) == 0
        assert aggregator.get_aggregate() is None

    def test_all_losses(self) -> None:
        """Test aggregate when all episodes are losses."""
        aggregator = RollingAggregator()

        for i in range(3):
            aggregator.add_episode(
                EpisodeMetrics(
                    episode_id=i,
                    total_reward=-5.0,
                    length=50,
                    kills=0,
                    deaths=3,
                    win=False,
                )
            )

        aggregate = aggregator.get_aggregate()
        assert aggregate is not None
        assert aggregate.win_rate == 0.0
        assert aggregate.mean_reward == -5.0
