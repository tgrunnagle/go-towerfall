"""Unit tests for metrics Pydantic models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from bot.training.metrics.models import (
    AggregateMetrics,
    EpisodeMetrics,
    TrainingStepMetrics,
)


class TestEpisodeMetrics:
    """Tests for EpisodeMetrics model."""

    def test_basic_creation(self) -> None:
        """Test basic model creation with required fields."""
        metrics = EpisodeMetrics(
            episode_id=1,
            total_reward=10.5,
            length=500,
        )
        assert metrics.episode_id == 1
        assert metrics.total_reward == 10.5
        assert metrics.length == 500
        assert metrics.kills == 0
        assert metrics.deaths == 0
        assert metrics.win is False
        assert isinstance(metrics.timestamp, datetime)

    def test_full_creation(self) -> None:
        """Test model creation with all fields."""
        timestamp = datetime(2024, 1, 15, 12, 30, 0)
        metrics = EpisodeMetrics(
            episode_id=42,
            total_reward=25.7,
            length=1000,
            kills=5,
            deaths=2,
            win=True,
            timestamp=timestamp,
        )
        assert metrics.episode_id == 42
        assert metrics.total_reward == 25.7
        assert metrics.length == 1000
        assert metrics.kills == 5
        assert metrics.deaths == 2
        assert metrics.win is True
        assert metrics.timestamp == timestamp

    def test_negative_episode_id_rejected(self) -> None:
        """Test that negative episode_id is rejected."""
        with pytest.raises(ValidationError):
            EpisodeMetrics(episode_id=-1, total_reward=10.0, length=100)

    def test_negative_length_rejected(self) -> None:
        """Test that negative length is rejected."""
        with pytest.raises(ValidationError):
            EpisodeMetrics(episode_id=1, total_reward=10.0, length=-100)

    def test_negative_kills_rejected(self) -> None:
        """Test that negative kills is rejected."""
        with pytest.raises(ValidationError):
            EpisodeMetrics(episode_id=1, total_reward=10.0, length=100, kills=-1)

    def test_negative_deaths_rejected(self) -> None:
        """Test that negative deaths is rejected."""
        with pytest.raises(ValidationError):
            EpisodeMetrics(episode_id=1, total_reward=10.0, length=100, deaths=-1)

    def test_model_is_frozen(self) -> None:
        """Test that model is immutable."""
        metrics = EpisodeMetrics(episode_id=1, total_reward=10.0, length=100)
        with pytest.raises(ValidationError):
            metrics.episode_id = 2  # type: ignore[misc]

    def test_negative_reward_allowed(self) -> None:
        """Test that negative rewards are allowed."""
        metrics = EpisodeMetrics(episode_id=1, total_reward=-5.0, length=100)
        assert metrics.total_reward == -5.0


class TestTrainingStepMetrics:
    """Tests for TrainingStepMetrics model."""

    def test_basic_creation(self) -> None:
        """Test basic model creation."""
        metrics = TrainingStepMetrics(
            step=1,
            policy_loss=0.1,
            value_loss=0.2,
            entropy=0.5,
            kl_divergence=0.01,
            clip_fraction=0.15,
            learning_rate=3e-4,
            total_timesteps=2048,
        )
        assert metrics.step == 1
        assert metrics.policy_loss == 0.1
        assert metrics.value_loss == 0.2
        assert metrics.entropy == 0.5
        assert metrics.kl_divergence == 0.01
        assert metrics.clip_fraction == 0.15
        assert metrics.learning_rate == 3e-4
        assert metrics.total_timesteps == 2048

    def test_negative_step_rejected(self) -> None:
        """Test that negative step is rejected."""
        with pytest.raises(ValidationError):
            TrainingStepMetrics(
                step=-1,
                policy_loss=0.1,
                value_loss=0.2,
                entropy=0.5,
                kl_divergence=0.01,
                clip_fraction=0.15,
                learning_rate=3e-4,
                total_timesteps=2048,
            )

    def test_negative_kl_divergence_rejected(self) -> None:
        """Test that negative KL divergence is rejected."""
        with pytest.raises(ValidationError):
            TrainingStepMetrics(
                step=1,
                policy_loss=0.1,
                value_loss=0.2,
                entropy=0.5,
                kl_divergence=-0.01,
                clip_fraction=0.15,
                learning_rate=3e-4,
                total_timesteps=2048,
            )

    def test_clip_fraction_bounds(self) -> None:
        """Test that clip_fraction must be between 0 and 1."""
        with pytest.raises(ValidationError):
            TrainingStepMetrics(
                step=1,
                policy_loss=0.1,
                value_loss=0.2,
                entropy=0.5,
                kl_divergence=0.01,
                clip_fraction=1.5,  # > 1.0
                learning_rate=3e-4,
                total_timesteps=2048,
            )

    def test_zero_learning_rate_rejected(self) -> None:
        """Test that zero learning rate is rejected."""
        with pytest.raises(ValidationError):
            TrainingStepMetrics(
                step=1,
                policy_loss=0.1,
                value_loss=0.2,
                entropy=0.5,
                kl_divergence=0.01,
                clip_fraction=0.15,
                learning_rate=0.0,
                total_timesteps=2048,
            )

    def test_negative_losses_allowed(self) -> None:
        """Test that negative losses are allowed (can happen during training)."""
        metrics = TrainingStepMetrics(
            step=1,
            policy_loss=-0.1,
            value_loss=-0.2,
            entropy=-0.5,
            kl_divergence=0.01,
            clip_fraction=0.15,
            learning_rate=3e-4,
            total_timesteps=2048,
        )
        assert metrics.policy_loss == -0.1


class TestAggregateMetrics:
    """Tests for AggregateMetrics model."""

    def test_basic_creation(self) -> None:
        """Test basic model creation."""
        metrics = AggregateMetrics(
            mean_reward=15.5,
            std_reward=3.2,
            mean_length=450.0,
            win_rate=0.65,
            mean_kills=2.5,
            mean_deaths=1.2,
            mean_kd_ratio=2.08,
            episodes_count=100,
        )
        assert metrics.mean_reward == 15.5
        assert metrics.std_reward == 3.2
        assert metrics.mean_length == 450.0
        assert metrics.win_rate == 0.65
        assert metrics.mean_kills == 2.5
        assert metrics.mean_deaths == 1.2
        assert metrics.mean_kd_ratio == 2.08
        assert metrics.episodes_count == 100
        assert isinstance(metrics.timestamp, datetime)

    def test_win_rate_bounds(self) -> None:
        """Test that win_rate must be between 0 and 1."""
        with pytest.raises(ValidationError):
            AggregateMetrics(
                mean_reward=15.5,
                std_reward=3.2,
                mean_length=450.0,
                win_rate=1.5,  # > 1.0
                mean_kills=2.5,
                mean_deaths=1.2,
                mean_kd_ratio=2.08,
                episodes_count=100,
            )

    def test_negative_std_rejected(self) -> None:
        """Test that negative std_reward is rejected."""
        with pytest.raises(ValidationError):
            AggregateMetrics(
                mean_reward=15.5,
                std_reward=-3.2,  # Negative
                mean_length=450.0,
                win_rate=0.65,
                mean_kills=2.5,
                mean_deaths=1.2,
                mean_kd_ratio=2.08,
                episodes_count=100,
            )

    def test_negative_mean_reward_allowed(self) -> None:
        """Test that negative mean_reward is allowed."""
        metrics = AggregateMetrics(
            mean_reward=-5.0,
            std_reward=3.2,
            mean_length=450.0,
            win_rate=0.0,
            mean_kills=0.0,
            mean_deaths=3.0,
            mean_kd_ratio=0.0,
            episodes_count=100,
        )
        assert metrics.mean_reward == -5.0
