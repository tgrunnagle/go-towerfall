"""Unit tests for dashboard models."""

from datetime import datetime, timezone

import pytest

from bot.dashboard.models import DashboardConfig, GenerationMetrics


class TestGenerationMetrics:
    """Tests for GenerationMetrics model."""

    def test_create_valid_metrics(self):
        """Test creating valid generation metrics."""
        metrics = GenerationMetrics(
            generation_id=0,
            model_version="ppo_gen_000",
            opponent_type="baseline",
            total_episodes=1000,
            total_kills=2500,
            total_deaths=1500,
            kill_death_ratio=1.67,
            win_rate=0.6,
            avg_episode_reward=50.0,
            avg_episode_length=500.0,
            training_steps=100000,
            training_duration_seconds=3600.0,
            timestamp=datetime.now(timezone.utc),
        )

        assert metrics.generation_id == 0
        assert metrics.model_version == "ppo_gen_000"
        assert metrics.opponent_type == "baseline"
        assert metrics.kill_death_ratio == 1.67
        assert metrics.win_rate == 0.6

    def test_metrics_validation_negative_generation(self):
        """Test that negative generation_id is rejected."""
        with pytest.raises(ValueError):
            GenerationMetrics(
                generation_id=-1,
                model_version="ppo_gen_000",
                opponent_type="baseline",
                total_episodes=1000,
                total_kills=2500,
                total_deaths=1500,
                kill_death_ratio=1.67,
                win_rate=0.6,
                avg_episode_reward=50.0,
                avg_episode_length=500.0,
                training_steps=100000,
                training_duration_seconds=3600.0,
                timestamp=datetime.now(timezone.utc),
            )

    def test_metrics_validation_win_rate_range(self):
        """Test that win_rate must be between 0 and 1."""
        with pytest.raises(ValueError):
            GenerationMetrics(
                generation_id=0,
                model_version="ppo_gen_000",
                opponent_type="baseline",
                total_episodes=1000,
                total_kills=2500,
                total_deaths=1500,
                kill_death_ratio=1.67,
                win_rate=1.5,  # Invalid
                avg_episode_reward=50.0,
                avg_episode_length=500.0,
                training_steps=100000,
                training_duration_seconds=3600.0,
                timestamp=datetime.now(timezone.utc),
            )

    def test_metrics_frozen(self):
        """Test that metrics are immutable."""
        metrics = GenerationMetrics(
            generation_id=0,
            model_version="ppo_gen_000",
            opponent_type="baseline",
            total_episodes=1000,
            total_kills=2500,
            total_deaths=1500,
            kill_death_ratio=1.67,
            win_rate=0.6,
            avg_episode_reward=50.0,
            avg_episode_length=500.0,
            training_steps=100000,
            training_duration_seconds=3600.0,
            timestamp=datetime.now(timezone.utc),
        )

        with pytest.raises(Exception):
            metrics.generation_id = 1


class TestDashboardConfig:
    """Tests for DashboardConfig model."""

    def test_create_default_config(self):
        """Test creating config with defaults."""
        config = DashboardConfig()

        assert config.registry_path == "./model_registry"
        assert config.metrics_dir is None
        assert config.output_dir == "./reports"
        assert config.output_format == "html"
        assert config.generations is None
        assert config.title == "Training Generation Comparison"

    def test_create_custom_config(self):
        """Test creating config with custom values."""
        config = DashboardConfig(
            registry_path="/custom/registry",
            metrics_dir="/custom/metrics",
            output_dir="/custom/output",
            output_format="both",
            generations=(0, 5),
            title="Custom Dashboard",
        )

        assert config.registry_path == "/custom/registry"
        assert config.metrics_dir == "/custom/metrics"
        assert config.output_dir == "/custom/output"
        assert config.output_format == "both"
        assert config.generations == (0, 5)
        assert config.title == "Custom Dashboard"

    def test_config_frozen(self):
        """Test that config is immutable."""
        config = DashboardConfig()

        with pytest.raises(Exception):
            config.output_dir = "/new/path"
