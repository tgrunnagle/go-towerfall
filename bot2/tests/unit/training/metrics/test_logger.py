"""Unit tests for MetricsLogger."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bot.training.metrics.logger import MetricsLogger
from bot.training.metrics.models import EpisodeMetrics, TrainingStepMetrics
from bot.training.metrics.writers.base import MetricsWriter


class MockWriter(MetricsWriter):
    """Mock writer for testing."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.scalar_groups: list[tuple[str, dict[str, float], int]] = []
        self.flush_count = 0
        self.closed = False

    def write_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def write_scalars(
        self, main_tag: str, tag_scalar_dict: dict[str, float], step: int
    ) -> None:
        self.scalar_groups.append((main_tag, tag_scalar_dict, step))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


class TestMetricsLogger:
    """Tests for MetricsLogger class."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that initialization creates the log directory."""
        log_dir = tmp_path / "logs" / "nested"
        logger = MetricsLogger(log_dir=log_dir, writers=[])
        assert log_dir.exists()
        logger.close()

    def test_init_with_custom_writers(self, tmp_path: Path) -> None:
        """Test initialization with custom writers."""
        mock_writer = MockWriter()
        logger = MetricsLogger(log_dir=tmp_path, writers=[mock_writer])
        assert len(logger.writers) == 1
        logger.close()

    @patch("bot.training.metrics.logger.TensorBoardWriter")
    @patch("bot.training.metrics.logger.FileWriter")
    def test_init_default_writers(
        self,
        mock_file_writer: MagicMock,
        mock_tb_writer: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that default writers are created."""
        mock_file_writer.return_value = MagicMock()
        mock_tb_writer.return_value = MagicMock()

        logger = MetricsLogger(
            log_dir=tmp_path,
            enable_tensorboard=True,
            enable_file=True,
        )

        assert len(logger.writers) == 2
        mock_tb_writer.assert_called_once()
        mock_file_writer.assert_called_once()
        logger.close()

    @patch("bot.training.metrics.logger.TensorBoardWriter")
    @patch("bot.training.metrics.logger.FileWriter")
    def test_init_disable_tensorboard(
        self,
        mock_file_writer: MagicMock,
        mock_tb_writer: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test disabling TensorBoard writer."""
        mock_file_writer.return_value = MagicMock()

        logger = MetricsLogger(
            log_dir=tmp_path,
            enable_tensorboard=False,
            enable_file=True,
        )

        assert len(logger.writers) == 1
        mock_tb_writer.assert_not_called()
        logger.close()

    def test_log_episode(self, tmp_path: Path) -> None:
        """Test logging episode metrics."""
        mock_writer = MockWriter()
        logger = MetricsLogger(log_dir=tmp_path, writers=[mock_writer])

        episode = EpisodeMetrics(
            episode_id=1,
            total_reward=10.5,
            length=500,
            kills=3,
            deaths=1,
            win=True,
        )
        logger.log_episode(episode)

        assert logger.episode_count == 1
        assert len(mock_writer.scalar_groups) == 1

        main_tag, metrics, step = mock_writer.scalar_groups[0]
        assert main_tag == "episode"
        assert step == 1
        assert metrics["reward"] == 10.5
        assert metrics["length"] == 500.0
        assert metrics["kills"] == 3.0
        assert metrics["deaths"] == 1.0
        assert metrics["win"] == 1.0

        logger.close()

    def test_log_training_step(self, tmp_path: Path) -> None:
        """Test logging training step metrics."""
        mock_writer = MockWriter()
        logger = MetricsLogger(log_dir=tmp_path, writers=[mock_writer])

        step_metrics = TrainingStepMetrics(
            step=10,
            policy_loss=0.1,
            value_loss=0.2,
            entropy=0.5,
            kl_divergence=0.01,
            clip_fraction=0.15,
            learning_rate=3e-4,
            total_timesteps=20480,
        )
        logger.log_training_step(step_metrics)

        assert logger.step_count == 1

        # Should have one scalars group call and two scalar calls
        assert len(mock_writer.scalar_groups) == 1
        assert len(mock_writer.scalars) == 2

        main_tag, metrics, step = mock_writer.scalar_groups[0]
        assert main_tag == "train"
        assert step == 10
        assert metrics["policy_loss"] == 0.1
        assert metrics["value_loss"] == 0.2
        assert metrics["entropy"] == 0.5
        assert metrics["kl_divergence"] == 0.01
        assert metrics["clip_fraction"] == 0.15

        # Check individual scalar calls
        scalar_tags = {s[0] for s in mock_writer.scalars}
        assert "train/learning_rate" in scalar_tags
        assert "train/total_timesteps" in scalar_tags

        logger.close()

    def test_log_scalar(self, tmp_path: Path) -> None:
        """Test logging individual scalar."""
        mock_writer = MockWriter()
        logger = MetricsLogger(log_dir=tmp_path, writers=[mock_writer])

        logger.log_scalar("custom/metric", 42.0, 5)

        assert len(mock_writer.scalars) == 1
        assert mock_writer.scalars[0] == ("custom/metric", 42.0, 5)

        logger.close()

    def test_log_scalars(self, tmp_path: Path) -> None:
        """Test logging multiple scalars."""
        mock_writer = MockWriter()
        logger = MetricsLogger(log_dir=tmp_path, writers=[mock_writer])

        logger.log_scalars("custom", {"a": 1.0, "b": 2.0}, 5)

        assert len(mock_writer.scalar_groups) == 1
        assert mock_writer.scalar_groups[0] == ("custom", {"a": 1.0, "b": 2.0}, 5)

        logger.close()

    def test_get_summary_empty(self, tmp_path: Path) -> None:
        """Test get_summary with no episodes logged."""
        logger = MetricsLogger(log_dir=tmp_path, writers=[])
        summary = logger.get_summary()

        assert summary["total_episodes"] == 0.0
        assert summary["total_training_steps"] == 0.0
        assert "mean_reward" not in summary

        logger.close()

    def test_get_summary_with_episodes(self, tmp_path: Path) -> None:
        """Test get_summary with episodes logged."""
        logger = MetricsLogger(log_dir=tmp_path, writers=[])

        episodes = [
            EpisodeMetrics(
                episode_id=1, total_reward=10.0, length=100, kills=2, deaths=1, win=True
            ),
            EpisodeMetrics(
                episode_id=2, total_reward=20.0, length=200, kills=4, deaths=2, win=True
            ),
            EpisodeMetrics(
                episode_id=3,
                total_reward=15.0,
                length=150,
                kills=1,
                deaths=1,
                win=False,
            ),
        ]

        for ep in episodes:
            logger.log_episode(ep)

        summary = logger.get_summary()

        assert summary["total_episodes"] == 3.0
        assert summary["mean_reward"] == 15.0
        assert summary["win_rate"] == pytest.approx(2 / 3, rel=0.01)
        assert summary["window_episodes"] == 3.0

        logger.close()

    def test_get_aggregate_metrics(self, tmp_path: Path) -> None:
        """Test get_aggregate_metrics returns AggregateMetrics object."""
        logger = MetricsLogger(log_dir=tmp_path, writers=[])

        logger.log_episode(EpisodeMetrics(episode_id=1, total_reward=10.0, length=100))

        aggregate = logger.get_aggregate_metrics()
        assert aggregate is not None
        assert aggregate.mean_reward == 10.0
        assert aggregate.episodes_count == 1

        logger.close()

    def test_flush(self, tmp_path: Path) -> None:
        """Test flush calls flush on all writers."""
        mock_writer1 = MockWriter()
        mock_writer2 = MockWriter()
        logger = MetricsLogger(log_dir=tmp_path, writers=[mock_writer1, mock_writer2])

        logger.flush()

        assert mock_writer1.flush_count == 1
        assert mock_writer2.flush_count == 1

        logger.close()

    def test_close(self, tmp_path: Path) -> None:
        """Test close calls close on all writers."""
        mock_writer1 = MockWriter()
        mock_writer2 = MockWriter()
        logger = MetricsLogger(log_dir=tmp_path, writers=[mock_writer1, mock_writer2])

        logger.close()

        assert mock_writer1.closed
        assert mock_writer2.closed

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test context manager interface."""
        mock_writer = MockWriter()

        with MetricsLogger(log_dir=tmp_path, writers=[mock_writer]) as logger:
            logger.log_scalar("test", 1.0, 1)

        assert mock_writer.closed

    def test_write_after_close_ignored(self, tmp_path: Path) -> None:
        """Test that writes after close are ignored."""
        mock_writer = MockWriter()
        logger = MetricsLogger(log_dir=tmp_path, writers=[mock_writer])
        logger.close()

        # These should not raise
        logger.log_scalar("test", 1.0, 1)
        logger.log_episode(EpisodeMetrics(episode_id=1, total_reward=10.0, length=100))
        logger.log_training_step(
            TrainingStepMetrics(
                step=1,
                policy_loss=0.1,
                value_loss=0.2,
                entropy=0.5,
                kl_divergence=0.01,
                clip_fraction=0.15,
                learning_rate=3e-4,
                total_timesteps=2048,
            )
        )

        # No new writes should have been made after close
        assert len(mock_writer.scalars) == 0

    def test_custom_window_size(self, tmp_path: Path) -> None:
        """Test custom rolling window size."""
        logger = MetricsLogger(log_dir=tmp_path, writers=[], window_size=5)

        # Add 10 episodes
        for i in range(10):
            logger.log_episode(
                EpisodeMetrics(episode_id=i, total_reward=float(i), length=100)
            )

        summary = logger.get_summary()
        # Window should only contain last 5 episodes (5, 6, 7, 8, 9)
        assert summary["window_episodes"] == 5.0
        assert summary["mean_reward"] == 7.0  # (5 + 6 + 7 + 8 + 9) / 5

        logger.close()
