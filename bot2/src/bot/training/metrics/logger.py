"""Main MetricsLogger class for training metrics logging.

This module provides the MetricsLogger class that coordinates multiple
writers and maintains rolling statistics for monitoring training progress.
"""

import logging
from pathlib import Path

from bot.training.metrics.aggregators import RollingAggregator
from bot.training.metrics.models import (
    AggregateMetrics,
    EpisodeMetrics,
    TrainingStepMetrics,
)
from bot.training.metrics.writers.base import MetricsWriter
from bot.training.metrics.writers.file_writer import FileWriter
from bot.training.metrics.writers.tensorboard_writer import TensorBoardWriter

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Main logger for training metrics.

    Coordinates multiple writers and maintains rolling statistics for
    monitoring training progress. Supports logging episode metrics,
    training step metrics, and arbitrary scalar values.

    Example:
        logger = MetricsLogger(
            log_dir="logs/run_001",
            enable_tensorboard=True,
            enable_file=True,
        )

        # Log episode completion
        logger.log_episode(EpisodeMetrics(
            episode_id=1,
            total_reward=10.5,
            length=500,
            kills=3,
            deaths=1,
            win=True,
        ))

        # Log training step
        logger.log_training_step(TrainingStepMetrics(
            step=1,
            policy_loss=0.1,
            value_loss=0.2,
            entropy=0.5,
            kl_divergence=0.01,
            clip_fraction=0.15,
            learning_rate=3e-4,
            total_timesteps=2048,
        ))

        # Get summary
        summary = logger.get_summary()

        # Cleanup
        logger.close()

    Attributes:
        log_dir: Directory for log files
        writers: List of active writers
        aggregator: Rolling statistics aggregator
        episode_count: Total episodes logged
        step_count: Total training steps logged
    """

    def __init__(
        self,
        log_dir: str | Path,
        writers: list[MetricsWriter] | None = None,
        enable_tensorboard: bool = True,
        enable_file: bool = True,
        file_format: str = "json",
        window_size: int = 100,
    ) -> None:
        """Initialize the metrics logger.

        Args:
            log_dir: Directory for log files
            writers: Custom list of writers (overrides enable_* flags)
            enable_tensorboard: Enable TensorBoard logging (default True)
            enable_file: Enable file logging (default True)
            file_format: File format for FileWriter ("json" or "csv")
            window_size: Rolling window size for aggregates (default 100)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize writers
        if writers is not None:
            self.writers = writers
        else:
            self.writers = self._create_default_writers(
                enable_tensorboard, enable_file, file_format
            )

        # Initialize aggregator
        self.aggregator = RollingAggregator(window_size=window_size)

        # Counters
        self.episode_count = 0
        self.step_count = 0

        self._closed = False

        logger.info(
            "MetricsLogger initialized: %s (writers=%d)",
            self.log_dir,
            len(self.writers),
        )

    def _create_default_writers(
        self,
        enable_tensorboard: bool,
        enable_file: bool,
        file_format: str,
    ) -> list[MetricsWriter]:
        """Create default writers based on configuration.

        Args:
            enable_tensorboard: Whether to enable TensorBoard writer
            enable_file: Whether to enable file writer
            file_format: Format for file writer

        Returns:
            List of configured writers
        """
        writers: list[MetricsWriter] = []

        if enable_tensorboard:
            tb_dir = self.log_dir / "tensorboard"
            writers.append(TensorBoardWriter(log_dir=tb_dir))

        if enable_file:
            file_dir = self.log_dir / "files"
            writers.append(FileWriter(log_dir=file_dir, format=file_format))  # type: ignore[arg-type]

        return writers

    def log_episode(self, episode_data: EpisodeMetrics) -> None:
        """Log metrics for a completed episode.

        Args:
            episode_data: Episode metrics to log
        """
        if self._closed:
            return

        self.episode_count += 1
        self.aggregator.add_episode(episode_data)

        # Write to all writers
        step = self.episode_count
        for writer in self.writers:
            writer.write_scalars(
                "episode",
                {
                    "reward": episode_data.total_reward,
                    "length": float(episode_data.length),
                    "kills": float(episode_data.kills),
                    "deaths": float(episode_data.deaths),
                    "win": 1.0 if episode_data.win else 0.0,
                },
                step,
            )

    def log_training_step(self, step_data: TrainingStepMetrics) -> None:
        """Log metrics for a training step (PPO update).

        Args:
            step_data: Training step metrics to log
        """
        if self._closed:
            return

        self.step_count += 1

        # Write to all writers
        for writer in self.writers:
            writer.write_scalars(
                "train",
                {
                    "policy_loss": step_data.policy_loss,
                    "value_loss": step_data.value_loss,
                    "entropy": step_data.entropy,
                    "kl_divergence": step_data.kl_divergence,
                    "clip_fraction": step_data.clip_fraction,
                },
                step_data.step,
            )
            writer.write_scalar(
                "train/learning_rate",
                step_data.learning_rate,
                step_data.step,
            )
            writer.write_scalar(
                "train/total_timesteps",
                float(step_data.total_timesteps),
                step_data.step,
            )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar metric.

        Args:
            tag: Metric name/tag
            value: Scalar value to log
            step: Global step number
        """
        if self._closed:
            return

        for writer in self.writers:
            writer.write_scalar(tag, value, step)

    def log_scalars(
        self, main_tag: str, tag_scalar_dict: dict[str, float], step: int
    ) -> None:
        """Log multiple related scalar metrics.

        Args:
            main_tag: Group name for the metrics
            tag_scalar_dict: Dictionary mapping sub-tags to values
            step: Global step number
        """
        if self._closed:
            return

        for writer in self.writers:
            writer.write_scalars(main_tag, tag_scalar_dict, step)

    def get_summary(self) -> dict[str, float]:
        """Get aggregated summary statistics.

        Returns:
            Dictionary of summary metrics including rolling averages
            and total counts. Returns empty dict if no episodes logged.
        """
        summary: dict[str, float] = {
            "total_episodes": float(self.episode_count),
            "total_training_steps": float(self.step_count),
        }

        aggregate = self.aggregator.get_aggregate()
        if aggregate is not None:
            summary.update(
                {
                    "mean_reward": aggregate.mean_reward,
                    "std_reward": aggregate.std_reward,
                    "mean_length": aggregate.mean_length,
                    "win_rate": aggregate.win_rate,
                    "mean_kills": aggregate.mean_kills,
                    "mean_deaths": aggregate.mean_deaths,
                    "kd_ratio": aggregate.kd_ratio,
                    "window_episodes": float(aggregate.episodes_count),
                }
            )

        return summary

    def get_aggregate_metrics(self) -> AggregateMetrics | None:
        """Get the current aggregate metrics object.

        Returns:
            AggregateMetrics if episodes have been logged, None otherwise
        """
        return self.aggregator.get_aggregate()

    def flush(self) -> None:
        """Flush all writers."""
        if self._closed:
            return

        for writer in self.writers:
            writer.flush()

    def close(self) -> None:
        """Close all writers and release resources."""
        if self._closed:
            return

        for writer in self.writers:
            writer.close()

        self._closed = True
        logger.info("MetricsLogger closed: %s", self.log_dir)

    def __enter__(self) -> "MetricsLogger":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.close()
