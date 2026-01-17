"""Training metrics logging module.

This module provides a comprehensive metrics logging system for training
reinforcement learning agents. It supports:
- Episode metrics (rewards, lengths, kills, deaths, wins)
- Training step metrics (losses, entropy, KL divergence)
- Rolling aggregation for summary statistics
- Multiple output backends (TensorBoard, JSON/CSV files)

Usage:
    from bot.training.metrics import MetricsLogger, EpisodeMetrics, TrainingStepMetrics

    with MetricsLogger(log_dir="logs/run_001") as logger:
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
        print(f"Mean reward: {summary['mean_reward']}")
"""

from bot.training.metrics.aggregators import RollingAggregator
from bot.training.metrics.logger import MetricsLogger
from bot.training.metrics.models import (
    AggregateMetrics,
    EpisodeMetrics,
    TrainingStepMetrics,
)
from bot.training.metrics.writers import (
    FileWriter,
    MetricsWriter,
    TensorBoardWriter,
)

__all__ = [
    # Main logger
    "MetricsLogger",
    # Data models
    "EpisodeMetrics",
    "TrainingStepMetrics",
    "AggregateMetrics",
    # Writers
    "MetricsWriter",
    "FileWriter",
    "TensorBoardWriter",
    # Aggregation
    "RollingAggregator",
]
