"""Model Registry module for managing trained PPO models.

This module provides components for storing, versioning, and retrieving
trained models with associated metadata. It is a core part of the
successive training pipeline.

Usage:
    from bot.training.registry import ModelRegistry, TrainingMetrics

    registry = ModelRegistry("/path/to/registry")

    # Create training metrics
    metrics = TrainingMetrics(
        total_episodes=1000,
        total_timesteps=100000,
        average_reward=50.0,
        average_episode_length=500.0,
        win_rate=0.6,
        average_kills=2.5,
        average_deaths=1.5,
        kills_deaths_ratio=1.67,
    )

    # Register a trained model
    model_id = registry.register_model(
        model=trained_network,
        generation=0,
        opponent_model_id=None,
        training_metrics=metrics,
        hyperparameters=config.model_dump(),
        training_duration_seconds=3600.0,
    )

    # Retrieve a model
    network, metadata = registry.get_model(model_id)
"""

from bot.training.registry.model_metadata import (
    ModelMetadata,
    NetworkArchitecture,
    TrainingMetrics,
)
from bot.training.registry.model_registry import (
    ModelAlreadyExistsError,
    ModelNotFoundError,
    ModelRegistry,
)
from bot.training.registry.storage_backend import RegistryIndex, StorageBackend

__all__ = [
    # Main classes
    "ModelRegistry",
    "TrainingMetrics",
    "ModelMetadata",
    "NetworkArchitecture",
    # Storage
    "StorageBackend",
    "RegistryIndex",
    # Exceptions
    "ModelNotFoundError",
    "ModelAlreadyExistsError",
]
