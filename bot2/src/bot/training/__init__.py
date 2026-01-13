"""Training module for ML bot training infrastructure.

This module provides components for managing game server instances during
ML training, including game creation, lifecycle management, and cleanup.
It also provides the model registry for storing and retrieving trained models,
and the training orchestrator for coordinating the full training pipeline.

Usage:
    from bot.training import GameServerManager, TrainingGameConfig, GameInstance

    async with GameServerManager() as manager:
        config = TrainingGameConfig(room_name="Training Session")
        game = await manager.create_game(config, "Bot1")
        # Use game for training...

    from bot.training import ModelRegistry, TrainingMetrics

    registry = ModelRegistry("/path/to/registry")
    model_id = registry.register_model(...)

    from bot.training import TrainingOrchestrator, OrchestratorConfig

    config = OrchestratorConfig(num_envs=4, total_timesteps=500_000)
    async with TrainingOrchestrator(config) as orchestrator:
        metadata = await orchestrator.train()
"""

from bot.training.exceptions import (
    GameCreationError,
    GameNotFoundError,
    GameServerError,
    MaxGamesExceededError,
)
from bot.training.orchestrator import TrainingOrchestrator
from bot.training.orchestrator_config import OrchestratorConfig
from bot.training.registry import (
    ModelAlreadyExistsError,
    ModelMetadata,
    ModelNotFoundError,
    ModelRegistry,
    NetworkArchitecture,
    RegistryIndex,
    StorageBackend,
    TrainingMetrics,
)
from bot.training.server_manager import (
    GameInstance,
    GameServerManager,
    TrainingGameConfig,
)

__all__ = [
    # Orchestrator
    "TrainingOrchestrator",
    "OrchestratorConfig",
    # Server management
    "GameServerManager",
    "TrainingGameConfig",
    "GameInstance",
    # Model registry
    "ModelRegistry",
    "TrainingMetrics",
    "ModelMetadata",
    "NetworkArchitecture",
    "StorageBackend",
    "RegistryIndex",
    # Exceptions
    "GameServerError",
    "GameCreationError",
    "MaxGamesExceededError",
    "GameNotFoundError",
    "ModelNotFoundError",
    "ModelAlreadyExistsError",
]
