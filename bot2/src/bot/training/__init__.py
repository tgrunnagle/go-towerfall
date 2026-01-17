"""Training module for ML bot training infrastructure.

This module provides components for managing game server instances during
ML training, including game creation, lifecycle management, and cleanup.
It also provides the model registry for storing and retrieving trained models,
the training orchestrator for coordinating the full training pipeline,
and the metrics logging system for monitoring training progress.

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

    from bot.training.metrics import MetricsLogger, EpisodeMetrics

    with MetricsLogger(log_dir="logs/run_001") as logger:
        logger.log_episode(EpisodeMetrics(episode_id=1, total_reward=10.5, length=500))
"""

from bot.training.evaluation import (
    ComparisonResult,
    EvaluationManager,
    EvaluationResult,
)
from bot.training.exceptions import (
    GameCreationError,
    GameNotFoundError,
    GameServerError,
    MaxGamesExceededError,
)
from bot.training.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    FileWriter,
    MetricsLogger,
    MetricsWriter,
    RollingAggregator,
    TensorBoardWriter,
    TrainingStepMetrics,
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
from bot.training.successive_config import PromotionCriteria, SuccessiveTrainingConfig
from bot.training.successive_trainer import GenerationResult, SuccessiveTrainer

__all__ = [
    # Orchestrator
    "TrainingOrchestrator",
    "OrchestratorConfig",
    # Successive training
    "SuccessiveTrainer",
    "SuccessiveTrainingConfig",
    "PromotionCriteria",
    "GenerationResult",
    # Evaluation
    "EvaluationManager",
    "EvaluationResult",
    "ComparisonResult",
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
    # Metrics logging
    "MetricsLogger",
    "EpisodeMetrics",
    "TrainingStepMetrics",
    "AggregateMetrics",
    "MetricsWriter",
    "FileWriter",
    "TensorBoardWriter",
    "RollingAggregator",
    # Exceptions
    "GameServerError",
    "GameCreationError",
    "MaxGamesExceededError",
    "GameNotFoundError",
    "ModelNotFoundError",
    "ModelAlreadyExistsError",
]
