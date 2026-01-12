"""Training module for ML bot training infrastructure.

This module provides components for managing game server instances during
ML training, including game creation, lifecycle management, and cleanup.

Usage:
    from bot.training import GameServerManager, TrainingGameConfig, GameInstance

    async with GameServerManager() as manager:
        config = TrainingGameConfig(room_name="Training Session")
        game = await manager.create_game(config, "Bot1")
        # Use game for training...
"""

from bot.training.exceptions import (
    GameCreationError,
    GameNotFoundError,
    GameServerError,
    MaxGamesExceededError,
)
from bot.training.server_manager import (
    GameInstance,
    GameServerManager,
    TrainingGameConfig,
)

__all__ = [
    # Main classes
    "GameServerManager",
    "TrainingGameConfig",
    "GameInstance",
    # Exceptions
    "GameServerError",
    "GameCreationError",
    "MaxGamesExceededError",
    "GameNotFoundError",
]
