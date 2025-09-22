"""
Training module for RL bot system.

This module provides training session management, batch episode execution,
and APIs for managing accelerated training environments.
"""

from bot.rl_bot_system.training.training_session import (
    TrainingSession,
    TrainingConfig,
    TrainingMode,
    SessionStatus,
    SessionMetrics,
    EpisodeResult
)

from bot.rl_bot_system.training.session_manager import (
    SessionManager,
    ResourceLimits,
    ResourceStatus,
    SessionRequest
)

from bot.rl_bot_system.training.batch_episode_manager import (
    BatchEpisodeManager,
    EpisodeBatch,
    EpisodeTask,
    EpisodeStatus
)

from bot.rl_bot_system.training.training_api import (
    TrainingAPI
)

from bot.rl_bot_system.training.model_manager import (
    ModelManager,
    RLModel
)

__all__ = [
    # Training Session
    'TrainingSession',
    'TrainingConfig',
    'TrainingMode',
    'SessionStatus',
    'SessionMetrics',
    'EpisodeResult',
    
    # Session Manager
    'SessionManager',
    'ResourceLimits',
    'ResourceStatus',
    'SessionRequest',
    
    # Batch Episode Manager
    'BatchEpisodeManager',
    'EpisodeBatch',
    'EpisodeTask',
    'EpisodeStatus',
    
    # Training API
    'TrainingAPI',
    
    # Model Manager
    'ModelManager',
    'RLModel'
]