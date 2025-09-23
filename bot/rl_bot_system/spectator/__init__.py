"""
Spectator system for RL bot training observation.

This module provides functionality for creating and managing spectator sessions
that allow real-time observation of bot training and evaluation.
"""

from .spectator_manager import SpectatorManager, SpectatorSession, SpectatorMode
from .training_metrics_overlay import TrainingMetricsOverlay, MetricsData
from .room_manager import SpectatorRoomManager, RoomAccessControl, AccessLevel

__all__ = [
    'SpectatorManager',
    'SpectatorSession',
    'SpectatorMode',
    'TrainingMetricsOverlay',
    'MetricsData',
    'SpectatorRoomManager',
    'RoomAccessControl',
    'AccessLevel'
]