"""
FastAPI server for training metrics and spectator data.

This module provides a FastAPI server that serves training metrics data
to frontend overlays and manages WebSocket connections for real-time updates.
"""

from rl_bot_system.server.training_metrics_server import TrainingMetricsServer, ServerConfig
from rl_bot_system.server.data_models import (
    TrainingMetricsData,
    BotDecisionData,
    PerformanceGraphData,
    TrainingSessionInfo,
    SpectatorConnectionInfo
)
from rl_bot_system.server.websocket_manager import WebSocketManager, ConnectionManager

__all__ = [
    'TrainingMetricsServer',
    'ServerConfig',
    'TrainingMetricsData',
    'BotDecisionData',
    'PerformanceGraphData',
    'TrainingSessionInfo',
    'SpectatorConnectionInfo',
    'WebSocketManager',
    'ConnectionManager'
]