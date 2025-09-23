"""
FastAPI server for training metrics, episode replay, and spectator data.

This module provides a unified FastAPI server that serves training metrics data,
episode replay functionality, and manages WebSocket connections for real-time updates.
"""

# Import unified server components
from rl_bot_system.server.server import UnifiedServer, ServerConfig, create_server, run_server

# Legacy server class alias for backward compatibility
TrainingMetricsServer = UnifiedServer  # Use UnifiedServer as replacement

# Import data models
from rl_bot_system.server.data_models import (
    TrainingMetricsData,
    BotDecisionData,
    PerformanceGraphData,
    TrainingSessionInfo,
    SpectatorConnectionInfo,
    ServerStatus,
    TrainingStatus,
    MessageType
)

# Import WebSocket components
from rl_bot_system.server.websocket_manager import WebSocketManager, ConnectionManager

# Import API routers
from rl_bot_system.server.training_metrics_api import router as training_metrics_router
try:
    from rl_bot_system.server.replay_api import router as replay_router
except ImportError:
    replay_router = None

__all__ = [
    # Main server classes
    'UnifiedServer',
    'ServerConfig',
    'create_server',
    'run_server',
    
    # Legacy server (for backward compatibility)
    'TrainingMetricsServer',
    
    # Data models
    'TrainingMetricsData',
    'BotDecisionData',
    'PerformanceGraphData',
    'TrainingSessionInfo',
    'SpectatorConnectionInfo',
    'ServerStatus',
    'TrainingStatus',
    'MessageType',
    
    # WebSocket components
    'WebSocketManager',
    'ConnectionManager',
    
    # API routers
    'training_metrics_router',
    'replay_router'
]