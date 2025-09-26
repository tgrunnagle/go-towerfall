"""
FastAPI server for training metrics, episode replay, spectator data, and bot management.

This module provides a unified FastAPI server that serves training metrics data,
episode replay functionality, manages WebSocket connections for real-time updates,
and handles AI bot instances for game integration.
"""

# Import unified server components
from server.server import UnifiedServer, ServerConfig, create_server, run_server

# Legacy server class alias for backward compatibility
TrainingMetricsServer = UnifiedServer  # Use UnifiedServer as replacement

# Import data models
from server.data_models import (
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
from server.websocket_manager import WebSocketManager, ConnectionManager

# Import API routers
from server.training_metrics_api import router as training_metrics_router
try:
    from server.replay_api import router as replay_router
except ImportError:
    replay_router = None

# Import bot server components
from server.bot_server import (
    BotServer,
    BotServerConfig,
    BotConfig,
    BotType,
    BotStatus,
    BotInstance,
    GameClientPool,
    create_bot_server
)

# Import bot integration components
from server.bot_server_api import (
    BotServerApi,
    BotServerConfig
)

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
    'replay_router',
    
    # Bot server components
    'BotServer',
    'BotServerConfig',
    'BotConfig',
    'BotType',
    'BotStatus',
    'BotInstance',
    'GameClientPool',
    'create_bot_server',
    
    # Bot integration components
    'BotServerApi',
    'BotServerConfig'
]