"""
FastAPI server for training metrics, episode replay, spectator data, and bot management.

This module provides a unified FastAPI server that serves training metrics data,
episode replay functionality, manages WebSocket connections for real-time updates,
and handles AI bot instances for game integration.
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

# Import bot server components
from rl_bot_system.server.bot_server import (
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
from rl_bot_system.server.bot_server_api import (
    BotServerApi,
    initialize_bot_integration,
    cleanup_bot_integration,
    get_bot_integration,
    get_bot_router,
    add_bot_to_room,
    remove_bot_from_room,
    get_available_bots,
    get_room_bot_status
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
    'initialize_bot_integration',
    'cleanup_bot_integration',
    'get_bot_integration',
    'get_bot_router',
    'add_bot_to_room',
    'remove_bot_from_room',
    'get_available_bots',
    'get_room_bot_status'
]