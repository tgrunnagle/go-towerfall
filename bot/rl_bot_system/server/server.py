"""
Base FastAPI server that combines training metrics and replay APIs.

This module provides a unified server that includes both training metrics
and episode replay functionality with proper integration.
"""

import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rl_bot_system.server.data_models import (
    TrainingMetricsData,
    BotDecisionData,
    PerformanceGraphData,
    TrainingSessionInfo,
    ServerStatus,
    TrainingStatus,
    MessageType
)
from rl_bot_system.server.websocket_manager import ConnectionManager, WebSocketManager

# Import API routers
from rl_bot_system.server.training_metrics_api import (
    router as training_router,
    initialize_training_metrics_api
)

# Optional imports for replay functionality
try:
    from rl_bot_system.server.replay_api import (
        router as replay_router,
        initialize_replay_api
    )
    from rl_bot_system.replay.replay_manager import ReplayManager
    from rl_bot_system.spectator.episode_replay import EpisodeReplayManager
    REPLAY_AVAILABLE = True
except ImportError:
    replay_router = None
    initialize_replay_api = None
    ReplayManager = None
    EpisodeReplayManager = None
    REPLAY_AVAILABLE = False

# Optional imports for spectator functionality
try:
    from rl_bot_system.spectator.spectator_manager import SpectatorManager
    SPECTATOR_AVAILABLE = True
except ImportError:
    SpectatorManager = None
    SPECTATOR_AVAILABLE = False

# Optional imports for training engine
try:
    from rl_bot_system.training.training_engine import TrainingEngine
    TRAINING_ENGINE_AVAILABLE = True
except ImportError:
    TrainingEngine = None
    TRAINING_ENGINE_AVAILABLE = False

# Optional imports for bot server
try:
    from rl_bot_system.server.bot_server_api import (
        BotServerApi, BotServerConfig as BotConfig
    )
    BOT_SERVER_AVAILABLE = True
except ImportError:
    BotServerApi = None
    BotConfig = None
    BOT_SERVER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    """Configuration for the unified server."""
    host: str = "localhost"
    port: int = 4002
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:4000", "http://localhost:4001"]
    max_connections_per_session: int = 50
    metrics_history_size: int = 10000
    cleanup_interval_seconds: int = 300
    log_level: str = "INFO"
    
    # Integration settings
    game_server_url: str = "http://localhost:4000"
    enable_spectator_integration: bool = True
    enable_replay_integration: bool = True
    
    # Storage settings
    data_storage_path: str = "data/training_metrics"
    replay_storage_path: str = "data/replays"
    enable_data_persistence: bool = True
    
    # Bot server settings
    enable_bot_server: bool = True
    max_bots_per_room: int = 8
    max_total_bots: int = 50
    bot_timeout_seconds: int = 300
    models_dir: str = "bot/data/models"


class UnifiedServer:
    """
    Unified FastAPI server that combines training metrics and replay functionality.
    
    Provides a single server instance that includes:
    - Training metrics API endpoints
    - Episode replay API endpoints
    - WebSocket connections for real-time updates
    - Integration with spectator and training systems
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = FastAPI(
            title="RL Bot Training Server",
            description="Unified server for training metrics, episode replay, and spectator functionality",
            version="1.0.0"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        
        # Core components
        self.connection_manager = ConnectionManager()
        self.websocket_manager = WebSocketManager(self.connection_manager)
        
        # Data storage
        self._training_sessions: Dict[str, TrainingSessionInfo] = {}
        self._metrics_history: Dict[str, List[TrainingMetricsData]] = {}
        self._graph_data: Dict[str, Dict[str, PerformanceGraphData]] = {}
        
        # Integration components
        self.spectator_manager: Optional[SpectatorManager] = None
        self.training_engine: Optional[TrainingEngine] = None
        self.replay_manager: Optional[ReplayManager] = None
        self.episode_replay_manager: Optional[EpisodeReplayManager] = None
        self.bot_server_api: Optional[BotServerApi] = None
        
        # Server state
        self.start_time = datetime.now()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Setup routes and APIs
        self._setup_core_routes()
        self._setup_api_routers()
        self._setup_websocket_endpoint()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        
        logger.info(f"Unified Server initialized on {config.host}:{config.port}")
    
    def _setup_core_routes(self) -> None:
        """Setup core server routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/status", response_model=ServerStatus)
        async def get_server_status():
            """Get server status information."""
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_usage_mb = memory_info.rss / 1024 / 1024
                cpu_usage_percent = process.cpu_percent()
            except Exception:
                memory_usage_mb = 0.0
                cpu_usage_percent = 0.0
            
            return ServerStatus(
                status="running",
                active_sessions=len(self._training_sessions),
                total_connections=self.connection_manager.get_connection_count(),
                uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent
            )
        
        @self.app.get("/info")
        async def get_server_info():
            """Get server information and available features."""
            return {
                "name": "RL Bot Training Server",
                "version": "1.0.0",
                "features": {
                    "training_metrics": True,
                    "replay_system": REPLAY_AVAILABLE,
                    "spectator_integration": SPECTATOR_AVAILABLE,
                    "training_engine": TRAINING_ENGINE_AVAILABLE,
                    "bot_server": BOT_SERVER_AVAILABLE and self.config.enable_bot_server
                },
                "config": {
                    "max_connections_per_session": self.config.max_connections_per_session,
                    "metrics_history_size": self.config.metrics_history_size,
                    "data_persistence": self.config.enable_data_persistence,
                    "max_bots_per_room": self.config.max_bots_per_room if self.config.enable_bot_server else 0,
                    "max_total_bots": self.config.max_total_bots if self.config.enable_bot_server else 0
                }
            }
    
    def _setup_api_routers(self) -> None:
        """Setup API routers for different functionality."""
        
        # Initialize training metrics API
        initialize_training_metrics_api(
            self.websocket_manager,
            self._training_sessions,
            self._metrics_history,
            self._graph_data,
            self.config
        )
        
        # Include training metrics router
        self.app.include_router(training_router)
        logger.info("Training Metrics API router included")
        
        # Include replay API router if available
        if REPLAY_AVAILABLE and replay_router is not None:
            self.app.include_router(replay_router)
            logger.info("Replay API router included")
        else:
            logger.warning("Replay API not available - replay functionality disabled")
    
    def _setup_websocket_endpoint(self) -> None:
        """Setup WebSocket endpoint for real-time communication."""
        
        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(
            websocket: WebSocket,
            session_id: str,
            user_name: str = Query(...),
            user_id: Optional[str] = Query(None)
        ):
            """WebSocket endpoint for real-time training metrics and replay updates."""
            connection_id = None
            
            try:
                # Check if session exists
                if session_id not in self._training_sessions:
                    await websocket.close(code=4004, reason="Training session not found")
                    return
                
                # Check connection limit
                current_connections = self.connection_manager.get_connection_count(session_id)
                if current_connections >= self.config.max_connections_per_session:
                    await websocket.close(code=4003, reason="Too many connections")
                    return
                
                # Accept connection
                connection_id = await self.connection_manager.connect(
                    websocket, session_id, user_name, user_id
                )
                
                # Send current session info
                session_info = self._training_sessions[session_id]
                await self.connection_manager.send_to_connection(
                    connection_id,
                    MessageType.TRAINING_STATUS,
                    {"event": "session_info", "session": session_info.dict()}
                )
                
                # Send recent metrics if available
                recent_metrics = self._metrics_history.get(session_id, [])
                if recent_metrics:
                    latest_metrics = recent_metrics[-1]
                    await self.connection_manager.send_to_connection(
                        connection_id,
                        MessageType.TRAINING_METRICS,
                        latest_metrics.dict()
                    )
                
                # Send current graph data
                for graph in self._graph_data.get(session_id, {}).values():
                    await self.connection_manager.send_to_connection(
                        connection_id,
                        MessageType.GRAPH_UPDATE,
                        graph.dict()
                    )
                
                # Keep connection alive and handle messages
                while True:
                    try:
                        message = await websocket.receive_text()
                        await self.connection_manager.handle_message(connection_id, message)
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
                        break
            
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                if connection_id:
                    await self.connection_manager.disconnect(connection_id)
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for startup and shutdown."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize server components on startup."""
            logger.info("Starting Unified Server...")
            
            # Create data storage directories
            if self.config.enable_data_persistence:
                Path(self.config.data_storage_path).mkdir(parents=True, exist_ok=True)
                if REPLAY_AVAILABLE:
                    Path(self.config.replay_storage_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize spectator manager if enabled and available
            if self.config.enable_spectator_integration and SPECTATOR_AVAILABLE:
                self.spectator_manager = SpectatorManager(self.config.game_server_url)
                logger.info("Spectator manager initialized")
            elif self.config.enable_spectator_integration:
                logger.warning("SpectatorManager not available - spectator integration disabled")
            
            # Initialize replay system if enabled and available
            if self.config.enable_replay_integration and REPLAY_AVAILABLE:
                await self._initialize_replay_system()
            elif self.config.enable_replay_integration:
                logger.warning("Replay system not available - replay functionality disabled")
            
            # Initialize bot server if enabled and available
            if self.config.enable_bot_server and BOT_SERVER_AVAILABLE:
                await self._initialize_bot_server()
            elif self.config.enable_bot_server:
                logger.warning("Bot server not available - bot functionality disabled")
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            logger.info("Unified Server started successfully")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Clean up resources on shutdown."""
            logger.info("Shutting down Unified Server...")
            
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up connection manager
            await self.connection_manager.cleanup()
            
            # Clean up spectator manager
            if self.spectator_manager:
                await self.spectator_manager.cleanup()
            
            # Clean up replay system
            if self.episode_replay_manager:
                await self.episode_replay_manager.cleanup()
            
            # Clean up bot server
            if self.bot_server_api:
                await self.bot_server_api.cleanup()
            
            logger.info("Unified Server shutdown complete")
    
    async def _initialize_replay_system(self) -> None:
        """Initialize the replay system components."""
        try:
            # Initialize replay manager
            self.replay_manager = ReplayManager(
                storage_path=self.config.replay_storage_path
            )
            
            # Initialize episode replay manager
            self.episode_replay_manager = EpisodeReplayManager(
                self.replay_manager,
                self.spectator_manager
            )
            
            # Initialize replay API with manager instances
            if initialize_replay_api is not None:
                initialize_replay_api(
                    self.replay_manager,
                    self.spectator_manager,
                    self.episode_replay_manager
                )
                logger.info("Replay system initialized successfully")
            else:
                logger.warning("Replay API initialization function not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize replay system: {e}")
            self.replay_manager = None
            self.episode_replay_manager = None
    
    async def _initialize_bot_server(self) -> None:
        """Initialize the bot server components."""
        try:
            # Create bot server configuration
            bot_config = BotConfig(
                max_bots_per_room=self.config.max_bots_per_room,
                max_total_bots=self.config.max_total_bots,
                bot_timeout_seconds=self.config.bot_timeout_seconds,
                game_server_url=self.config.game_server_url,
                models_dir=self.config.models_dir
            )
            
            # Initialize bot integration
            self.bot_server_api = BotServerApi(bot_config)
            await self.bot_server_api.initialize()
            
            # Include bot API router
            self.app.include_router(self.bot_server_api.router)
            logger.info("Bot server initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize bot server: {e}")
            self.bot_server_api = None
    
    async def _periodic_cleanup(self) -> None:
        """Background task for periodic cleanup."""
        while True:
            try:
                current_time = datetime.now()
                
                # Clean up old metrics data
                for session_id in list(self._metrics_history.keys()):
                    if session_id not in self._training_sessions:
                        # Session no longer exists, clean up its data
                        del self._metrics_history[session_id]
                        if session_id in self._graph_data:
                            del self._graph_data[session_id]
                        logger.info(f"Cleaned up data for deleted session: {session_id}")
                
                # Clean up completed sessions older than 24 hours
                sessions_to_remove = []
                for session_id, session in self._training_sessions.items():
                    if (session.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.STOPPED] and
                        session.end_time and 
                        current_time - session.end_time > timedelta(hours=24)):
                        sessions_to_remove.append(session_id)
                
                for session_id in sessions_to_remove:
                    del self._training_sessions[session_id]
                    if session_id in self._metrics_history:
                        del self._metrics_history[session_id]
                    if session_id in self._graph_data:
                        del self._graph_data[session_id]
                    logger.info(f"Cleaned up old completed session: {session_id}")
                
                # Update spectator counts
                for session_id, session in self._training_sessions.items():
                    session.spectator_count = self.connection_manager.get_connection_count(session_id)
                
                # Sleep until next cleanup
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    # Integration methods
    async def integrate_with_training_engine(self, training_engine) -> None:
        """
        Integrate with a training engine for automatic metrics collection.
        
        Args:
            training_engine: TrainingEngine instance to integrate with
        """
        self.training_engine = training_engine
        
        # Register callbacks for training events
        # This would depend on the training engine's callback system
        logger.info("Integrated with training engine")
    
    async def integrate_with_spectator_manager(self, spectator_manager) -> None:
        """
        Integrate with spectator manager for coordinated spectator handling.
        
        Args:
            spectator_manager: SpectatorManager instance to integrate with
        """
        self.spectator_manager = spectator_manager
        
        # Register callbacks for spectator events
        # This would coordinate with the existing spectator system
        logger.info("Integrated with spectator manager")
    
    def run(self) -> None:
        """Run the server."""
        import uvicorn
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level.lower()
        )


# Convenience functions
def create_server(config: Optional[ServerConfig] = None) -> UnifiedServer:
    """
    Create a unified server instance.
    
    Args:
        config: Optional server configuration
        
    Returns:
        UnifiedServer instance
    """
    if config is None:
        config = ServerConfig()
    
    return UnifiedServer(config)


def run_server(config: Optional[ServerConfig] = None) -> None:
    """
    Create and run a unified server.
    
    Args:
        config: Optional server configuration
    """
    server = create_server(config)
    server.run()


if __name__ == "__main__":
    # Run server with default configuration
    run_server()