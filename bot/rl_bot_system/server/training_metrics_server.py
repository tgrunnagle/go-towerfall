"""
FastAPI server for training metrics data.

This module implements a FastAPI server that serves training metrics data
to frontend overlays and manages WebSocket connections for real-time updates.
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rl_bot_system.server.data_models import (
    TrainingMetricsData,
    BotDecisionData,
    PerformanceGraphData,
    TrainingSessionInfo,
    TrainingSessionRequest,
    TrainingSessionUpdate,
    HistoricalDataRequest,
    HistoricalDataResponse,
    ServerStatus,
    ErrorResponse,
    TrainingStatus,
    MessageType
)
from rl_bot_system.server.websocket_manager import ConnectionManager, WebSocketManager

# Optional imports - these may fail due to complex dependencies
try:
    from rl_bot_system.spectator.spectator_manager import SpectatorManager
except ImportError:
    SpectatorManager = None

try:
    from rl_bot_system.training.training_engine import TrainingEngine
except ImportError:
    TrainingEngine = None

logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    """Configuration for the training metrics server."""
    host: str = "localhost"
    port: int = 8000
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:4000"]
    max_connections_per_session: int = 50
    metrics_history_size: int = 10000
    cleanup_interval_seconds: int = 300
    log_level: str = "INFO"
    
    # Integration settings
    game_server_url: str = "http://localhost:4000"
    enable_spectator_integration: bool = True
    
    # Storage settings
    data_storage_path: str = "data/training_metrics"
    enable_data_persistence: bool = True


class TrainingMetricsServer:
    """
    FastAPI server for training metrics and spectator data.
    
    Provides REST API endpoints and WebSocket connections for real-time
    training metrics, bot decisions, and performance graphs.
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = FastAPI(
            title="Training Metrics Server",
            description="Real-time training metrics and spectator data API",
            version="1.0.0"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
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
        
        # Server state
        self.start_time = datetime.now()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Setup routes
        self._setup_routes()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        
        logger.info(f"Training Metrics Server initialized on {config.host}:{config.port}")
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # Server status
        @self.app.get("/status", response_model=ServerStatus)
        async def get_server_status():
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return ServerStatus(
                status="running",
                active_sessions=len(self._training_sessions),
                total_connections=self.connection_manager.get_connection_count(),
                uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
                memory_usage_mb=memory_info.rss / 1024 / 1024,
                cpu_usage_percent=process.cpu_percent()
            )
        
        # Training session management
        @self.app.post("/api/training/sessions", response_model=TrainingSessionInfo)
        async def create_training_session(request: TrainingSessionRequest):
            """Create a new training session."""
            session_info = TrainingSessionInfo(
                session_id=request.training_session_id,
                training_session_id=request.training_session_id,
                model_generation=request.model_generation,
                algorithm=request.algorithm,
                status=TrainingStatus.STARTING,
                start_time=datetime.now(),
                current_episode=0,
                total_episodes=request.total_episodes,
                spectator_count=0,
                room_code=request.room_code
            )
            
            self._training_sessions[request.training_session_id] = session_info
            self._metrics_history[request.training_session_id] = []
            self._graph_data[request.training_session_id] = {}
            
            logger.info(f"Created training session: {request.training_session_id}")
            
            # Broadcast session creation
            await self.websocket_manager.broadcast_training_status(
                request.training_session_id,
                {"event": "session_created", "session": session_info.dict()}
            )
            
            return session_info
        
        @self.app.get("/api/training/sessions", response_model=List[TrainingSessionInfo])
        async def list_training_sessions():
            """List all active training sessions."""
            return list(self._training_sessions.values())
        
        @self.app.get("/api/training/sessions/{session_id}", response_model=TrainingSessionInfo)
        async def get_training_session(session_id: str):
            """Get information about a specific training session."""
            if session_id not in self._training_sessions:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            return self._training_sessions[session_id]
        
        @self.app.put("/api/training/sessions/{session_id}", response_model=TrainingSessionInfo)
        async def update_training_session(session_id: str, update: TrainingSessionUpdate):
            """Update a training session."""
            if session_id not in self._training_sessions:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            session = self._training_sessions[session_id]
            
            if update.status is not None:
                session.status = update.status
            if update.current_episode is not None:
                session.current_episode = update.current_episode
            if update.end_time is not None:
                session.end_time = update.end_time
            
            # Broadcast session update
            await self.websocket_manager.broadcast_training_status(
                session_id,
                {"event": "session_updated", "session": session.dict()}
            )
            
            return session
        
        @self.app.delete("/api/training/sessions/{session_id}")
        async def delete_training_session(session_id: str):
            """Delete a training session."""
            if session_id not in self._training_sessions:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            # Clean up data
            del self._training_sessions[session_id]
            if session_id in self._metrics_history:
                del self._metrics_history[session_id]
            if session_id in self._graph_data:
                del self._graph_data[session_id]
            
            # Broadcast session deletion
            await self.websocket_manager.broadcast_training_status(
                session_id,
                {"event": "session_deleted", "session_id": session_id}
            )
            
            logger.info(f"Deleted training session: {session_id}")
            
            return {"message": "Training session deleted"}
        
        # Metrics endpoints
        @self.app.post("/api/training/sessions/{session_id}/metrics")
        async def update_training_metrics(session_id: str, metrics: TrainingMetricsData):
            """Update training metrics for a session."""
            if session_id not in self._training_sessions:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            # Store metrics
            if session_id not in self._metrics_history:
                self._metrics_history[session_id] = []
            
            self._metrics_history[session_id].append(metrics)
            
            # Limit history size
            max_size = self.config.metrics_history_size
            if len(self._metrics_history[session_id]) > max_size:
                self._metrics_history[session_id] = self._metrics_history[session_id][-max_size:]
            
            # Update session info
            session = self._training_sessions[session_id]
            session.current_episode = metrics.episode
            session.spectator_count = self.connection_manager.get_connection_count(session_id)
            
            # Broadcast metrics update
            await self.websocket_manager.broadcast_training_metrics(session_id, metrics)
            
            return {"message": "Metrics updated"}
        
        @self.app.post("/api/training/sessions/{session_id}/bot_decision")
        async def update_bot_decision(session_id: str, decision: BotDecisionData):
            """Update bot decision data for a session."""
            if session_id not in self._training_sessions:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            # Broadcast bot decision update
            await self.websocket_manager.broadcast_bot_decision(session_id, decision)
            
            return {"message": "Bot decision updated"}
        
        @self.app.post("/api/training/sessions/{session_id}/graph_update")
        async def update_performance_graph(session_id: str, graph: PerformanceGraphData):
            """Update performance graph data for a session."""
            if session_id not in self._training_sessions:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            # Store graph data
            if session_id not in self._graph_data:
                self._graph_data[session_id] = {}
            
            self._graph_data[session_id][graph.graph_id] = graph
            
            # Broadcast graph update
            await self.websocket_manager.broadcast_graph_update(session_id, graph)
            
            return {"message": "Graph updated"}
        
        # Historical data endpoints
        @self.app.get("/api/training/sessions/{session_id}/history", response_model=HistoricalDataResponse)
        async def get_historical_data(
            session_id: str,
            start_time: Optional[datetime] = Query(None),
            end_time: Optional[datetime] = Query(None),
            max_points: Optional[int] = Query(1000),
            metrics: Optional[str] = Query(None)  # Comma-separated list
        ):
            """Get historical training data for a session."""
            if session_id not in self._training_sessions:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            # Get metrics history
            metrics_data = self._metrics_history.get(session_id, [])
            
            # Filter by time range
            if start_time or end_time:
                filtered_metrics = []
                for metric in metrics_data:
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    filtered_metrics.append(metric)
                metrics_data = filtered_metrics
            
            # Limit number of points
            if max_points and len(metrics_data) > max_points:
                step = len(metrics_data) // max_points
                metrics_data = metrics_data[::step]
            
            # Get graph data
            graph_data = list(self._graph_data.get(session_id, {}).values())
            
            return HistoricalDataResponse(
                session_id=session_id,
                metrics_data=metrics_data,
                graph_data=graph_data,
                total_points=len(metrics_data)
            )
        
        # WebSocket endpoint
        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(
            websocket: WebSocket,
            session_id: str,
            user_name: str = Query(...),
            user_id: Optional[str] = Query(None)
        ):
            """WebSocket endpoint for real-time training metrics."""
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
            logger.info("Starting Training Metrics Server...")
            
            # Create data storage directory
            if self.config.enable_data_persistence:
                Path(self.config.data_storage_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize spectator manager if enabled and available
            if self.config.enable_spectator_integration and SpectatorManager is not None:
                self.spectator_manager = SpectatorManager(self.config.game_server_url)
            elif self.config.enable_spectator_integration:
                logger.warning("SpectatorManager not available - spectator integration disabled")
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            logger.info("Training Metrics Server started successfully")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Clean up resources on shutdown."""
            logger.info("Shutting down Training Metrics Server...")
            
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
            
            logger.info("Training Metrics Server shutdown complete")
    
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
                
                # Sleep until next cleanup
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def run(self) -> None:
        """Run the server."""
        import uvicorn
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level.lower()
        )


# Convenience function for creating and running server
def create_server(config: Optional[ServerConfig] = None) -> TrainingMetricsServer:
    """
    Create a training metrics server instance.
    
    Args:
        config: Optional server configuration
        
    Returns:
        TrainingMetricsServer instance
    """
    if config is None:
        config = ServerConfig()
    
    return TrainingMetricsServer(config)


def run_server(config: Optional[ServerConfig] = None) -> None:
    """
    Create and run a training metrics server.
    
    Args:
        config: Optional server configuration
    """
    server = create_server(config)
    server.run()


if __name__ == "__main__":
    # Run server with default configuration
    run_server()