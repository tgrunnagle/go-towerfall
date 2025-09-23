#!/usr/bin/env python3
"""
Training Metrics Server Startup Script

This script starts the FastAPI training metrics server with proper import handling.
It can be run directly from the bot/ directory without import issues.
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Now we can import our modules
try:
    import asyncio
    import json
    from datetime import datetime, timedelta
    from typing import Dict, Any, List, Optional
    
    # Try to import FastAPI dependencies
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
        FASTAPI_AVAILABLE = True
    except ImportError as e:
        print(f"FastAPI dependencies not available: {e}")
        print("Please install: pip install fastapi uvicorn pydantic")
        FASTAPI_AVAILABLE = False
    
    # Try to import psutil
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        print("psutil not available, server status will be limited")
        PSUTIL_AVAILABLE = False
    
    # Import our server components directly
    try:
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
    except ImportError as e:
        print(f"Failed to import core server components: {e}")
        raise
    
    # Optional imports - these may fail due to dependencies
    SpectatorManager = None
    TrainingEngine = None
    
    # Don't import these as they have complex dependencies that require 'bot' package
    # They can be imported later if needed for full integration
    print("Note: SpectatorManager and TrainingEngine integration disabled in standalone mode")
    print("      Use the full server implementation for complete integration")

except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you're running this script from the bot/ directory")
    sys.exit(1)


class SimpleServerConfig:
    """Simple server configuration without complex dependencies."""
    
    def __init__(self, **kwargs):
        self.host = kwargs.get('host', 'localhost')
        self.port = kwargs.get('port', 4002)
        self.cors_origins = kwargs.get('cors_origins', ["http://localhost:3000", "http://localhost:4000"])
        self.max_connections_per_session = kwargs.get('max_connections_per_session', 50)
        self.metrics_history_size = kwargs.get('metrics_history_size', 10000)
        self.cleanup_interval_seconds = kwargs.get('cleanup_interval_seconds', 300)
        self.log_level = kwargs.get('log_level', 'INFO')
        self.game_server_url = kwargs.get('game_server_url', 'http://localhost:4000')
        self.enable_spectator_integration = kwargs.get('enable_spectator_integration', True)
        self.data_storage_path = kwargs.get('data_storage_path', 'data/training_metrics')
        self.enable_data_persistence = kwargs.get('enable_data_persistence', True)


class SimpleTrainingMetricsServer:
    """Simplified training metrics server that can run standalone."""
    
    def __init__(self, config: SimpleServerConfig):
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI dependencies are required to run the server")
        
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
        
        # Server state
        self.start_time = datetime.now()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Setup routes
        self._setup_routes()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        
        print(f"Training Metrics Server initialized on {config.host}:{config.port}")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/status")
        async def get_server_status():
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_usage_mb = memory_info.rss / 1024 / 1024
                cpu_usage_percent = process.cpu_percent()
            else:
                memory_usage_mb = 0.0
                cpu_usage_percent = 0.0
            
            return {
                "status": "running",
                "active_sessions": len(self._training_sessions),
                "total_connections": self.connection_manager.get_connection_count(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "memory_usage_mb": memory_usage_mb,
                "cpu_usage_percent": cpu_usage_percent
            }
        
        @self.app.post("/api/training/sessions")
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
            
            print(f"Created training session: {request.training_session_id}")
            
            return session_info
        
        @self.app.get("/api/training/sessions")
        async def list_training_sessions():
            """List all active training sessions."""
            return list(self._training_sessions.values())
        
        @self.app.get("/api/training/sessions/{session_id}")
        async def get_training_session(session_id: str):
            """Get information about a specific training session."""
            if session_id not in self._training_sessions:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            return self._training_sessions[session_id]
        
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
                
                # Keep connection alive
                while True:
                    try:
                        message = await websocket.receive_text()
                        await self.connection_manager.handle_message(connection_id, message)
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        print(f"Error handling WebSocket message: {e}")
                        break
            
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                if connection_id:
                    await self.connection_manager.disconnect(connection_id)
    
    def _setup_event_handlers(self):
        """Setup event handlers for startup and shutdown."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize server components on startup."""
            print("Starting Training Metrics Server...")
            
            # Create data storage directory
            if self.config.enable_data_persistence:
                Path(self.config.data_storage_path).mkdir(parents=True, exist_ok=True)
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            print("Training Metrics Server started successfully")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Clean up resources on shutdown."""
            print("Shutting down Training Metrics Server...")
            
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up connection manager
            await self.connection_manager.cleanup()
            
            print("Training Metrics Server shutdown complete")
    
    async def _periodic_cleanup(self):
        """Background task for periodic cleanup."""
        while True:
            try:
                current_time = datetime.now()
                
                # Clean up old metrics data
                for session_id in list(self._metrics_history.keys()):
                    if session_id not in self._training_sessions:
                        del self._metrics_history[session_id]
                        if session_id in self._graph_data:
                            del self._graph_data[session_id]
                
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)
    
    def run(self):
        """Run the server."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level.lower()
        )


def main():
    """Main entry point for the training metrics server."""
    parser = argparse.ArgumentParser(description="Training Metrics Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=4002, help="Server port")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Create configuration
    config_kwargs = {
        'host': args.host,
        'port': args.port,
        'log_level': args.log_level
    }
    
    # Load config file if provided
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config_kwargs.update(file_config)
        except Exception as e:
            print(f"Failed to load config file: {e}")
    
    config = SimpleServerConfig(**config_kwargs)
    
    # Create and run server
    try:
        server = SimpleTrainingMetricsServer(config)
        print(f"Starting server on {config.host}:{config.port}")
        server.run()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()