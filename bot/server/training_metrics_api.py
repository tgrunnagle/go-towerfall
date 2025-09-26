"""
FastAPI endpoints for training metrics functionality.

This module provides REST API endpoints for managing training sessions,
metrics updates, and historical data retrieval.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from server.data_models import (
    TrainingMetricsData,
    BotDecisionData,
    PerformanceGraphData,
    TrainingSessionInfo,
    TrainingSessionRequest,
    TrainingSessionUpdate,
    HistoricalDataRequest,
    HistoricalDataResponse,
    TrainingStatus,
    MessageType
)
from server.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/training", tags=["training"])

# Global instances (would be injected in production)
websocket_manager: Optional[WebSocketManager] = None
training_sessions: Optional[Dict[str, TrainingSessionInfo]] = None
metrics_history: Optional[Dict[str, List[TrainingMetricsData]]] = None
graph_data: Optional[Dict[str, Dict[str, PerformanceGraphData]]] = None
config: Optional[Any] = None


# Dependency injection
def get_websocket_manager() -> WebSocketManager:
    """Get WebSocket manager instance."""
    if websocket_manager is None:
        raise HTTPException(status_code=500, detail="WebSocket manager not initialized")
    return websocket_manager


def get_training_sessions() -> Dict[str, TrainingSessionInfo]:
    """Get training sessions storage."""
    if training_sessions is None:
        raise HTTPException(status_code=500, detail="Training sessions storage not initialized")
    return training_sessions


def get_metrics_history() -> Dict[str, List[TrainingMetricsData]]:
    """Get metrics history storage."""
    if metrics_history is None:
        raise HTTPException(status_code=500, detail="Metrics history storage not initialized")
    return metrics_history


def get_graph_data() -> Dict[str, Dict[str, PerformanceGraphData]]:
    """Get graph data storage."""
    if graph_data is None:
        raise HTTPException(status_code=500, detail="Graph data storage not initialized")
    return graph_data


def get_config() -> Any:
    """Get server configuration."""
    if config is None:
        raise HTTPException(status_code=500, detail="Server configuration not initialized")
    return config


# API Endpoints
@router.post("/sessions", response_model=TrainingSessionInfo)
async def create_training_session(
    request: TrainingSessionRequest,
    sessions: Dict[str, TrainingSessionInfo] = Depends(get_training_sessions),
    metrics_hist: Dict[str, List[TrainingMetricsData]] = Depends(get_metrics_history),
    graphs: Dict[str, Dict[str, PerformanceGraphData]] = Depends(get_graph_data),
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
) -> TrainingSessionInfo:
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
    
    sessions[request.training_session_id] = session_info
    metrics_hist[request.training_session_id] = []
    graphs[request.training_session_id] = {}
    
    logger.info(f"Created training session: {request.training_session_id}")
    
    # Broadcast session creation
    await ws_manager.broadcast_training_status(
        request.training_session_id,
        {"event": "session_created", "session": session_info.dict()}
    )
    
    return session_info


@router.get("/sessions", response_model=List[TrainingSessionInfo])
async def list_training_sessions(
    sessions: Dict[str, TrainingSessionInfo] = Depends(get_training_sessions)
) -> List[TrainingSessionInfo]:
    """List all active training sessions."""
    return list(sessions.values())


@router.get("/sessions/{session_id}", response_model=TrainingSessionInfo)
async def get_training_session(
    session_id: str,
    sessions: Dict[str, TrainingSessionInfo] = Depends(get_training_sessions)
) -> TrainingSessionInfo:
    """Get information about a specific training session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return sessions[session_id]


@router.put("/sessions/{session_id}", response_model=TrainingSessionInfo)
async def update_training_session(
    session_id: str,
    update: TrainingSessionUpdate,
    sessions: Dict[str, TrainingSessionInfo] = Depends(get_training_sessions),
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
) -> TrainingSessionInfo:
    """Update a training session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = sessions[session_id]
    
    if update.status is not None:
        session.status = update.status
    if update.current_episode is not None:
        session.current_episode = update.current_episode
    if update.end_time is not None:
        session.end_time = update.end_time
    
    # Broadcast session update
    await ws_manager.broadcast_training_status(
        session_id,
        {"event": "session_updated", "session": session.dict()}
    )
    
    return session


@router.delete("/sessions/{session_id}")
async def delete_training_session(
    session_id: str,
    sessions: Dict[str, TrainingSessionInfo] = Depends(get_training_sessions),
    metrics_hist: Dict[str, List[TrainingMetricsData]] = Depends(get_metrics_history),
    graphs: Dict[str, Dict[str, PerformanceGraphData]] = Depends(get_graph_data),
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, str]:
    """Delete a training session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    # Clean up data
    del sessions[session_id]
    if session_id in metrics_hist:
        del metrics_hist[session_id]
    if session_id in graphs:
        del graphs[session_id]
    
    # Broadcast session deletion
    await ws_manager.broadcast_training_status(
        session_id,
        {"event": "session_deleted", "session_id": session_id}
    )
    
    logger.info(f"Deleted training session: {session_id}")
    
    return {"message": "Training session deleted"}


@router.post("/sessions/{session_id}/metrics")
async def update_training_metrics(
    session_id: str,
    metrics: TrainingMetricsData,
    sessions: Dict[str, TrainingSessionInfo] = Depends(get_training_sessions),
    metrics_hist: Dict[str, List[TrainingMetricsData]] = Depends(get_metrics_history),
    ws_manager: WebSocketManager = Depends(get_websocket_manager),
    server_config: Any = Depends(get_config)
) -> Dict[str, str]:
    """Update training metrics for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    # Store metrics
    if session_id not in metrics_hist:
        metrics_hist[session_id] = []
    
    metrics_hist[session_id].append(metrics)
    
    # Limit history size
    max_size = getattr(server_config, 'metrics_history_size', 10000)
    if len(metrics_hist[session_id]) > max_size:
        metrics_hist[session_id] = metrics_hist[session_id][-max_size:]
    
    # Update session info
    session = sessions[session_id]
    session.current_episode = metrics.episode
    # Note: spectator_count would be updated by the connection manager
    
    # Broadcast metrics update
    await ws_manager.broadcast_training_metrics(session_id, metrics)
    
    return {"message": "Metrics updated"}


@router.post("/sessions/{session_id}/bot_decision")
async def update_bot_decision(
    session_id: str,
    decision: BotDecisionData,
    sessions: Dict[str, TrainingSessionInfo] = Depends(get_training_sessions),
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, str]:
    """Update bot decision data for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    # Broadcast bot decision update
    await ws_manager.broadcast_bot_decision(session_id, decision)
    
    return {"message": "Bot decision updated"}


@router.post("/sessions/{session_id}/graph_update")
async def update_performance_graph(
    session_id: str,
    graph: PerformanceGraphData,
    sessions: Dict[str, TrainingSessionInfo] = Depends(get_training_sessions),
    graphs: Dict[str, Dict[str, PerformanceGraphData]] = Depends(get_graph_data),
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, str]:
    """Update performance graph data for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    # Store graph data
    if session_id not in graphs:
        graphs[session_id] = {}
    
    graphs[session_id][graph.graph_id] = graph
    
    # Broadcast graph update
    await ws_manager.broadcast_graph_update(session_id, graph)
    
    return {"message": "Graph updated"}


@router.get("/sessions/{session_id}/history", response_model=HistoricalDataResponse)
async def get_historical_data(
    session_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    max_points: Optional[int] = Query(1000),
    metrics: Optional[str] = Query(None),  # Comma-separated list
    sessions: Dict[str, TrainingSessionInfo] = Depends(get_training_sessions),
    metrics_hist: Dict[str, List[TrainingMetricsData]] = Depends(get_metrics_history),
    graphs: Dict[str, Dict[str, PerformanceGraphData]] = Depends(get_graph_data)
) -> HistoricalDataResponse:
    """Get historical training data for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    # Get metrics history
    metrics_data = metrics_hist.get(session_id, [])
    
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
    graph_data_list = list(graphs.get(session_id, {}).values())
    
    return HistoricalDataResponse(
        session_id=session_id,
        metrics_data=metrics_data,
        graph_data=graph_data_list,
        total_points=len(metrics_data)
    )


# Initialization function
def initialize_training_metrics_api(
    ws_manager: WebSocketManager,
    sessions: Dict[str, TrainingSessionInfo],
    metrics_hist: Dict[str, List[TrainingMetricsData]],
    graphs: Dict[str, Dict[str, PerformanceGraphData]],
    server_config: Any
) -> None:
    """
    Initialize the training metrics API with required instances.
    
    Args:
        ws_manager: WebSocketManager instance
        sessions: Training sessions storage
        metrics_hist: Metrics history storage
        graphs: Graph data storage
        server_config: Server configuration
    """
    global websocket_manager, training_sessions, metrics_history, graph_data, config
    
    websocket_manager = ws_manager
    training_sessions = sessions
    metrics_history = metrics_hist
    graph_data = graphs
    config = server_config
    
    logger.info("Training Metrics API initialized successfully")