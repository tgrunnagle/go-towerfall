"""
Data models for training metrics server.

This module defines Pydantic models for training metrics, bot decisions,
and performance graphs used by the FastAPI server.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class TrainingStatus(str, Enum):
    """Training session status."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class MessageType(str, Enum):
    """WebSocket message types."""
    TRAINING_METRICS = "training_metrics"
    BOT_DECISION = "bot_decision"
    GRAPH_UPDATE = "graph_update"
    TRAINING_STATUS = "training_status"
    CONNECTION_STATUS = "connection_status"
    ERROR = "error"


class TrainingMetricsData(BaseModel):
    """Training metrics data model."""
    timestamp: datetime
    episode: int
    total_episodes: int
    current_reward: float
    average_reward: float
    best_reward: float
    episode_length: int
    win_rate: float
    loss_value: Optional[float] = None
    learning_rate: float
    epsilon: Optional[float] = None  # For DQN
    model_generation: int
    algorithm: str
    training_time_elapsed: float
    
    # Performance metrics
    actions_per_second: Optional[float] = None
    frames_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class BotDecisionData(BaseModel):
    """Bot decision visualization data model."""
    timestamp: datetime
    action_probabilities: Dict[str, float]
    state_values: Optional[float] = None
    q_values: Optional[List[float]] = None
    selected_action: Optional[str] = None
    confidence_score: Optional[float] = None
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class GraphDataPoint(BaseModel):
    """Single data point for performance graphs."""
    timestamp: datetime
    value: float
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class PerformanceGraphData(BaseModel):
    """Performance graph data model."""
    graph_id: str
    title: str
    y_label: str
    metrics: List[str]
    data_points: Dict[str, List[GraphDataPoint]]
    max_points: int = 1000
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class TrainingSessionInfo(BaseModel):
    """Training session information."""
    session_id: str
    training_session_id: str
    model_generation: int
    algorithm: str
    status: TrainingStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    current_episode: int
    total_episodes: int
    spectator_count: int
    room_code: Optional[str] = None
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class SpectatorConnectionInfo(BaseModel):
    """Spectator connection information."""
    connection_id: str
    user_name: str
    user_id: Optional[str] = None
    connected_at: datetime
    session_id: str
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class WebSocketMessage(BaseModel):
    """WebSocket message wrapper."""
    type: MessageType
    data: Union[
        TrainingMetricsData,
        BotDecisionData,
        PerformanceGraphData,
        TrainingSessionInfo,
        SpectatorConnectionInfo,
        Dict[str, Any]
    ]
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class TrainingSessionRequest(BaseModel):
    """Request to create a training session."""
    training_session_id: str
    model_generation: int
    algorithm: str
    total_episodes: int
    room_code: Optional[str] = None
    enable_spectators: bool = True


class TrainingSessionUpdate(BaseModel):
    """Update to a training session."""
    status: Optional[TrainingStatus] = None
    current_episode: Optional[int] = None
    end_time: Optional[datetime] = None


class HistoricalDataRequest(BaseModel):
    """Request for historical training data."""
    session_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_points: Optional[int] = 1000
    metrics: Optional[List[str]] = None


class HistoricalDataResponse(BaseModel):
    """Response with historical training data."""
    session_id: str
    metrics_data: List[TrainingMetricsData]
    graph_data: List[PerformanceGraphData]
    total_points: int
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class ServerStatus(BaseModel):
    """Server status information."""
    status: str
    active_sessions: int
    total_connections: int
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}