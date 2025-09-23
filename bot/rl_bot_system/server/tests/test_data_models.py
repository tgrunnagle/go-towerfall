"""
Tests for training metrics server data models.

This module tests the Pydantic models used for training metrics,
bot decisions, and performance graphs.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from rl_bot_system.server.data_models import (
    TrainingMetricsData,
    BotDecisionData,
    PerformanceGraphData,
    TrainingSessionInfo,
    SpectatorConnectionInfo,
    WebSocketMessage,
    TrainingStatus,
    MessageType,
    GraphDataPoint,
    HistoricalDataRequest,
    HistoricalDataResponse,
    ServerStatus,
    ErrorResponse
)


class TestTrainingMetricsData:
    """Test TrainingMetricsData model."""
    
    def test_create_training_metrics_data(self):
        """Test creating TrainingMetricsData with required fields."""
        timestamp = datetime.now()
        
        metrics = TrainingMetricsData(
            timestamp=timestamp,
            episode=100,
            total_episodes=1000,
            current_reward=15.5,
            average_reward=12.3,
            best_reward=20.1,
            episode_length=250,
            win_rate=75.5,
            learning_rate=0.001,
            model_generation=2,
            algorithm="DQN",
            training_time_elapsed=3600.0
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.episode == 100
        assert metrics.total_episodes == 1000
        assert metrics.current_reward == 15.5
        assert metrics.average_reward == 12.3
        assert metrics.best_reward == 20.1
        assert metrics.episode_length == 250
        assert metrics.win_rate == 75.5
        assert metrics.learning_rate == 0.001
        assert metrics.model_generation == 2
        assert metrics.algorithm == "DQN"
        assert metrics.training_time_elapsed == 3600.0
    
    def test_training_metrics_data_with_optional_fields(self):
        """Test TrainingMetricsData with optional fields."""
        timestamp = datetime.now()
        
        metrics = TrainingMetricsData(
            timestamp=timestamp,
            episode=50,
            total_episodes=500,
            current_reward=10.0,
            average_reward=8.5,
            best_reward=15.0,
            episode_length=200,
            win_rate=60.0,
            learning_rate=0.0005,
            model_generation=1,
            algorithm="PPO",
            training_time_elapsed=1800.0,
            loss_value=0.05,
            epsilon=0.1,
            actions_per_second=30.5,
            frames_per_second=60.0,
            memory_usage_mb=512.0
        )
        
        assert metrics.loss_value == 0.05
        assert metrics.epsilon == 0.1
        assert metrics.actions_per_second == 30.5
        assert metrics.frames_per_second == 60.0
        assert metrics.memory_usage_mb == 512.0
    
    def test_training_metrics_data_json_serialization(self):
        """Test JSON serialization of TrainingMetricsData."""
        timestamp = datetime.now()
        
        metrics = TrainingMetricsData(
            timestamp=timestamp,
            episode=25,
            total_episodes=100,
            current_reward=5.0,
            average_reward=4.5,
            best_reward=8.0,
            episode_length=150,
            win_rate=45.0,
            learning_rate=0.001,
            model_generation=1,
            algorithm="A2C",
            training_time_elapsed=900.0
        )
        
        json_data = metrics.model_dump_json()
        assert isinstance(json_data, str)
        
        # Parse back and verify
        import json
        parsed = json.loads(json_data)
        assert parsed['episode'] == 25
        assert parsed['algorithm'] == "A2C"
        assert 'timestamp' in parsed


class TestBotDecisionData:
    """Test BotDecisionData model."""
    
    def test_create_bot_decision_data(self):
        """Test creating BotDecisionData."""
        timestamp = datetime.now()
        action_probs = {"move_left": 0.3, "move_right": 0.2, "jump": 0.1, "shoot": 0.4}
        
        decision = BotDecisionData(
            timestamp=timestamp,
            action_probabilities=action_probs,
            state_values=5.5,
            q_values=[1.2, 2.3, 0.8, 3.1],
            selected_action="shoot",
            confidence_score=0.85
        )
        
        assert decision.timestamp == timestamp
        assert decision.action_probabilities == action_probs
        assert decision.state_values == 5.5
        assert decision.q_values == [1.2, 2.3, 0.8, 3.1]
        assert decision.selected_action == "shoot"
        assert decision.confidence_score == 0.85
    
    def test_bot_decision_data_minimal(self):
        """Test BotDecisionData with minimal required fields."""
        timestamp = datetime.now()
        action_probs = {"action_a": 0.6, "action_b": 0.4}
        
        decision = BotDecisionData(
            timestamp=timestamp,
            action_probabilities=action_probs
        )
        
        assert decision.timestamp == timestamp
        assert decision.action_probabilities == action_probs
        assert decision.state_values is None
        assert decision.q_values is None
        assert decision.selected_action is None
        assert decision.confidence_score is None


class TestPerformanceGraphData:
    """Test PerformanceGraphData model."""
    
    def test_create_performance_graph_data(self):
        """Test creating PerformanceGraphData."""
        timestamp1 = datetime.now()
        timestamp2 = datetime.now()
        
        data_points = {
            "reward": [
                GraphDataPoint(timestamp=timestamp1, value=10.0),
                GraphDataPoint(timestamp=timestamp2, value=12.5)
            ],
            "win_rate": [
                GraphDataPoint(timestamp=timestamp1, value=60.0),
                GraphDataPoint(timestamp=timestamp2, value=65.0)
            ]
        }
        
        graph = PerformanceGraphData(
            graph_id="reward_progress",
            title="Reward Progress",
            y_label="Reward",
            metrics=["reward", "win_rate"],
            data_points=data_points,
            max_points=1000
        )
        
        assert graph.graph_id == "reward_progress"
        assert graph.title == "Reward Progress"
        assert graph.y_label == "Reward"
        assert graph.metrics == ["reward", "win_rate"]
        assert len(graph.data_points["reward"]) == 2
        assert len(graph.data_points["win_rate"]) == 2
        assert graph.max_points == 1000


class TestTrainingSessionInfo:
    """Test TrainingSessionInfo model."""
    
    def test_create_training_session_info(self):
        """Test creating TrainingSessionInfo."""
        start_time = datetime.now()
        
        session = TrainingSessionInfo(
            session_id="session_123",
            training_session_id="training_456",
            model_generation=3,
            algorithm="DQN",
            status=TrainingStatus.RUNNING,
            start_time=start_time,
            current_episode=150,
            total_episodes=1000,
            spectator_count=5,
            room_code="ABC123"
        )
        
        assert session.session_id == "session_123"
        assert session.training_session_id == "training_456"
        assert session.model_generation == 3
        assert session.algorithm == "DQN"
        assert session.status == TrainingStatus.RUNNING
        assert session.start_time == start_time
        assert session.current_episode == 150
        assert session.total_episodes == 1000
        assert session.spectator_count == 5
        assert session.room_code == "ABC123"
        assert session.end_time is None
    
    def test_training_session_info_with_end_time(self):
        """Test TrainingSessionInfo with end time."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        session = TrainingSessionInfo(
            session_id="session_789",
            training_session_id="training_101",
            model_generation=1,
            algorithm="PPO",
            status=TrainingStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            current_episode=500,
            total_episodes=500,
            spectator_count=0
        )
        
        assert session.status == TrainingStatus.COMPLETED
        assert session.end_time == end_time


class TestWebSocketMessage:
    """Test WebSocketMessage model."""
    
    def test_create_websocket_message_with_metrics(self):
        """Test creating WebSocketMessage with TrainingMetricsData."""
        timestamp = datetime.now()
        
        metrics = TrainingMetricsData(
            timestamp=timestamp,
            episode=10,
            total_episodes=100,
            current_reward=5.0,
            average_reward=4.0,
            best_reward=7.0,
            episode_length=100,
            win_rate=50.0,
            learning_rate=0.001,
            model_generation=1,
            algorithm="DQN",
            training_time_elapsed=600.0
        )
        
        message = WebSocketMessage(
            type=MessageType.TRAINING_METRICS,
            data=metrics
        )
        
        assert message.type == MessageType.TRAINING_METRICS
        assert isinstance(message.data, TrainingMetricsData)
        assert message.data.episode == 10
        assert isinstance(message.timestamp, datetime)
    
    def test_create_websocket_message_with_dict(self):
        """Test creating WebSocketMessage with dictionary data."""
        data = {"status": "connected", "session_id": "test_session"}
        
        message = WebSocketMessage(
            type=MessageType.CONNECTION_STATUS,
            data=data
        )
        
        assert message.type == MessageType.CONNECTION_STATUS
        assert message.data == data
    
    def test_websocket_message_json_serialization(self):
        """Test JSON serialization of WebSocketMessage."""
        data = {"error": "test_error", "message": "Test error message"}
        
        message = WebSocketMessage(
            type=MessageType.ERROR,
            data=data
        )
        
        json_data = message.model_dump_json()
        assert isinstance(json_data, str)
        
        # Parse back and verify
        import json
        parsed = json.loads(json_data)
        assert parsed['type'] == MessageType.ERROR
        assert parsed['data'] == data
        assert 'timestamp' in parsed


class TestHistoricalDataModels:
    """Test historical data request and response models."""
    
    def test_historical_data_request(self):
        """Test HistoricalDataRequest model."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        request = HistoricalDataRequest(
            session_id="test_session",
            start_time=start_time,
            end_time=end_time,
            max_points=500,
            metrics=["reward", "win_rate"]
        )
        
        assert request.session_id == "test_session"
        assert request.start_time == start_time
        assert request.end_time == end_time
        assert request.max_points == 500
        assert request.metrics == ["reward", "win_rate"]
    
    def test_historical_data_request_minimal(self):
        """Test HistoricalDataRequest with minimal fields."""
        request = HistoricalDataRequest(session_id="minimal_session")
        
        assert request.session_id == "minimal_session"
        assert request.start_time is None
        assert request.end_time is None
        assert request.max_points == 1000  # Default value
        assert request.metrics is None
    
    def test_historical_data_response(self):
        """Test HistoricalDataResponse model."""
        timestamp = datetime.now()
        
        metrics_data = [
            TrainingMetricsData(
                timestamp=timestamp,
                episode=1,
                total_episodes=10,
                current_reward=1.0,
                average_reward=1.0,
                best_reward=1.0,
                episode_length=50,
                win_rate=10.0,
                learning_rate=0.001,
                model_generation=1,
                algorithm="DQN",
                training_time_elapsed=60.0
            )
        ]
        
        graph_data = [
            PerformanceGraphData(
                graph_id="test_graph",
                title="Test Graph",
                y_label="Value",
                metrics=["test_metric"],
                data_points={"test_metric": [GraphDataPoint(timestamp=timestamp, value=1.0)]}
            )
        ]
        
        response = HistoricalDataResponse(
            session_id="test_session",
            metrics_data=metrics_data,
            graph_data=graph_data,
            total_points=1
        )
        
        assert response.session_id == "test_session"
        assert len(response.metrics_data) == 1
        assert len(response.graph_data) == 1
        assert response.total_points == 1


class TestServerStatus:
    """Test ServerStatus model."""
    
    def test_create_server_status(self):
        """Test creating ServerStatus."""
        status = ServerStatus(
            status="running",
            active_sessions=5,
            total_connections=25,
            uptime_seconds=3600.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=15.5
        )
        
        assert status.status == "running"
        assert status.active_sessions == 5
        assert status.total_connections == 25
        assert status.uptime_seconds == 3600.0
        assert status.memory_usage_mb == 256.0
        assert status.cpu_usage_percent == 15.5


class TestErrorResponse:
    """Test ErrorResponse model."""
    
    def test_create_error_response(self):
        """Test creating ErrorResponse."""
        error = ErrorResponse(
            error="validation_error",
            message="Invalid input data"
        )
        
        assert error.error == "validation_error"
        assert error.message == "Invalid input data"
        assert isinstance(error.timestamp, datetime)
    
    def test_error_response_with_timestamp(self):
        """Test ErrorResponse with custom timestamp."""
        custom_timestamp = datetime.now()
        
        error = ErrorResponse(
            error="server_error",
            message="Internal server error",
            timestamp=custom_timestamp
        )
        
        assert error.timestamp == custom_timestamp


class TestEnums:
    """Test enum values."""
    
    def test_training_status_enum(self):
        """Test TrainingStatus enum values."""
        assert TrainingStatus.STARTING == "starting"
        assert TrainingStatus.RUNNING == "running"
        assert TrainingStatus.PAUSED == "paused"
        assert TrainingStatus.COMPLETED == "completed"
        assert TrainingStatus.FAILED == "failed"
        assert TrainingStatus.STOPPED == "stopped"
    
    def test_message_type_enum(self):
        """Test MessageType enum values."""
        assert MessageType.TRAINING_METRICS == "training_metrics"
        assert MessageType.BOT_DECISION == "bot_decision"
        assert MessageType.GRAPH_UPDATE == "graph_update"
        assert MessageType.TRAINING_STATUS == "training_status"
        assert MessageType.CONNECTION_STATUS == "connection_status"
        assert MessageType.ERROR == "error"


if __name__ == "__main__":
    pytest.main([__file__])