"""
Tests for WebSocket connection manager.

This module tests the WebSocket connection management, message broadcasting,
and subscription handling for the training metrics server.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from server.websocket_manager import (
    ConnectionManager,
    WebSocketManager,
    Connection
)
from server.data_models import (
    TrainingMetricsData,
    BotDecisionData,
    PerformanceGraphData,
    MessageType,
    GraphDataPoint
)


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.closed = False
        self.messages = []
        self.accept_called = False
    
    async def accept(self):
        """Mock accept method."""
        self.accept_called = True
    
    async def send_text(self, message: str):
        """Mock send_text method."""
        if self.closed:
            raise Exception("WebSocket is closed")
        self.messages.append(message)
    
    async def receive_text(self):
        """Mock receive_text method."""
        # This would normally block waiting for messages
        await asyncio.sleep(0.1)
        return '{"type": "ping"}'
    
    def close(self):
        """Mock close method."""
        self.closed = True


@pytest.fixture
def connection_manager():
    """Create a ConnectionManager instance for testing."""
    return ConnectionManager()


@pytest.fixture
def websocket_manager(connection_manager):
    """Create a WebSocketManager instance for testing."""
    return WebSocketManager(connection_manager)


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket for testing."""
    return MockWebSocket()


class TestConnectionManager:
    """Test ConnectionManager functionality."""
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self, connection_manager, mock_websocket):
        """Test connecting a WebSocket."""
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            session_id="test_session",
            user_name="test_user",
            user_id="user_123"
        )
        
        assert connection_id is not None
        assert mock_websocket.accept_called
        assert connection_manager.get_connection_count() == 1
        assert connection_manager.get_connection_count("test_session") == 1
        
        # Check connection info
        conn_info = connection_manager.get_connection_info(connection_id)
        assert conn_info is not None
        assert conn_info.user_name == "test_user"
        assert conn_info.user_id == "user_123"
        assert conn_info.session_id == "test_session"
    
    @pytest.mark.asyncio
    async def test_connect_with_custom_subscriptions(self, connection_manager, mock_websocket):
        """Test connecting with custom subscriptions."""
        subscriptions = [MessageType.TRAINING_METRICS, MessageType.BOT_DECISION]
        
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            session_id="test_session",
            user_name="test_user",
            subscriptions=subscriptions
        )
        
        # Verify connection confirmation message was sent
        assert len(mock_websocket.messages) == 1
        
        import json
        message = json.loads(mock_websocket.messages[0])
        assert message["type"] == MessageType.CONNECTION_STATUS
        assert message["data"]["subscriptions"] == subscriptions
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, connection_manager, mock_websocket):
        """Test disconnecting a WebSocket."""
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            session_id="test_session",
            user_name="test_user"
        )
        
        assert connection_manager.get_connection_count() == 1
        
        await connection_manager.disconnect(connection_id)
        
        assert connection_manager.get_connection_count() == 0
        assert connection_manager.get_connection_info(connection_id) is None
    
    @pytest.mark.asyncio
    async def test_send_to_connection(self, connection_manager, mock_websocket):
        """Test sending a message to a specific connection."""
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            session_id="test_session",
            user_name="test_user"
        )
        
        # Clear connection confirmation message
        mock_websocket.messages.clear()
        
        success = await connection_manager.send_to_connection(
            connection_id,
            MessageType.TRAINING_METRICS,
            {"episode": 10, "reward": 5.0}
        )
        
        assert success
        assert len(mock_websocket.messages) == 1
        
        import json
        message = json.loads(mock_websocket.messages[0])
        assert message["type"] == MessageType.TRAINING_METRICS
        assert message["data"]["episode"] == 10
    
    @pytest.mark.asyncio
    async def test_send_to_nonexistent_connection(self, connection_manager):
        """Test sending a message to a non-existent connection."""
        success = await connection_manager.send_to_connection(
            "nonexistent_id",
            MessageType.TRAINING_METRICS,
            {"test": "data"}
        )
        
        assert not success
    
    @pytest.mark.asyncio
    async def test_send_to_unsubscribed_connection(self, connection_manager, mock_websocket):
        """Test sending a message type the connection isn't subscribed to."""
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            session_id="test_session",
            user_name="test_user",
            subscriptions=[MessageType.TRAINING_METRICS]  # Only subscribed to metrics
        )
        
        # Clear connection confirmation message
        mock_websocket.messages.clear()
        
        # Try to send bot decision (not subscribed)
        success = await connection_manager.send_to_connection(
            connection_id,
            MessageType.BOT_DECISION,
            {"action": "move"}
        )
        
        assert success  # Returns True but doesn't send message
        assert len(mock_websocket.messages) == 0  # No message sent
    
    @pytest.mark.asyncio
    async def test_broadcast_to_session(self, connection_manager):
        """Test broadcasting a message to all connections in a session."""
        # Create multiple connections in the same session
        websockets = [MockWebSocket() for _ in range(3)]
        connection_ids = []
        
        for i, ws in enumerate(websockets):
            conn_id = await connection_manager.connect(
                websocket=ws,
                session_id="test_session",
                user_name=f"user_{i}"
            )
            connection_ids.append(conn_id)
            ws.messages.clear()  # Clear connection confirmation
        
        # Broadcast message
        sent_count = await connection_manager.broadcast_to_session(
            "test_session",
            MessageType.TRAINING_METRICS,
            {"episode": 20, "reward": 10.0}
        )
        
        assert sent_count == 3
        
        # Verify all connections received the message
        for ws in websockets:
            assert len(ws.messages) == 1
            import json
            message = json.loads(ws.messages[0])
            assert message["type"] == MessageType.TRAINING_METRICS
            assert message["data"]["episode"] == 20
    
    @pytest.mark.asyncio
    async def test_broadcast_to_subscription(self, connection_manager):
        """Test broadcasting to all connections with a specific subscription."""
        # Create connections with different subscriptions
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()
        
        # Connection 1: subscribed to metrics
        await connection_manager.connect(
            websocket=ws1,
            session_id="session_1",
            user_name="user_1",
            subscriptions=[MessageType.TRAINING_METRICS]
        )
        
        # Connection 2: subscribed to metrics and decisions
        await connection_manager.connect(
            websocket=ws2,
            session_id="session_2",
            user_name="user_2",
            subscriptions=[MessageType.TRAINING_METRICS, MessageType.BOT_DECISION]
        )
        
        # Connection 3: subscribed to decisions only
        await connection_manager.connect(
            websocket=ws3,
            session_id="session_3",
            user_name="user_3",
            subscriptions=[MessageType.BOT_DECISION]
        )
        
        # Clear connection confirmations
        for ws in [ws1, ws2, ws3]:
            ws.messages.clear()
        
        # Broadcast metrics message
        sent_count = await connection_manager.broadcast_to_subscription(
            MessageType.TRAINING_METRICS,
            {"episode": 30}
        )
        
        assert sent_count == 2  # Only ws1 and ws2 should receive it
        
        # Check messages
        assert len(ws1.messages) == 1
        assert len(ws2.messages) == 1
        assert len(ws3.messages) == 0  # Not subscribed to metrics
    
    @pytest.mark.asyncio
    async def test_handle_message(self, connection_manager, mock_websocket):
        """Test handling incoming messages from WebSocket."""
        callback_called = False
        received_message = None
        
        async def test_callback(connection_id: str, message_data: dict):
            nonlocal callback_called, received_message
            callback_called = True
            received_message = message_data
        
        # Register callback
        connection_manager.register_message_callback("test_message", test_callback)
        
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            session_id="test_session",
            user_name="test_user"
        )
        
        # Handle a message
        test_message = '{"type": "test_message", "data": {"key": "value"}}'
        await connection_manager.handle_message(connection_id, test_message)
        
        assert callback_called
        assert received_message["type"] == "test_message"
        assert received_message["data"]["key"] == "value"
    
    @pytest.mark.asyncio
    async def test_handle_invalid_json_message(self, connection_manager, mock_websocket):
        """Test handling invalid JSON messages."""
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            session_id="test_session",
            user_name="test_user"
        )
        
        # Clear connection confirmation
        mock_websocket.messages.clear()
        
        # Send invalid JSON
        await connection_manager.handle_message(connection_id, "invalid json")
        
        # Should receive error message
        assert len(mock_websocket.messages) == 1
        import json
        error_message = json.loads(mock_websocket.messages[0])
        assert error_message["type"] == MessageType.ERROR
        assert "invalid_json" in error_message["data"]["error"]
    
    @pytest.mark.asyncio
    async def test_get_session_connections(self, connection_manager):
        """Test getting all connections for a session."""
        # Create connections in different sessions
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()
        
        await connection_manager.connect(ws1, "session_1", "user_1")
        await connection_manager.connect(ws2, "session_1", "user_2")
        await connection_manager.connect(ws3, "session_2", "user_3")
        
        # Get connections for session_1
        session_1_connections = connection_manager.get_session_connections("session_1")
        assert len(session_1_connections) == 2
        
        user_names = [conn.user_name for conn in session_1_connections]
        assert "user_1" in user_names
        assert "user_2" in user_names
        
        # Get connections for session_2
        session_2_connections = connection_manager.get_session_connections("session_2")
        assert len(session_2_connections) == 1
        assert session_2_connections[0].user_name == "user_3"
    
    @pytest.mark.asyncio
    async def test_connection_callbacks(self, connection_manager, mock_websocket):
        """Test connection and disconnection callbacks."""
        connect_called = False
        disconnect_called = False
        callback_connection = None
        
        async def connect_callback(connection: Connection):
            nonlocal connect_called, callback_connection
            connect_called = True
            callback_connection = connection
        
        async def disconnect_callback(connection: Connection):
            nonlocal disconnect_called
            disconnect_called = True
        
        # Register callbacks
        connection_manager.register_connect_callback(connect_callback)
        connection_manager.register_disconnect_callback(disconnect_callback)
        
        # Connect
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            session_id="test_session",
            user_name="test_user"
        )
        
        assert connect_called
        assert callback_connection is not None
        assert callback_connection.user_name == "test_user"
        
        # Disconnect
        await connection_manager.disconnect(connection_id)
        
        assert disconnect_called
    
    @pytest.mark.asyncio
    async def test_cleanup(self, connection_manager):
        """Test cleanup of all connections."""
        # Create multiple connections
        websockets = [MockWebSocket() for _ in range(3)]
        
        for i, ws in enumerate(websockets):
            await connection_manager.connect(
                websocket=ws,
                session_id=f"session_{i}",
                user_name=f"user_{i}"
            )
        
        assert connection_manager.get_connection_count() == 3
        
        # Cleanup
        await connection_manager.cleanup()
        
        assert connection_manager.get_connection_count() == 0


class TestWebSocketManager:
    """Test WebSocketManager functionality."""
    
    @pytest.mark.asyncio
    async def test_broadcast_training_metrics(self, websocket_manager, connection_manager):
        """Test broadcasting training metrics."""
        # Create a connection
        mock_ws = MockWebSocket()
        await connection_manager.connect(
            websocket=mock_ws,
            session_id="test_session",
            user_name="test_user"
        )
        mock_ws.messages.clear()
        
        # Create metrics data
        metrics = TrainingMetricsData(
            timestamp=datetime.now(),
            episode=50,
            total_episodes=100,
            current_reward=15.0,
            average_reward=12.0,
            best_reward=20.0,
            episode_length=200,
            win_rate=75.0,
            learning_rate=0.001,
            model_generation=2,
            algorithm="DQN",
            training_time_elapsed=1800.0
        )
        
        # Broadcast metrics
        sent_count = await websocket_manager.broadcast_training_metrics("test_session", metrics)
        
        assert sent_count == 1
        assert len(mock_ws.messages) == 1
        
        import json
        message = json.loads(mock_ws.messages[0])
        assert message["type"] == MessageType.TRAINING_METRICS
        assert message["data"]["episode"] == 50
        assert message["data"]["algorithm"] == "DQN"
    
    @pytest.mark.asyncio
    async def test_broadcast_bot_decision(self, websocket_manager, connection_manager):
        """Test broadcasting bot decision data."""
        # Create a connection
        mock_ws = MockWebSocket()
        await connection_manager.connect(
            websocket=mock_ws,
            session_id="test_session",
            user_name="test_user"
        )
        mock_ws.messages.clear()
        
        # Create decision data
        decision = BotDecisionData(
            timestamp=datetime.now(),
            action_probabilities={"move_left": 0.3, "move_right": 0.7},
            state_values=5.5,
            selected_action="move_right",
            confidence_score=0.85
        )
        
        # Broadcast decision
        sent_count = await websocket_manager.broadcast_bot_decision("test_session", decision)
        
        assert sent_count == 1
        assert len(mock_ws.messages) == 1
        
        import json
        message = json.loads(mock_ws.messages[0])
        assert message["type"] == MessageType.BOT_DECISION
        assert message["data"]["selected_action"] == "move_right"
        assert message["data"]["confidence_score"] == 0.85
    
    @pytest.mark.asyncio
    async def test_broadcast_graph_update(self, websocket_manager, connection_manager):
        """Test broadcasting performance graph updates."""
        # Create a connection
        mock_ws = MockWebSocket()
        await connection_manager.connect(
            websocket=mock_ws,
            session_id="test_session",
            user_name="test_user"
        )
        mock_ws.messages.clear()
        
        # Create graph data
        timestamp = datetime.now()
        graph = PerformanceGraphData(
            graph_id="reward_graph",
            title="Reward Progress",
            y_label="Reward",
            metrics=["reward"],
            data_points={
                "reward": [GraphDataPoint(timestamp=timestamp, value=10.0)]
            }
        )
        
        # Broadcast graph update
        sent_count = await websocket_manager.broadcast_graph_update("test_session", graph)
        
        assert sent_count == 1
        assert len(mock_ws.messages) == 1
        
        import json
        message = json.loads(mock_ws.messages[0])
        assert message["type"] == MessageType.GRAPH_UPDATE
        assert message["data"]["graph_id"] == "reward_graph"
        assert message["data"]["title"] == "Reward Progress"
    
    @pytest.mark.asyncio
    async def test_broadcast_training_status(self, websocket_manager, connection_manager):
        """Test broadcasting training status updates."""
        # Create a connection
        mock_ws = MockWebSocket()
        await connection_manager.connect(
            websocket=mock_ws,
            session_id="test_session",
            user_name="test_user"
        )
        mock_ws.messages.clear()
        
        # Broadcast status update
        status_data = {"event": "training_started", "episode": 1}
        sent_count = await websocket_manager.broadcast_training_status("test_session", status_data)
        
        assert sent_count == 1
        assert len(mock_ws.messages) == 1
        
        import json
        message = json.loads(mock_ws.messages[0])
        assert message["type"] == MessageType.TRAINING_STATUS
        assert message["data"]["event"] == "training_started"
        assert message["data"]["episode"] == 1


if __name__ == "__main__":
    pytest.main([__file__])