"""
Tests for TrainingMetricsOverlay functionality.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from bot.rl_bot_system.spectator.training_metrics_overlay import (
    TrainingMetricsOverlay, MetricsData, PerformanceGraph
)


class TestTrainingMetricsOverlay:
    """Test cases for TrainingMetricsOverlay."""
    
    @pytest.fixture
    async def metrics_overlay(self):
        """Create a TrainingMetricsOverlay instance for testing."""
        overlay = TrainingMetricsOverlay(
            session_id="test_session",
            enable_graphs=True,
            enable_decision_viz=True
        )
        yield overlay
        await overlay.cleanup()
    
    @pytest.fixture
    def sample_metrics_data(self):
        """Create sample metrics data for testing."""
        return MetricsData(
            timestamp=datetime.now(),
            episode=50,
            total_episodes=1000,
            current_reward=75.5,
            average_reward=65.3,
            best_reward=100.0,
            episode_length=300,
            win_rate=0.6,
            loss_value=0.08,
            learning_rate=0.001,
            epsilon=0.2,
            model_generation=1,
            algorithm="PPO",
            training_time_elapsed=1800.0,
            action_probabilities={"move_left": 0.2, "move_right": 0.5, "shoot": 0.3},
            state_values=0.75,
            selected_action="move_right",
            actions_per_second=10.5,
            frames_per_second=60.0,
            memory_usage_mb=256.0
        )
    
    @pytest.fixture
    def mock_game_client(self):
        """Create a mock GameClient for testing."""
        client = Mock()
        client.websocket = Mock()
        client.websocket.closed = False
        client.websocket.send = AsyncMock()
        return client
    
    @pytest.mark.asyncio
    async def test_register_spectator(self, metrics_overlay, mock_game_client, sample_metrics_data):
        """Test registering a spectator client."""
        # Set current metrics
        metrics_overlay._current_metrics = sample_metrics_data
        
        # Register spectator
        await metrics_overlay.register_spectator(mock_game_client)
        
        # Verify client is registered
        assert mock_game_client in metrics_overlay._spectator_clients
        
        # Verify current metrics were sent
        mock_game_client.websocket.send.assert_called()
        
        # Check that graph config was sent (since graphs are enabled)
        call_args = [call[0][0] for call in mock_game_client.websocket.send.call_args_list]
        
        # Should have sent metrics and graph config
        assert len(call_args) >= 2
        
        # Parse messages
        messages = [json.loads(arg) for arg in call_args]
        message_types = [msg['type'] for msg in messages]
        
        assert 'training_metrics' in message_types
        assert 'graph_config' in message_types
    
    @pytest.mark.asyncio
    async def test_unregister_spectator(self, metrics_overlay, mock_game_client):
        """Test unregistering a spectator client."""
        # Register first
        await metrics_overlay.register_spectator(mock_game_client)
        assert mock_game_client in metrics_overlay._spectator_clients
        
        # Unregister
        await metrics_overlay.unregister_spectator(mock_game_client)
        assert mock_game_client not in metrics_overlay._spectator_clients
    
    @pytest.mark.asyncio
    async def test_update_metrics(self, metrics_overlay, mock_game_client, sample_metrics_data):
        """Test updating training metrics."""
        # Register spectator
        await metrics_overlay.register_spectator(mock_game_client)
        
        # Clear previous calls
        mock_game_client.websocket.send.reset_mock()
        
        # Update metrics
        await metrics_overlay.update_metrics(sample_metrics_data)
        
        # Verify metrics are stored
        assert metrics_overlay._current_metrics == sample_metrics_data
        assert len(metrics_overlay._metrics_history) == 1
        assert metrics_overlay._metrics_history[0] == sample_metrics_data
        
        # Verify metrics were broadcast
        mock_game_client.websocket.send.assert_called()
        
        # Check message content
        call_args = mock_game_client.websocket.send.call_args_list
        assert len(call_args) >= 1
        
        # Parse the metrics message
        metrics_message = None
        for call in call_args:
            message = json.loads(call[0][0])
            if message['type'] == 'training_metrics':
                metrics_message = message
                break
        
        assert metrics_message is not None
        assert metrics_message['data']['episode'] == 50
        assert metrics_message['data']['algorithm'] == "PPO"
    
    @pytest.mark.asyncio
    async def test_send_bot_decision_data(self, metrics_overlay, mock_game_client):
        """Test sending bot decision visualization data."""
        # Register spectator
        await metrics_overlay.register_spectator(mock_game_client)
        
        # Clear previous calls
        mock_game_client.websocket.send.reset_mock()
        
        # Send decision data
        action_probs = {"move_left": 0.3, "move_right": 0.4, "shoot": 0.3}
        await metrics_overlay.send_bot_decision_data(
            action_probabilities=action_probs,
            state_values=0.85,
            q_values=[0.1, 0.8, 0.6],
            selected_action="move_right"
        )
        
        # Verify message was sent
        mock_game_client.websocket.send.assert_called_once()
        
        # Parse message
        message = json.loads(mock_game_client.websocket.send.call_args[0][0])
        
        assert message['type'] == 'bot_decision'
        assert message['action_probabilities'] == action_probs
        assert message['state_values'] == 0.85
        assert message['q_values'] == [0.1, 0.8, 0.6]
        assert message['selected_action'] == "move_right"
    
    @pytest.mark.asyncio
    async def test_decision_viz_disabled(self, mock_game_client):
        """Test that decision data is not sent when visualization is disabled."""
        overlay = TrainingMetricsOverlay(
            session_id="test_session",
            enable_decision_viz=False
        )
        
        try:
            # Register spectator
            await overlay.register_spectator(mock_game_client)
            
            # Clear previous calls
            mock_game_client.websocket.send.reset_mock()
            
            # Send decision data
            await overlay.send_bot_decision_data(
                action_probabilities={"move_left": 0.5, "shoot": 0.5}
            )
            
            # Verify no message was sent
            mock_game_client.websocket.send.assert_not_called()
            
        finally:
            await overlay.cleanup()
    
    def test_get_metrics_history(self, metrics_overlay):
        """Test retrieving metrics history."""
        # Add some metrics to history
        base_time = datetime.now()
        
        for i in range(5):
            metrics = MetricsData(
                timestamp=base_time + timedelta(seconds=i),
                episode=i,
                total_episodes=100,
                current_reward=float(i * 10),
                average_reward=float(i * 5),
                best_reward=50.0,
                episode_length=100,
                win_rate=0.5,
                loss_value=0.1,
                learning_rate=0.001,
                epsilon=0.1,
                model_generation=1,
                algorithm="DQN",
                training_time_elapsed=float(i * 60)
            )
            metrics_overlay._metrics_history.append(metrics)
        
        # Get all history
        history = metrics_overlay.get_metrics_history()
        assert len(history) == 5
        
        # Get history with time filter
        start_time = base_time + timedelta(seconds=2)
        end_time = base_time + timedelta(seconds=4)
        
        filtered_history = metrics_overlay.get_metrics_history(
            start_time=start_time,
            end_time=end_time
        )
        assert len(filtered_history) == 3  # episodes 2, 3, 4
        
        # Get history with max points limit
        limited_history = metrics_overlay.get_metrics_history(max_points=3)
        assert len(limited_history) == 3
    
    def test_get_current_metrics(self, metrics_overlay, sample_metrics_data):
        """Test getting current metrics."""
        # Initially no metrics
        assert metrics_overlay.get_current_metrics() is None
        
        # Set current metrics
        metrics_overlay._current_metrics = sample_metrics_data
        
        # Get current metrics
        current = metrics_overlay.get_current_metrics()
        assert current == sample_metrics_data
    
    def test_performance_graphs_setup(self, metrics_overlay):
        """Test that performance graphs are set up correctly."""
        graphs = metrics_overlay._performance_graphs
        
        # Should have default graphs
        assert len(graphs) > 0
        
        # Check specific graphs
        graph_ids = [graph.graph_id for graph in graphs]
        
        expected_graphs = [
            "reward_progress",
            "win_rate", 
            "episode_length",
            "learning_metrics",
            "performance_stats"
        ]
        
        for expected_id in expected_graphs:
            assert expected_id in graph_ids
        
        # Check graph structure
        reward_graph = next(g for g in graphs if g.graph_id == "reward_progress")
        assert reward_graph.title == "Reward Progress"
        assert "current_reward" in reward_graph.metrics
        assert "average_reward" in reward_graph.metrics
        assert "best_reward" in reward_graph.metrics
    
    @pytest.mark.asyncio
    async def test_broadcast_with_disconnected_client(self, metrics_overlay, sample_metrics_data):
        """Test broadcasting when a client is disconnected."""
        # Create mock clients - one connected, one disconnected
        connected_client = Mock()
        connected_client.websocket = Mock()
        connected_client.websocket.closed = False
        connected_client.websocket.send = AsyncMock()
        
        disconnected_client = Mock()
        disconnected_client.websocket = Mock()
        disconnected_client.websocket.closed = True
        
        # Register both clients
        metrics_overlay._spectator_clients = [connected_client, disconnected_client]
        
        # Update metrics
        await metrics_overlay.update_metrics(sample_metrics_data)
        
        # Verify only connected client received message
        connected_client.websocket.send.assert_called()
        
        # Verify disconnected client was removed from list
        assert disconnected_client not in metrics_overlay._spectator_clients
        assert connected_client in metrics_overlay._spectator_clients
    
    @pytest.mark.asyncio
    async def test_send_message_error_handling(self, metrics_overlay):
        """Test error handling when sending messages to clients."""
        # Create client that will raise exception
        error_client = Mock()
        error_client.websocket = Mock()
        error_client.websocket.closed = False
        error_client.websocket.send = AsyncMock(side_effect=Exception("Connection error"))
        
        # Register client
        metrics_overlay._spectator_clients = [error_client]
        
        # Try to send message - should not raise exception
        await metrics_overlay._send_message_to_client(error_client, '{"test": "message"}')
        
        # Client should be removed from list due to error
        assert error_client not in metrics_overlay._spectator_clients
    
    @pytest.mark.asyncio
    async def test_graph_updates(self, metrics_overlay, mock_game_client, sample_metrics_data):
        """Test performance graph updates."""
        # Register spectator
        await metrics_overlay.register_spectator(mock_game_client)
        
        # Clear previous calls
        mock_game_client.websocket.send.reset_mock()
        
        # Update metrics (should trigger graph updates)
        await metrics_overlay.update_metrics(sample_metrics_data)
        
        # Find graph update message
        call_args = mock_game_client.websocket.send.call_args_list
        graph_message = None
        
        for call in call_args:
            message = json.loads(call[0][0])
            if message['type'] == 'graph_update':
                graph_message = message
                break
        
        assert graph_message is not None
        assert 'graphs' in graph_message
        assert len(graph_message['graphs']) > 0
        
        # Check that graph data contains expected metrics
        for graph_data in graph_message['graphs']:
            assert 'graph_id' in graph_data
            assert 'timestamp' in graph_data
            assert 'data_points' in graph_data
    
    def test_metrics_data_serialization(self, sample_metrics_data):
        """Test MetricsData serialization to dictionary."""
        data_dict = sample_metrics_data.to_dict()
        
        # Check that all fields are present
        assert 'timestamp' in data_dict
        assert 'episode' in data_dict
        assert 'algorithm' in data_dict
        assert 'action_probabilities' in data_dict
        
        # Check timestamp is ISO format string
        assert isinstance(data_dict['timestamp'], str)
        
        # Check numeric fields
        assert data_dict['episode'] == 50
        assert data_dict['current_reward'] == 75.5
        assert data_dict['algorithm'] == "PPO"
    
    def test_performance_graph_dataclass(self):
        """Test PerformanceGraph dataclass."""
        graph = PerformanceGraph(
            graph_id="test_graph",
            title="Test Graph",
            y_label="Test Values",
            metrics=["metric1", "metric2"],
            max_points=500,
            update_frequency=2.0
        )
        
        assert graph.graph_id == "test_graph"
        assert graph.title == "Test Graph"
        assert graph.y_label == "Test Values"
        assert graph.metrics == ["metric1", "metric2"]
        assert graph.max_points == 500
        assert graph.update_frequency == 2.0
    
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_game_client):
        """Test cleanup functionality."""
        overlay = TrainingMetricsOverlay(
            session_id="test_cleanup",
            enable_graphs=True
        )
        
        # Register spectator
        await overlay.register_spectator(mock_game_client)
        assert len(overlay._spectator_clients) == 1
        
        # Cleanup
        await overlay.cleanup()
        
        # Verify clients are cleared
        assert len(overlay._spectator_clients) == 0
        
        # Verify update task is cancelled
        assert overlay._update_task is None or overlay._update_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_graphs_disabled(self, mock_game_client, sample_metrics_data):
        """Test overlay behavior when graphs are disabled."""
        overlay = TrainingMetricsOverlay(
            session_id="test_no_graphs",
            enable_graphs=False,
            enable_decision_viz=True
        )
        
        try:
            # Register spectator
            await overlay.register_spectator(mock_game_client)
            
            # Should not send graph config when graphs are disabled
            call_args = [call[0][0] for call in mock_game_client.websocket.send.call_args_list]
            messages = [json.loads(arg) for arg in call_args]
            message_types = [msg['type'] for msg in messages]
            
            assert 'graph_config' not in message_types
            
            # Clear calls
            mock_game_client.websocket.send.reset_mock()
            
            # Update metrics
            await overlay.update_metrics(sample_metrics_data)
            
            # Should not send graph updates
            call_args = [call[0][0] for call in mock_game_client.websocket.send.call_args_list]
            messages = [json.loads(arg) for arg in call_args]
            message_types = [msg['type'] for msg in messages]
            
            assert 'graph_update' not in message_types
            assert 'training_metrics' in message_types  # Should still send metrics
            
        finally:
            await overlay.cleanup()