"""
Tests for WebSocket connection monitoring system.

This module tests the WebSocket monitoring functionality including
connection tracking, health monitoring, and failure detection.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

import sys
import os

from server.websocket_monitor import (
    WebSocketConnectionMonitor, WebSocketConnectionInfo, WebSocketState,
    ConnectionFailureReason, WebSocketMessage
)
from server.monitored_game_client import MonitoredGameClient
from server.connection_diagnostics import (
    ConnectionDiagnostics, ConnectionIssueType, IssueSeverity
)
from server.diagnostics import get_diagnostic_tracker


class TestWebSocketConnectionMonitor:
    """Test WebSocket connection monitoring functionality."""
    
    @pytest.fixture
    async def monitor(self):
        """Create a WebSocket monitor for testing."""
        monitor = WebSocketConnectionMonitor()
        await monitor.start()
        yield monitor
        await monitor.stop()
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, monitor):
        """Test starting monitoring for a bot."""
        bot_id = "test_bot_1"
        ws_url = "ws://localhost:4000/ws"
        
        await monitor.start_monitoring(bot_id, ws_url)
        
        # Check that connection info was created
        connection_info = monitor.get_connection_info(bot_id)
        assert connection_info is not None
        assert connection_info.bot_id == bot_id
        assert connection_info.websocket_url == ws_url
        assert connection_info.connection_state == WebSocketState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_track_connection_success(self, monitor):
        """Test tracking successful connection."""
        bot_id = "test_bot_2"
        ws_url = "ws://localhost:4000/ws"
        
        await monitor.start_monitoring(bot_id, ws_url)
        await monitor.track_connection_success(bot_id)
        
        connection_info = monitor.get_connection_info(bot_id)
        assert connection_info.connection_state == WebSocketState.CONNECTED
        assert connection_info.connected_at is not None
        assert connection_info.failure_reason is None
    
    @pytest.mark.asyncio
    async def test_track_connection_failure(self, monitor):
        """Test tracking connection failure."""
        bot_id = "test_bot_3"
        ws_url = "ws://localhost:4000/ws"
        
        await monitor.start_monitoring(bot_id, ws_url)
        
        error = ConnectionError("Connection refused")
        failure_reason = ConnectionFailureReason.SERVER_UNAVAILABLE
        
        await monitor.track_connection_failure(bot_id, error, failure_reason)
        
        connection_info = monitor.get_connection_info(bot_id)
        assert connection_info.connection_state == WebSocketState.FAILED
        assert connection_info.failure_reason == failure_reason
        assert str(error) in connection_info.error_messages
    
    @pytest.mark.asyncio
    async def test_track_messages(self, monitor):
        """Test tracking sent and received messages."""
        bot_id = "test_bot_4"
        ws_url = "ws://localhost:4000/ws"
        
        await monitor.start_monitoring(bot_id, ws_url)
        
        # Track sent message
        sent_message = '{"type": "Key", "payload": {"key": "W", "isDown": true}}'
        await monitor.track_message_sent(bot_id, sent_message)
        
        # Track received message
        received_message = '{"type": "GameState", "payload": {"players": []}}'
        await monitor.track_message_received(bot_id, received_message)
        
        connection_info = monitor.get_connection_info(bot_id)
        assert connection_info.messages_sent == 1
        assert connection_info.messages_received == 1
        assert connection_info.bytes_sent > 0
        assert connection_info.bytes_received > 0
        
        # Check message history
        message_history = monitor.get_message_history(bot_id)
        assert len(message_history) == 2
        assert message_history[0].direction == 'sent'
        assert message_history[1].direction == 'received'
    
    @pytest.mark.asyncio
    async def test_connection_health(self, monitor):
        """Test connection health assessment."""
        bot_id = "test_bot_5"
        ws_url = "ws://localhost:4000/ws"
        
        await monitor.start_monitoring(bot_id, ws_url)
        await monitor.track_connection_success(bot_id)
        
        # Simulate recent activity
        await monitor.track_message_received(bot_id, '{"type": "ping"}')
        
        health = monitor.get_connection_health(bot_id)
        assert health['healthy'] is True
        assert health['connected'] is True
        assert health['connection_state'] == 'connected'
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, monitor):
        """Test stopping monitoring for a bot."""
        bot_id = "test_bot_6"
        ws_url = "ws://localhost:4000/ws"
        
        await monitor.start_monitoring(bot_id, ws_url)
        await monitor.stop_monitoring(bot_id)
        
        # Check that connection info was removed
        connection_info = monitor.get_connection_info(bot_id)
        assert connection_info is None


class TestMonitoredGameClient:
    """Test MonitoredGameClient functionality."""
    
    @pytest.fixture
    def monitored_client(self):
        """Create a monitored game client for testing."""
        return MonitoredGameClient(
            bot_id="test_bot",
            ws_url="ws://localhost:4000/ws",
            http_url="http://localhost:4000"
        )
    
    def test_initialization(self, monitored_client):
        """Test MonitoredGameClient initialization."""
        assert monitored_client.bot_id == "test_bot"
        assert monitored_client.ws_url == "ws://localhost:4000/ws"
        assert monitored_client.http_url == "http://localhost:4000"
        assert monitored_client.websocket_monitor is not None
        assert monitored_client.diagnostic_tracker is not None
    
    def test_inheritance(self, monitored_client):
        """Test that MonitoredGameClient properly inherits from GameClient."""
        # Test that it's an instance of GameClient
        from core.game_client import GameClient
        assert isinstance(monitored_client, GameClient)
        
        # Test that properties are accessible (inherited)
        assert hasattr(monitored_client, 'game_state')
        assert hasattr(monitored_client, 'player_id')
        assert hasattr(monitored_client, 'room_id')
    
    def test_connection_health_methods(self, monitored_client):
        """Test connection health monitoring methods."""
        # These should not raise exceptions even with no connection
        health = monitored_client.get_connection_health()
        assert isinstance(health, dict)
        
        connection_info = monitored_client.get_connection_info()
        # Should be None since no connection has been established
        
        message_history = monitored_client.get_message_history()
        assert isinstance(message_history, list)


class TestConnectionDiagnostics:
    """Test connection diagnostics functionality."""
    
    @pytest.fixture
    async def diagnostics(self):
        """Create connection diagnostics for testing."""
        diagnostics = ConnectionDiagnostics()
        await diagnostics.start()
        yield diagnostics
        await diagnostics.stop()
    
    @pytest.mark.asyncio
    async def test_issue_detection(self, diagnostics):
        """Test connection issue detection."""
        bot_id = "test_bot_diag"
        
        # Mock connection pattern with high failure rate
        with patch.object(diagnostics, 'analyze_bot_connection') as mock_analyze:
            from server.connection_diagnostics import ConnectionPattern
            
            mock_pattern = ConnectionPattern(
                bot_id=bot_id,
                analysis_period=timedelta(hours=1),
                total_connections=10,
                successful_connections=3,
                failed_connections=7,
                average_connection_time=2.0,
                average_connection_duration=30.0,
                disconnection_frequency=8.0,  # High disconnection frequency
                common_failure_reasons=[(ConnectionFailureReason.SERVER_UNAVAILABLE, 5)],
                message_patterns={'Key': 10, 'Click': 5},
                latency_stats={'current_latency_ms': 100}
            )
            mock_analyze.return_value = mock_pattern
            
            # Detect issues
            issues = await diagnostics.detect_connection_issues(bot_id)
            
            # Should detect frequent disconnections
            assert len(issues) > 0
            frequent_disconnection_issues = [
                issue for issue in issues 
                if issue.issue_type == ConnectionIssueType.FREQUENT_DISCONNECTIONS
            ]
            assert len(frequent_disconnection_issues) > 0
            
            issue = frequent_disconnection_issues[0]
            assert issue.severity == IssueSeverity.HIGH
            assert "frequent disconnections" in issue.description.lower()
            assert len(issue.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_issue_resolution(self, diagnostics):
        """Test marking issues as resolved."""
        bot_id = "test_bot_resolve"
        
        # Create a mock issue
        with patch.object(diagnostics, 'analyze_bot_connection') as mock_analyze:
            from server.connection_diagnostics import ConnectionPattern
            
            mock_pattern = ConnectionPattern(
                bot_id=bot_id,
                analysis_period=timedelta(hours=1),
                total_connections=5,
                successful_connections=1,
                failed_connections=4,
                average_connection_time=15.0,  # Slow connection
                average_connection_duration=10.0,
                disconnection_frequency=2.0,
                common_failure_reasons=[],
                message_patterns={},
                latency_stats={}
            )
            mock_analyze.return_value = mock_pattern
            
            # Detect issues
            issues = await diagnostics.detect_connection_issues(bot_id)
            assert len(issues) > 0
            
            # Resolve the first issue
            issue_id = issues[0].issue_id
            success = diagnostics.resolve_issue(bot_id, issue_id)
            assert success is True
            
            # Check that issue is marked as resolved
            resolved_issues = diagnostics.get_bot_issues(bot_id, include_resolved=True)
            resolved_issue = next((i for i in resolved_issues if i.issue_id == issue_id), None)
            assert resolved_issue is not None
            assert resolved_issue.resolved is True
            assert resolved_issue.resolved_at is not None
    
    @pytest.mark.asyncio
    async def test_connection_report_generation(self, diagnostics):
        """Test generating comprehensive connection reports."""
        bot_id = "test_bot_report"
        
        # Generate report (should not raise exception even with no data)
        report = diagnostics.generate_connection_report(bot_id)
        
        assert isinstance(report, dict)
        assert report['bot_id'] == bot_id
        assert 'report_generated_at' in report
        assert 'connection_status' in report
        assert 'active_issues' in report
        assert 'issue_summary' in report
        assert 'recommendations' in report


@pytest.mark.asyncio
async def test_integration_websocket_monitoring():
    """Integration test for WebSocket monitoring system."""
    # Create monitoring components
    monitor = WebSocketConnectionMonitor()
    diagnostics = ConnectionDiagnostics(websocket_monitor=monitor)
    
    try:
        await monitor.start()
        await diagnostics.start()
        
        bot_id = "integration_test_bot"
        ws_url = "ws://localhost:4000/ws"
        
        # Start monitoring
        await monitor.start_monitoring(bot_id, ws_url)
        
        # Simulate connection success
        await monitor.track_connection_success(bot_id)
        
        # Simulate some message activity
        await monitor.track_message_sent(bot_id, '{"type": "Key", "payload": {}}')
        await monitor.track_message_received(bot_id, '{"type": "GameState", "payload": {}}')
        
        # Check connection health
        health = monitor.get_connection_health(bot_id)
        assert health['healthy'] is True
        
        # Generate diagnostic report
        report = diagnostics.generate_connection_report(bot_id)
        assert report['bot_id'] == bot_id
        assert report['connection_status']['is_healthy'] is True
        
    finally:
        await monitor.stop()
        await diagnostics.stop()


if __name__ == "__main__":
    # Run a simple test
    asyncio.run(test_integration_websocket_monitoring())
    print("WebSocket monitoring integration test passed!")