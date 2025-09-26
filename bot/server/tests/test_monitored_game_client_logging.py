"""
Tests for MonitoredGameClient diagnostic logging functionality.

This module tests the log file creation, naming convention, periodic flushing,
and diagnostic logging features of the MonitoredGameClient.
"""

import pytest
import pytest_asyncio
import asyncio
import os
import tempfile
import shutil
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from server.monitored_game_client import MonitoredGameClient
from server.websocket_monitor import WebSocketConnectionMonitor
from server.diagnostics import BotDiagnosticTracker


class TestMonitoredGameClientLogging:
    """Test suite for MonitoredGameClient logging functionality."""
    
    @pytest.fixture
    def temp_logs_dir(self):
        """Create a temporary directory for test logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_websocket_monitor(self):
        """Create a mock WebSocket monitor."""
        monitor = Mock(spec=WebSocketConnectionMonitor)
        monitor.start_monitoring = AsyncMock()
        monitor.stop_monitoring = AsyncMock()
        monitor.track_connection_success = AsyncMock()
        monitor.track_connection_failure = AsyncMock()
        monitor.track_message_sent = AsyncMock()
        monitor.track_message_received = AsyncMock()
        monitor.track_connection_attempt = AsyncMock()
        monitor.get_connection_health = Mock(return_value={})
        monitor.get_connection_info = Mock(return_value={})
        monitor.get_message_history = Mock(return_value=[])
        return monitor
    
    @pytest.fixture
    def mock_diagnostic_tracker(self):
        """Create a mock diagnostic tracker."""
        tracker = Mock(spec=BotDiagnosticTracker)
        tracker.log_event = Mock()
        tracker.record_bot_action = Mock()
        return tracker
    
    def test_log_filename_generation(self):
        """Test that log filenames are generated correctly."""
        bot_id = "test_bot_001"
        
        # Mock the logs directory to use a temporary location
        with patch('server.monitored_game_client.os.path.dirname') as mock_dirname:
            mock_dirname.return_value = "/fake/server"
            
            with patch('server.monitored_game_client.os.makedirs') as mock_makedirs:
                with patch('server.monitored_game_client.os.path.exists', return_value=False):
                    filename = MonitoredGameClient._generate_log_filename(bot_id)
                    
                    # Check filename format
                    expected_pattern = datetime.now().strftime("%Y_%m_%d")
                    assert expected_pattern in filename
                    assert bot_id in filename
                    assert "diagnostics.log" in filename
                    assert "_1_" in filename  # First file of the day
    
    def test_log_filename_increments(self):
        """Test that log filenames increment when files exist."""
        bot_id = "test_bot_002"
        
        with patch('server.monitored_game_client.os.path.dirname') as mock_dirname:
            mock_dirname.return_value = "/fake/server"
            
            with patch('server.monitored_game_client.os.makedirs') as mock_makedirs:
                # Mock that first two files exist, third doesn't
                def mock_exists(path):
                    return "_1_" in path or "_2_" in path
                
                with patch('server.monitored_game_client.os.path.exists', side_effect=mock_exists):
                    filename = MonitoredGameClient._generate_log_filename(bot_id)
                    
                    # Should be the third file
                    assert "_3_" in filename
    
    @pytest_asyncio.fixture
    async def client_with_temp_logs(self, temp_logs_dir, mock_websocket_monitor, mock_diagnostic_tracker):
        """Create a MonitoredGameClient with temporary log directory."""
        bot_id = "test_client"
        
        # Patch the log directory to use our temp directory
        with patch('server.monitored_game_client.os.path.dirname') as mock_dirname:
            mock_dirname.return_value = temp_logs_dir
            
            client = MonitoredGameClient(
                bot_id=bot_id,
                websocket_monitor=mock_websocket_monitor,
                diagnostic_tracker=mock_diagnostic_tracker
            )
            
            yield client
            
            # Cleanup
            try:
                await client.close()
            except Exception:
                pass
    
    @pytest.mark.asyncio
    async def test_log_file_creation(self, client_with_temp_logs, temp_logs_dir):
        """Test that log files are created correctly."""
        client = client_with_temp_logs
        
        # Get the log file path
        log_path = client.get_log_file_path()
        assert log_path is not None
        
        # Check that the file exists
        assert os.path.exists(log_path)
        
        # Check that it's in the correct directory
        assert temp_logs_dir in log_path
        
        # Check filename format
        filename = os.path.basename(log_path)
        today = datetime.now().strftime("%Y_%m_%d")
        assert today in filename
        assert "test_client" in filename
        assert "diagnostics.log" in filename
    
    @pytest.mark.asyncio
    async def test_diagnostic_logging_methods(self, client_with_temp_logs):
        """Test the diagnostic logging methods."""
        client = client_with_temp_logs
        log_path = client.get_log_file_path()
        
        # Test various logging methods
        client.log_diagnostic_info("Test diagnostic message")
        client.log_diagnostic_info("Test with details", {"key": "value", "number": 42})
        client.log_performance_metric("test_metric", 123.45, "ms")
        client.log_performance_metric("count_metric", 5)
        client.log_connection_stats()
        
        # Flush logs to ensure they're written
        client.flush_logs()
        
        # Read the log file and verify content
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that our messages are in the log
        assert "Test diagnostic message" in content
        assert "Test with details" in content
        assert "METRIC - test_metric: 123.45 ms" in content
        assert "METRIC - count_metric: 5" in content
        assert "CONNECTION STATISTICS" in content
        assert "Diagnostic logging initialized" in content
    
    @pytest.mark.asyncio
    async def test_periodic_flush_task(self, client_with_temp_logs):
        """Test that the periodic flush task is working."""
        client = client_with_temp_logs
        
        # Check that flush task is created
        assert client._flush_task is not None
        assert not client._flush_task.done()
        
        # Log something
        client.logger.info("Test message for periodic flush")
        
        # Wait a bit for the periodic flush (it runs every 5 seconds, but we'll wait less)
        await asyncio.sleep(0.1)
        
        # The task should still be running
        assert not client._flush_task.done()
    
    @pytest.mark.asyncio
    async def test_flush_on_close(self, client_with_temp_logs):
        """Test that logs are flushed when client is closed."""
        client = client_with_temp_logs
        log_path = client.get_log_file_path()
        
        # Log something
        client.logger.info("Message before close")
        
        # Close the client
        await client.close()
        
        # Check that flush task is cancelled or done (allow some time for cancellation)
        await asyncio.sleep(0.1)  # Give time for cancellation to complete
        assert client._flush_task.cancelled() or client._flush_task.done()
        
        # Check that the message was written to file
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "Message before close" in content
        assert "Connection closed successfully" in content
    
    @pytest.mark.asyncio
    async def test_no_console_output(self, client_with_temp_logs, capsys):
        """Test that logging doesn't produce console output."""
        client = client_with_temp_logs
        
        # Log various messages
        client.logger.info("Info message")
        client.logger.debug("Debug message")
        client.logger.error("Error message")
        client.log_diagnostic_info("Diagnostic message")
        
        # Flush logs
        client.flush_logs()
        
        # Check that nothing was printed to console
        captured = capsys.readouterr()
        assert "Info message" not in captured.out
        assert "Debug message" not in captured.out
        assert "Error message" not in captured.out
        assert "Diagnostic message" not in captured.out
        assert "Info message" not in captured.err
        assert "Debug message" not in captured.err
        assert "Error message" not in captured.err
        assert "Diagnostic message" not in captured.err
    
    @pytest.mark.asyncio
    async def test_input_logging(self, client_with_temp_logs, mock_websocket_monitor, mock_diagnostic_tracker):
        """Test that input actions are logged."""
        client = client_with_temp_logs
        log_path = client.get_log_file_path()
        
        # Mock the parent class methods to avoid actual WebSocket operations but allow logging
        with patch.object(client.__class__.__bases__[0], 'send_keyboard_input', new_callable=AsyncMock) as mock_keyboard:
            with patch.object(client.__class__.__bases__[0], 'send_mouse_input', new_callable=AsyncMock) as mock_mouse:
                # Test keyboard input logging - this should log even if the parent method fails
                try:
                    await client.send_keyboard_input("w", True)
                except Exception:
                    pass  # Expected due to mocking
                
                # Test mouse input logging
                try:
                    await client.send_mouse_input("left", True, 100.0, 200.0)
                except Exception:
                    pass  # Expected due to mocking
        
        # Flush and check logs
        client.flush_logs()
        
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that the debug logging happened (the logging occurs before the parent method call)
        assert "sending keyboard input" in content.lower() or "sending mouse input" in content.lower()
    
    def test_get_log_file_path(self, temp_logs_dir, mock_websocket_monitor, mock_diagnostic_tracker):
        """Test getting the log file path."""
        with patch('server.monitored_game_client.os.path.dirname') as mock_dirname:
            mock_dirname.return_value = temp_logs_dir
            
            client = MonitoredGameClient(
                bot_id="path_test",
                websocket_monitor=mock_websocket_monitor,
                diagnostic_tracker=mock_diagnostic_tracker
            )
            
            log_path = client.get_log_file_path()
            assert log_path is not None
            assert "path_test" in log_path
            assert "diagnostics.log" in log_path
    
    @pytest.mark.asyncio
    async def test_error_handling_in_logging_setup(self, mock_websocket_monitor, mock_diagnostic_tracker):
        """Test error handling when log file setup fails."""
        # Mock os.makedirs to raise an exception
        with patch('server.monitored_game_client.os.makedirs', side_effect=PermissionError("Access denied")):
            with patch('server.monitored_game_client.logging.StreamHandler') as mock_handler:
                mock_console_handler = Mock()
                mock_console_handler.level = logging.ERROR  # Set proper level attribute
                mock_console_handler.setLevel = Mock()
                mock_handler.return_value = mock_console_handler
                
                # This should not raise an exception, but fall back to console logging
                client = MonitoredGameClient(
                    bot_id="error_test",
                    websocket_monitor=mock_websocket_monitor,
                    diagnostic_tracker=mock_diagnostic_tracker
                )
                
                # Should have fallen back to console handler
                mock_handler.assert_called_once()
                mock_console_handler.setLevel.assert_called_once_with(logging.ERROR)
                
                # Cleanup
                try:
                    await client.close()
                except Exception:
                    pass


# Integration test that can be run separately
@pytest.mark.integration
class TestMonitoredGameClientLoggingIntegration:
    """Integration tests for logging that use real file system."""
    
    @pytest.mark.asyncio
    async def test_real_log_file_creation(self):
        """Test creating actual log files in the server/logs directory."""
        # This test uses the real logs directory
        bot_id = f"integration_test_{datetime.now().strftime('%H%M%S')}"
        
        # Create client (this will create real log file)
        client = MonitoredGameClient(bot_id=bot_id)
        
        try:
            # Get log path
            log_path = client.get_log_file_path()
            assert log_path is not None
            
            # Check that file exists
            assert os.path.exists(log_path)
            
            # Check that it's in server/logs
            assert "server" + os.sep + "logs" in log_path
            
            # Test logging
            client.log_diagnostic_info("Integration test message")
            client.log_performance_metric("integration_metric", 999.99, "test_units")
            client.flush_logs()
            
            # Verify content
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "Integration test message" in content
            assert "METRIC - integration_metric: 999.99 test_units" in content
            assert bot_id in content
            
            print(f"✓ Integration test passed. Log file created at: {log_path}")
            
        finally:
            # Cleanup
            await client.close()
            
            # Optionally remove the test log file
            if log_path and os.path.exists(log_path):
                try:
                    os.remove(log_path)
                    print(f"✓ Test log file cleaned up: {log_path}")
                except Exception as e:
                    print(f"Warning: Could not clean up test log file: {e}")