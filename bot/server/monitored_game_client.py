"""
Monitored GameClient wrapper with WebSocket connection monitoring.

This module provides a wrapper around the GameClient that integrates
with the WebSocket connection monitoring system to track connection
establishment, health, and failures.
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, Optional, Any, Callable, Awaitable
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from core.game_client import GameClient, TrainingMode
from .websocket_monitor import (
    WebSocketConnectionMonitor, WebSocketState, ConnectionFailureReason,
    get_websocket_monitor
)
from .diagnostics import (
    BotDiagnosticTracker, BotLifecycleEvent, DiagnosticLevel,
    ConnectionStatus, get_diagnostic_tracker
)


class MonitoredGameClient(GameClient):
    """
    GameClient subclass with integrated WebSocket connection monitoring.
    
    Inherits all GameClient functionality while adding connection health tracking,
    message flow monitoring, and failure detection for diagnostic purposes.
    """
    
    @staticmethod
    def _generate_log_filename(bot_id: str) -> str:
        """
        Generate a unique log filename with format YYYY_MM_DD_<incrementing number>.
        
        Args:
            bot_id: Bot identifier to include in filename
            
        Returns:
            Unique log filename
        """
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        today = datetime.now()
        date_prefix = today.strftime("%Y_%m_%d")
        
        filename = f"{date_prefix}_client_diagnostics_{bot_id}.log"
        filepath = os.path.join(logs_dir, filename)
        if not os.path.exists(filepath):
            return filepath
    
    def __init__(self, bot_id: str, ws_url: str = "ws://localhost:4000/ws", 
                 http_url: str = "http://localhost:4000",
                 websocket_monitor: Optional[WebSocketConnectionMonitor] = None,
                 diagnostic_tracker: Optional[BotDiagnosticTracker] = None):
        """
        Initialize the monitored game client.
        
        Args:
            bot_id: Bot identifier for monitoring
            ws_url: WebSocket URL
            http_url: HTTP API URL
            websocket_monitor: WebSocket monitor instance
            diagnostic_tracker: Diagnostic tracker instance
        """
        # Initialize parent GameClient
        super().__init__(ws_url=ws_url, http_url=http_url)
        
        self.bot_id = bot_id
        
        # Initialize monitoring components
        self.websocket_monitor = websocket_monitor or get_websocket_monitor()
        self.diagnostic_tracker = diagnostic_tracker or get_diagnostic_tracker()
        
        # Connection state tracking
        self._connection_established = False
        self._connection_start_time: Optional[datetime] = None
        self._last_message_time: Optional[datetime] = None
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{bot_id}")
        
        # Initialize flush task attribute
        self._flush_task: Optional[asyncio.Task] = None
        
        # Set up file logging for diagnostics
        self._setup_diagnostic_logging()
    
    def _setup_diagnostic_logging(self) -> None:
        """Set up file logging for diagnostic output."""
        try:
            # Generate unique log filename
            log_filepath = self._generate_log_filename(self.bot_id)
            
            # Create file handler with buffering
            file_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # Configure logger to only use file handler (no console output)
            self.logger.handlers.clear()  # Remove any existing handlers
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.DEBUG)
            self.logger.propagate = False  # Prevent propagation to parent loggers
            
            # Store file handler for cleanup and flushing
            self._log_file_handler = file_handler
            
            # Log initialization
            self.logger.info(f"Diagnostic logging initialized for bot {self.bot_id}")
            self.logger.info(f"Log file: {log_filepath}")
            
            # Start periodic flush task
            self._start_periodic_flush()
            
        except Exception as e:
            # Create a minimal console logger as fallback
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            self.logger.addHandler(console_handler)
            self.logger.error(f"Failed to set up file logging: {e}")
    
    def _start_periodic_flush(self) -> None:
        """Start the periodic flush task."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._periodic_flush_loop())
    
    async def _periodic_flush_loop(self) -> None:
        """Periodically flush the log file to ensure data is written."""
        try:
            while True:
                await asyncio.sleep(5.0)  # Flush every 5 seconds
                if hasattr(self, '_log_file_handler') and self._log_file_handler:
                    self._log_file_handler.flush()
        except asyncio.CancelledError:
            # Task was cancelled, which is expected during cleanup
            pass
        except Exception as e:
            # Log error but don't crash the flush loop
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in periodic flush: {e}")
    
    def _stop_periodic_flush(self) -> None:
        """Stop the periodic flush task."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
    
    async def connect(self, room_code: str, player_name: str, room_password: Optional[str] = None) -> None:
        """
        Connect to a game room with monitoring.
        
        Args:
            room_code: Room code to join
            player_name: Player name
            room_password: Optional room password
        """
        self._connection_start_time = datetime.now()
        
        # Start WebSocket monitoring
        await self.websocket_monitor.start_monitoring(self.bot_id, self.ws_url)
        
        # Log connection attempt
        self.logger.info(f"Starting connection to room {room_code} as {player_name}")
        self.logger.debug(f"Connection details - WS URL: {self.ws_url}, HTTP URL: {self.http_url}")
        
        self.diagnostic_tracker.log_event(
            bot_id=self.bot_id,
            event_type=BotLifecycleEvent.BOT_CONNECTING,
            level=DiagnosticLevel.INFO,
            message=f"Starting connection to room {room_code} as {player_name}",
            details={
                'room_code': room_code,
                'player_name': player_name,
                'ws_url': self.ws_url,
                'http_url': self.http_url
            }
        )
        
        try:
            # Attempt connection through parent GameClient
            listener_task = await super().connect(room_code, player_name, room_password)
            
            # Set up message monitoring
            await self._setup_message_monitoring()
            
            # Track successful connection
            await self.websocket_monitor.track_connection_success(self.bot_id)
            self._connection_established = True
            
            # Log successful connection
            connection_duration = (datetime.now() - self._connection_start_time).total_seconds()
            self.logger.info(f"Successfully connected to room {room_code} in {connection_duration:.2f}s")
            self.logger.debug(f"Player ID: {self.player_id}, Room ID: {self.room_id}")
            
            self.diagnostic_tracker.log_event(
                bot_id=self.bot_id,
                event_type=BotLifecycleEvent.BOT_GAME_JOINED,
                level=DiagnosticLevel.INFO,
                message=f"Successfully connected to room {room_code}",
                details={
                    'room_code': room_code,
                    'player_id': self.player_id,
                    'room_id': self.room_id,
                    'connection_time_seconds': connection_duration
                }
            )
            
            # Flush logs after successful connection
            self.flush_logs()
            
            return listener_task
            
        except Exception as e:
            # Determine failure reason
            failure_reason = self._classify_connection_error(e)
            
            # Track connection failure
            await self.websocket_monitor.track_connection_failure(
                self.bot_id, e, failure_reason
            )
            
            # Log connection failure
            connection_duration = (datetime.now() - self._connection_start_time).total_seconds()
            self.logger.error(f"Failed to connect to room {room_code} after {connection_duration:.2f}s: {str(e)}")
            self.logger.debug(f"Error type: {type(e).__name__}, Failure reason: {failure_reason.value}")
            
            self.diagnostic_tracker.log_event(
                bot_id=self.bot_id,
                event_type=BotLifecycleEvent.BOT_GAME_JOIN_FAILED,
                level=DiagnosticLevel.ERROR,
                message=f"Failed to connect to room {room_code}: {str(e)}",
                details={
                    'room_code': room_code,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'failure_reason': failure_reason.value,
                    'connection_time_seconds': connection_duration
                }
            )
            
            # Flush logs after connection failure
            self.flush_logs()
            
            raise
    
    async def send_keyboard_input(self, key: str, pressed: bool) -> None:
        """Send keyboard input with monitoring."""
        try:
            self.logger.debug(f"Sending keyboard input: {key} {'pressed' if pressed else 'released'}")
            await super().send_keyboard_input(key, pressed)
            
            # Track message sent
            message = json.dumps({
                "type": "Key",
                "payload": {"key": key, "isDown": pressed}
            })
            await self.websocket_monitor.track_message_sent(self.bot_id, message)
            
            # Record bot action
            self.diagnostic_tracker.record_bot_action(
                bot_id=self.bot_id,
                action_type="keyboard_input",
                success=True,
                action_details={'key': key, 'pressed': pressed}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send keyboard input {key}: {e}")
            # Record failed action
            self.diagnostic_tracker.record_bot_action(
                bot_id=self.bot_id,
                action_type="keyboard_input",
                success=False,
                action_details={'key': key, 'pressed': pressed},
                error_message=str(e)
            )
            raise
    
    async def send_mouse_input(self, button: str, pressed: bool, x: float, y: float) -> None:
        """Send mouse input with monitoring."""
        try:
            self.logger.debug(f"Sending mouse input: {button} {'pressed' if pressed else 'released'} at ({x}, {y})")
            await super().send_mouse_input(button, pressed, x, y)
            
            # Track message sent
            button_code = 0 if button.lower() == "left" else 2
            message = json.dumps({
                "type": "Click",
                "payload": {"x": x, "y": y, "button": button_code, "isDown": pressed}
            })
            await self.websocket_monitor.track_message_sent(self.bot_id, message)
            
            # Record bot action
            self.diagnostic_tracker.record_bot_action(
                bot_id=self.bot_id,
                action_type="mouse_input",
                success=True,
                action_details={'button': button, 'pressed': pressed, 'x': x, 'y': y}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send mouse input {button} at ({x}, {y}): {e}")
            # Record failed action
            self.diagnostic_tracker.record_bot_action(
                bot_id=self.bot_id,
                action_type="mouse_input",
                success=False,
                action_details={'button': button, 'pressed': pressed, 'x': x, 'y': y},
                error_message=str(e)
            )
            raise
    
    async def exit_game(self) -> None:
        """Exit game with monitoring cleanup."""
        try:
            self.logger.info("Bot exiting game")
            
            # Log exit attempt
            self.diagnostic_tracker.log_event(
                bot_id=self.bot_id,
                event_type=BotLifecycleEvent.BOT_DISCONNECTING,
                level=DiagnosticLevel.INFO,
                message="Bot exiting game"
            )
            
            await super().exit_game()
            
            # Stop monitoring
            await self.websocket_monitor.stop_monitoring(self.bot_id)
            
            self._connection_established = False
            self.logger.info("Bot successfully exited game")
            
            # Flush logs after exit
            self.flush_logs()
            
        except Exception as e:
            self.logger.error(f"Error exiting game: {e}")
            self.flush_logs()  # Ensure error is flushed
            raise
    
    async def close(self) -> None:
        """Close connection with monitoring cleanup."""
        try:
            self.logger.info("Closing connection")
            await super().close()
            
            # Stop monitoring
            await self.websocket_monitor.stop_monitoring(self.bot_id)
            
            self._connection_established = False
            
            self.logger.info("Connection closed successfully")
            
            # Stop periodic flush task
            self._stop_periodic_flush()
            
            # Final flush and clean up file handler
            if hasattr(self, '_log_file_handler') and self._log_file_handler:
                self._log_file_handler.flush()  # Final flush
                self.logger.removeHandler(self._log_file_handler)
                self._log_file_handler.close()
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error closing connection: {e}")
            # Stop flush task even if there was an error
            self._stop_periodic_flush()
    
    def register_message_handler(self, handler) -> None:
        """Register message handler with monitoring wrapper."""
        # Wrap the handler to track received messages
        async def monitored_handler(message_data):
            try:
                # Track message received
                message = json.dumps(message_data)
                await self.websocket_monitor.track_message_received(self.bot_id, message)
                
                # Update last message time
                self._last_message_time = datetime.now()
                
                # Log message received (debug level to avoid spam)
                message_type = message_data.get('type', 'unknown')
                self.logger.debug(f"Received message: {message_type}")
                
                # Call original handler
                await handler(message_data)
                
            except Exception as e:
                self.logger.error(f"Error in message handler: {e}")
                # Still call original handler
                try:
                    await handler(message_data)
                except Exception as handler_error:
                    self.logger.error(f"Error in original handler: {handler_error}")
        
        super().register_message_handler(monitored_handler)
    
    # Training mode methods with monitoring
    
    async def enable_training_mode(self, speed_multiplier: float = 10.0, 
                                 headless: bool = False,
                                 training_session_id: Optional[str] = None) -> None:
        """Enable training mode with monitoring."""
        self.logger.info(f"Enabling training mode - Speed: {speed_multiplier}x, Headless: {headless}")
        if training_session_id:
            self.logger.debug(f"Training session ID: {training_session_id}")
            
        await super().enable_training_mode(speed_multiplier, headless, training_session_id)
        
        # Log training mode activation
        self.diagnostic_tracker.log_event(
            bot_id=self.bot_id,
            event_type=BotLifecycleEvent.BOT_ACTIVE,
            level=DiagnosticLevel.INFO,
            message=f"Training mode enabled - Speed: {speed_multiplier}x, Headless: {headless}",
            details={
                'speed_multiplier': speed_multiplier,
                'headless': headless,
                'training_session_id': training_session_id
            }
        )
    
    async def disable_training_mode(self) -> None:
        """Disable training mode with monitoring."""
        self.logger.info("Disabling training mode")
        await super().disable_training_mode()
        
        # Log training mode deactivation
        self.diagnostic_tracker.log_event(
            bot_id=self.bot_id,
            event_type=BotLifecycleEvent.BOT_ACTIVE,
            level=DiagnosticLevel.INFO,
            message="Training mode disabled"
        )
    
    async def get_direct_state(self) -> Dict[str, Any]:
        """Get direct state with monitoring."""
        try:
            self.logger.debug("Retrieving direct state")
            state = await super().get_direct_state()
            
            # Record successful state retrieval
            self.diagnostic_tracker.record_bot_action(
                bot_id=self.bot_id,
                action_type="get_direct_state",
                success=True
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to get direct state: {e}")
            # Record failed state retrieval
            self.diagnostic_tracker.record_bot_action(
                bot_id=self.bot_id,
                action_type="get_direct_state",
                success=False,
                error_message=str(e)
            )
            raise
    
    # Connection health methods
    
    def is_connected(self) -> bool:
        """Check if connection is established."""
        return self._connection_established and self.websocket is not None
    
    def get_connection_health(self) -> Dict[str, Any]:
        """Get connection health information."""
        health = self.websocket_monitor.get_connection_health(self.bot_id)
        self.logger.debug(f"Connection health: {health}")
        return health
    
    def get_connection_info(self):
        """Get detailed connection information."""
        info = self.websocket_monitor.get_connection_info(self.bot_id)
        self.logger.debug(f"Connection info retrieved")
        return info
    
    def get_message_history(self, limit: Optional[int] = None):
        """Get message history."""
        history = self.websocket_monitor.get_message_history(self.bot_id, limit)
        self.logger.debug(f"Retrieved message history (limit: {limit})")
        return history
    
    def log_diagnostic_info(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log diagnostic information to the file.
        
        Args:
            message: Diagnostic message
            details: Optional additional details
        """
        if details:
            self.logger.info(f"{message} - Details: {json.dumps(details, default=str)}")
        else:
            self.logger.info(message)
    
    def log_performance_metric(self, metric_name: str, value: Any, unit: Optional[str] = None) -> None:
        """
        Log performance metrics.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Optional unit of measurement
        """
        unit_str = f" {unit}" if unit else ""
        self.logger.info(f"METRIC - {metric_name}: {value}{unit_str}")
    
    def log_connection_stats(self) -> None:
        """Log current connection statistics."""
        try:
            health = self.get_connection_health()
            info = self.get_connection_info()
            
            self.logger.info("=== CONNECTION STATISTICS ===")
            self.logger.info(f"Connection established: {self._connection_established}")
            self.logger.info(f"Last message time: {self._last_message_time}")
            
            if health:
                self.logger.info(f"Connection health: {json.dumps(health, default=str)}")
            
            if info:
                self.logger.info(f"Connection info: {json.dumps(info, default=str)}")
                
        except Exception as e:
            self.logger.error(f"Failed to log connection stats: {e}")
    
    def flush_logs(self) -> None:
        """Manually flush logs to file immediately."""
        try:
            if hasattr(self, '_log_file_handler') and self._log_file_handler:
                self._log_file_handler.flush()
                self.logger.debug("Logs flushed to file")
        except Exception as e:
            self.logger.error(f"Failed to flush logs: {e}")
    
    def get_log_file_path(self) -> Optional[str]:
        """
        Get the path to the current log file.
        
        Returns:
            Path to the log file, or None if not set up
        """
        try:
            if hasattr(self, '_log_file_handler') and self._log_file_handler:
                return self._log_file_handler.baseFilename
        except Exception:
            pass
        return None
    
    async def _setup_message_monitoring(self) -> None:
        """Set up message monitoring for the WebSocket connection."""
        if not self.websocket:
            return
        
        # Track the WebSocket connection
        await self.websocket_monitor.track_connection_attempt(
            self.bot_id, self.websocket
        )
    
    def _classify_connection_error(self, error: Exception) -> ConnectionFailureReason:
        """
        Classify connection error to determine failure reason.
        
        Args:
            error: Exception that occurred
            
        Returns:
            ConnectionFailureReason enum value
        """
        error_str = str(error).lower()
        
        if isinstance(error, ConnectionClosed):
            return ConnectionFailureReason.UNEXPECTED_CLOSE
        elif isinstance(error, WebSocketException):
            return ConnectionFailureReason.PROTOCOL_ERROR
        elif isinstance(error, asyncio.TimeoutError):
            return ConnectionFailureReason.TIMEOUT
        elif "authentication" in error_str or "unauthorized" in error_str:
            return ConnectionFailureReason.AUTHENTICATION_FAILED
        elif "connection refused" in error_str or "unreachable" in error_str:
            return ConnectionFailureReason.SERVER_UNAVAILABLE
        elif "invalid" in error_str and "url" in error_str:
            return ConnectionFailureReason.INVALID_URL
        elif "permission" in error_str or "forbidden" in error_str:
            return ConnectionFailureReason.PERMISSION_DENIED
        else:
            return ConnectionFailureReason.NETWORK_ERROR