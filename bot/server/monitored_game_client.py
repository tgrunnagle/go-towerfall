"""
Monitored GameClient wrapper with WebSocket connection monitoring.

This module provides a wrapper around the GameClient that integrates
with the WebSocket connection monitoring system to track connection
establishment, health, and failures.
"""

import asyncio
import logging
import json
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
        
        self.logger = logging.getLogger(f"{__name__}.{bot_id}")
    
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
            
            raise
    
    async def send_keyboard_input(self, key: str, pressed: bool) -> None:
        """Send keyboard input with monitoring."""
        try:
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
            
        except Exception as e:
            self.logger.error(f"Error exiting game: {e}")
            raise
    
    async def close(self) -> None:
        """Close connection with monitoring cleanup."""
        try:
            await super().close()
            
            # Stop monitoring
            await self.websocket_monitor.stop_monitoring(self.bot_id)
            
            self._connection_established = False
            
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")
    
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
            state = await super().get_direct_state()
            
            # Record successful state retrieval
            self.diagnostic_tracker.record_bot_action(
                bot_id=self.bot_id,
                action_type="get_direct_state",
                success=True
            )
            
            return state
            
        except Exception as e:
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
        return self.websocket_monitor.get_connection_health(self.bot_id)
    
    def get_connection_info(self):
        """Get detailed connection information."""
        return self.websocket_monitor.get_connection_info(self.bot_id)
    
    def get_message_history(self, limit: Optional[int] = None):
        """Get message history."""
        return self.websocket_monitor.get_message_history(self.bot_id, limit)
    
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