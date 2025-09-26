"""
WebSocket connection monitoring system for bot game clients.

This module provides comprehensive monitoring of WebSocket connections,
including connection establishment tracking, health monitoring, and
failure detection for bot game clients.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set
from dataclasses import dataclass, field
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
import aiohttp

from .diagnostics import (
    BotDiagnosticTracker, BotLifecycleEvent, DiagnosticLevel,
    ConnectionStatus, get_diagnostic_tracker
)


class WebSocketState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


class ConnectionFailureReason(Enum):
    """Reasons for WebSocket connection failures."""
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_FAILED = "authentication_failed"
    SERVER_UNAVAILABLE = "server_unavailable"
    TIMEOUT = "timeout"
    PROTOCOL_ERROR = "protocol_error"
    UNEXPECTED_CLOSE = "unexpected_close"
    INVALID_URL = "invalid_url"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class WebSocketConnectionInfo:
    """Information about a WebSocket connection."""
    bot_id: str
    websocket_url: str
    connection_state: WebSocketState
    connected_at: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None
    last_ping_sent: Optional[datetime] = None
    last_pong_received: Optional[datetime] = None
    last_message_sent: Optional[datetime] = None
    last_message_received: Optional[datetime] = None
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    connection_attempts: int = 0
    reconnection_attempts: int = 0
    failure_reason: Optional[ConnectionFailureReason] = None
    error_messages: List[str] = field(default_factory=list)
    latency_ms: Optional[float] = None
    
    @property
    def connection_duration(self) -> Optional[timedelta]:
        """Get the duration of the current connection."""
        if self.connected_at and self.connection_state == WebSocketState.CONNECTED:
            return datetime.now() - self.connected_at
        elif self.connected_at and self.disconnected_at:
            return self.disconnected_at - self.connected_at
        return None
    
    @property
    def is_healthy(self) -> bool:
        """Check if the connection is healthy."""
        if self.connection_state != WebSocketState.CONNECTED:
            return False
        
        # Check for recent activity
        now = datetime.now()
        if self.last_message_received:
            time_since_last_message = now - self.last_message_received
            if time_since_last_message > timedelta(minutes=5):  # No messages for 5 minutes
                return False
        
        # Check ping/pong health
        if self.last_ping_sent and not self.last_pong_received:
            time_since_ping = now - self.last_ping_sent
            if time_since_ping > timedelta(seconds=30):  # No pong for 30 seconds
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'bot_id': self.bot_id,
            'websocket_url': self.websocket_url,
            'connection_state': self.connection_state.value,
            'connected_at': self.connected_at.isoformat() if self.connected_at else None,
            'disconnected_at': self.disconnected_at.isoformat() if self.disconnected_at else None,
            'last_ping_sent': self.last_ping_sent.isoformat() if self.last_ping_sent else None,
            'last_pong_received': self.last_pong_received.isoformat() if self.last_pong_received else None,
            'last_message_sent': self.last_message_sent.isoformat() if self.last_message_sent else None,
            'last_message_received': self.last_message_received.isoformat() if self.last_message_received else None,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'connection_attempts': self.connection_attempts,
            'reconnection_attempts': self.reconnection_attempts,
            'failure_reason': self.failure_reason.value if self.failure_reason else None,
            'error_messages': self.error_messages,
            'latency_ms': self.latency_ms,
            'connection_duration': str(self.connection_duration) if self.connection_duration else None,
            'is_healthy': self.is_healthy
        }


@dataclass
class WebSocketMessage:
    """Represents a WebSocket message for monitoring."""
    bot_id: str
    timestamp: datetime
    direction: str  # 'sent' or 'received'
    message_type: str
    message_size: int
    content: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WebSocketConnectionMonitor:
    """
    Monitors WebSocket connections for bot game clients.
    
    Provides connection establishment tracking, health monitoring,
    and failure detection with detailed logging and diagnostics.
    """
    
    def __init__(self, diagnostic_tracker: Optional[BotDiagnosticTracker] = None,
                 ping_interval: float = 30.0, ping_timeout: float = 10.0,
                 max_reconnect_attempts: int = 5, reconnect_delay: float = 5.0):
        """
        Initialize the WebSocket connection monitor.
        
        Args:
            diagnostic_tracker: Diagnostic tracker instance
            ping_interval: Interval between ping messages (seconds)
            ping_timeout: Timeout for ping responses (seconds)
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Delay between reconnection attempts (seconds)
        """
        self.diagnostic_tracker = diagnostic_tracker or get_diagnostic_tracker()
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        # Connection tracking
        self._connections: Dict[str, WebSocketConnectionInfo] = {}
        self._websockets: Dict[str, websockets.WebSocketServerProtocol] = {}
        self._connection_tasks: Dict[str, asyncio.Task] = {}
        self._ping_tasks: Dict[str, asyncio.Task] = {}
        
        # Message monitoring
        self._message_history: Dict[str, List[WebSocketMessage]] = {}
        self._max_message_history = 100
        
        # Event callbacks
        self._connection_callbacks: List[Callable[[str, WebSocketState], Awaitable[None]]] = []
        self._message_callbacks: List[Callable[[WebSocketMessage], Awaitable[None]]] = []
        self._health_callbacks: List[Callable[[str, bool], Awaitable[None]]] = []
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the WebSocket connection monitor."""
        if self._running:
            return
        
        self._running = True
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._periodic_health_check())
        
        self.logger.info("WebSocket connection monitor started")
    
    async def stop(self) -> None:
        """Stop the WebSocket connection monitor."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for bot_id in list(self._connections.keys()):
            await self.stop_monitoring(bot_id)
        
        self.logger.info("WebSocket connection monitor stopped")
    
    async def start_monitoring(self, bot_id: str, websocket_url: str) -> None:
        """
        Start monitoring a WebSocket connection for a bot.
        
        Args:
            bot_id: Bot identifier
            websocket_url: WebSocket URL to monitor
        """
        if bot_id in self._connections:
            self.logger.warning(f"Already monitoring connection for bot {bot_id}")
            return
        
        # Create connection info
        connection_info = WebSocketConnectionInfo(
            bot_id=bot_id,
            websocket_url=websocket_url,
            connection_state=WebSocketState.DISCONNECTED
        )
        
        self._connections[bot_id] = connection_info
        self._message_history[bot_id] = []
        
        # Log monitoring start
        self.diagnostic_tracker.log_event(
            bot_id=bot_id,
            event_type=BotLifecycleEvent.BOT_WEBSOCKET_CONNECTING,
            level=DiagnosticLevel.INFO,
            message=f"Started WebSocket connection monitoring for {websocket_url}",
            details={
                'websocket_url': websocket_url,
                'ping_interval': self.ping_interval,
                'max_reconnect_attempts': self.max_reconnect_attempts
            }
        )
        
        self.logger.info(f"Started monitoring WebSocket connection for bot {bot_id}")
    
    async def stop_monitoring(self, bot_id: str) -> None:
        """
        Stop monitoring a WebSocket connection for a bot.
        
        Args:
            bot_id: Bot identifier
        """
        if bot_id not in self._connections:
            return
        
        # Cancel connection task
        if bot_id in self._connection_tasks:
            self._connection_tasks[bot_id].cancel()
            try:
                await self._connection_tasks[bot_id]
            except asyncio.CancelledError:
                pass
            del self._connection_tasks[bot_id]
        
        # Cancel ping task
        if bot_id in self._ping_tasks:
            self._ping_tasks[bot_id].cancel()
            try:
                await self._ping_tasks[bot_id]
            except asyncio.CancelledError:
                pass
            del self._ping_tasks[bot_id]
        
        # Close WebSocket if open
        if bot_id in self._websockets:
            try:
                await self._websockets[bot_id].close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket for bot {bot_id}: {e}")
            del self._websockets[bot_id]
        
        # Update connection state
        if bot_id in self._connections:
            self._connections[bot_id].connection_state = WebSocketState.CLOSED
            self._connections[bot_id].disconnected_at = datetime.now()
        
        # Clean up
        del self._connections[bot_id]
        del self._message_history[bot_id]
        
        self.logger.info(f"Stopped monitoring WebSocket connection for bot {bot_id}")
    
    async def track_connection_attempt(self, bot_id: str, websocket: websockets.WebSocketServerProtocol) -> None:
        """
        Track a WebSocket connection attempt.
        
        Args:
            bot_id: Bot identifier
            websocket: WebSocket connection
        """
        if bot_id not in self._connections:
            return
        
        connection_info = self._connections[bot_id]
        connection_info.connection_attempts += 1
        connection_info.connection_state = WebSocketState.CONNECTING
        
        # Store WebSocket reference
        self._websockets[bot_id] = websocket
        
        # Update diagnostic tracker
        self.diagnostic_tracker.update_connection_status(
            bot_id=bot_id,
            connection_status=ConnectionStatus.CONNECTING,
            websocket_connected=False,
            websocket_url=connection_info.websocket_url
        )
        
        # Log connection attempt
        self.diagnostic_tracker.log_event(
            bot_id=bot_id,
            event_type=BotLifecycleEvent.BOT_WEBSOCKET_CONNECTING,
            level=DiagnosticLevel.INFO,
            message=f"WebSocket connection attempt #{connection_info.connection_attempts}",
            details={
                'websocket_url': connection_info.websocket_url,
                'attempt_number': connection_info.connection_attempts
            }
        )
        
        # Start connection monitoring task
        self._connection_tasks[bot_id] = asyncio.create_task(
            self._monitor_connection(bot_id, websocket)
        )
    
    async def track_connection_success(self, bot_id: str) -> None:
        """
        Track successful WebSocket connection.
        
        Args:
            bot_id: Bot identifier
        """
        if bot_id not in self._connections:
            return
        
        connection_info = self._connections[bot_id]
        connection_info.connection_state = WebSocketState.CONNECTED
        connection_info.connected_at = datetime.now()
        connection_info.failure_reason = None
        
        # Update diagnostic tracker
        self.diagnostic_tracker.update_connection_status(
            bot_id=bot_id,
            connection_status=ConnectionStatus.CONNECTED,
            websocket_connected=True,
            websocket_url=connection_info.websocket_url
        )
        
        # Log successful connection
        self.diagnostic_tracker.log_event(
            bot_id=bot_id,
            event_type=BotLifecycleEvent.BOT_WEBSOCKET_CONNECTED,
            level=DiagnosticLevel.INFO,
            message="WebSocket connection established successfully",
            details={
                'websocket_url': connection_info.websocket_url,
                'connection_attempts': connection_info.connection_attempts,
                'reconnection_attempts': connection_info.reconnection_attempts
            }
        )
        
        # Start ping monitoring
        self._ping_tasks[bot_id] = asyncio.create_task(self._ping_monitor(bot_id))
        
        # Notify callbacks
        await self._notify_connection_callbacks(bot_id, WebSocketState.CONNECTED)
        await self._notify_health_callbacks(bot_id, True)
    
    async def track_connection_failure(self, bot_id: str, error: Exception,
                                     failure_reason: ConnectionFailureReason) -> None:
        """
        Track WebSocket connection failure.
        
        Args:
            bot_id: Bot identifier
            error: Exception that caused the failure
            failure_reason: Reason for the failure
        """
        if bot_id not in self._connections:
            return
        
        connection_info = self._connections[bot_id]
        connection_info.connection_state = WebSocketState.FAILED
        connection_info.disconnected_at = datetime.now()
        connection_info.failure_reason = failure_reason
        connection_info.error_messages.append(str(error))
        
        # Update diagnostic tracker
        self.diagnostic_tracker.update_connection_status(
            bot_id=bot_id,
            connection_status=ConnectionStatus.FAILED,
            websocket_connected=False,
            websocket_url=connection_info.websocket_url,
            error_message=str(error)
        )
        
        # Log connection failure
        self.diagnostic_tracker.log_event(
            bot_id=bot_id,
            event_type=BotLifecycleEvent.BOT_WEBSOCKET_FAILED,
            level=DiagnosticLevel.ERROR,
            message=f"WebSocket connection failed: {failure_reason.value}",
            details={
                'websocket_url': connection_info.websocket_url,
                'failure_reason': failure_reason.value,
                'error_message': str(error),
                'connection_attempts': connection_info.connection_attempts,
                'reconnection_attempts': connection_info.reconnection_attempts
            }
        )
        
        # Notify callbacks
        await self._notify_connection_callbacks(bot_id, WebSocketState.FAILED)
        await self._notify_health_callbacks(bot_id, False)
        
        # Attempt reconnection if within limits
        if connection_info.reconnection_attempts < self.max_reconnect_attempts:
            await self._schedule_reconnection(bot_id)
    
    async def track_message_sent(self, bot_id: str, message: str) -> None:
        """
        Track a message sent through WebSocket.
        
        Args:
            bot_id: Bot identifier
            message: Message content
        """
        if bot_id not in self._connections:
            return
        
        connection_info = self._connections[bot_id]
        connection_info.messages_sent += 1
        connection_info.bytes_sent += len(message.encode('utf-8'))
        connection_info.last_message_sent = datetime.now()
        
        # Parse message for monitoring
        try:
            message_data = json.loads(message)
            message_type = message_data.get('type', 'unknown')
        except json.JSONDecodeError:
            message_type = 'raw'
            message_data = None
        
        # Create message record
        ws_message = WebSocketMessage(
            bot_id=bot_id,
            timestamp=datetime.now(),
            direction='sent',
            message_type=message_type,
            message_size=len(message),
            content=message_data
        )
        
        # Store in history
        self._message_history[bot_id].append(ws_message)
        if len(self._message_history[bot_id]) > self._max_message_history:
            self._message_history[bot_id] = self._message_history[bot_id][-self._max_message_history:]
        
        # Notify callbacks
        await self._notify_message_callbacks(ws_message)
    
    async def track_message_received(self, bot_id: str, message: str) -> None:
        """
        Track a message received through WebSocket.
        
        Args:
            bot_id: Bot identifier
            message: Message content
        """
        if bot_id not in self._connections:
            return
        
        connection_info = self._connections[bot_id]
        connection_info.messages_received += 1
        connection_info.bytes_received += len(message.encode('utf-8'))
        connection_info.last_message_received = datetime.now()
        
        # Parse message for monitoring
        try:
            message_data = json.loads(message)
            message_type = message_data.get('type', 'unknown')
        except json.JSONDecodeError:
            message_type = 'raw'
            message_data = None
        
        # Create message record
        ws_message = WebSocketMessage(
            bot_id=bot_id,
            timestamp=datetime.now(),
            direction='received',
            message_type=message_type,
            message_size=len(message),
            content=message_data
        )
        
        # Store in history
        self._message_history[bot_id].append(ws_message)
        if len(self._message_history[bot_id]) > self._max_message_history:
            self._message_history[bot_id] = self._message_history[bot_id][-self._max_message_history:]
        
        # Handle pong messages
        if message_type == 'pong':
            connection_info.last_pong_received = datetime.now()
            if connection_info.last_ping_sent:
                latency = (connection_info.last_pong_received - connection_info.last_ping_sent).total_seconds() * 1000
                connection_info.latency_ms = latency
        
        # Notify callbacks
        await self._notify_message_callbacks(ws_message)
    
    def get_connection_info(self, bot_id: str) -> Optional[WebSocketConnectionInfo]:
        """
        Get connection information for a bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            WebSocketConnectionInfo or None if not found
        """
        return self._connections.get(bot_id)
    
    def get_connection_health(self, bot_id: str) -> Dict[str, Any]:
        """
        Get connection health status for a bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            Dictionary with health information
        """
        if bot_id not in self._connections:
            return {'healthy': False, 'reason': 'not_monitored'}
        
        connection_info = self._connections[bot_id]
        
        health_status = {
            'healthy': connection_info.is_healthy,
            'connection_state': connection_info.connection_state.value,
            'connected': connection_info.connection_state == WebSocketState.CONNECTED,
            'connection_duration': str(connection_info.connection_duration) if connection_info.connection_duration else None,
            'latency_ms': connection_info.latency_ms,
            'messages_sent': connection_info.messages_sent,
            'messages_received': connection_info.messages_received,
            'bytes_sent': connection_info.bytes_sent,
            'bytes_received': connection_info.bytes_received,
            'connection_attempts': connection_info.connection_attempts,
            'reconnection_attempts': connection_info.reconnection_attempts,
            'failure_reason': connection_info.failure_reason.value if connection_info.failure_reason else None,
            'error_count': len(connection_info.error_messages),
            'last_error': connection_info.error_messages[-1] if connection_info.error_messages else None
        }
        
        # Add health issues
        issues = []
        if connection_info.connection_state != WebSocketState.CONNECTED:
            issues.append(f"not_connected_{connection_info.connection_state.value}")
        
        if connection_info.last_message_received:
            time_since_message = datetime.now() - connection_info.last_message_received
            if time_since_message > timedelta(minutes=5):
                issues.append("no_recent_messages")
        
        if connection_info.last_ping_sent and not connection_info.last_pong_received:
            time_since_ping = datetime.now() - connection_info.last_ping_sent
            if time_since_ping > timedelta(seconds=30):
                issues.append("ping_timeout")
        
        if connection_info.latency_ms and connection_info.latency_ms > 1000:
            issues.append("high_latency")
        
        health_status['issues'] = issues
        
        return health_status
    
    def get_message_history(self, bot_id: str, limit: Optional[int] = None) -> List[WebSocketMessage]:
        """
        Get message history for a bot.
        
        Args:
            bot_id: Bot identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of WebSocket messages
        """
        if bot_id not in self._message_history:
            return []
        
        messages = self._message_history[bot_id]
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_all_connections(self) -> Dict[str, WebSocketConnectionInfo]:
        """
        Get information for all monitored connections.
        
        Returns:
            Dictionary mapping bot_id to WebSocketConnectionInfo
        """
        return self._connections.copy()
    
    def register_connection_callback(self, callback: Callable[[str, WebSocketState], Awaitable[None]]) -> None:
        """Register a callback for connection state changes."""
        self._connection_callbacks.append(callback)
    
    def register_message_callback(self, callback: Callable[[WebSocketMessage], Awaitable[None]]) -> None:
        """Register a callback for WebSocket messages."""
        self._message_callbacks.append(callback)
    
    def register_health_callback(self, callback: Callable[[str, bool], Awaitable[None]]) -> None:
        """Register a callback for connection health changes."""
        self._health_callbacks.append(callback)
    
    async def _monitor_connection(self, bot_id: str, websocket: websockets.WebSocketServerProtocol) -> None:
        """
        Monitor a WebSocket connection for a bot.
        
        Args:
            bot_id: Bot identifier
            websocket: WebSocket connection to monitor
        """
        try:
            # Wait for connection to be established
            await websocket.wait_closed()
            
        except ConnectionClosed as e:
            await self.track_connection_failure(
                bot_id, e, ConnectionFailureReason.UNEXPECTED_CLOSE
            )
        except WebSocketException as e:
            await self.track_connection_failure(
                bot_id, e, ConnectionFailureReason.PROTOCOL_ERROR
            )
        except Exception as e:
            await self.track_connection_failure(
                bot_id, e, ConnectionFailureReason.NETWORK_ERROR
            )
    
    async def _ping_monitor(self, bot_id: str) -> None:
        """
        Monitor connection health with ping/pong messages.
        
        Args:
            bot_id: Bot identifier
        """
        while self._running and bot_id in self._connections:
            try:
                connection_info = self._connections[bot_id]
                
                if (connection_info.connection_state == WebSocketState.CONNECTED and 
                    bot_id in self._websockets):
                    
                    # Send ping
                    websocket = self._websockets[bot_id]
                    await websocket.ping()
                    connection_info.last_ping_sent = datetime.now()
                    
                    # Wait for ping interval
                    await asyncio.sleep(self.ping_interval)
                    
                    # Check if pong was received
                    if connection_info.last_ping_sent and connection_info.last_pong_received:
                        time_since_pong = connection_info.last_ping_sent - connection_info.last_pong_received
                        if time_since_pong > timedelta(seconds=self.ping_timeout):
                            # Ping timeout
                            await self.track_connection_failure(
                                bot_id, 
                                TimeoutError("Ping timeout"),
                                ConnectionFailureReason.TIMEOUT
                            )
                            break
                else:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in ping monitor for bot {bot_id}: {e}")
                break
    
    async def _schedule_reconnection(self, bot_id: str) -> None:
        """
        Schedule a reconnection attempt for a bot.
        
        Args:
            bot_id: Bot identifier
        """
        if bot_id not in self._connections:
            return
        
        connection_info = self._connections[bot_id]
        connection_info.reconnection_attempts += 1
        connection_info.connection_state = WebSocketState.RECONNECTING
        
        # Log reconnection attempt
        self.diagnostic_tracker.log_event(
            bot_id=bot_id,
            event_type=BotLifecycleEvent.BOT_RECONNECT_ATTEMPT,
            level=DiagnosticLevel.WARN,
            message=f"Scheduling reconnection attempt #{connection_info.reconnection_attempts}",
            details={
                'reconnection_attempt': connection_info.reconnection_attempts,
                'max_attempts': self.max_reconnect_attempts,
                'delay_seconds': self.reconnect_delay
            }
        )
        
        # Wait before reconnecting
        await asyncio.sleep(self.reconnect_delay)
        
        # This would trigger the bot to attempt reconnection
        # The actual reconnection logic would be handled by the GameClient
        await self._notify_connection_callbacks(bot_id, WebSocketState.RECONNECTING)
    
    async def _periodic_health_check(self) -> None:
        """Periodic health check for all connections."""
        while self._running:
            try:
                for bot_id, connection_info in self._connections.items():
                    previous_health = connection_info.is_healthy
                    current_health = connection_info.is_healthy
                    
                    if previous_health != current_health:
                        await self._notify_health_callbacks(bot_id, current_health)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in periodic health check: {e}")
                await asyncio.sleep(30)
    
    async def _notify_connection_callbacks(self, bot_id: str, state: WebSocketState) -> None:
        """Notify connection state change callbacks."""
        for callback in self._connection_callbacks:
            try:
                await callback(bot_id, state)
            except Exception as e:
                self.logger.error(f"Error in connection callback: {e}")
    
    async def _notify_message_callbacks(self, message: WebSocketMessage) -> None:
        """Notify message callbacks."""
        for callback in self._message_callbacks:
            try:
                await callback(message)
            except Exception as e:
                self.logger.error(f"Error in message callback: {e}")
    
    async def _notify_health_callbacks(self, bot_id: str, healthy: bool) -> None:
        """Notify health change callbacks."""
        for callback in self._health_callbacks:
            try:
                await callback(bot_id, healthy)
            except Exception as e:
                self.logger.error(f"Error in health callback: {e}")


# Global WebSocket monitor instance
_websocket_monitor: Optional[WebSocketConnectionMonitor] = None


def get_websocket_monitor() -> WebSocketConnectionMonitor:
    """Get the global WebSocket connection monitor instance."""
    global _websocket_monitor
    if _websocket_monitor is None:
        _websocket_monitor = WebSocketConnectionMonitor()
    return _websocket_monitor


def set_websocket_monitor(monitor: WebSocketConnectionMonitor) -> None:
    """Set the global WebSocket connection monitor instance."""
    global _websocket_monitor
    _websocket_monitor = monitor