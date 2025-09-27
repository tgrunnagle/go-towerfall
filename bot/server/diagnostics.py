"""
Diagnostic infrastructure for bot lifecycle tracking.

This module provides comprehensive diagnostic tools for monitoring bot health,
activity, and lifecycle events to help identify and resolve bot inactivity issues.
"""

import logging
import asyncio
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Awaitable
from dataclasses import dataclass, field, asdict
import json
import uuid


class DiagnosticLevel(Enum):
    """Diagnostic logging levels focused on problem identification."""
    ERROR = "error"      # Critical failures and errors
    WARN = "warn"        # Potential issues and warnings
    INFO = "info"        # Important lifecycle events
    DEBUG = "debug"      # Detailed troubleshooting information


class BotLifecycleEvent(Enum):
    """Critical bot lifecycle events to track."""
    BOT_SPAWN_REQUESTED = "bot_spawn_requested"
    BOT_INITIALIZING = "bot_initializing"
    BOT_AI_LOADING = "bot_ai_loading"
    BOT_AI_LOADED = "bot_ai_loaded"
    BOT_AI_LOAD_FAILED = "bot_ai_load_failed"
    BOT_CONNECTING = "bot_connecting"
    BOT_WEBSOCKET_CONNECTING = "bot_websocket_connecting"
    BOT_WEBSOCKET_CONNECTED = "bot_websocket_connected"
    BOT_WEBSOCKET_FAILED = "bot_websocket_failed"
    BOT_GAME_JOINED = "bot_game_joined"
    BOT_GAME_JOIN_FAILED = "bot_game_join_failed"
    BOT_ACTIVE = "bot_active"
    BOT_AI_DECISION_MADE = "bot_ai_decision_made"
    BOT_ACTION_SENT = "bot_action_sent"
    BOT_ACTION_FAILED = "bot_action_failed"
    BOT_DISCONNECTING = "bot_disconnecting"
    BOT_TERMINATED = "bot_terminated"
    BOT_ERROR = "bot_error"
    BOT_RECONNECT_ATTEMPT = "bot_reconnect_attempt"
    BOT_RECONNECT_SUCCESS = "bot_reconnect_success"
    BOT_RECONNECT_FAILED = "bot_reconnect_failed"


class ConnectionStatus(Enum):
    """WebSocket connection status."""
    NOT_CONNECTED = "not_connected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    RECONNECTING = "reconnecting"


class AIStatus(Enum):
    """Bot AI execution status."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class BotDiagnosticInfo:
    """Comprehensive diagnostic information for a bot instance."""
    bot_id: str
    bot_name: str
    bot_type: str
    difficulty: str
    status: str
    connection_status: ConnectionStatus
    ai_status: AIStatus
    room_id: Optional[str]
    created_at: datetime
    last_activity: datetime
    last_decision_time: Optional[datetime]
    error_messages: List[str] = field(default_factory=list)
    websocket_connected: bool = False
    websocket_url: Optional[str] = None
    game_client_status: str = "unknown"
    actions_sent: int = 0
    decisions_made: int = 0
    game_state_updates_received: int = 0
    connection_errors: List[str] = field(default_factory=list)
    reconnection_attempts: int = 0
    uptime_seconds: float = 0.0
    performance_issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'bot_id': self.bot_id,
            'bot_name': self.bot_name,
            'bot_type': self.bot_type,
            'difficulty': self.difficulty,
            'status': self.status,
            'connection_status': self.connection_status.value,
            'ai_status': self.ai_status.value,
            'room_id': self.room_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'last_decision_time': self.last_decision_time.isoformat() if self.last_decision_time else None,
            'error_messages': self.error_messages,
            'websocket_connected': self.websocket_connected,
            'websocket_url': self.websocket_url,
            'game_client_status': self.game_client_status,
            'actions_sent': self.actions_sent,
            'decisions_made': self.decisions_made,
            'game_state_updates_received': self.game_state_updates_received,
            'connection_errors': self.connection_errors,
            'reconnection_attempts': self.reconnection_attempts,
            'uptime_seconds': self.uptime_seconds,
            'performance_issues': self.performance_issues
        }


@dataclass
class ConnectionHealth:
    """WebSocket connection health monitoring."""
    websocket_connected: bool
    connection_status: ConnectionStatus
    last_ping: Optional[datetime]
    last_pong: Optional[datetime]
    connection_errors: List[str] = field(default_factory=list)
    reconnection_attempts: int = 0
    message_queue_size: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    connection_duration: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'websocket_connected': self.websocket_connected,
            'connection_status': self.connection_status.value,
            'last_ping': self.last_ping.isoformat() if self.last_ping else None,
            'last_pong': self.last_pong.isoformat() if self.last_pong else None,
            'connection_errors': self.connection_errors,
            'reconnection_attempts': self.reconnection_attempts,
            'message_queue_size': self.message_queue_size,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'connection_duration': str(self.connection_duration) if self.connection_duration else None
        }


@dataclass
class BotActivityMetrics:
    """Bot activity and performance metrics."""
    decisions_made: int = 0
    actions_executed: int = 0
    actions_failed: int = 0
    game_state_updates: int = 0
    errors_encountered: int = 0
    uptime_seconds: float = 0.0
    last_decision_time: Optional[datetime] = None
    last_action_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    average_decision_time_ms: float = 0.0
    decision_success_rate: float = 0.0
    action_success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'decisions_made': self.decisions_made,
            'actions_executed': self.actions_executed,
            'actions_failed': self.actions_failed,
            'game_state_updates': self.game_state_updates,
            'errors_encountered': self.errors_encountered,
            'uptime_seconds': self.uptime_seconds,
            'last_decision_time': self.last_decision_time.isoformat() if self.last_decision_time else None,
            'last_action_time': self.last_action_time.isoformat() if self.last_action_time else None,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'average_decision_time_ms': self.average_decision_time_ms,
            'decision_success_rate': self.decision_success_rate,
            'action_success_rate': self.action_success_rate
        }


@dataclass
class DiagnosticEvent:
    """Individual diagnostic event record."""
    event_id: str
    bot_id: str
    event_type: BotLifecycleEvent
    timestamp: datetime
    level: DiagnosticLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'event_id': self.event_id,
            'bot_id': self.bot_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'message': self.message,
            'details': self.details,
            'correlation_id': self.correlation_id
        }


class BotDiagnosticTracker:
    """
    Comprehensive diagnostic tracking system for bot lifecycle events.
    
    Provides enhanced logging, status tracking, and health monitoring
    focused on identifying bot inactivity issues.
    """
    
    def __init__(self, max_events_per_bot: int = 1000, cleanup_interval_hours: int = 24):
        """
        Initialize the diagnostic tracker.
        
        Args:
            max_events_per_bot: Maximum diagnostic events to keep per bot
            cleanup_interval_hours: Hours between diagnostic data cleanup
        """
        self.max_events_per_bot = max_events_per_bot
        self.cleanup_interval_hours = cleanup_interval_hours
        
        # Diagnostic data storage
        self._bot_diagnostics: Dict[str, BotDiagnosticInfo] = {}
        self._connection_health: Dict[str, ConnectionHealth] = {}
        self._activity_metrics: Dict[str, BotActivityMetrics] = {}
        self._diagnostic_events: Dict[str, List[DiagnosticEvent]] = {}  # bot_id -> events
        
        # Event callbacks
        self._event_callbacks: List[Callable[[DiagnosticEvent], Awaitable[None]]] = []
        self._health_callbacks: List[Callable[[str, BotDiagnosticInfo], Awaitable[None]]] = []
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Logger setup
        self.logger = logging.getLogger(f"{__name__}.BotDiagnosticTracker")
        self._setup_diagnostic_logger()
    
    def _setup_diagnostic_logger(self) -> None:
        """Setup specialized diagnostic logger for file-only output."""
        # Create diagnostic-specific logger
        self.diagnostic_logger = logging.getLogger("bot_diagnostics")
        self.diagnostic_logger.setLevel(logging.INFO)
        
        # Prevent propagation to parent loggers (prevents console output)
        self.diagnostic_logger.propagate = False
        
        # Create formatter for diagnostic events
        formatter = logging.Formatter(
            '%(asctime)s - BOT_DIAGNOSTIC - %(levelname)s - %(message)s'
        )
        
        # Add file handler if not already present
        if not self.diagnostic_logger.handlers:
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Create file handler for diagnostic events
            log_file = os.path.join(logs_dir, 'bot_diagnostics.log')
            handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            
            self.diagnostic_logger.addHandler(handler)
            
            # Log initialization message
            self.diagnostic_logger.info("Bot diagnostic logger initialized - file-only output")
    
    async def start(self) -> None:
        """Start the diagnostic tracker."""
        if self._running:
            return
        
        self._running = True
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        self.logger.info("Bot diagnostic tracker started")
    
    async def stop(self) -> None:
        """Stop the diagnostic tracker."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Bot diagnostic tracker stopped")
    
    def register_bot(self, bot_id: str, bot_name: str, bot_type: str, 
                    difficulty: str, room_id: Optional[str] = None) -> None:
        """
        Register a new bot for diagnostic tracking.
        
        Args:
            bot_id: Unique bot identifier
            bot_name: Bot display name
            bot_type: Type of bot (rules_based, rl_generation, etc.)
            difficulty: Bot difficulty level
            room_id: Room the bot is joining
        """
        current_time = datetime.now()
        
        # Create diagnostic info
        self._bot_diagnostics[bot_id] = BotDiagnosticInfo(
            bot_id=bot_id,
            bot_name=bot_name,
            bot_type=bot_type,
            difficulty=difficulty,
            status="initializing",
            connection_status=ConnectionStatus.NOT_CONNECTED,
            ai_status=AIStatus.NOT_LOADED,
            room_id=room_id,
            created_at=current_time,
            last_activity=current_time,
            last_decision_time=None
        )
        
        # Create connection health tracking
        self._connection_health[bot_id] = ConnectionHealth(
            websocket_connected=False,
            connection_status=ConnectionStatus.NOT_CONNECTED,
            last_ping=None,
            last_pong=None
        )
        
        # Create activity metrics
        self._activity_metrics[bot_id] = BotActivityMetrics()
        
        # Initialize event list
        self._diagnostic_events[bot_id] = []
        
        # Log registration
        self.log_event(
            bot_id=bot_id,
            event_type=BotLifecycleEvent.BOT_SPAWN_REQUESTED,
            level=DiagnosticLevel.INFO,
            message=f"Bot {bot_name} ({bot_type}, {difficulty}) registered for room {room_id}",
            details={
                'bot_name': bot_name,
                'bot_type': bot_type,
                'difficulty': difficulty,
                'room_id': room_id
            }
        )
    
    def unregister_bot(self, bot_id: str) -> None:
        """
        Unregister a bot from diagnostic tracking.
        
        Args:
            bot_id: Bot identifier to unregister
        """
        if bot_id in self._bot_diagnostics:
            self.log_event(
                bot_id=bot_id,
                event_type=BotLifecycleEvent.BOT_TERMINATED,
                level=DiagnosticLevel.INFO,
                message=f"Bot {bot_id} unregistered from diagnostic tracking"
            )
            
            # Clean up diagnostic data
            del self._bot_diagnostics[bot_id]
            del self._connection_health[bot_id]
            del self._activity_metrics[bot_id]
            del self._diagnostic_events[bot_id]
    
    def log_event(self, bot_id: str, event_type: BotLifecycleEvent, 
                  level: DiagnosticLevel, message: str, 
                  details: Optional[Dict[str, Any]] = None,
                  correlation_id: Optional[str] = None) -> None:
        """
        Log a diagnostic event for a bot.
        
        Args:
            bot_id: Bot identifier
            event_type: Type of lifecycle event
            level: Diagnostic level
            message: Event message
            details: Additional event details
            correlation_id: Optional correlation ID for tracking related events
        """
        if bot_id not in self._bot_diagnostics:
            return
        
        # Create diagnostic event
        event = DiagnosticEvent(
            event_id=str(uuid.uuid4()),
            bot_id=bot_id,
            event_type=event_type,
            timestamp=datetime.now(),
            level=level,
            message=message,
            details=details or {},
            correlation_id=correlation_id
        )
        
        # Store event
        if bot_id not in self._diagnostic_events:
            self._diagnostic_events[bot_id] = []
        
        self._diagnostic_events[bot_id].append(event)
        
        # Limit events per bot
        if len(self._diagnostic_events[bot_id]) > self.max_events_per_bot:
            self._diagnostic_events[bot_id] = self._diagnostic_events[bot_id][-self.max_events_per_bot:]
        
        # Update last activity
        if bot_id in self._bot_diagnostics:
            self._bot_diagnostics[bot_id].last_activity = event.timestamp
        
        # Log to diagnostic logger
        log_level = getattr(logging, level.value.upper())
        self.diagnostic_logger.log(
            log_level,
            f"[{bot_id}] {event_type.value}: {message} | Details: {json.dumps(details or {})}"
        )
        
        # Notify callbacks
        asyncio.create_task(self._notify_event_callbacks(event))
    
    def update_bot_status(self, bot_id: str, status: str, 
                         error_message: Optional[str] = None) -> None:
        """
        Update bot status and log the change.
        
        Args:
            bot_id: Bot identifier
            status: New bot status
            error_message: Optional error message if status is error
        """
        if bot_id not in self._bot_diagnostics:
            return
        
        old_status = self._bot_diagnostics[bot_id].status
        self._bot_diagnostics[bot_id].status = status
        self._bot_diagnostics[bot_id].last_activity = datetime.now()
        
        if error_message:
            self._bot_diagnostics[bot_id].error_messages.append(error_message)
        
        # Log status change
        level = DiagnosticLevel.ERROR if status == "error" else DiagnosticLevel.INFO
        event_type = BotLifecycleEvent.BOT_ERROR if status == "error" else BotLifecycleEvent.BOT_ACTIVE
        
        self.log_event(
            bot_id=bot_id,
            event_type=event_type,
            level=level,
            message=f"Bot status changed from {old_status} to {status}",
            details={
                'old_status': old_status,
                'new_status': status,
                'error_message': error_message
            }
        )
    
    def update_connection_status(self, bot_id: str, connection_status: ConnectionStatus,
                               websocket_connected: bool = False,
                               websocket_url: Optional[str] = None,
                               error_message: Optional[str] = None) -> None:
        """
        Update bot connection status.
        
        Args:
            bot_id: Bot identifier
            connection_status: New connection status
            websocket_connected: Whether WebSocket is connected
            websocket_url: WebSocket URL if applicable
            error_message: Optional error message
        """
        if bot_id not in self._connection_health:
            return
        
        old_status = self._connection_health[bot_id].connection_status
        self._connection_health[bot_id].connection_status = connection_status
        self._connection_health[bot_id].websocket_connected = websocket_connected
        
        # Update bot diagnostic info
        if bot_id in self._bot_diagnostics:
            self._bot_diagnostics[bot_id].connection_status = connection_status
            self._bot_diagnostics[bot_id].websocket_connected = websocket_connected
            self._bot_diagnostics[bot_id].websocket_url = websocket_url
            self._bot_diagnostics[bot_id].last_activity = datetime.now()
        
        if error_message:
            self._connection_health[bot_id].connection_errors.append(error_message)
            if bot_id in self._bot_diagnostics:
                self._bot_diagnostics[bot_id].connection_errors.append(error_message)
        
        # Log connection status change
        level = DiagnosticLevel.ERROR if connection_status == ConnectionStatus.FAILED else DiagnosticLevel.INFO
        event_type = (BotLifecycleEvent.BOT_WEBSOCKET_FAILED if connection_status == ConnectionStatus.FAILED 
                     else BotLifecycleEvent.BOT_WEBSOCKET_CONNECTED)
        
        self.log_event(
            bot_id=bot_id,
            event_type=event_type,
            level=level,
            message=f"Connection status changed from {old_status.value} to {connection_status.value}",
            details={
                'old_status': old_status.value,
                'new_status': connection_status.value,
                'websocket_connected': websocket_connected,
                'websocket_url': websocket_url,
                'error_message': error_message
            }
        )
    
    def update_ai_status(self, bot_id: str, ai_status: AIStatus,
                        error_message: Optional[str] = None) -> None:
        """
        Update bot AI status.
        
        Args:
            bot_id: Bot identifier
            ai_status: New AI status
            error_message: Optional error message
        """
        if bot_id not in self._bot_diagnostics:
            return
        
        old_status = self._bot_diagnostics[bot_id].ai_status
        self._bot_diagnostics[bot_id].ai_status = ai_status
        self._bot_diagnostics[bot_id].last_activity = datetime.now()
        
        if error_message:
            self._bot_diagnostics[bot_id].error_messages.append(error_message)
        
        # Log AI status change
        level = DiagnosticLevel.ERROR if ai_status == AIStatus.ERROR else DiagnosticLevel.INFO
        event_type = (BotLifecycleEvent.BOT_AI_LOAD_FAILED if ai_status == AIStatus.ERROR 
                     else BotLifecycleEvent.BOT_AI_LOADED)
        
        self.log_event(
            bot_id=bot_id,
            event_type=event_type,
            level=level,
            message=f"AI status changed from {old_status.value} to {ai_status.value}",
            details={
                'old_status': old_status.value,
                'new_status': ai_status.value,
                'error_message': error_message
            }
        )
    
    def record_bot_decision(self, bot_id: str, decision_details: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a bot AI decision.
        
        Args:
            bot_id: Bot identifier
            decision_details: Optional details about the decision
        """
        if bot_id not in self._activity_metrics:
            return
        
        current_time = datetime.now()
        
        # Update metrics
        self._activity_metrics[bot_id].decisions_made += 1
        self._activity_metrics[bot_id].last_decision_time = current_time
        
        # Update diagnostic info
        if bot_id in self._bot_diagnostics:
            self._bot_diagnostics[bot_id].decisions_made += 1
            self._bot_diagnostics[bot_id].last_decision_time = current_time
            self._bot_diagnostics[bot_id].last_activity = current_time
        
        # Log decision (only for debugging level to avoid spam)
        self.log_event(
            bot_id=bot_id,
            event_type=BotLifecycleEvent.BOT_AI_DECISION_MADE,
            level=DiagnosticLevel.DEBUG,
            message="Bot made AI decision",
            details=decision_details or {}
        )
    
    def record_bot_action(self, bot_id: str, action_type: str, success: bool,
                         action_details: Optional[Dict[str, Any]] = None,
                         error_message: Optional[str] = None) -> None:
        """
        Record a bot action execution.
        
        Args:
            bot_id: Bot identifier
            action_type: Type of action performed
            success: Whether the action was successful
            action_details: Optional details about the action
            error_message: Optional error message if action failed
        """
        if bot_id not in self._activity_metrics:
            return
        
        current_time = datetime.now()
        
        # Update metrics
        if success:
            self._activity_metrics[bot_id].actions_executed += 1
            self._activity_metrics[bot_id].last_action_time = current_time
        else:
            self._activity_metrics[bot_id].actions_failed += 1
            self._activity_metrics[bot_id].errors_encountered += 1
            self._activity_metrics[bot_id].last_error_time = current_time
        
        # Update diagnostic info
        if bot_id in self._bot_diagnostics:
            self._bot_diagnostics[bot_id].actions_sent += 1
            self._bot_diagnostics[bot_id].last_activity = current_time
            
            if error_message:
                self._bot_diagnostics[bot_id].error_messages.append(error_message)
        
        # Log action
        level = DiagnosticLevel.WARN if not success else DiagnosticLevel.DEBUG
        event_type = BotLifecycleEvent.BOT_ACTION_FAILED if not success else BotLifecycleEvent.BOT_ACTION_SENT
        
        self.log_event(
            bot_id=bot_id,
            event_type=event_type,
            level=level,
            message=f"Bot action {action_type} {'succeeded' if success else 'failed'}",
            details={
                'action_type': action_type,
                'success': success,
                'error_message': error_message,
                **(action_details or {})
            }
        )
    
    def get_bot_diagnostics(self, bot_id: str) -> Optional[BotDiagnosticInfo]:
        """
        Get comprehensive diagnostic information for a bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            BotDiagnosticInfo or None if bot not found
        """
        if bot_id not in self._bot_diagnostics:
            return None
        
        # Update uptime
        diagnostic_info = self._bot_diagnostics[bot_id]
        diagnostic_info.uptime_seconds = (datetime.now() - diagnostic_info.created_at).total_seconds()
        
        return diagnostic_info
    
    def get_connection_health(self, bot_id: str) -> Optional[ConnectionHealth]:
        """
        Get connection health information for a bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            ConnectionHealth or None if bot not found
        """
        return self._connection_health.get(bot_id)
    
    def get_activity_metrics(self, bot_id: str) -> Optional[BotActivityMetrics]:
        """
        Get activity metrics for a bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            BotActivityMetrics or None if bot not found
        """
        if bot_id not in self._activity_metrics:
            return None
        
        # Update calculated metrics
        metrics = self._activity_metrics[bot_id]
        
        # Calculate success rates
        total_actions = metrics.actions_executed + metrics.actions_failed
        if total_actions > 0:
            metrics.action_success_rate = metrics.actions_executed / total_actions
        
        if metrics.decisions_made > 0:
            metrics.decision_success_rate = (metrics.decisions_made - metrics.errors_encountered) / metrics.decisions_made
        
        return metrics
    
    def get_diagnostic_events(self, bot_id: str, limit: Optional[int] = None,
                            event_type: Optional[BotLifecycleEvent] = None,
                            level: Optional[DiagnosticLevel] = None) -> List[DiagnosticEvent]:
        """
        Get diagnostic events for a bot.
        
        Args:
            bot_id: Bot identifier
            limit: Maximum number of events to return
            event_type: Filter by event type
            level: Filter by diagnostic level
            
        Returns:
            List of diagnostic events
        """
        if bot_id not in self._diagnostic_events:
            return []
        
        events = self._diagnostic_events[bot_id]
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if level:
            events = [e for e in events if e.level == level]
        
        # Sort by timestamp (most recent first)
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            events = events[:limit]
        
        return events
    
    def get_all_bot_diagnostics(self) -> Dict[str, BotDiagnosticInfo]:
        """
        Get diagnostic information for all tracked bots.
        
        Returns:
            Dictionary mapping bot_id to BotDiagnosticInfo
        """
        result = {}
        for bot_id in self._bot_diagnostics:
            result[bot_id] = self.get_bot_diagnostics(bot_id)
        return result
    
    def register_event_callback(self, callback: Callable[[DiagnosticEvent], Awaitable[None]]) -> None:
        """Register a callback for diagnostic events."""
        self._event_callbacks.append(callback)
    
    def register_health_callback(self, callback: Callable[[str, BotDiagnosticInfo], Awaitable[None]]) -> None:
        """Register a callback for bot health updates."""
        self._health_callbacks.append(callback)
    
    async def _notify_event_callbacks(self, event: DiagnosticEvent) -> None:
        """Notify all event callbacks about a new diagnostic event."""
        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                self.logger.error(f"Error in diagnostic event callback: {e}")
    
    async def _notify_health_callbacks(self, bot_id: str, diagnostic_info: BotDiagnosticInfo) -> None:
        """Notify all health callbacks about bot health updates."""
        for callback in self._health_callbacks:
            try:
                await callback(bot_id, diagnostic_info)
            except Exception as e:
                self.logger.error(f"Error in diagnostic health callback: {e}")
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old diagnostic data."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval_hours * 3600)  # Convert hours to seconds
                
                current_time = datetime.now()
                cleanup_threshold = current_time - timedelta(hours=self.cleanup_interval_hours * 2)
                
                # Clean up old events
                for bot_id in list(self._diagnostic_events.keys()):
                    events = self._diagnostic_events[bot_id]
                    self._diagnostic_events[bot_id] = [
                        e for e in events if e.timestamp > cleanup_threshold
                    ]
                
                self.logger.info("Completed periodic diagnostic data cleanup")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")


# Global diagnostic tracker instance
_diagnostic_tracker: Optional[BotDiagnosticTracker] = None


def get_diagnostic_tracker() -> BotDiagnosticTracker:
    """Get the global diagnostic tracker instance."""
    global _diagnostic_tracker
    if _diagnostic_tracker is None:
        _diagnostic_tracker = BotDiagnosticTracker()
    return _diagnostic_tracker


async def initialize_diagnostics() -> None:
    """Initialize the global diagnostic tracker."""
    tracker = get_diagnostic_tracker()
    await tracker.start()


async def cleanup_diagnostics() -> None:
    """Cleanup the global diagnostic tracker."""
    global _diagnostic_tracker
    if _diagnostic_tracker:
        await _diagnostic_tracker.stop()
        _diagnostic_tracker = None