"""
Bot Server for managing AI bot instances in game rooms.

This module provides the BotServer class that manages bot instances, handles bot lifecycle,
and coordinates with the game server for bot placement and management.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Awaitable
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from core.game_client import GameClient, TrainingMode
from rl_bot_system.rules_based.rules_based_bot import RulesBasedBot, DifficultyLevel
from rl_bot_system.training.model_manager import ModelManager
from server.diagnostics import (
    BotDiagnosticTracker, BotLifecycleEvent, DiagnosticLevel, 
    ConnectionStatus, AIStatus, get_diagnostic_tracker
)
from server.monitored_game_client import MonitoredGameClient
from server.websocket_monitor import get_websocket_monitor
from server.connection_diagnostics import get_connection_diagnostics


class BotType(Enum):
    """Types of bots available for spawning."""
    RULES_BASED = "rules_based"
    RL_GENERATION = "rl_generation"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class BotStatus(Enum):
    """Status of a bot instance."""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    ACTIVE = "active"
    DISCONNECTING = "disconnecting"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class BotConfig:
    """Configuration for a bot instance."""
    bot_type: BotType
    difficulty: DifficultyLevel
    name: str
    generation: Optional[int] = None  # For RL bots
    custom_config: Optional[Dict[str, Any]] = None
    training_mode: bool = False
    auto_cleanup: bool = True


@dataclass
class BotInstance:
    """Represents an active bot instance."""
    bot_id: str
    config: BotConfig
    status: BotStatus
    room_id: Optional[str]
    game_client: Optional[GameClient]
    bot_ai: Optional[Any]  # RulesBasedBot or RL model
    created_at: datetime
    last_activity: datetime
    performance_stats: Dict[str, Any]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'bot_id': self.bot_id,
            'config': {
                'bot_type': self.config.bot_type.value,
                'difficulty': self.config.difficulty.value,
                'name': self.config.name,
                'generation': self.config.generation,
                'training_mode': self.config.training_mode,
                'auto_cleanup': self.config.auto_cleanup
            },
            'status': self.status.value,
            'room_id': self.room_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'performance_stats': self.performance_stats,
            'error_message': self.error_message
        }


@dataclass
class BotServerConfig:
    """Configuration for the bot server."""
    max_bots_per_room: int = 8
    max_total_bots: int = 50
    bot_timeout_seconds: int = 300  # 5 minutes
    cleanup_interval_seconds: int = 60
    game_server_url: str = "http://localhost:4000"
    models_dir: str = "data/models"
    enable_performance_tracking: bool = True
    auto_cleanup_empty_rooms: bool = True
    bot_reconnect_attempts: int = 3
    bot_reconnect_delay: float = 5.0


class GameClientPool:
    """Pool manager for MonitoredGameClient connections with WebSocket monitoring."""
    
    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self._available_clients: List[MonitoredGameClient] = []
        self._active_clients: Dict[str, MonitoredGameClient] = {}  # bot_id -> client
        self._client_creation_lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def get_client(self, bot_id: str, game_server_url: str) -> MonitoredGameClient:
        """
        Get a MonitoredGameClient for a bot, either from pool or create new one.
        
        Args:
            bot_id: ID of the bot requesting the client
            game_server_url: URL of the game server
            
        Returns:
            MonitoredGameClient instance
            
        Raises:
            RuntimeError: If max connections exceeded
        """
        async with self._client_creation_lock:
            if len(self._active_clients) >= self.max_connections:
                raise RuntimeError(f"Maximum connections ({self.max_connections}) exceeded")
            
            # Try to reuse an available client
            if self._available_clients:
                client = self._available_clients.pop()
                # Update bot_id for monitoring
                client.bot_id = bot_id
                self.logger.debug(f"Reusing pooled monitored client for bot {bot_id}")
            else:
                # Create new monitored client
                ws_url = game_server_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
                client = MonitoredGameClient(bot_id=bot_id, ws_url=ws_url, http_url=game_server_url)
                self.logger.debug(f"Created new monitored client for bot {bot_id}")
            
            self._active_clients[bot_id] = client
            return client
    
    async def return_client(self, bot_id: str) -> None:
        """
        Return a MonitoredGameClient to the pool for reuse.
        
        Args:
            bot_id: ID of the bot returning the client
        """
        if bot_id in self._active_clients:
            client = self._active_clients.pop(bot_id)
            
            try:
                # Clean up client state
                await client.close()
                
                # Reset GameClient state for reuse (now inherited)
                client.game_state = {}
                client.player_id = None
                client.player_token = None
                client.room_id = None
                client.training_mode = TrainingMode.NORMAL
                client.speed_multiplier = 1.0
                client.direct_state_access = False
                client._state_cache = {}
                client._training_session_id = None
                client._state_update_callbacks.clear()
                
                # Reset monitoring state
                client._connection_established = False
                client._connection_start_time = None
                client._last_message_time = None
                
                # Return to pool if not at capacity
                if len(self._available_clients) < self.max_connections // 2:
                    self._available_clients.append(client)
                    self.logger.debug(f"Returned monitored client to pool from bot {bot_id}")
                else:
                    self.logger.debug(f"Discarded monitored client from bot {bot_id} (pool full)")
                    
            except Exception as e:
                self.logger.error(f"Error returning monitored client from bot {bot_id}: {e}")
    
    def get_pool_stats(self) -> Dict[str, int]:
        """Get statistics about the client pool."""
        return {
            'available_clients': len(self._available_clients),
            'active_clients': len(self._active_clients),
            'max_connections': self.max_connections,
            'utilization_percent': int((len(self._active_clients) / self.max_connections) * 100)
        }


class BotServer:
    """
    Server for managing AI bot instances and their lifecycle.
    
    Handles bot spawning, termination, resource allocation, and integration
    with the game server for bot placement in rooms.
    """
    
    def __init__(self, config: BotServerConfig):
        """
        Initialize the BotServer.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Bot management
        self._bots: Dict[str, BotInstance] = {}
        self._room_bots: Dict[str, Set[str]] = {}  # room_id -> set of bot_ids
        self._bot_creation_lock = asyncio.Lock()
        
        # Resource management
        self.client_pool = GameClientPool(config.max_total_bots)
        self.model_manager = ModelManager(config.models_dir)
        self._model_cache: Dict[int, Any] = {}  # generation -> loaded model
        
        # Diagnostic tracking
        self.diagnostic_tracker = get_diagnostic_tracker()
        
        # WebSocket monitoring
        self.websocket_monitor = get_websocket_monitor()
        self.connection_diagnostics = get_connection_diagnostics()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Event callbacks
        self._bot_status_callbacks: List[Callable[[str, BotStatus], Awaitable[None]]] = []
        self._room_empty_callbacks: List[Callable[[str], Awaitable[None]]] = []
        
        self.logger.info(f"BotServer initialized with max {config.max_total_bots} bots")
    
    async def start(self) -> None:
        """Start the bot server and background tasks."""
        if self._running:
            return
        
        self._running = True
        self._start_time = datetime.now()
        
        # Start diagnostic tracker
        await self.diagnostic_tracker.start()
        
        # Start WebSocket monitoring
        await self.websocket_monitor.start()
        await self.connection_diagnostics.start()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        self.logger.info("BotServer started")
    
    async def stop(self) -> None:
        """Stop the bot server and clean up resources."""
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
        
        # Terminate all bots
        bot_ids = list(self._bots.keys())
        for bot_id in bot_ids:
            await self.terminate_bot(bot_id)
        
        # Stop diagnostic tracker
        await self.diagnostic_tracker.stop()
        
        # Stop WebSocket monitoring
        await self.websocket_monitor.stop()
        await self.connection_diagnostics.stop()
        
        self.logger.info("BotServer stopped")
    
    async def spawn_bot(self, bot_config: BotConfig, room_info: Dict[str, Any]) -> str:
        """
        Spawn a new bot instance and connect it to a game room.
        
        Args:
            bot_config: Configuration for the bot
            room_info: Information about the room to join (room_code, password, etc.)
            
        Returns:
            str: Bot ID of the spawned bot
            
        Raises:
            RuntimeError: If bot cannot be spawned due to limits or errors
        """
        spawn_start_time = datetime.now()
        correlation_id = str(uuid.uuid4())
        
        # Log spawn request with detailed context
        self.logger.info(f"Bot spawn requested - Type: {bot_config.bot_type.value}, "
                        f"Difficulty: {bot_config.difficulty.value}, Name: {bot_config.name}, "
                        f"Room: {room_info.get('room_id', 'unknown')}, "
                        f"Training: {bot_config.training_mode}")
        
        async with self._bot_creation_lock:
            try:
                # Detailed limit checking with diagnostic logging
                current_bot_count = len(self._bots)
                self.logger.debug(f"Current bot count: {current_bot_count}/{self.config.max_total_bots}")
                
                if current_bot_count >= self.config.max_total_bots:
                    error_msg = f"Maximum total bots ({self.config.max_total_bots}) exceeded. Current: {current_bot_count}"
                    self.logger.error(f"Bot spawn failed - {error_msg}")
                    raise RuntimeError(error_msg)
                
                room_id = room_info.get('room_id')
                if room_id:
                    room_bot_count = len(self._room_bots.get(room_id, set()))
                    self.logger.debug(f"Room {room_id} bot count: {room_bot_count}/{self.config.max_bots_per_room}")
                    
                    if room_bot_count >= self.config.max_bots_per_room:
                        error_msg = f"Maximum bots per room ({self.config.max_bots_per_room}) exceeded for room {room_id}. Current: {room_bot_count}"
                        self.logger.error(f"Bot spawn failed - {error_msg}")
                        raise RuntimeError(error_msg)
                
                # Generate bot ID and log creation step
                bot_id = str(uuid.uuid4())
                self.logger.info(f"Generated bot ID: {bot_id} for spawn request")
                
                # Validate bot configuration before creation
                validation_result = self._validate_bot_config(bot_config, room_info)
                if not validation_result['valid']:
                    error_msg = f"Bot configuration validation failed: {validation_result['error']}"
                    self.logger.error(f"Bot spawn failed - {error_msg}")
                    raise RuntimeError(error_msg)
                
                self.logger.debug(f"Bot configuration validated successfully for {bot_id}")
                
                # Create bot instance with detailed logging
                self.logger.debug(f"Creating bot instance for {bot_id}")
                bot_instance = BotInstance(
                    bot_id=bot_id,
                    config=bot_config,
                    status=BotStatus.INITIALIZING,
                    room_id=room_id,
                    game_client=None,
                    bot_ai=None,
                    created_at=datetime.now(),
                    last_activity=datetime.now(),
                    performance_stats={
                        'games_played': 0,
                        'wins': 0,
                        'losses': 0,
                        'kills': 0,
                        'deaths': 0,
                        'accuracy': 0.0,
                        'total_playtime': 0.0
                    }
                )
                
                self._bots[bot_id] = bot_instance
                self.logger.debug(f"Bot instance created and stored for {bot_id}")
                
                # Track room assignment with logging
                if room_id:
                    if room_id not in self._room_bots:
                        self._room_bots[room_id] = set()
                        self.logger.debug(f"Created new room tracking for {room_id}")
                    
                    self._room_bots[room_id].add(bot_id)
                    self.logger.debug(f"Added bot {bot_id} to room {room_id} tracking")
                
                # Register bot with diagnostic tracker
                self.diagnostic_tracker.register_bot(
                    bot_id=bot_id,
                    bot_name=bot_config.name,
                    bot_type=bot_config.bot_type.value,
                    difficulty=bot_config.difficulty.value,
                    room_id=room_id
                )
                
                # Log detailed spawn success metrics
                spawn_duration = (datetime.now() - spawn_start_time).total_seconds()
                self.logger.info(f"Bot spawn setup completed successfully - ID: {bot_id}, "
                               f"Type: {bot_config.bot_type.value}, Duration: {spawn_duration:.3f}s")
                
                # Log spawn completion event with correlation ID
                self.diagnostic_tracker.log_event(
                    bot_id=bot_id,
                    event_type=BotLifecycleEvent.BOT_SPAWN_REQUESTED,
                    level=DiagnosticLevel.INFO,
                    message=f"Bot spawn setup completed, starting initialization",
                    details={
                        'bot_type': bot_config.bot_type.value,
                        'difficulty': bot_config.difficulty.value,
                        'training_mode': bot_config.training_mode,
                        'room_id': room_id,
                        'room_code': room_info.get('room_code'),
                        'spawn_duration_ms': spawn_duration * 1000,
                        'total_bots_after': len(self._bots),
                        'room_bots_after': len(self._room_bots.get(room_id, set())) if room_id else 0
                    },
                    correlation_id=correlation_id
                )
                
                # Initialize bot in background with correlation tracking
                asyncio.create_task(self._initialize_bot(bot_id, room_info, correlation_id))
                
                return bot_id
                
            except Exception as e:
                # Log detailed spawn failure
                spawn_duration = (datetime.now() - spawn_start_time).total_seconds()
                self.logger.error(f"Bot spawn failed after {spawn_duration:.3f}s - "
                                f"Type: {bot_config.bot_type.value}, "
                                f"Name: {bot_config.name}, "
                                f"Error: {str(e)}")
                raise
    
    async def terminate_bot(self, bot_id: str) -> bool:
        """
        Terminate a bot instance and clean up resources.
        
        Args:
            bot_id: ID of the bot to terminate
            
        Returns:
            bool: True if termination successful
        """
        if bot_id not in self._bots:
            self.logger.warning(f"Bot {bot_id} not found for termination")
            return False
        
        bot_instance = self._bots[bot_id]
        
        try:
            # Update status
            await self._update_bot_status(bot_id, BotStatus.DISCONNECTING)
            
            # Disconnect from game
            if bot_instance.game_client:
                try:
                    await bot_instance.game_client.exit_game()
                except Exception as e:
                    self.logger.error(f"Error exiting game for bot {bot_id}: {e}")
                
                # Return client to pool
                await self.client_pool.return_client(bot_id)
            
            # Remove from room tracking
            if bot_instance.room_id and bot_instance.room_id in self._room_bots:
                self._room_bots[bot_instance.room_id].discard(bot_id)
                
                # Check if room is now empty
                if not self._room_bots[bot_instance.room_id]:
                    del self._room_bots[bot_instance.room_id]
                    
                    # Notify callbacks about empty room
                    if self.config.auto_cleanup_empty_rooms:
                        for callback in self._room_empty_callbacks:
                            try:
                                await callback(bot_instance.room_id)
                            except Exception as e:
                                self.logger.error(f"Error in room empty callback: {e}")
            
            # Update final status
            await self._update_bot_status(bot_id, BotStatus.TERMINATED)
            
            # Remove from active bots
            del self._bots[bot_id]
            
            # Unregister from diagnostic tracker
            self.diagnostic_tracker.unregister_bot(bot_id)
            
            self.logger.info(f"Terminated bot {bot_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error terminating bot {bot_id}: {e}")
            bot_instance.status = BotStatus.ERROR
            bot_instance.error_message = str(e)
            return False
    
    async def configure_bot_difficulty(self, bot_id: str, difficulty: DifficultyLevel) -> bool:
        """
        Adjust bot difficulty level mid-game.
        
        Args:
            bot_id: ID of the bot to configure
            difficulty: New difficulty level
            
        Returns:
            bool: True if configuration successful
        """
        if bot_id not in self._bots:
            return False
        
        bot_instance = self._bots[bot_id]
        
        try:
            old_difficulty = bot_instance.config.difficulty
            
            # Update config
            bot_instance.config.difficulty = difficulty
            
            # Update bot AI if it's rules-based
            if (bot_instance.config.bot_type == BotType.RULES_BASED and 
                isinstance(bot_instance.bot_ai, RulesBasedBot)):
                bot_instance.bot_ai.set_difficulty_level(difficulty)
                
                # Apply real-time difficulty adjustments
                await self._apply_realtime_difficulty_changes(bot_id, old_difficulty, difficulty)
            
            bot_instance.last_activity = datetime.now()
            
            # Notify callbacks about difficulty change
            for callback in self._bot_status_callbacks:
                try:
                    await callback(bot_id, bot_instance.status)
                except Exception as e:
                    self.logger.error(f"Error in difficulty change callback: {e}")
            
            self.logger.info(f"Updated bot {bot_id} difficulty from {old_difficulty.value} to {difficulty.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring bot {bot_id} difficulty: {e}")
            return False
    
    def get_bot_status(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a specific bot.
        
        Args:
            bot_id: ID of the bot
            
        Returns:
            Optional[Dict[str, Any]]: Bot status information or None if not found
        """
        if bot_id not in self._bots:
            return None
        
        return self._bots[bot_id].to_dict()
    
    def get_room_bots(self, room_id: str) -> List[Dict[str, Any]]:
        """
        Get all bots in a specific room.
        
        Args:
            room_id: ID of the room
            
        Returns:
            List[Dict[str, Any]]: List of bot information dictionaries
        """
        if room_id not in self._room_bots:
            return []
        
        bots = []
        for bot_id in self._room_bots[room_id]:
            if bot_id in self._bots:
                bots.append(self._bots[bot_id].to_dict())
        
        return bots
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get overall server status and statistics.
        
        Returns:
            Dict[str, Any]: Server status information
        """
        # Count bots by status
        status_counts = {}
        for status in BotStatus:
            status_counts[status.value] = 0
        
        for bot in self._bots.values():
            status_counts[bot.status.value] += 1
        
        # Count bots by type
        type_counts = {}
        for bot_type in BotType:
            type_counts[bot_type.value] = 0
        
        for bot in self._bots.values():
            type_counts[bot.config.bot_type.value] += 1
        
        # Get WebSocket monitoring stats
        websocket_connections = self.websocket_monitor.get_all_connections()
        connection_stats = {
            'total_monitored_connections': len(websocket_connections),
            'healthy_connections': len([c for c in websocket_connections.values() if c.is_healthy]),
            'failed_connections': len([c for c in websocket_connections.values() 
                                     if c.connection_state.value == 'failed']),
            'connecting_connections': len([c for c in websocket_connections.values() 
                                         if c.connection_state.value == 'connecting'])
        }
        
        return {
            'running': self._running,
            'total_bots': len(self._bots),
            'max_total_bots': self.config.max_total_bots,
            'active_rooms': len(self._room_bots),
            'bots_by_status': status_counts,
            'bots_by_type': type_counts,
            'client_pool_stats': self.client_pool.get_pool_stats(),
            'websocket_stats': connection_stats,
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds() if hasattr(self, '_start_time') else 0
        }
    
    def get_available_bot_types(self) -> List[Dict[str, Any]]:
        """
        Get list of available bot types and their configurations.
        
        Returns:
            List[Dict[str, Any]]: Available bot types with metadata
        """
        bot_types = []
        
        # Rules-based bots
        bot_types.append({
            'type': BotType.RULES_BASED.value,
            'name': 'Rules-Based Bot',
            'description': 'Traditional AI with configurable difficulty levels',
            'difficulties': [d.value for d in DifficultyLevel],
            'supports_training_mode': True
        })
        
        # RL generation bots
        available_generations = self.model_manager.list_models()
        if available_generations:
            bot_types.append({
                'type': BotType.RL_GENERATION.value,
                'name': 'RL Generation Bot',
                'description': 'Reinforcement learning trained models',
                'available_generations': [gen for gen, _ in available_generations],
                'difficulties': [d.value for d in DifficultyLevel],
                'supports_training_mode': True
            })
        
        return bot_types
    
    def register_bot_status_callback(self, callback: Callable[[str, BotStatus], Awaitable[None]]) -> None:
        """Register a callback for bot status changes."""
        self._bot_status_callbacks.append(callback)
    
    def register_room_empty_callback(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Register a callback for when rooms become empty of bots."""
        self._room_empty_callbacks.append(callback)
    
    # WebSocket Connection Monitoring Methods
    
    def get_bot_connection_health(self, bot_id: str) -> Dict[str, Any]:
        """
        Get WebSocket connection health for a specific bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            Dictionary with connection health information
        """
        return self.websocket_monitor.get_connection_health(bot_id)
    
    def get_bot_connection_info(self, bot_id: str) -> Optional[Any]:
        """
        Get detailed WebSocket connection information for a bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            WebSocketConnectionInfo or None if not found
        """
        return self.websocket_monitor.get_connection_info(bot_id)
    
    def get_bot_message_history(self, bot_id: str, limit: Optional[int] = None) -> List[Any]:
        """
        Get WebSocket message history for a bot.
        
        Args:
            bot_id: Bot identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of WebSocket messages
        """
        return self.websocket_monitor.get_message_history(bot_id, limit)
    
    def get_all_connection_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get connection health for all bots.
        
        Returns:
            Dictionary mapping bot_id to health information
        """
        result = {}
        connections = self.websocket_monitor.get_all_connections()
        for bot_id in connections.keys():
            result[bot_id] = self.websocket_monitor.get_connection_health(bot_id)
        return result
    
    def get_connection_issues(self, bot_id: str) -> List[Dict[str, Any]]:
        """
        Get connection issues detected for a specific bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            List of connection issues
        """
        issues = self.connection_diagnostics.get_bot_issues(bot_id)
        return [issue.to_dict() for issue in issues]
    
    def get_all_connection_issues(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get connection issues for all bots.
        
        Returns:
            Dictionary mapping bot_id to list of issues
        """
        all_issues = self.connection_diagnostics.get_all_issues()
        result = {}
        for bot_id, issues in all_issues.items():
            result[bot_id] = [issue.to_dict() for issue in issues]
        return result
    
    def generate_connection_report(self, bot_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive connection report for a bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            Detailed connection report
        """
        return self.connection_diagnostics.generate_connection_report(bot_id)
    
    def resolve_connection_issue(self, bot_id: str, issue_id: str) -> bool:
        """
        Mark a connection issue as resolved.
        
        Args:
            bot_id: Bot identifier
            issue_id: Issue identifier
            
        Returns:
            True if issue was found and resolved
        """
        return self.connection_diagnostics.resolve_issue(bot_id, issue_id)
    
    async def check_room_human_players(self, room_id: str) -> bool:
        """
        Check if a room has any human players remaining.
        
        Args:
            room_id: ID of the room to check
            
        Returns:
            bool: True if human players are present, False otherwise
        """
        try:
            # This would need to integrate with the game server API
            # For now, we'll implement a placeholder that can be extended
            # In a real implementation, this would query the game server
            
            # Placeholder implementation - assume room has human players
            # This should be replaced with actual game server integration
            self.logger.debug(f"Checking human players in room {room_id} (placeholder implementation)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking human players in room {room_id}: {e}")
            return True  # Assume players present on error to avoid premature cleanup
    
    async def cleanup_room_bots(self, room_id: str, reason: str = "room_empty") -> int:
        """
        Clean up all bots in a room.
        
        Args:
            room_id: ID of the room to clean up
            reason: Reason for cleanup
            
        Returns:
            int: Number of bots cleaned up
        """
        if room_id not in self._room_bots:
            return 0
        
        bot_ids = list(self._room_bots[room_id])
        cleanup_count = 0
        
        self.logger.info(f"Cleaning up {len(bot_ids)} bots from room {room_id} (reason: {reason})")
        
        for bot_id in bot_ids:
            try:
                success = await self.terminate_bot(bot_id)
                if success:
                    cleanup_count += 1
            except Exception as e:
                self.logger.error(f"Error cleaning up bot {bot_id} from room {room_id}: {e}")
        
        return cleanup_count
    
    async def monitor_bot_health(self, bot_id: str) -> Dict[str, Any]:
        """
        Monitor the health status of a specific bot.
        
        Args:
            bot_id: ID of the bot to monitor
            
        Returns:
            Dict[str, Any]: Health status information
        """
        if bot_id not in self._bots:
            return {"status": "not_found", "healthy": False}
        
        # Get comprehensive diagnostic information
        diagnostic_info = self.diagnostic_tracker.get_bot_diagnostics(bot_id)
        connection_health = self.diagnostic_tracker.get_connection_health(bot_id)
        activity_metrics = self.diagnostic_tracker.get_activity_metrics(bot_id)
        
        if not diagnostic_info:
            return {"status": "not_found", "healthy": False}
        
        bot_instance = self._bots[bot_id]
        current_time = datetime.now()
        
        # Calculate time since last activity
        time_since_activity = (current_time - bot_instance.last_activity).total_seconds()
        
        # Build comprehensive health status using diagnostic data
        health_status = {
            "bot_id": bot_id,
            "bot_name": diagnostic_info.bot_name,
            "bot_type": diagnostic_info.bot_type,
            "difficulty": diagnostic_info.difficulty,
            "status": diagnostic_info.status,
            "healthy": True,
            "last_activity_seconds": time_since_activity,
            "connection_status": diagnostic_info.connection_status.value,
            "ai_status": diagnostic_info.ai_status.value,
            "websocket_connected": diagnostic_info.websocket_connected,
            "websocket_url": diagnostic_info.websocket_url,
            "actions_sent": diagnostic_info.actions_sent,
            "decisions_made": diagnostic_info.decisions_made,
            "game_state_updates_received": diagnostic_info.game_state_updates_received,
            "reconnection_attempts": diagnostic_info.reconnection_attempts,
            "uptime_seconds": diagnostic_info.uptime_seconds,
            "performance_issues": diagnostic_info.performance_issues.copy(),
            "error_messages": diagnostic_info.error_messages.copy(),
            "connection_errors": diagnostic_info.connection_errors.copy()
        }
        
        # Add activity metrics if available
        if activity_metrics:
            health_status.update({
                "activity_metrics": {
                    "decisions_made": activity_metrics.decisions_made,
                    "actions_executed": activity_metrics.actions_executed,
                    "actions_failed": activity_metrics.actions_failed,
                    "errors_encountered": activity_metrics.errors_encountered,
                    "decision_success_rate": activity_metrics.decision_success_rate,
                    "action_success_rate": activity_metrics.action_success_rate,
                    "average_decision_time_ms": activity_metrics.average_decision_time_ms,
                    "last_decision_time": activity_metrics.last_decision_time.isoformat() if activity_metrics.last_decision_time else None,
                    "last_action_time": activity_metrics.last_action_time.isoformat() if activity_metrics.last_action_time else None
                }
            })
        
        # Add connection health if available
        if connection_health:
            health_status["connection_health"] = {
                "websocket_connected": connection_health.websocket_connected,
                "connection_status": connection_health.connection_status.value,
                "reconnection_attempts": connection_health.reconnection_attempts,
                "messages_sent": connection_health.messages_sent,
                "messages_received": connection_health.messages_received,
                "connection_duration": str(connection_health.connection_duration) if connection_health.connection_duration else None
            }
        
        # Determine overall health based on diagnostic data
        health_status["healthy"] = len(diagnostic_info.performance_issues) == 0 and len(diagnostic_info.error_messages) == 0
        
        # Check if bot is responsive
        if time_since_activity > self.config.bot_timeout_seconds:
            health_status["healthy"] = False
            if "inactive_timeout" not in health_status["performance_issues"]:
                health_status["performance_issues"].append("inactive_timeout")
        
        # Check for critical issues
        if diagnostic_info.connection_status in [ConnectionStatus.FAILED, ConnectionStatus.DISCONNECTED]:
            health_status["healthy"] = False
        
        if diagnostic_info.ai_status in [AIStatus.ERROR, AIStatus.NOT_LOADED]:
            health_status["healthy"] = False
        
        if diagnostic_info.status == "error":
            health_status["healthy"] = False
        
        return health_status
    
    async def attempt_bot_reconnection(self, bot_id: str) -> bool:
        """
        Attempt to reconnect a bot that has lost connection.
        
        Args:
            bot_id: ID of the bot to reconnect
            
        Returns:
            bool: True if reconnection successful
        """
        if bot_id not in self._bots:
            return False
        
        bot_instance = self._bots[bot_id]
        
        try:
            self.logger.info(f"Attempting to reconnect bot {bot_id}")
            
            # Update status to indicate reconnection attempt
            await self._update_bot_status(bot_id, BotStatus.CONNECTING)
            
            # Get fresh game client
            if bot_instance.game_client:
                await self.client_pool.return_client(bot_id)
            
            bot_instance.game_client = await self.client_pool.get_client(
                bot_id, self.config.game_server_url
            )
            
            # Configure training mode if needed
            if bot_instance.config.training_mode:
                await bot_instance.game_client.enable_training_mode(
                    speed_multiplier=10.0,
                    headless=True
                )
            
            # Reconnect to the same room
            if bot_instance.room_id:
                # This would need room information to reconnect
                # For now, we'll mark as active and let the AI loop handle reconnection
                pass
            
            # Restart bot AI loop
            asyncio.create_task(self._run_bot_ai_loop(bot_id))
            
            await self._update_bot_status(bot_id, BotStatus.ACTIVE)
            bot_instance.last_activity = datetime.now()
            bot_instance.error_message = None
            
            self.logger.info(f"Successfully reconnected bot {bot_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reconnect bot {bot_id}: {e}")
            bot_instance.status = BotStatus.ERROR
            bot_instance.error_message = f"Reconnection failed: {e}"
            return False
    
    async def get_all_bot_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health status for all active bots.
        
        Returns:
            Dict[str, Dict[str, Any]]: Health status for each bot
        """
        health_statuses = {}
        
        for bot_id in self._bots.keys():
            health_statuses[bot_id] = await self.monitor_bot_health(bot_id)
        
        return health_statuses
    
    # Private methods
    
    async def _update_bot_status(self, bot_id: str, status: BotStatus, error_message: Optional[str] = None) -> None:
        """Update bot status and notify diagnostic tracker."""
        if bot_id not in self._bots:
            return
        
        old_status = self._bots[bot_id].status
        self._bots[bot_id].status = status
        self._bots[bot_id].last_activity = datetime.now()
        
        if error_message:
            self._bots[bot_id].error_message = error_message
        
        # Update diagnostic tracker
        self.diagnostic_tracker.update_bot_status(
            bot_id=bot_id,
            status=status.value,
            error_message=error_message
        )
        
        # Notify callbacks
        for callback in self._bot_status_callbacks:
            try:
                await callback(bot_id, status)
            except Exception as e:
                self.logger.error(f"Error in bot status callback: {e}")
    
    async def _initialize_bot(self, bot_id: str, room_info: Dict[str, Any], correlation_id: Optional[str] = None) -> None:
        """Initialize a bot instance in the background with detailed diagnostic tracking."""
        if bot_id not in self._bots:
            self.logger.error(f"Bot {bot_id} not found in _bots during initialization")
            return
        
        bot_instance = self._bots[bot_id]
        init_start_time = datetime.now()
        
        try:
            # Log initialization start with detailed context
            self.logger.info(f"Starting bot initialization - ID: {bot_id}, "
                           f"Type: {bot_instance.config.bot_type.value}, "
                           f"Room: {room_info.get('room_code', 'unknown')}")
            
            self.diagnostic_tracker.log_event(
                bot_id=bot_id,
                event_type=BotLifecycleEvent.BOT_INITIALIZING,
                level=DiagnosticLevel.INFO,
                message="Starting bot initialization process",
                details={
                    'room_code': room_info.get('room_code'),
                    'room_password_provided': bool(room_info.get('room_password')),
                    'training_mode': bot_instance.config.training_mode,
                    'auto_cleanup': bot_instance.config.auto_cleanup
                },
                correlation_id=correlation_id
            )
            await self._update_bot_status(bot_id, BotStatus.INITIALIZING)
            
            # Step 1: Load bot AI with detailed tracking
            ai_load_start = datetime.now()
            self.logger.debug(f"Loading AI for bot {bot_id} - Type: {bot_instance.config.bot_type.value}")
            
            self.diagnostic_tracker.update_ai_status(bot_id, AIStatus.LOADING)
            self.diagnostic_tracker.log_event(
                bot_id=bot_id,
                event_type=BotLifecycleEvent.BOT_AI_LOADING,
                level=DiagnosticLevel.INFO,
                message=f"Loading {bot_instance.config.bot_type.value} AI",
                details={
                    'bot_type': bot_instance.config.bot_type.value,
                    'difficulty': bot_instance.config.difficulty.value,
                    'generation': bot_instance.config.generation
                },
                correlation_id=correlation_id
            )
            
            try:
                bot_instance.bot_ai = await self._load_bot_ai(bot_instance.config, bot_id, correlation_id)
                ai_load_duration = (datetime.now() - ai_load_start).total_seconds()
                
                self.diagnostic_tracker.update_ai_status(bot_id, AIStatus.LOADED)
                self.diagnostic_tracker.log_event(
                    bot_id=bot_id,
                    event_type=BotLifecycleEvent.BOT_AI_LOADED,
                    level=DiagnosticLevel.INFO,
                    message=f"Bot AI loaded successfully ({bot_instance.config.bot_type.value})",
                    details={
                        'load_duration_ms': ai_load_duration * 1000,
                        'ai_type': type(bot_instance.bot_ai).__name__
                    },
                    correlation_id=correlation_id
                )
                
                self.logger.info(f"AI loaded for bot {bot_id} in {ai_load_duration:.3f}s")
                
            except Exception as ai_error:
                ai_load_duration = (datetime.now() - ai_load_start).total_seconds()
                error_msg = f"Failed to load AI: {str(ai_error)}"
                
                self.diagnostic_tracker.update_ai_status(bot_id, AIStatus.ERROR, error_msg)
                self.diagnostic_tracker.log_event(
                    bot_id=bot_id,
                    event_type=BotLifecycleEvent.BOT_AI_LOAD_FAILED,
                    level=DiagnosticLevel.ERROR,
                    message=error_msg,
                    details={
                        'error_type': type(ai_error).__name__,
                        'error_message': str(ai_error),
                        'load_duration_ms': ai_load_duration * 1000,
                        'bot_type': bot_instance.config.bot_type.value,
                        'difficulty': bot_instance.config.difficulty.value
                    },
                    correlation_id=correlation_id
                )
                
                self.logger.error(f"AI loading failed for bot {bot_id} after {ai_load_duration:.3f}s: {ai_error}")
                raise ai_error
            
            # Step 2: Get game client with detailed tracking
            client_start = datetime.now()
            self.logger.debug(f"Acquiring game client for bot {bot_id}")
            
            self.diagnostic_tracker.log_event(
                bot_id=bot_id,
                event_type=BotLifecycleEvent.BOT_CONNECTING,
                level=DiagnosticLevel.INFO,
                message="Starting game client acquisition",
                details={
                    'game_server_url': self.config.game_server_url,
                    'client_pool_stats': self.client_pool.get_pool_stats()
                },
                correlation_id=correlation_id
            )
            await self._update_bot_status(bot_id, BotStatus.CONNECTING)
            
            # Update connection status
            websocket_url = self.config.game_server_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
            self.diagnostic_tracker.update_connection_status(
                bot_id=bot_id,
                connection_status=ConnectionStatus.CONNECTING,
                websocket_url=websocket_url
            )
            
            try:
                bot_instance.game_client = await self.client_pool.get_client(
                    bot_id, self.config.game_server_url
                )
                client_duration = (datetime.now() - client_start).total_seconds()
                
                self.logger.info(f"Game client acquired for bot {bot_id} in {client_duration:.3f}s")
                
                # Step 3: Configure training mode if needed
                if bot_instance.config.training_mode:
                    training_start = datetime.now()
                    self.logger.debug(f"Enabling training mode for bot {bot_id}")
                    
                    await bot_instance.game_client.enable_training_mode(
                        speed_multiplier=10.0,
                        headless=True
                    )
                    
                    training_duration = (datetime.now() - training_start).total_seconds()
                    self.logger.debug(f"Training mode enabled for bot {bot_id} in {training_duration:.3f}s")
                
                # Step 4: Connect to game room with detailed tracking
                room_code = room_info.get('room_code')
                room_password = room_info.get('room_password', '')
                
                if room_code:
                    connect_start = datetime.now()
                    self.logger.info(f"Connecting bot {bot_id} to room {room_code}")
                    
                    self.diagnostic_tracker.log_event(
                        bot_id=bot_id,
                        event_type=BotLifecycleEvent.BOT_WEBSOCKET_CONNECTING,
                        level=DiagnosticLevel.INFO,
                        message=f"Connecting to game room {room_code}",
                        details={
                            'room_code': room_code, 
                            'has_password': bool(room_password),
                            'player_name': bot_instance.config.name,
                            'websocket_url': websocket_url
                        },
                        correlation_id=correlation_id
                    )
                    
                    await bot_instance.game_client.connect(
                        room_code=room_code,
                        player_name=bot_instance.config.name,
                        room_password=room_password
                    )
                    
                    connect_duration = (datetime.now() - connect_start).total_seconds()
                    
                    self.diagnostic_tracker.log_event(
                        bot_id=bot_id,
                        event_type=BotLifecycleEvent.BOT_GAME_JOINED,
                        level=DiagnosticLevel.INFO,
                        message=f"Successfully joined game room {room_code}",
                        details={
                            'connect_duration_ms': connect_duration * 1000,
                            'player_id': getattr(bot_instance.game_client, 'player_id', None),
                            'room_id': getattr(bot_instance.game_client, 'room_id', None)
                        },
                        correlation_id=correlation_id
                    )
                    
                    self.logger.info(f"Bot {bot_id} joined room {room_code} in {connect_duration:.3f}s")
                
                # Update connection status to connected
                self.diagnostic_tracker.update_connection_status(
                    bot_id=bot_id,
                    connection_status=ConnectionStatus.CONNECTED,
                    websocket_connected=True,
                    websocket_url=websocket_url
                )
                
            except Exception as conn_error:
                client_duration = (datetime.now() - client_start).total_seconds()
                error_msg = f"Game client connection failed: {str(conn_error)}"
                
                self.diagnostic_tracker.update_connection_status(
                    bot_id=bot_id,
                    connection_status=ConnectionStatus.FAILED,
                    error_message=error_msg
                )
                self.diagnostic_tracker.log_event(
                    bot_id=bot_id,
                    event_type=BotLifecycleEvent.BOT_GAME_JOIN_FAILED,
                    level=DiagnosticLevel.ERROR,
                    message=error_msg,
                    details={
                        'error_type': type(conn_error).__name__,
                        'error_message': str(conn_error),
                        'connection_duration_ms': client_duration * 1000,
                        'room_code': room_info.get('room_code'),
                        'websocket_url': websocket_url,
                        'game_server_url': self.config.game_server_url
                    },
                    correlation_id=correlation_id
                )
                
                self.logger.error(f"Connection failed for bot {bot_id} after {client_duration:.3f}s: {conn_error}")
                raise conn_error
            
            # Step 5: Start bot AI loop with tracking
            self.logger.debug(f"Starting AI loop for bot {bot_id}")
            asyncio.create_task(self._run_bot_ai_loop(bot_id, correlation_id))
            
            # Update AI status to active
            self.diagnostic_tracker.update_ai_status(bot_id, AIStatus.ACTIVE)
            
            await self._update_bot_status(bot_id, BotStatus.ACTIVE)
            bot_instance.last_activity = datetime.now()
            
            # Log successful initialization with comprehensive metrics
            total_init_duration = (datetime.now() - init_start_time).total_seconds()
            
            self.diagnostic_tracker.log_event(
                bot_id=bot_id,
                event_type=BotLifecycleEvent.BOT_ACTIVE,
                level=DiagnosticLevel.INFO,
                message="Bot initialization completed successfully",
                details={
                    'total_init_duration_ms': total_init_duration * 1000,
                    'final_status': BotStatus.ACTIVE.value,
                    'ai_status': AIStatus.ACTIVE.value,
                    'connection_status': ConnectionStatus.CONNECTED.value,
                    'websocket_connected': True,
                    'player_id': getattr(bot_instance.game_client, 'player_id', None),
                    'room_id': getattr(bot_instance.game_client, 'room_id', None)
                },
                correlation_id=correlation_id
            )
            
            self.logger.info(f"Bot {bot_id} initialization completed successfully in {total_init_duration:.3f}s")
            
        except Exception as e:
            # Log detailed initialization failure
            total_init_duration = (datetime.now() - init_start_time).total_seconds()
            error_msg = f"Bot initialization failed: {str(e)}"
            
            self.diagnostic_tracker.log_event(
                bot_id=bot_id,
                event_type=BotLifecycleEvent.BOT_ERROR,
                level=DiagnosticLevel.ERROR,
                message=error_msg,
                details={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'init_duration_ms': total_init_duration * 1000,
                    'final_status': BotStatus.ERROR.value,
                    'room_code': room_info.get('room_code'),
                    'bot_type': bot_instance.config.bot_type.value
                },
                correlation_id=correlation_id
            )
            
            self.logger.error(f"Bot {bot_id} initialization failed after {total_init_duration:.3f}s: {e}")
            await self._update_bot_status(bot_id, BotStatus.ERROR, str(e))
            
            self.diagnostic_tracker.log_event(
                bot_id=bot_id,
                event_type=BotLifecycleEvent.BOT_ERROR,
                level=DiagnosticLevel.ERROR,
                message=f"Bot initialization failed: {e}",
                details={'error': str(e), 'error_type': type(e).__name__}
            )
            
            # Clean up on failure
            if bot_instance.game_client:
                await self.client_pool.return_client(bot_id)
    
    async def _load_bot_ai(self, config: BotConfig, bot_id: str, correlation_id: Optional[str] = None) -> Any:
        """Load the appropriate AI for a bot configuration with detailed diagnostic tracking."""
        load_start = datetime.now()
        
        try:
            if config.bot_type == BotType.RULES_BASED:
                self.logger.debug(f"Loading rules-based AI for bot {bot_id} with difficulty {config.difficulty.value}")
                
                # Create rules-based bot with validation
                try:
                    bot_ai = RulesBasedBot(config.difficulty)
                    
                    # Validate the bot AI was created successfully
                    if not hasattr(bot_ai, 'difficulty'):
                        raise RuntimeError("Rules-based bot creation failed - missing difficulty attribute")
                    
                    load_duration = (datetime.now() - load_start).total_seconds()
                    self.logger.debug(f"Rules-based AI loaded successfully for bot {bot_id} in {load_duration:.3f}s")
                    
                    return bot_ai
                    
                except Exception as rules_error:
                    load_duration = (datetime.now() - load_start).total_seconds()
                    error_msg = f"Failed to create rules-based bot: {str(rules_error)}"
                    self.logger.error(f"Rules-based AI loading failed for bot {bot_id} after {load_duration:.3f}s: {rules_error}")
                    raise RuntimeError(error_msg) from rules_error
            
            elif config.bot_type == BotType.RL_GENERATION:
                if config.generation is None:
                    raise ValueError("Generation required for RL bot")
                
                self.logger.debug(f"Loading RL generation {config.generation} AI for bot {bot_id}")
                
                # Load from cache or model manager
                if config.generation not in self._model_cache:
                    try:
                        self.logger.debug(f"RL generation {config.generation} not in cache, attempting to load from model manager")
                        
                        # Check if model exists
                        available_models = self.model_manager.list_models()
                        available_generations = [gen for gen, _ in available_models]
                        
                        if config.generation not in available_generations:
                            self.logger.warning(f"RL generation {config.generation} not found in available models: {available_generations}")
                            self.logger.info(f"Falling back to rules-based bot for bot {bot_id}")
                            return RulesBasedBot(config.difficulty)
                        
                        # This would need to be implemented based on the actual RL model structure
                        # For now, return a placeholder with detailed logging
                        self.logger.warning(f"RL model loading not fully implemented for generation {config.generation}")
                        self.logger.info(f"Falling back to rules-based bot for bot {bot_id}")
                        
                        # Fallback to rules-based for now
                        fallback_bot = RulesBasedBot(config.difficulty)
                        load_duration = (datetime.now() - load_start).total_seconds()
                        self.logger.debug(f"Fallback rules-based AI loaded for bot {bot_id} in {load_duration:.3f}s")
                        
                        return fallback_bot
                        
                    except Exception as e:
                        load_duration = (datetime.now() - load_start).total_seconds()
                        self.logger.error(f"Failed to load RL model generation {config.generation} for bot {bot_id} after {load_duration:.3f}s: {e}")
                        self.logger.info(f"Falling back to rules-based bot for bot {bot_id}")
                        
                        # Fallback to rules-based
                        fallback_bot = RulesBasedBot(config.difficulty)
                        return fallback_bot
                
                # Return cached model
                cached_model = self._model_cache[config.generation]
                load_duration = (datetime.now() - load_start).total_seconds()
                self.logger.debug(f"Cached RL model loaded for bot {bot_id} in {load_duration:.3f}s")
                
                return cached_model
            
            else:
                error_msg = f"Unsupported bot type: {config.bot_type}"
                self.logger.error(f"AI loading failed for bot {bot_id}: {error_msg}")
                raise ValueError(error_msg)
                
        except Exception as e:
            load_duration = (datetime.now() - load_start).total_seconds()
            self.logger.error(f"AI loading failed for bot {bot_id} after {load_duration:.3f}s: {e}")
            raise
    
    def _validate_bot_config(self, config: BotConfig, room_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate bot configuration before spawning.
        
        Args:
            config: Bot configuration to validate
            room_info: Room information for validation
            
        Returns:
            Dict with 'valid' boolean and 'error' message if invalid
        """
        try:
            # Validate bot type
            if not isinstance(config.bot_type, BotType):
                return {'valid': False, 'error': f'Invalid bot type: {config.bot_type}'}
            
            # Validate difficulty
            if not isinstance(config.difficulty, DifficultyLevel):
                return {'valid': False, 'error': f'Invalid difficulty level: {config.difficulty}'}
            
            # Validate bot name
            if not config.name or not isinstance(config.name, str) or len(config.name.strip()) == 0:
                return {'valid': False, 'error': 'Bot name is required and must be a non-empty string'}
            
            if len(config.name) > 50:
                return {'valid': False, 'error': 'Bot name must be 50 characters or less'}
            
            # Validate RL generation if applicable
            if config.bot_type == BotType.RL_GENERATION:
                if config.generation is None:
                    return {'valid': False, 'error': 'Generation is required for RL bots'}
                
                if not isinstance(config.generation, int) or config.generation < 0:
                    return {'valid': False, 'error': 'Generation must be a non-negative integer'}
            
            # Validate room information
            room_code = room_info.get('room_code')
            if room_code and (not isinstance(room_code, str) or len(room_code.strip()) == 0):
                return {'valid': False, 'error': 'Room code must be a non-empty string if provided'}
            
            # Validate training mode settings
            if config.training_mode and not isinstance(config.training_mode, bool):
                return {'valid': False, 'error': 'Training mode must be a boolean'}
            
            # Validate auto cleanup settings
            if not isinstance(config.auto_cleanup, bool):
                return {'valid': False, 'error': 'Auto cleanup must be a boolean'}
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': f'Configuration validation error: {str(e)}'}

    async def _run_bot_ai_loop(self, bot_id: str, correlation_id: Optional[str] = None) -> None:
        """Run the main AI decision loop for a bot."""
        if bot_id not in self._bots:
            return
        
        bot_instance = self._bots[bot_id]
        
        try:
            while (bot_instance.status == BotStatus.ACTIVE and 
                   bot_instance.game_client and 
                   bot_instance.bot_ai):
                
                # Get current game state
                if bot_instance.config.training_mode:
                    game_state = await bot_instance.game_client.get_direct_state()
                else:
                    game_state = bot_instance.game_client.game_state
                
                # Track game state updates
                if game_state:
                    if bot_id in self._bots:
                        self._bots[bot_id].performance_stats['total_playtime'] = (
                            datetime.now() - bot_instance.created_at
                        ).total_seconds()
                    
                    # Record game state update for diagnostics
                    if bot_id in self.diagnostic_tracker._activity_metrics:
                        self.diagnostic_tracker._activity_metrics[bot_id].game_state_updates += 1
                
                if game_state and isinstance(bot_instance.bot_ai, RulesBasedBot):
                    # Analyze game state and select action
                    decision_start = datetime.now()
                    analysis = bot_instance.bot_ai.analyze_game_state(game_state)
                    action = bot_instance.bot_ai.select_action(analysis)
                    decision_time = (datetime.now() - decision_start).total_seconds() * 1000  # ms
                    
                    # Record decision for diagnostics
                    self.diagnostic_tracker.record_bot_decision(
                        bot_id=bot_id,
                        decision_details={
                            'decision_time_ms': decision_time,
                            'has_action': action is not None,
                            'analysis_summary': str(analysis) if hasattr(analysis, '__str__') else 'analysis_completed'
                        }
                    )
                    
                    # Execute action
                    if action:
                        await self._execute_bot_action(bot_id, action)
                
                # Update activity timestamp
                bot_instance.last_activity = datetime.now()
                
                # Sleep based on bot difficulty (reaction time)
                if isinstance(bot_instance.bot_ai, RulesBasedBot):
                    sleep_time = bot_instance.bot_ai.config.get('decision_frequency', 0.1)
                else:
                    sleep_time = 0.1
                
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            self.logger.error(f"Error in bot AI loop for {bot_id}: {e}")
            await self._update_bot_status(bot_id, BotStatus.ERROR, str(e))
            
            self.diagnostic_tracker.log_event(
                bot_id=bot_id,
                event_type=BotLifecycleEvent.BOT_ERROR,
                level=DiagnosticLevel.ERROR,
                message=f"Bot AI loop error: {e}",
                details={'error': str(e), 'error_type': type(e).__name__}
            )
    
    async def _execute_bot_action(self, bot_id: str, action: Any) -> None:
        """Execute a bot action through the game client."""
        if bot_id not in self._bots:
            return
        
        bot_instance = self._bots[bot_id]
        if not bot_instance.game_client:
            return
        
        action_success = False
        error_message = None
        action_type = getattr(action, 'action_type', 'unknown')
        
        try:
            params = getattr(action, 'parameters', {})
            
            if action_type in ['move_left', 'move_right', 'jump', 'crouch']:
                # Keyboard actions
                key = params.get('key')
                pressed = params.get('pressed', True)
                if key:
                    await bot_instance.game_client.send_keyboard_input(key, pressed)
                    action_success = True
            
            elif action_type == 'shoot_at_enemy':
                # Mouse actions
                button = params.get('button', 'left')
                pressed = params.get('pressed', True)
                x = params.get('x', 0)
                y = params.get('y', 0)
                await bot_instance.game_client.send_mouse_input(button, pressed, x, y)
                action_success = True
            
            # Hold action for specified duration
            if hasattr(action, 'duration') and action.duration and action.duration > 0:
                await asyncio.sleep(action.duration)
                
                # Release action if it was a press
                if action_type in ['move_left', 'move_right', 'jump', 'crouch']:
                    key = params.get('key')
                    if key:
                        await bot_instance.game_client.send_keyboard_input(key, False)
                elif action_type == 'shoot_at_enemy':
                    button = params.get('button', 'left')
                    x = params.get('x', 0)
                    y = params.get('y', 0)
                    await bot_instance.game_client.send_mouse_input(button, False, x, y)
            
            if not action_success:
                error_message = f"Unknown action type: {action_type}"
            
        except Exception as e:
            self.logger.error(f"Error executing action for bot {bot_id}: {e}")
            action_success = False
            error_message = str(e)
        
        # Record action execution for diagnostics
        self.diagnostic_tracker.record_bot_action(
            bot_id=bot_id,
            action_type=action_type,
            success=action_success,
            action_details={
                'parameters': getattr(action, 'parameters', {}),
                'duration': getattr(action, 'duration', None)
            },
            error_message=error_message
        )
    

    
    async def _apply_realtime_difficulty_changes(self, bot_id: str, old_difficulty: DifficultyLevel, new_difficulty: DifficultyLevel) -> None:
        """
        Apply real-time difficulty changes to a bot's behavior.
        
        Args:
            bot_id: ID of the bot
            old_difficulty: Previous difficulty level
            new_difficulty: New difficulty level
        """
        if bot_id not in self._bots:
            return
        
        bot_instance = self._bots[bot_id]
        
        try:
            # Update reaction times and decision frequencies
            if isinstance(bot_instance.bot_ai, RulesBasedBot):
                # Adjust decision frequency based on difficulty
                decision_frequencies = {
                    DifficultyLevel.BEGINNER: 0.3,
                    DifficultyLevel.INTERMEDIATE: 0.2,
                    DifficultyLevel.ADVANCED: 0.15,
                    DifficultyLevel.EXPERT: 0.1
                }
                
                new_frequency = decision_frequencies.get(new_difficulty, 0.2)
                bot_instance.bot_ai.config['decision_frequency'] = new_frequency
                
                # Adjust accuracy and reaction time modifiers
                accuracy_modifiers = {
                    DifficultyLevel.BEGINNER: 0.6,
                    DifficultyLevel.INTERMEDIATE: 0.75,
                    DifficultyLevel.ADVANCED: 0.9,
                    DifficultyLevel.EXPERT: 0.95
                }
                
                new_accuracy = accuracy_modifiers.get(new_difficulty, 0.75)
                bot_instance.bot_ai.config['accuracy_modifier'] = new_accuracy
                
                self.logger.debug(f"Applied real-time difficulty changes to bot {bot_id}: "
                                f"frequency={new_frequency}, accuracy={new_accuracy}")
        
        except Exception as e:
            self.logger.error(f"Error applying real-time difficulty changes to bot {bot_id}: {e}")
    
    async def _check_and_cleanup_empty_rooms(self) -> None:
        """
        Check for rooms that have no human players and clean up bots if configured to do so.
        """
        try:
            rooms_to_check = list(self._room_bots.keys())
            
            for room_id in rooms_to_check:
                # Check if room has human players
                has_human_players = await self.check_room_human_players(room_id)
                
                if not has_human_players:
                    # Get bots with auto_cleanup enabled
                    bots_to_cleanup = []
                    
                    for bot_id in self._room_bots.get(room_id, set()):
                        if bot_id in self._bots:
                            bot_instance = self._bots[bot_id]
                            if bot_instance.config.auto_cleanup:
                                bots_to_cleanup.append(bot_id)
                    
                    if bots_to_cleanup:
                        self.logger.info(f"Room {room_id} has no human players, cleaning up {len(bots_to_cleanup)} bots")
                        
                        for bot_id in bots_to_cleanup:
                            await self.terminate_bot(bot_id)
                        
                        # Notify callbacks about room cleanup
                        for callback in self._room_empty_callbacks:
                            try:
                                await callback(room_id)
                            except Exception as e:
                                self.logger.error(f"Error in room empty callback for {room_id}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error checking and cleaning up empty rooms: {e}")
    
    async def _periodic_cleanup(self) -> None:
        """Background task for periodic cleanup of inactive bots."""
        while self._running:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.config.bot_timeout_seconds)
                
                # Find inactive bots
                inactive_bots = []
                unhealthy_bots = []
                
                for bot_id, bot_instance in self._bots.items():
                    # Check for timeout
                    if (bot_instance.last_activity < timeout_threshold and 
                        bot_instance.status in [BotStatus.ACTIVE, BotStatus.ERROR]):
                        inactive_bots.append(bot_id)
                    
                    # Check bot health
                    health_status = await self.monitor_bot_health(bot_id)
                    if not health_status["healthy"] and bot_instance.status == BotStatus.ACTIVE:
                        unhealthy_bots.append((bot_id, health_status))
                
                # Attempt reconnection for unhealthy bots before terminating
                for bot_id, health_status in unhealthy_bots:
                    if "connection_lost" in health_status.get("performance_issues", []):
                        self.logger.info(f"Attempting reconnection for unhealthy bot {bot_id}")
                        
                        # Try reconnection up to configured attempts
                        bot_instance = self._bots[bot_id]
                        reconnect_attempts = getattr(bot_instance, '_reconnect_attempts', 0)
                        
                        if reconnect_attempts < self.config.bot_reconnect_attempts:
                            bot_instance._reconnect_attempts = reconnect_attempts + 1
                            
                            # Wait before reconnection attempt
                            await asyncio.sleep(self.config.bot_reconnect_delay)
                            
                            success = await self.attempt_bot_reconnection(bot_id)
                            if success:
                                bot_instance._reconnect_attempts = 0  # Reset on success
                            else:
                                self.logger.warning(f"Reconnection attempt {reconnect_attempts + 1} failed for bot {bot_id}")
                        else:
                            self.logger.error(f"Bot {bot_id} exceeded max reconnection attempts, terminating")
                            inactive_bots.append(bot_id)
                
                # Terminate inactive bots
                for bot_id in inactive_bots:
                    self.logger.info(f"Terminating inactive bot {bot_id}")
                    await self.terminate_bot(bot_id)
                
                # Auto-cleanup: Check for empty rooms if enabled
                if self.config.auto_cleanup_empty_rooms:
                    await self._check_and_cleanup_empty_rooms()
                
                # Sleep until next cleanup
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


# Convenience functions

def create_bot_server(config: Optional[BotServerConfig] = None) -> BotServer:
    """
    Create a BotServer instance.
    
    Args:
        config: Optional server configuration
        
    Returns:
        BotServer instance
    """
    if config is None:
        config = BotServerConfig()
    
    return BotServer(config)


async def run_bot_server(config: Optional[BotServerConfig] = None) -> None:
    """
    Create and run a BotServer.
    
    Args:
        config: Optional server configuration
    """
    server = create_bot_server(config)
    
    try:
        await server.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    # Run server with default configuration
    asyncio.run(run_bot_server())