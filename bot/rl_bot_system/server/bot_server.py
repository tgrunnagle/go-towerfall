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

from game_client import GameClient, TrainingMode
from rl_bot_system.rules_based.rules_based_bot import RulesBasedBot, DifficultyLevel
from rl_bot_system.training.model_manager import ModelManager


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
    models_dir: str = "bot/data/models"
    enable_performance_tracking: bool = True
    auto_cleanup_empty_rooms: bool = True
    bot_reconnect_attempts: int = 3
    bot_reconnect_delay: float = 5.0


class GameClientPool:
    """Pool manager for GameClient connections."""
    
    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self._available_clients: List[GameClient] = []
        self._active_clients: Dict[str, GameClient] = {}  # bot_id -> client
        self._client_creation_lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def get_client(self, bot_id: str, game_server_url: str) -> GameClient:
        """
        Get a GameClient for a bot, either from pool or create new one.
        
        Args:
            bot_id: ID of the bot requesting the client
            game_server_url: URL of the game server
            
        Returns:
            GameClient instance
            
        Raises:
            RuntimeError: If max connections exceeded
        """
        async with self._client_creation_lock:
            if len(self._active_clients) >= self.max_connections:
                raise RuntimeError(f"Maximum connections ({self.max_connections}) exceeded")
            
            # Try to reuse an available client
            if self._available_clients:
                client = self._available_clients.pop()
                self.logger.debug(f"Reusing pooled client for bot {bot_id}")
            else:
                # Create new client
                ws_url = game_server_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
                client = GameClient(ws_url=ws_url, http_url=game_server_url)
                self.logger.debug(f"Created new client for bot {bot_id}")
            
            self._active_clients[bot_id] = client
            return client
    
    async def return_client(self, bot_id: str) -> None:
        """
        Return a GameClient to the pool for reuse.
        
        Args:
            bot_id: ID of the bot returning the client
        """
        if bot_id in self._active_clients:
            client = self._active_clients.pop(bot_id)
            
            try:
                # Clean up client state
                await client.close()
                
                # Reset client for reuse
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
                
                # Return to pool if not at capacity
                if len(self._available_clients) < self.max_connections // 2:
                    self._available_clients.append(client)
                    self.logger.debug(f"Returned client to pool from bot {bot_id}")
                else:
                    self.logger.debug(f"Discarded client from bot {bot_id} (pool full)")
                    
            except Exception as e:
                self.logger.error(f"Error returning client from bot {bot_id}: {e}")
    
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
        async with self._bot_creation_lock:
            # Check limits
            if len(self._bots) >= self.config.max_total_bots:
                raise RuntimeError(f"Maximum total bots ({self.config.max_total_bots}) exceeded")
            
            room_id = room_info.get('room_id')
            if room_id:
                room_bot_count = len(self._room_bots.get(room_id, set()))
                if room_bot_count >= self.config.max_bots_per_room:
                    raise RuntimeError(f"Maximum bots per room ({self.config.max_bots_per_room}) exceeded")
            
            # Generate bot ID
            bot_id = str(uuid.uuid4())
            
            # Create bot instance
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
            
            # Track room assignment
            if room_id:
                if room_id not in self._room_bots:
                    self._room_bots[room_id] = set()
                self._room_bots[room_id].add(bot_id)
            
            self.logger.info(f"Created bot {bot_id} ({bot_config.bot_type.value}) for room {room_id}")
            
            # Initialize bot in background
            asyncio.create_task(self._initialize_bot(bot_id, room_info))
            
            return bot_id
    
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
        
        return {
            'running': self._running,
            'total_bots': len(self._bots),
            'max_total_bots': self.config.max_total_bots,
            'active_rooms': len(self._room_bots),
            'bots_by_status': status_counts,
            'bots_by_type': type_counts,
            'client_pool_stats': self.client_pool.get_pool_stats(),
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
        
        bot_instance = self._bots[bot_id]
        current_time = datetime.now()
        
        # Calculate time since last activity
        time_since_activity = (current_time - bot_instance.last_activity).total_seconds()
        
        # Check various health indicators
        health_status = {
            "bot_id": bot_id,
            "status": bot_instance.status.value,
            "healthy": True,
            "last_activity_seconds": time_since_activity,
            "connection_status": "unknown",
            "ai_status": "unknown",
            "performance_issues": []
        }
        
        # Check if bot is responsive
        if time_since_activity > self.config.bot_timeout_seconds:
            health_status["healthy"] = False
            health_status["performance_issues"].append("inactive_timeout")
        
        # Check connection status
        if bot_instance.game_client:
            try:
                # Check if WebSocket connection is still alive
                if hasattr(bot_instance.game_client, 'websocket') and bot_instance.game_client.websocket:
                    if bot_instance.game_client.websocket.client_state.name == 'CONNECTED':
                        health_status["connection_status"] = "connected"
                    else:
                        health_status["connection_status"] = "disconnected"
                        health_status["healthy"] = False
                        health_status["performance_issues"].append("connection_lost")
                else:
                    health_status["connection_status"] = "no_websocket"
            except Exception as e:
                health_status["connection_status"] = f"error: {e}"
                health_status["performance_issues"].append("connection_check_failed")
        else:
            health_status["connection_status"] = "no_client"
            health_status["healthy"] = False
            health_status["performance_issues"].append("no_game_client")
        
        # Check AI status
        if bot_instance.bot_ai:
            health_status["ai_status"] = "loaded"
        else:
            health_status["ai_status"] = "not_loaded"
            health_status["healthy"] = False
            health_status["performance_issues"].append("no_ai_loaded")
        
        # Check for error status
        if bot_instance.status == BotStatus.ERROR:
            health_status["healthy"] = False
            health_status["performance_issues"].append("bot_error_status")
            if bot_instance.error_message:
                health_status["error_message"] = bot_instance.error_message
        
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
    
    async def _initialize_bot(self, bot_id: str, room_info: Dict[str, Any]) -> None:
        """Initialize a bot instance in the background."""
        if bot_id not in self._bots:
            return
        
        bot_instance = self._bots[bot_id]
        
        try:
            # Load bot AI
            await self._update_bot_status(bot_id, BotStatus.INITIALIZING)
            bot_instance.bot_ai = await self._load_bot_ai(bot_instance.config)
            
            # Get game client
            await self._update_bot_status(bot_id, BotStatus.CONNECTING)
            bot_instance.game_client = await self.client_pool.get_client(
                bot_id, self.config.game_server_url
            )
            
            # Configure training mode if needed
            if bot_instance.config.training_mode:
                await bot_instance.game_client.enable_training_mode(
                    speed_multiplier=10.0,
                    headless=True
                )
            
            # Connect to game room
            room_code = room_info.get('room_code')
            room_password = room_info.get('room_password', '')
            
            if room_code:
                await bot_instance.game_client.connect(
                    room_code=room_code,
                    player_name=bot_instance.config.name,
                    room_password=room_password
                )
            
            # Start bot AI loop
            asyncio.create_task(self._run_bot_ai_loop(bot_id))
            
            await self._update_bot_status(bot_id, BotStatus.ACTIVE)
            bot_instance.last_activity = datetime.now()
            
            self.logger.info(f"Bot {bot_id} initialized and connected successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bot {bot_id}: {e}")
            bot_instance.status = BotStatus.ERROR
            bot_instance.error_message = str(e)
            
            # Clean up on failure
            if bot_instance.game_client:
                await self.client_pool.return_client(bot_id)
    
    async def _load_bot_ai(self, config: BotConfig) -> Any:
        """Load the appropriate AI for a bot configuration."""
        if config.bot_type == BotType.RULES_BASED:
            return RulesBasedBot(config.difficulty)
        
        elif config.bot_type == BotType.RL_GENERATION:
            if config.generation is None:
                raise ValueError("Generation required for RL bot")
            
            # Load from cache or model manager
            if config.generation not in self._model_cache:
                try:
                    # This would need to be implemented based on the actual RL model structure
                    # For now, return a placeholder
                    self.logger.warning(f"RL model loading not fully implemented for generation {config.generation}")
                    # Fallback to rules-based for now
                    return RulesBasedBot(config.difficulty)
                except Exception as e:
                    self.logger.error(f"Failed to load RL model generation {config.generation}: {e}")
                    # Fallback to rules-based
                    return RulesBasedBot(config.difficulty)
            
            return self._model_cache[config.generation]
        
        else:
            raise ValueError(f"Unsupported bot type: {config.bot_type}")
    
    async def _run_bot_ai_loop(self, bot_id: str) -> None:
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
                
                if game_state and isinstance(bot_instance.bot_ai, RulesBasedBot):
                    # Analyze game state and select action
                    analysis = bot_instance.bot_ai.analyze_game_state(game_state)
                    action = bot_instance.bot_ai.select_action(analysis)
                    
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
            bot_instance.status = BotStatus.ERROR
            bot_instance.error_message = str(e)
    
    async def _execute_bot_action(self, bot_id: str, action: Any) -> None:
        """Execute a bot action through the game client."""
        if bot_id not in self._bots:
            return
        
        bot_instance = self._bots[bot_id]
        if not bot_instance.game_client:
            return
        
        try:
            action_type = action.action_type
            params = action.parameters
            
            if action_type in ['move_left', 'move_right', 'jump', 'crouch']:
                # Keyboard actions
                key = params.get('key')
                pressed = params.get('pressed', True)
                if key:
                    await bot_instance.game_client.send_keyboard_input(key, pressed)
            
            elif action_type == 'shoot_at_enemy':
                # Mouse actions
                button = params.get('button', 'left')
                pressed = params.get('pressed', True)
                x = params.get('x', 0)
                y = params.get('y', 0)
                await bot_instance.game_client.send_mouse_input(button, pressed, x, y)
            
            # Hold action for specified duration
            if action.duration and action.duration > 0:
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
            
        except Exception as e:
            self.logger.error(f"Error executing action for bot {bot_id}: {e}")
    
    async def _update_bot_status(self, bot_id: str, status: BotStatus) -> None:
        """Update bot status and notify callbacks."""
        if bot_id not in self._bots:
            return
        
        old_status = self._bots[bot_id].status
        self._bots[bot_id].status = status
        
        if old_status != status:
            # Notify callbacks
            for callback in self._bot_status_callbacks:
                try:
                    await callback(bot_id, status)
                except Exception as e:
                    self.logger.error(f"Error in bot status callback: {e}")
    
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