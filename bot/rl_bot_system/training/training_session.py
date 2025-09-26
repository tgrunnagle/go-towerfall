"""
Training session management for accelerated RL bot training.

This module provides the TrainingSession class and related components for managing
accelerated game instances, headless mode configuration, and batch episode management.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor
import aiohttp

from core.game_client import GameClient


class TrainingMode(Enum):
    """Training mode configurations for different speed requirements."""
    REALTIME = "realtime"  # 1x speed for human evaluation
    TRAINING = "training"  # 10-50x speed with visual feedback
    HEADLESS = "headless"  # 50-100x speed with no rendering
    BATCH = "batch"  # Parallel episode execution


class SessionStatus(Enum):
    """Training session status states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Configuration for training sessions."""
    speed_multiplier: float = 1.0
    headless_mode: bool = False
    max_episodes: int = 1000
    episode_timeout: int = 300  # seconds
    parallel_episodes: int = 1
    training_mode: TrainingMode = TrainingMode.REALTIME
    room_password: Optional[str] = None
    spectator_enabled: bool = False
    auto_cleanup: bool = True


@dataclass
class EpisodeResult:
    """Result data from a completed training episode."""
    episode_id: str
    start_time: datetime
    end_time: datetime
    duration: float
    total_reward: float
    episode_length: int
    game_result: str  # 'win', 'loss', 'draw', 'timeout'
    final_state: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class SessionMetrics:
    """Real-time metrics for training sessions."""
    episodes_completed: int = 0
    episodes_failed: int = 0
    total_reward: float = 0.0
    average_reward: float = 0.0
    best_reward: float = float('-inf')
    worst_reward: float = float('inf')
    average_episode_length: float = 0.0
    success_rate: float = 0.0
    episodes_per_minute: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class TrainingSession:
    """
    Manages accelerated game instances for RL bot training.
    
    Provides functionality for:
    - Creating training rooms with speed multipliers
    - Managing headless mode for maximum training speed
    - Batch episode management for parallel training
    - Real-time metrics and progress tracking
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        config: Optional[TrainingConfig] = None,
        game_server_url: str = "http://localhost:4000",
        ws_url: str = "ws://localhost:4000/ws"
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.config = config or TrainingConfig()
        self.game_server_url = game_server_url
        self.ws_url = ws_url
        
        self.status = SessionStatus.INITIALIZING
        self.metrics = SessionMetrics()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Training room management
        self.room_code: Optional[str] = None
        self.room_id: Optional[str] = None
        self.spectator_room_code: Optional[str] = None
        
        # Episode management
        self.active_episodes: Dict[str, asyncio.Task] = {}
        self.completed_episodes: List[EpisodeResult] = []
        self.episode_queue: asyncio.Queue = asyncio.Queue()
        
        # Event handlers
        self.episode_complete_handlers: List[Callable[[EpisodeResult], Awaitable[None]]] = []
        self.metrics_update_handlers: List[Callable[[SessionMetrics], Awaitable[None]]] = []
        
        # Resource management
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_episodes)
        self._shutdown_event = asyncio.Event()
        
        self._logger = logging.getLogger(f"{__name__}.{self.session_id}")

    async def initialize(self) -> None:
        """Initialize the training session and create training room."""
        try:
            self._logger.info(f"Initializing training session {self.session_id}")
            self.start_time = datetime.now()
            
            # Create training room with appropriate configuration
            await self._create_training_room()
            
            # Set up spectator room if enabled
            if self.config.spectator_enabled:
                await self._create_spectator_room()
            
            self.status = SessionStatus.RUNNING
            self._logger.info(f"Training session {self.session_id} initialized successfully")
            
        except Exception as e:
            self.status = SessionStatus.FAILED
            self._logger.error(f"Failed to initialize training session: {e}")
            raise

    async def start_training(self) -> None:
        """Start the training session with batch episode management."""
        if self.status != SessionStatus.RUNNING:
            raise RuntimeError(f"Cannot start training in status: {self.status}")
        
        try:
            self._logger.info(f"Starting training with {self.config.max_episodes} episodes")
            
            # Start episode management tasks
            episode_manager_task = asyncio.create_task(self._manage_episodes())
            metrics_updater_task = asyncio.create_task(self._update_metrics_loop())
            
            # Wait for completion or shutdown
            await asyncio.gather(
                episode_manager_task,
                metrics_updater_task,
                return_exceptions=True
            )
            
        except Exception as e:
            self.status = SessionStatus.FAILED
            self._logger.error(f"Training session failed: {e}")
            raise
        finally:
            await self._cleanup()

    async def pause(self) -> None:
        """Pause the training session."""
        if self.status == SessionStatus.RUNNING:
            self.status = SessionStatus.PAUSED
            self._logger.info("Training session paused")

    async def resume(self) -> None:
        """Resume a paused training session."""
        if self.status == SessionStatus.PAUSED:
            self.status = SessionStatus.RUNNING
            self._logger.info("Training session resumed")

    async def stop(self) -> None:
        """Stop the training session gracefully."""
        self._logger.info("Stopping training session")
        self._shutdown_event.set()
        
        # Cancel active episodes
        for episode_id, task in self.active_episodes.items():
            if not task.done():
                task.cancel()
                self._logger.debug(f"Cancelled episode {episode_id}")
        
        # Wait for episodes to complete
        if self.active_episodes:
            await asyncio.gather(*self.active_episodes.values(), return_exceptions=True)
        
        self.status = SessionStatus.COMPLETED
        self.end_time = datetime.now()

    async def request_training_room(
        self,
        speed_multiplier: float,
        headless: bool = False
    ) -> Dict[str, Any]:
        """
        Request a training room with specified speed multiplier.
        
        Args:
            speed_multiplier: Game speed multiplier (1.0 to 100.0)
            headless: Whether to run in headless mode
            
        Returns:
            Room information including room code and configuration
        """
        try:
            room_config = {
                "roomType": "training",
                "speedMultiplier": speed_multiplier,
                "headlessMode": headless,
                "maxPlayers": self.config.parallel_episodes + 1,  # +1 for training bot
                "password": self.config.room_password,
                "spectatorEnabled": self.config.spectator_enabled
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.game_server_url}/api/createTrainingRoom",
                    json=room_config
                ) as response:
                    response.raise_for_status()
                    room_data = await response.json()
            
            self.room_code = room_data["roomCode"]
            self.room_id = room_data["roomId"]
            
            self._logger.info(
                f"Created training room {self.room_code} with {speed_multiplier}x speed"
            )
            
            return room_data
            
        except Exception as e:
            self._logger.error(f"Failed to create training room: {e}")
            raise

    def register_episode_handler(
        self,
        handler: Callable[[EpisodeResult], Awaitable[None]]
    ) -> None:
        """Register a handler for episode completion events."""
        self.episode_complete_handlers.append(handler)

    def register_metrics_handler(
        self,
        handler: Callable[[SessionMetrics], Awaitable[None]]
    ) -> None:
        """Register a handler for metrics update events."""
        self.metrics_update_handlers.append(handler)

    async def get_session_info(self) -> Dict[str, Any]:
        """Get current session information and metrics."""
        return {
            "sessionId": self.session_id,
            "status": self.status.value,
            "config": {
                "speedMultiplier": self.config.speed_multiplier,
                "headlessMode": self.config.headless_mode,
                "maxEpisodes": self.config.max_episodes,
                "parallelEpisodes": self.config.parallel_episodes,
                "trainingMode": self.config.training_mode.value
            },
            "roomInfo": {
                "roomCode": self.room_code,
                "roomId": self.room_id,
                "spectatorRoomCode": self.spectator_room_code
            },
            "metrics": {
                "episodesCompleted": self.metrics.episodes_completed,
                "episodesFailed": self.metrics.episodes_failed,
                "totalReward": self.metrics.total_reward,
                "averageReward": self.metrics.average_reward,
                "bestReward": self.metrics.best_reward,
                "successRate": self.metrics.success_rate,
                "episodesPerMinute": self.metrics.episodes_per_minute
            },
            "timing": {
                "startTime": self.start_time.isoformat() if self.start_time else None,
                "endTime": self.end_time.isoformat() if self.end_time else None,
                "duration": (
                    (self.end_time or datetime.now()) - self.start_time
                ).total_seconds() if self.start_time else 0
            }
        }

    async def _create_training_room(self) -> None:
        """Create a training room with the configured settings."""
        room_data = await self.request_training_room(
            self.config.speed_multiplier,
            self.config.headless_mode
        )
        
        # Store additional room configuration
        self._logger.debug(f"Training room created: {room_data}")

    async def _create_spectator_room(self) -> None:
        """Create a spectator room for training observation."""
        try:
            spectator_config = {
                "trainingSessionId": self.session_id,
                "parentRoomId": self.room_id,
                "spectatorMode": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.game_server_url}/api/createSpectatorRoom",
                    json=spectator_config
                ) as response:
                    response.raise_for_status()
                    spectator_data = await response.json()
            
            self.spectator_room_code = spectator_data["roomCode"]
            self._logger.info(f"Created spectator room: {self.spectator_room_code}")
            
        except Exception as e:
            self._logger.warning(f"Failed to create spectator room: {e}")

    async def _manage_episodes(self) -> None:
        """Manage the execution of training episodes."""
        episode_count = 0
        
        while (
            episode_count < self.config.max_episodes and
            not self._shutdown_event.is_set() and
            self.status == SessionStatus.RUNNING
        ):
            # Maintain parallel episodes up to the configured limit
            while (
                len(self.active_episodes) < self.config.parallel_episodes and
                episode_count < self.config.max_episodes and
                not self._shutdown_event.is_set()
            ):
                episode_id = f"episode_{episode_count:06d}"
                episode_task = asyncio.create_task(
                    self._run_episode(episode_id)
                )
                self.active_episodes[episode_id] = episode_task
                episode_count += 1
                
                self._logger.debug(f"Started episode {episode_id}")
            
            # Wait for at least one episode to complete
            if self.active_episodes:
                done, pending = await asyncio.wait(
                    self.active_episodes.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0
                )
                
                # Process completed episodes
                for task in done:
                    episode_id = None
                    for eid, etask in self.active_episodes.items():
                        if etask == task:
                            episode_id = eid
                            break
                    
                    if episode_id:
                        del self.active_episodes[episode_id]
                        
                        try:
                            result = await task
                            await self._handle_episode_completion(result)
                        except Exception as e:
                            self._logger.error(f"Episode {episode_id} failed: {e}")
                            await self._handle_episode_failure(episode_id, str(e))
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
        
        # Wait for remaining episodes to complete
        if self.active_episodes:
            await asyncio.gather(*self.active_episodes.values(), return_exceptions=True)
        
        self.status = SessionStatus.COMPLETED
        self.end_time = datetime.now()

    async def _run_episode(self, episode_id: str) -> EpisodeResult:
        """Run a single training episode."""
        start_time = datetime.now()
        
        try:
            # Create game client for this episode
            game_client = GameClient(self.ws_url, self.game_server_url)
            
            # Connect to training room
            await game_client.connect(
                self.room_code,
                f"training_bot_{episode_id}",
                self.config.room_password
            )
            
            # Run episode logic (placeholder - will be implemented by training engine)
            episode_result = await self._simulate_episode(game_client, episode_id)
            
            # Clean up
            await game_client.exit_game()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return EpisodeResult(
                episode_id=episode_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                total_reward=episode_result.get("reward", 0.0),
                episode_length=episode_result.get("length", 0),
                game_result=episode_result.get("result", "unknown"),
                final_state=episode_result.get("final_state", {}),
                error=None
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return EpisodeResult(
                episode_id=episode_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                total_reward=0.0,
                episode_length=0,
                game_result="error",
                final_state={},
                error=str(e)
            )

    async def _simulate_episode(
        self,
        game_client: GameClient,
        episode_id: str
    ) -> Dict[str, Any]:
        """
        Simulate a training episode.
        
        This is a placeholder implementation that will be replaced by the actual
        training engine integration.
        """
        # Placeholder episode simulation
        await asyncio.sleep(0.1)  # Simulate episode duration
        
        return {
            "reward": 10.0,  # Placeholder reward
            "length": 100,   # Placeholder episode length
            "result": "win", # Placeholder result
            "final_state": {"health": 100, "score": 150}
        }

    async def _handle_episode_completion(self, result: EpisodeResult) -> None:
        """Handle the completion of a training episode."""
        self.completed_episodes.append(result)
        
        # Update metrics
        if result.error is None:
            self.metrics.episodes_completed += 1
            self.metrics.total_reward += result.total_reward
            self.metrics.average_reward = (
                self.metrics.total_reward / self.metrics.episodes_completed
            )
            self.metrics.best_reward = max(self.metrics.best_reward, result.total_reward)
            self.metrics.worst_reward = min(self.metrics.worst_reward, result.total_reward)
        else:
            self.metrics.episodes_failed += 1
        
        # Calculate success rate
        total_episodes = self.metrics.episodes_completed + self.metrics.episodes_failed
        if total_episodes > 0:
            self.metrics.success_rate = self.metrics.episodes_completed / total_episodes
        
        # Notify handlers
        for handler in self.episode_complete_handlers:
            try:
                await handler(result)
            except Exception as e:
                self._logger.error(f"Error in episode completion handler: {e}")

    async def _handle_episode_failure(self, episode_id: str, error: str) -> None:
        """Handle episode failure."""
        self.metrics.episodes_failed += 1
        self._logger.warning(f"Episode {episode_id} failed: {error}")

    async def _update_metrics_loop(self) -> None:
        """Continuously update and broadcast training metrics."""
        last_episode_count = 0
        last_update_time = time.time()
        
        while not self._shutdown_event.is_set() and self.status == SessionStatus.RUNNING:
            current_time = time.time()
            time_delta = current_time - last_update_time
            
            if time_delta >= 1.0:  # Update every second
                # Calculate episodes per minute
                episode_delta = self.metrics.episodes_completed - last_episode_count
                if time_delta > 0:
                    self.metrics.episodes_per_minute = (episode_delta / time_delta) * 60
                
                # Update average episode length
                if self.completed_episodes:
                    total_length = sum(ep.episode_length for ep in self.completed_episodes)
                    self.metrics.average_episode_length = total_length / len(self.completed_episodes)
                
                self.metrics.last_updated = datetime.now()
                
                # Notify metrics handlers
                for handler in self.metrics_update_handlers:
                    try:
                        await handler(self.metrics)
                    except Exception as e:
                        self._logger.error(f"Error in metrics handler: {e}")
                
                last_episode_count = self.metrics.episodes_completed
                last_update_time = current_time
            
            await asyncio.sleep(0.5)

    async def _cleanup(self) -> None:
        """Clean up resources when session ends."""
        try:
            if self.config.auto_cleanup:
                # Clean up training room
                if self.room_id:
                    await self._cleanup_training_room()
                
                # Clean up spectator room
                if self.spectator_room_code:
                    await self._cleanup_spectator_room()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self._logger.info(f"Training session {self.session_id} cleaned up")
            
        except Exception as e:
            self._logger.error(f"Error during cleanup: {e}")

    async def _cleanup_training_room(self) -> None:
        """Clean up the training room."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.game_server_url}/api/room/{self.room_id}"
                ) as response:
                    if response.status == 200:
                        self._logger.info(f"Training room {self.room_code} cleaned up")
                    else:
                        self._logger.warning(f"Failed to cleanup room: {response.status}")
        except Exception as e:
            self._logger.warning(f"Error cleaning up training room: {e}")

    async def _cleanup_spectator_room(self) -> None:
        """Clean up the spectator room."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.game_server_url}/api/spectatorRoom/{self.spectator_room_code}"
                ) as response:
                    if response.status == 200:
                        self._logger.info(f"Spectator room {self.spectator_room_code} cleaned up")
        except Exception as e:
            self._logger.warning(f"Error cleaning up spectator room: {e}")