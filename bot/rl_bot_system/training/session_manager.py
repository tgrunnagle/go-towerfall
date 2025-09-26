"""
Training session manager for coordinating multiple training sessions.

This module provides the SessionManager class for managing multiple concurrent
training sessions, resource allocation, and session lifecycle management.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

from rl_bot_system.training.training_session import (
    TrainingSession,
    TrainingConfig,
    SessionStatus,
    TrainingMode,
    SessionMetrics,
    EpisodeResult
)


class ResourceStatus(Enum):
    """System resource status levels."""
    AVAILABLE = "available"
    LIMITED = "limited"
    EXHAUSTED = "exhausted"


@dataclass
class ResourceLimits:
    """System resource limits configuration."""
    max_concurrent_sessions: int = 5
    max_parallel_episodes_per_session: int = 4
    max_total_parallel_episodes: int = 16
    memory_limit_mb: int = 8192
    cpu_limit_percent: int = 80


@dataclass
class SessionRequest:
    """Request for creating a new training session."""
    session_id: Optional[str] = None
    config: Optional[TrainingConfig] = None
    priority: int = 0  # Higher numbers = higher priority
    requested_at: datetime = None
    
    def __post_init__(self):
        if self.requested_at is None:
            self.requested_at = datetime.now()


class SessionManager:
    """
    Manages multiple training sessions with resource allocation and queuing.
    
    Provides functionality for:
    - Managing multiple concurrent training sessions
    - Resource allocation and monitoring
    - Session queuing when resources are limited
    - Global metrics and monitoring across all sessions
    """

    def __init__(
        self,
        resource_limits: Optional[ResourceLimits] = None,
        game_server_url: str = "http://localhost:4000",
        ws_url: str = "ws://localhost:4000/ws"
    ):
        self.resource_limits = resource_limits or ResourceLimits()
        self.game_server_url = game_server_url
        self.ws_url = ws_url
        
        # Session management
        self.active_sessions: Dict[str, TrainingSession] = {}
        self.session_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.completed_sessions: List[str] = []
        
        # Resource monitoring
        self.current_parallel_episodes = 0
        self.resource_status = ResourceStatus.AVAILABLE
        
        # Event handlers
        self.session_start_handlers: List[Callable[[str], Awaitable[None]]] = []
        self.session_complete_handlers: List[Callable[[str, SessionMetrics], Awaitable[None]]] = []
        self.resource_alert_handlers: List[Callable[[ResourceStatus], Awaitable[None]]] = []
        
        # Background tasks
        self._session_processor_task: Optional[asyncio.Task] = None
        self._resource_monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self._logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """Start the session manager and background tasks."""
        self._logger.info("Starting session manager")
        
        # Start background tasks
        self._session_processor_task = asyncio.create_task(self._process_session_queue())
        self._resource_monitor_task = asyncio.create_task(self._monitor_resources())
        
        self._logger.info("Session manager started")

    async def stop(self) -> None:
        """Stop the session manager and clean up all sessions."""
        self._logger.info("Stopping session manager")
        self._shutdown_event.set()
        
        # Stop all active sessions
        stop_tasks = []
        for session in self.active_sessions.values():
            stop_tasks.append(session.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Cancel background tasks
        if self._session_processor_task:
            self._session_processor_task.cancel()
        if self._resource_monitor_task:
            self._resource_monitor_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self._session_processor_task, self._resource_monitor_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self._logger.info("Session manager stopped")

    async def create_session(
        self,
        config: Optional[TrainingConfig] = None,
        priority: int = 0,
        session_id: Optional[str] = None
    ) -> str:
        """
        Create a new training session.
        
        Args:
            config: Training configuration
            priority: Session priority (higher = more important)
            session_id: Optional custom session ID
            
        Returns:
            Session ID of the created session
        """
        request = SessionRequest(
            session_id=session_id,
            config=config or TrainingConfig(),
            priority=priority
        )
        
        # Add to queue (negative priority for max-heap behavior)
        await self.session_queue.put((-priority, request.requested_at, request))
        
        session_id = request.session_id or f"session_{len(self.active_sessions) + len(self.completed_sessions)}"
        self._logger.info(f"Queued training session {session_id} with priority {priority}")
        
        return session_id

    async def get_session(self, session_id: str) -> Optional[TrainingSession]:
        """Get an active training session by ID."""
        return self.active_sessions.get(session_id)

    async def pause_session(self, session_id: str) -> bool:
        """Pause a training session."""
        session = self.active_sessions.get(session_id)
        if session:
            await session.pause()
            self._logger.info(f"Paused session {session_id}")
            return True
        return False

    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused training session."""
        session = self.active_sessions.get(session_id)
        if session:
            await session.resume()
            self._logger.info(f"Resumed session {session_id}")
            return True
        return False

    async def stop_session(self, session_id: str) -> bool:
        """Stop a training session."""
        session = self.active_sessions.get(session_id)
        if session:
            await session.stop()
            self._logger.info(f"Stopped session {session_id}")
            return True
        return False

    async def get_all_sessions_info(self) -> Dict[str, Any]:
        """Get information about all sessions."""
        active_info = {}
        for session_id, session in self.active_sessions.items():
            active_info[session_id] = await session.get_session_info()
        
        return {
            "activeSessions": active_info,
            "completedSessions": self.completed_sessions,
            "queuedSessions": self.session_queue.qsize(),
            "resourceStatus": {
                "status": self.resource_status.value,
                "currentParallelEpisodes": self.current_parallel_episodes,
                "maxParallelEpisodes": self.resource_limits.max_total_parallel_episodes,
                "activeSessions": len(self.active_sessions),
                "maxConcurrentSessions": self.resource_limits.max_concurrent_sessions
            }
        }

    async def get_global_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all sessions."""
        total_episodes = 0
        total_rewards = 0.0
        total_failures = 0
        session_count = len(self.active_sessions)
        
        for session in self.active_sessions.values():
            metrics = session.metrics
            total_episodes += metrics.episodes_completed
            total_rewards += metrics.total_reward
            total_failures += metrics.episodes_failed
        
        return {
            "totalSessions": session_count,
            "totalEpisodes": total_episodes,
            "totalRewards": total_rewards,
            "totalFailures": total_failures,
            "averageRewardPerEpisode": total_rewards / max(total_episodes, 1),
            "globalSuccessRate": total_episodes / max(total_episodes + total_failures, 1),
            "resourceUtilization": {
                "parallelEpisodes": self.current_parallel_episodes,
                "maxParallelEpisodes": self.resource_limits.max_total_parallel_episodes,
                "utilizationPercent": (
                    self.current_parallel_episodes / 
                    max(self.resource_limits.max_total_parallel_episodes, 1)
                ) * 100
            }
        }

    def register_session_start_handler(
        self,
        handler: Callable[[str], Awaitable[None]]
    ) -> None:
        """Register a handler for session start events."""
        self.session_start_handlers.append(handler)

    def register_session_complete_handler(
        self,
        handler: Callable[[str, SessionMetrics], Awaitable[None]]
    ) -> None:
        """Register a handler for session completion events."""
        self.session_complete_handlers.append(handler)

    def register_resource_alert_handler(
        self,
        handler: Callable[[ResourceStatus], Awaitable[None]]
    ) -> None:
        """Register a handler for resource status alerts."""
        self.resource_alert_handlers.append(handler)

    async def _process_session_queue(self) -> None:
        """Process queued training sessions based on resource availability."""
        while not self._shutdown_event.is_set():
            try:
                # Check if we can start a new session
                if not self._can_start_new_session():
                    await asyncio.sleep(1.0)
                    continue
                
                # Get next session from queue (with timeout to allow shutdown)
                try:
                    priority, requested_at, request = await asyncio.wait_for(
                        self.session_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Create and start the session
                session = TrainingSession(
                    session_id=request.session_id,
                    config=request.config,
                    game_server_url=self.game_server_url,
                    ws_url=self.ws_url
                )
                
                # Register session event handlers
                session.register_episode_handler(self._handle_episode_complete)
                session.register_metrics_handler(self._handle_metrics_update)
                
                # Initialize and start the session
                await session.initialize()
                
                # Add to active sessions
                self.active_sessions[session.session_id] = session
                
                # Start training in background
                training_task = asyncio.create_task(
                    self._run_session_with_cleanup(session)
                )
                
                # Notify handlers
                for handler in self.session_start_handlers:
                    try:
                        await handler(session.session_id)
                    except Exception as e:
                        self._logger.error(f"Error in session start handler: {e}")
                
                self._logger.info(f"Started training session {session.session_id}")
                
            except Exception as e:
                self._logger.error(f"Error processing session queue: {e}")
                await asyncio.sleep(1.0)

    async def _run_session_with_cleanup(self, session: TrainingSession) -> None:
        """Run a session and handle cleanup when it completes."""
        try:
            await session.start_training()
        except Exception as e:
            self._logger.error(f"Session {session.session_id} failed: {e}")
        finally:
            # Clean up session
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
                self.completed_sessions.append(session.session_id)
                
                # Notify completion handlers
                for handler in self.session_complete_handlers:
                    try:
                        await handler(session.session_id, session.metrics)
                    except Exception as e:
                        self._logger.error(f"Error in session complete handler: {e}")
                
                self._logger.info(f"Session {session.session_id} completed and cleaned up")

    async def _monitor_resources(self) -> None:
        """Monitor system resources and update status."""
        while not self._shutdown_event.is_set():
            try:
                # Update current parallel episodes count
                self.current_parallel_episodes = sum(
                    len(session.active_episodes) for session in self.active_sessions.values()
                )
                
                # Determine resource status
                old_status = self.resource_status
                
                if (
                    len(self.active_sessions) >= self.resource_limits.max_concurrent_sessions or
                    self.current_parallel_episodes >= self.resource_limits.max_total_parallel_episodes
                ):
                    self.resource_status = ResourceStatus.EXHAUSTED
                elif (
                    len(self.active_sessions) >= self.resource_limits.max_concurrent_sessions * 0.8 or
                    self.current_parallel_episodes >= self.resource_limits.max_total_parallel_episodes * 0.8
                ):
                    self.resource_status = ResourceStatus.LIMITED
                else:
                    self.resource_status = ResourceStatus.AVAILABLE
                
                # Notify if status changed
                if old_status != self.resource_status:
                    self._logger.info(f"Resource status changed: {old_status.value} -> {self.resource_status.value}")
                    
                    for handler in self.resource_alert_handlers:
                        try:
                            await handler(self.resource_status)
                        except Exception as e:
                            self._logger.error(f"Error in resource alert handler: {e}")
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self._logger.error(f"Error monitoring resources: {e}")
                await asyncio.sleep(5.0)

    def _can_start_new_session(self) -> bool:
        """Check if a new session can be started based on resource limits."""
        return (
            len(self.active_sessions) < self.resource_limits.max_concurrent_sessions and
            self.current_parallel_episodes < self.resource_limits.max_total_parallel_episodes
        )

    async def _handle_episode_complete(self, result: EpisodeResult) -> None:
        """Handle episode completion from any session."""
        # This could be used for global episode tracking or analysis
        pass

    async def _handle_metrics_update(self, metrics: SessionMetrics) -> None:
        """Handle metrics updates from any session."""
        # This could be used for global metrics aggregation
        pass