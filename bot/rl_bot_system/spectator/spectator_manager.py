"""
Spectator Manager for RL bot training observation.

This module manages spectator sessions, room creation, and access control
for observing bot training and evaluation sessions.
"""

import asyncio
import logging
import uuid
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Callable, Awaitable
from dataclasses import dataclass, asdict
from enum import Enum

from core.game_client import GameClient, TrainingMode
from rl_bot_system.spectator.training_metrics_overlay import TrainingMetricsOverlay, MetricsData

logger = logging.getLogger(__name__)


class SpectatorMode(Enum):
    """Spectator viewing modes."""
    LIVE_TRAINING = "live_training"
    REPLAY = "replay"
    COMPARISON = "comparison"


@dataclass
class SpectatorSession:
    """Represents an active spectator session."""
    session_id: str
    training_session_id: str
    room_code: str
    room_password: Optional[str]
    spectator_mode: SpectatorMode
    created_at: datetime
    expires_at: datetime
    max_spectators: int
    current_spectators: int
    access_control: Dict[str, Any]
    metrics_overlay: bool
    performance_graphs: bool
    decision_visualization: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'training_session_id': self.training_session_id,
            'room_code': self.room_code,
            'room_password': self.room_password,
            'spectator_mode': self.spectator_mode.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'max_spectators': self.max_spectators,
            'current_spectators': self.current_spectators,
            'access_control': self.access_control,
            'metrics_overlay': self.metrics_overlay,
            'performance_graphs': self.performance_graphs,
            'decision_visualization': self.decision_visualization
        }


class SpectatorManager:
    """
    Manages spectator sessions for RL bot training observation.
    
    Provides functionality to create spectator rooms, manage access control,
    and coordinate real-time training observation.
    """
    
    def __init__(self, game_server_url: str = "http://localhost:4000"):
        self.game_server_url = game_server_url
        self._active_sessions: Dict[str, SpectatorSession] = {}
        self._session_clients: Dict[str, List[GameClient]] = {}
        self._metrics_overlays: Dict[str, TrainingMetricsOverlay] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_callbacks: Dict[str, List[Callable[[MetricsData], Awaitable[None]]]] = {}
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        
    async def create_spectator_session(
        self,
        training_session_id: str,
        spectator_mode: SpectatorMode = SpectatorMode.LIVE_TRAINING,
        max_spectators: int = 10,
        session_duration_hours: int = 24,
        password_protected: bool = False,
        enable_metrics_overlay: bool = True,
        enable_performance_graphs: bool = True,
        enable_decision_visualization: bool = True,
        access_control: Optional[Dict[str, Any]] = None
    ) -> SpectatorSession:
        """
        Create a new spectator session for a training session.
        
        Args:
            training_session_id: ID of the training session to observe
            spectator_mode: Type of spectator viewing mode
            max_spectators: Maximum number of concurrent spectators
            session_duration_hours: How long the session should remain active
            password_protected: Whether to require a password for access
            enable_metrics_overlay: Enable training metrics overlay
            enable_performance_graphs: Enable real-time performance graphs
            enable_decision_visualization: Enable bot decision visualization
            access_control: Additional access control settings
            
        Returns:
            Created SpectatorSession object
        """
        session_id = str(uuid.uuid4())
        room_code = self._generate_room_code()
        room_password = secrets.token_urlsafe(8) if password_protected else None
        
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=session_duration_hours)
        
        if access_control is None:
            access_control = {
                'allow_anonymous': True,
                'require_approval': False,
                'allowed_users': [],
                'blocked_users': []
            }
        
        session = SpectatorSession(
            session_id=session_id,
            training_session_id=training_session_id,
            room_code=room_code,
            room_password=room_password,
            spectator_mode=spectator_mode,
            created_at=created_at,
            expires_at=expires_at,
            max_spectators=max_spectators,
            current_spectators=0,
            access_control=access_control,
            metrics_overlay=enable_metrics_overlay,
            performance_graphs=enable_performance_graphs,
            decision_visualization=enable_decision_visualization
        )
        
        self._active_sessions[session_id] = session
        self._session_clients[session_id] = []
        self._metrics_callbacks[session_id] = []
        
        # Create metrics overlay if enabled
        if enable_metrics_overlay:
            self._metrics_overlays[session_id] = TrainingMetricsOverlay(
                session_id=session_id,
                enable_graphs=enable_performance_graphs,
                enable_decision_viz=enable_decision_visualization
            )
        
        logger.info(f"Created spectator session {session_id} for training {training_session_id}")
        logger.info(f"Room code: {room_code}, Password: {'Yes' if room_password else 'No'}")
        
        return session
    
    async def join_spectator_session(
        self,
        session_id: str,
        spectator_name: str,
        password: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Join an existing spectator session.
        
        Args:
            session_id: ID of the spectator session to join
            spectator_name: Name for the spectator
            password: Password if session is password protected
            user_id: Optional user ID for access control
            
        Returns:
            Connection information for the spectator
            
        Raises:
            ValueError: If session doesn't exist or access is denied
            RuntimeError: If session is full or expired
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Spectator session {session_id} not found")
        
        session = self._active_sessions[session_id]
        
        # Check if session is expired
        if datetime.now() > session.expires_at:
            raise RuntimeError(f"Spectator session {session_id} has expired")
        
        # Check if session is full
        if session.current_spectators >= session.max_spectators:
            raise RuntimeError(f"Spectator session {session_id} is full")
        
        # Check password if required
        if session.room_password and password != session.room_password:
            raise ValueError("Invalid password for spectator session")
        
        # Check access control
        if not self._check_access_permission(session, user_id):
            raise ValueError("Access denied to spectator session")
        
        # Create game client for spectator
        client = GameClient(http_url=self.game_server_url)
        
        try:
            # Join the training room as spectator
            join_result = await client.connect(
                room_code=session.room_code,
                player_name=f"[SPECTATOR] {spectator_name}",
                room_password=session.room_password
            )
            
            # Add to session
            self._session_clients[session_id].append(client)
            session.current_spectators += 1
            
            # Register for metrics updates if overlay is enabled
            if session.metrics_overlay and session_id in self._metrics_overlays:
                overlay = self._metrics_overlays[session_id]
                await overlay.register_spectator(client)
            
            connection_info = {
                'session_id': session_id,
                'room_code': session.room_code,
                'spectator_mode': session.spectator_mode.value,
                'metrics_overlay': session.metrics_overlay,
                'performance_graphs': session.performance_graphs,
                'decision_visualization': session.decision_visualization,
                'training_session_id': session.training_session_id,
                'client': client
            }
            
            logger.info(f"Spectator {spectator_name} joined session {session_id}")
            return connection_info
            
        except Exception as e:
            logger.error(f"Failed to join spectator session: {e}")
            raise
    
    async def leave_spectator_session(
        self,
        session_id: str,
        client: GameClient
    ) -> None:
        """
        Leave a spectator session.
        
        Args:
            session_id: ID of the spectator session
            client: GameClient instance to disconnect
        """
        if session_id not in self._active_sessions:
            return
        
        session = self._active_sessions[session_id]
        
        # Remove client from session
        if client in self._session_clients[session_id]:
            self._session_clients[session_id].remove(client)
            session.current_spectators -= 1
            
            # Unregister from metrics overlay
            if session.metrics_overlay and session_id in self._metrics_overlays:
                overlay = self._metrics_overlays[session_id]
                await overlay.unregister_spectator(client)
            
            # Close client connection
            await client.close()
            
            logger.info(f"Spectator left session {session_id}")
    
    async def update_training_metrics(
        self,
        session_id: str,
        metrics_data: MetricsData
    ) -> None:
        """
        Update training metrics for a spectator session.
        
        Args:
            session_id: ID of the spectator session
            metrics_data: Updated training metrics
        """
        if session_id not in self._active_sessions:
            return
        
        session = self._active_sessions[session_id]
        
        # Update metrics overlay
        if session.metrics_overlay and session_id in self._metrics_overlays:
            overlay = self._metrics_overlays[session_id]
            await overlay.update_metrics(metrics_data)
        
        # Notify callbacks
        if session_id in self._metrics_callbacks:
            for callback in self._metrics_callbacks[session_id]:
                try:
                    await callback(metrics_data)
                except Exception as e:
                    logger.error(f"Error in metrics callback: {e}")
    
    def register_metrics_callback(
        self,
        session_id: str,
        callback: Callable[[MetricsData], Awaitable[None]]
    ) -> None:
        """
        Register a callback for training metrics updates.
        
        Args:
            session_id: ID of the spectator session
            callback: Async function to call with metrics updates
        """
        if session_id not in self._metrics_callbacks:
            self._metrics_callbacks[session_id] = []
        
        self._metrics_callbacks[session_id].append(callback)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a spectator session.
        
        Args:
            session_id: ID of the spectator session
            
        Returns:
            Session information dictionary or None if not found
        """
        if session_id not in self._active_sessions:
            return None
        
        session = self._active_sessions[session_id]
        return session.to_dict()
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active spectator sessions.
        
        Returns:
            List of session information dictionaries
        """
        return [session.to_dict() for session in self._active_sessions.values()]
    
    async def close_session(self, session_id: str) -> None:
        """
        Close a spectator session and disconnect all spectators.
        
        Args:
            session_id: ID of the spectator session to close
        """
        if session_id not in self._active_sessions:
            return
        
        # Disconnect all spectators
        if session_id in self._session_clients:
            for client in self._session_clients[session_id][:]:  # Copy list to avoid modification during iteration
                await self.leave_spectator_session(session_id, client)
        
        # Clean up resources
        if session_id in self._metrics_overlays:
            await self._metrics_overlays[session_id].cleanup()
            del self._metrics_overlays[session_id]
        
        if session_id in self._metrics_callbacks:
            del self._metrics_callbacks[session_id]
        
        if session_id in self._session_clients:
            del self._session_clients[session_id]
        
        del self._active_sessions[session_id]
        
        logger.info(f"Closed spectator session {session_id}")
    
    async def cleanup(self) -> None:
        """Clean up all resources and close all sessions."""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all sessions
        session_ids = list(self._active_sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id)
        
        logger.info("SpectatorManager cleanup completed")
    
    def _generate_room_code(self) -> str:
        """Generate a unique room code for spectator access."""
        # Generate a 6-character alphanumeric code
        import string
        import random
        
        chars = string.ascii_uppercase + string.digits
        code = ''.join(random.choices(chars, k=6))
        
        # Ensure uniqueness (simple check)
        while any(session.room_code == code for session in self._active_sessions.values()):
            code = ''.join(random.choices(chars, k=6))
        
        return code
    
    def _check_access_permission(
        self,
        session: SpectatorSession,
        user_id: Optional[str]
    ) -> bool:
        """
        Check if a user has permission to access a spectator session.
        
        Args:
            session: SpectatorSession to check access for
            user_id: Optional user ID to check
            
        Returns:
            True if access is allowed, False otherwise
        """
        access_control = session.access_control
        
        # Check if user is blocked
        if user_id and user_id in access_control.get('blocked_users', []):
            return False
        
        # Check if anonymous access is allowed
        if not user_id and not access_control.get('allow_anonymous', True):
            return False
        
        # Check if user is in allowed list (if specified)
        allowed_users = access_control.get('allowed_users', [])
        if allowed_users and user_id not in allowed_users:
            return False
        
        return True
    
    async def _cleanup_expired_sessions(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            try:
                current_time = datetime.now()
                expired_sessions = [
                    session_id for session_id, session in self._active_sessions.items()
                    if current_time > session.expires_at
                ]
                
                for session_id in expired_sessions:
                    logger.info(f"Cleaning up expired spectator session {session_id}")
                    await self.close_session(session_id)
                
                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying