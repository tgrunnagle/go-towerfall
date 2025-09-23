"""
Integration utilities for training metrics server.

This module provides integration helpers for connecting the training metrics
server with existing training engines and spectator managers.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Awaitable

from bot.rl_bot_system.server.training_metrics_server import TrainingMetricsServer
from bot.rl_bot_system.server.data_models import (
    TrainingMetricsData,
    BotDecisionData,
    PerformanceGraphData,
    TrainingStatus
)
from bot.rl_bot_system.spectator.spectator_manager import SpectatorManager
from bot.rl_bot_system.spectator.training_metrics_overlay import MetricsData

logger = logging.getLogger(__name__)


class TrainingEngineIntegration:
    """
    Integration adapter for training engines.
    
    Connects training engines with the metrics server to automatically
    collect and broadcast training data.
    """
    
    def __init__(self, metrics_server: TrainingMetricsServer):
        self.metrics_server = metrics_server
        self._active_sessions: Dict[str, str] = {}  # training_id -> session_id
        self._callbacks_registered = False
    
    async def register_training_session(
        self,
        training_id: str,
        model_generation: int,
        algorithm: str,
        total_episodes: int,
        room_code: Optional[str] = None
    ) -> str:
        """
        Register a new training session with the metrics server.
        
        Args:
            training_id: Unique training session identifier
            model_generation: Model generation number
            algorithm: RL algorithm being used
            total_episodes: Total number of episodes to train
            room_code: Optional room code for spectators
            
        Returns:
            Session ID created in the metrics server
        """
        # Create session in metrics server
        session_request = {
            "training_session_id": training_id,
            "model_generation": model_generation,
            "algorithm": algorithm,
            "total_episodes": total_episodes,
            "room_code": room_code,
            "enable_spectators": True
        }
        
        # This would call the metrics server API
        # For now, we'll simulate the session creation
        session_id = training_id
        self._active_sessions[training_id] = session_id
        
        logger.info(f"Registered training session {training_id} with metrics server")
        return session_id
    
    async def update_training_metrics(
        self,
        training_id: str,
        episode: int,
        total_episodes: int,
        current_reward: float,
        average_reward: float,
        best_reward: float,
        episode_length: int,
        win_rate: float,
        loss_value: Optional[float] = None,
        learning_rate: float = 0.001,
        epsilon: Optional[float] = None,
        model_generation: int = 1,
        algorithm: str = "DQN",
        training_time_elapsed: float = 0.0,
        **kwargs
    ) -> None:
        """
        Update training metrics for a session.
        
        Args:
            training_id: Training session identifier
            episode: Current episode number
            total_episodes: Total number of episodes
            current_reward: Current episode reward
            average_reward: Average reward over recent episodes
            best_reward: Best reward achieved so far
            episode_length: Length of current episode
            win_rate: Current win rate percentage
            loss_value: Current loss value (optional)
            learning_rate: Current learning rate
            epsilon: Current epsilon value for exploration (optional)
            model_generation: Model generation number
            algorithm: RL algorithm name
            training_time_elapsed: Total training time in seconds
            **kwargs: Additional metrics
        """
        if training_id not in self._active_sessions:
            logger.warning(f"Training session {training_id} not registered")
            return
        
        session_id = self._active_sessions[training_id]
        
        metrics_data = TrainingMetricsData(
            timestamp=datetime.now(),
            episode=episode,
            total_episodes=total_episodes,
            current_reward=current_reward,
            average_reward=average_reward,
            best_reward=best_reward,
            episode_length=episode_length,
            win_rate=win_rate,
            loss_value=loss_value,
            learning_rate=learning_rate,
            epsilon=epsilon,
            model_generation=model_generation,
            algorithm=algorithm,
            training_time_elapsed=training_time_elapsed,
            **{k: v for k, v in kwargs.items() if k in ['actions_per_second', 'frames_per_second', 'memory_usage_mb']}
        )
        
        # Send to metrics server (this would be an HTTP request in practice)
        await self._send_metrics_to_server(session_id, metrics_data)
    
    async def update_bot_decision(
        self,
        training_id: str,
        action_probabilities: Dict[str, float],
        state_values: Optional[float] = None,
        q_values: Optional[list] = None,
        selected_action: Optional[str] = None,
        confidence_score: Optional[float] = None
    ) -> None:
        """
        Update bot decision data for visualization.
        
        Args:
            training_id: Training session identifier
            action_probabilities: Probability distribution over actions
            state_values: Current state value estimate
            q_values: Q-values for each action
            selected_action: The action that was selected
            confidence_score: Confidence in the decision
        """
        if training_id not in self._active_sessions:
            return
        
        session_id = self._active_sessions[training_id]
        
        decision_data = BotDecisionData(
            timestamp=datetime.now(),
            action_probabilities=action_probabilities,
            state_values=state_values,
            q_values=q_values,
            selected_action=selected_action,
            confidence_score=confidence_score
        )
        
        # Send to metrics server
        await self._send_bot_decision_to_server(session_id, decision_data)
    
    async def update_training_status(
        self,
        training_id: str,
        status: TrainingStatus,
        end_time: Optional[datetime] = None
    ) -> None:
        """
        Update training session status.
        
        Args:
            training_id: Training session identifier
            status: New training status
            end_time: End time if training is completed
        """
        if training_id not in self._active_sessions:
            return
        
        session_id = self._active_sessions[training_id]
        
        update_data = {
            "status": status,
            "end_time": end_time
        }
        
        # Send to metrics server
        await self._send_status_update_to_server(session_id, update_data)
    
    async def unregister_training_session(self, training_id: str) -> None:
        """
        Unregister a training session.
        
        Args:
            training_id: Training session identifier
        """
        if training_id in self._active_sessions:
            session_id = self._active_sessions[training_id]
            del self._active_sessions[training_id]
            
            # Optionally clean up session in metrics server
            logger.info(f"Unregistered training session {training_id}")
    
    async def _send_metrics_to_server(self, session_id: str, metrics_data: TrainingMetricsData) -> None:
        """Send metrics data to the server."""
        # In a real implementation, this would make an HTTP request
        # For now, we'll directly call the server method if available
        if hasattr(self.metrics_server, '_metrics_history'):
            if session_id not in self.metrics_server._metrics_history:
                self.metrics_server._metrics_history[session_id] = []
            self.metrics_server._metrics_history[session_id].append(metrics_data)
            
            # Broadcast to WebSocket connections
            await self.metrics_server.websocket_manager.broadcast_training_metrics(session_id, metrics_data)
    
    async def _send_bot_decision_to_server(self, session_id: str, decision_data: BotDecisionData) -> None:
        """Send bot decision data to the server."""
        # Broadcast to WebSocket connections
        await self.metrics_server.websocket_manager.broadcast_bot_decision(session_id, decision_data)
    
    async def _send_status_update_to_server(self, session_id: str, update_data: Dict[str, Any]) -> None:
        """Send status update to the server."""
        # Update session status
        if hasattr(self.metrics_server, '_training_sessions') and session_id in self.metrics_server._training_sessions:
            session = self.metrics_server._training_sessions[session_id]
            if 'status' in update_data:
                session.status = update_data['status']
            if 'end_time' in update_data:
                session.end_time = update_data['end_time']
            
            # Broadcast status update
            await self.metrics_server.websocket_manager.broadcast_training_status(
                session_id,
                {"event": "status_updated", "session": session.dict()}
            )


class SpectatorManagerIntegration:
    """
    Integration adapter for spectator managers.
    
    Bridges the existing spectator system with the new metrics server.
    """
    
    def __init__(self, metrics_server: TrainingMetricsServer, spectator_manager: SpectatorManager):
        self.metrics_server = metrics_server
        self.spectator_manager = spectator_manager
        self._session_mapping: Dict[str, str] = {}  # spectator_session_id -> metrics_session_id
    
    async def create_integrated_session(
        self,
        training_session_id: str,
        model_generation: int,
        algorithm: str,
        total_episodes: int,
        **spectator_kwargs
    ) -> Dict[str, str]:
        """
        Create a coordinated session in both systems.
        
        Args:
            training_session_id: Training session identifier
            model_generation: Model generation number
            algorithm: RL algorithm name
            total_episodes: Total number of episodes
            **spectator_kwargs: Additional arguments for spectator session
            
        Returns:
            Dictionary with session IDs from both systems
        """
        # Create spectator session
        spectator_session = await self.spectator_manager.create_spectator_session(
            training_session_id=training_session_id,
            **spectator_kwargs
        )
        
        # Create metrics server session
        metrics_session_request = {
            "training_session_id": training_session_id,
            "model_generation": model_generation,
            "algorithm": algorithm,
            "total_episodes": total_episodes,
            "room_code": spectator_session.room_code,
            "enable_spectators": True
        }
        
        # Store mapping
        self._session_mapping[spectator_session.session_id] = training_session_id
        
        # Register metrics callback to forward to metrics server
        self.spectator_manager.register_metrics_callback(
            spectator_session.session_id,
            self._forward_metrics_to_server
        )
        
        return {
            "spectator_session_id": spectator_session.session_id,
            "metrics_session_id": training_session_id,
            "room_code": spectator_session.room_code,
            "room_password": spectator_session.room_password
        }
    
    async def _forward_metrics_to_server(self, metrics_data: MetricsData) -> None:
        """
        Forward metrics from spectator manager to metrics server.
        
        Args:
            metrics_data: Metrics data from spectator system
        """
        # Convert MetricsData to TrainingMetricsData
        training_metrics = TrainingMetricsData(
            timestamp=metrics_data.timestamp,
            episode=metrics_data.episode,
            total_episodes=metrics_data.total_episodes,
            current_reward=metrics_data.current_reward,
            average_reward=metrics_data.average_reward,
            best_reward=metrics_data.best_reward,
            episode_length=metrics_data.episode_length,
            win_rate=metrics_data.win_rate,
            loss_value=metrics_data.loss_value,
            learning_rate=metrics_data.learning_rate,
            epsilon=metrics_data.epsilon,
            model_generation=metrics_data.model_generation,
            algorithm=metrics_data.algorithm,
            training_time_elapsed=metrics_data.training_time_elapsed,
            actions_per_second=metrics_data.actions_per_second,
            frames_per_second=metrics_data.frames_per_second,
            memory_usage_mb=metrics_data.memory_usage_mb
        )
        
        # Find corresponding metrics session
        for spectator_session_id, metrics_session_id in self._session_mapping.items():
            # Forward to metrics server
            await self.metrics_server.websocket_manager.broadcast_training_metrics(
                metrics_session_id, training_metrics
            )
            
            # Also forward bot decision data if available
            if (metrics_data.action_probabilities or 
                metrics_data.state_values is not None or 
                metrics_data.q_values):
                
                bot_decision = BotDecisionData(
                    timestamp=metrics_data.timestamp,
                    action_probabilities=metrics_data.action_probabilities or {},
                    state_values=metrics_data.state_values,
                    q_values=metrics_data.q_values,
                    selected_action=metrics_data.selected_action
                )
                
                await self.metrics_server.websocket_manager.broadcast_bot_decision(
                    metrics_session_id, bot_decision
                )


class MetricsCollector:
    """
    Utility class for collecting and aggregating training metrics.
    
    Provides helper methods for common metrics calculations and data collection.
    """
    
    def __init__(self, integration: TrainingEngineIntegration):
        self.integration = integration
        self._episode_rewards: Dict[str, list] = {}
        self._episode_lengths: Dict[str, list] = {}
        self._win_counts: Dict[str, int] = {}
        self._total_episodes: Dict[str, int] = {}
    
    async def record_episode_start(self, training_id: str) -> None:
        """Record the start of a new episode."""
        if training_id not in self._episode_rewards:
            self._episode_rewards[training_id] = []
            self._episode_lengths[training_id] = []
            self._win_counts[training_id] = 0
            self._total_episodes[training_id] = 0
    
    async def record_episode_end(
        self,
        training_id: str,
        reward: float,
        episode_length: int,
        won: bool,
        **additional_metrics
    ) -> None:
        """
        Record the end of an episode and update metrics.
        
        Args:
            training_id: Training session identifier
            reward: Episode reward
            episode_length: Number of steps in episode
            won: Whether the episode was won
            **additional_metrics: Additional metrics to record
        """
        if training_id not in self._episode_rewards:
            await self.record_episode_start(training_id)
        
        # Record episode data
        self._episode_rewards[training_id].append(reward)
        self._episode_lengths[training_id].append(episode_length)
        self._total_episodes[training_id] += 1
        
        if won:
            self._win_counts[training_id] += 1
        
        # Calculate aggregated metrics
        rewards = self._episode_rewards[training_id]
        lengths = self._episode_lengths[training_id]
        
        # Keep only recent history (last 100 episodes)
        if len(rewards) > 100:
            rewards = rewards[-100:]
            lengths = lengths[-100:]
            self._episode_rewards[training_id] = rewards
            self._episode_lengths[training_id] = lengths
        
        average_reward = sum(rewards) / len(rewards)
        best_reward = max(rewards)
        average_length = sum(lengths) / len(lengths)
        win_rate = (self._win_counts[training_id] / self._total_episodes[training_id]) * 100
        
        # Update metrics in server
        await self.integration.update_training_metrics(
            training_id=training_id,
            episode=self._total_episodes[training_id],
            total_episodes=additional_metrics.get('total_episodes', 1000),
            current_reward=reward,
            average_reward=average_reward,
            best_reward=best_reward,
            episode_length=episode_length,
            win_rate=win_rate,
            **additional_metrics
        )
    
    def get_session_stats(self, training_id: str) -> Dict[str, Any]:
        """
        Get current statistics for a training session.
        
        Args:
            training_id: Training session identifier
            
        Returns:
            Dictionary with current statistics
        """
        if training_id not in self._episode_rewards:
            return {}
        
        rewards = self._episode_rewards[training_id]
        lengths = self._episode_lengths[training_id]
        
        return {
            'total_episodes': self._total_episodes[training_id],
            'total_wins': self._win_counts[training_id],
            'win_rate': (self._win_counts[training_id] / max(1, self._total_episodes[training_id])) * 100,
            'average_reward': sum(rewards) / len(rewards) if rewards else 0,
            'best_reward': max(rewards) if rewards else 0,
            'average_episode_length': sum(lengths) / len(lengths) if lengths else 0,
            'recent_rewards': rewards[-10:] if len(rewards) >= 10 else rewards
        }