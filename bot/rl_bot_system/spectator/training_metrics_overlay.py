"""
Training Metrics Overlay for spectator interface.

This module provides real-time training metrics display, performance graphs,
and bot decision visualization for spectators observing training sessions.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

from game_client import GameClient

logger = logging.getLogger(__name__)


@dataclass
class MetricsData:
    """Training metrics data for display."""
    timestamp: datetime
    episode: int
    total_episodes: int
    current_reward: float
    average_reward: float
    best_reward: float
    episode_length: int
    win_rate: float
    loss_value: Optional[float]
    learning_rate: float
    epsilon: Optional[float]  # For DQN
    model_generation: int
    algorithm: str
    training_time_elapsed: float
    
    # Bot decision data
    action_probabilities: Optional[Dict[str, float]] = None
    state_values: Optional[float] = None
    q_values: Optional[List[float]] = None
    selected_action: Optional[str] = None
    
    # Performance metrics
    actions_per_second: Optional[float] = None
    frames_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceGraph:
    """Configuration for a performance graph."""
    graph_id: str
    title: str
    y_label: str
    metrics: List[str]  # Which metrics to plot
    max_points: int = 1000
    update_frequency: float = 1.0  # seconds
    
    
class TrainingMetricsOverlay:
    """
    Manages training metrics overlay for spectator interface.
    
    Provides real-time display of training progress, performance graphs,
    and bot decision visualization for spectators.
    """
    
    def __init__(
        self,
        session_id: str,
        enable_graphs: bool = True,
        enable_decision_viz: bool = True,
        max_history_points: int = 1000
    ):
        self.session_id = session_id
        self.enable_graphs = enable_graphs
        self.enable_decision_viz = enable_decision_viz
        self.max_history_points = max_history_points
        
        # Spectator clients
        self._spectator_clients: List[GameClient] = []
        
        # Metrics history
        self._metrics_history: deque = deque(maxlen=max_history_points)
        self._current_metrics: Optional[MetricsData] = None
        
        # Performance graphs configuration
        self._performance_graphs = self._setup_performance_graphs()
        
        # Update tracking
        self._last_update = datetime.now()
        self._update_frequency = 1.0  # seconds
        
        # Background tasks
        self._update_task: Optional[asyncio.Task] = None
        
        if enable_graphs:
            self._update_task = asyncio.create_task(self._periodic_updates())
    
    async def register_spectator(self, client: GameClient) -> None:
        """
        Register a spectator client for metrics updates.
        
        Args:
            client: GameClient instance for the spectator
        """
        self._spectator_clients.append(client)
        
        # Send current metrics to new spectator
        if self._current_metrics:
            await self._send_metrics_to_client(client, self._current_metrics)
        
        # Send graph configuration
        if self.enable_graphs:
            await self._send_graph_config_to_client(client)
        
        logger.info(f"Registered spectator for metrics overlay in session {self.session_id}")
    
    async def unregister_spectator(self, client: GameClient) -> None:
        """
        Unregister a spectator client.
        
        Args:
            client: GameClient instance to unregister
        """
        if client in self._spectator_clients:
            self._spectator_clients.remove(client)
            logger.info(f"Unregistered spectator from metrics overlay in session {self.session_id}")
    
    async def update_metrics(self, metrics_data: MetricsData) -> None:
        """
        Update training metrics and broadcast to spectators.
        
        Args:
            metrics_data: New training metrics data
        """
        self._current_metrics = metrics_data
        self._metrics_history.append(metrics_data)
        
        # Broadcast to all spectators
        await self._broadcast_metrics_update(metrics_data)
        
        # Update performance graphs if enabled
        if self.enable_graphs:
            await self._update_performance_graphs(metrics_data)
    
    async def send_bot_decision_data(
        self,
        action_probabilities: Dict[str, float],
        state_values: Optional[float] = None,
        q_values: Optional[List[float]] = None,
        selected_action: Optional[str] = None
    ) -> None:
        """
        Send bot decision visualization data to spectators.
        
        Args:
            action_probabilities: Probability distribution over actions
            state_values: Current state value estimate
            q_values: Q-values for each action (if applicable)
            selected_action: The action that was selected
        """
        if not self.enable_decision_viz:
            return
        
        decision_data = {
            'type': 'bot_decision',
            'timestamp': datetime.now().isoformat(),
            'action_probabilities': action_probabilities,
            'state_values': state_values,
            'q_values': q_values,
            'selected_action': selected_action
        }
        
        await self._broadcast_to_spectators(decision_data)
    
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_points: Optional[int] = None
    ) -> List[MetricsData]:
        """
        Get historical metrics data.
        
        Args:
            start_time: Start time for filtering (optional)
            end_time: End time for filtering (optional)
            max_points: Maximum number of points to return (optional)
            
        Returns:
            List of MetricsData objects
        """
        history = list(self._metrics_history)
        
        # Filter by time range
        if start_time or end_time:
            filtered_history = []
            for metrics in history:
                if start_time and metrics.timestamp < start_time:
                    continue
                if end_time and metrics.timestamp > end_time:
                    continue
                filtered_history.append(metrics)
            history = filtered_history
        
        # Limit number of points
        if max_points and len(history) > max_points:
            # Sample evenly across the range
            step = len(history) // max_points
            history = history[::step]
        
        return history
    
    def get_current_metrics(self) -> Optional[MetricsData]:
        """Get the current training metrics."""
        return self._current_metrics
    
    async def cleanup(self) -> None:
        """Clean up resources and stop background tasks."""
        # Cancel update task
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        # Clear spectator clients
        self._spectator_clients.clear()
        
        logger.info(f"Cleaned up metrics overlay for session {self.session_id}")
    
    def _setup_performance_graphs(self) -> List[PerformanceGraph]:
        """Set up default performance graphs."""
        return [
            PerformanceGraph(
                graph_id="reward_progress",
                title="Reward Progress",
                y_label="Reward",
                metrics=["current_reward", "average_reward", "best_reward"]
            ),
            PerformanceGraph(
                graph_id="win_rate",
                title="Win Rate",
                y_label="Win Rate (%)",
                metrics=["win_rate"]
            ),
            PerformanceGraph(
                graph_id="episode_length",
                title="Episode Length",
                y_label="Steps",
                metrics=["episode_length"]
            ),
            PerformanceGraph(
                graph_id="learning_metrics",
                title="Learning Metrics",
                y_label="Value",
                metrics=["loss_value", "learning_rate", "epsilon"]
            ),
            PerformanceGraph(
                graph_id="performance_stats",
                title="Performance Stats",
                y_label="Rate",
                metrics=["actions_per_second", "frames_per_second"]
            )
        ]
    
    async def _broadcast_metrics_update(self, metrics_data: MetricsData) -> None:
        """Broadcast metrics update to all spectators."""
        message = {
            'type': 'training_metrics',
            'data': metrics_data.to_dict()
        }
        
        await self._broadcast_to_spectators(message)
    
    async def _broadcast_to_spectators(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all registered spectators."""
        if not self._spectator_clients:
            return
        
        message_json = json.dumps(message)
        
        # Send to all spectators (in parallel)
        tasks = []
        for client in self._spectator_clients[:]:  # Copy list to avoid modification during iteration
            if client.websocket and not client.websocket.closed:
                tasks.append(self._send_message_to_client(client, message_json))
            else:
                # Remove disconnected clients
                self._spectator_clients.remove(client)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_message_to_client(self, client: GameClient, message: str) -> None:
        """Send a message to a specific client."""
        try:
            if client.websocket and not client.websocket.closed:
                await client.websocket.send(message)
        except Exception as e:
            logger.error(f"Failed to send message to spectator: {e}")
            # Remove client from list
            if client in self._spectator_clients:
                self._spectator_clients.remove(client)
    
    async def _send_metrics_to_client(self, client: GameClient, metrics_data: MetricsData) -> None:
        """Send current metrics to a specific client."""
        message = {
            'type': 'training_metrics',
            'data': metrics_data.to_dict()
        }
        
        await self._send_message_to_client(client, json.dumps(message))
    
    async def _send_graph_config_to_client(self, client: GameClient) -> None:
        """Send graph configuration to a specific client."""
        message = {
            'type': 'graph_config',
            'graphs': [
                {
                    'graph_id': graph.graph_id,
                    'title': graph.title,
                    'y_label': graph.y_label,
                    'metrics': graph.metrics,
                    'max_points': graph.max_points
                }
                for graph in self._performance_graphs
            ]
        }
        
        await self._send_message_to_client(client, json.dumps(message))
    
    async def _update_performance_graphs(self, metrics_data: MetricsData) -> None:
        """Update performance graphs with new metrics data."""
        # Prepare graph data for each configured graph
        graph_updates = []
        
        for graph in self._performance_graphs:
            graph_data = {
                'graph_id': graph.graph_id,
                'timestamp': metrics_data.timestamp.isoformat(),
                'data_points': {}
            }
            
            # Extract relevant metrics for this graph
            metrics_dict = metrics_data.to_dict()
            for metric in graph.metrics:
                if metric in metrics_dict and metrics_dict[metric] is not None:
                    graph_data['data_points'][metric] = metrics_dict[metric]
            
            if graph_data['data_points']:  # Only send if we have data
                graph_updates.append(graph_data)
        
        if graph_updates:
            message = {
                'type': 'graph_update',
                'graphs': graph_updates
            }
            
            await self._broadcast_to_spectators(message)
    
    async def _periodic_updates(self) -> None:
        """Background task for periodic updates."""
        while True:
            try:
                # Send periodic status updates
                if self._current_metrics:
                    status_message = {
                        'type': 'training_status',
                        'session_id': self.session_id,
                        'active_spectators': len(self._spectator_clients),
                        'last_update': datetime.now().isoformat(),
                        'metrics_history_size': len(self._metrics_history)
                    }
                    
                    await self._broadcast_to_spectators(status_message)
                
                # Wait for next update
                await asyncio.sleep(self._update_frequency)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
                await asyncio.sleep(5)  # Wait before retrying