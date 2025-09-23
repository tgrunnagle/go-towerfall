"""
WebSocket connection manager for training metrics server.

This module manages WebSocket connections for real-time training metrics,
bot decisions, and performance graph updates.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Callable, Awaitable
from collections import defaultdict
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect
from rl_bot_system.server.data_models import (
    WebSocketMessage,
    MessageType,
    SpectatorConnectionInfo,
    TrainingMetricsData,
    BotDecisionData,
    PerformanceGraphData
)

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """WebSocket connection information."""
    websocket: WebSocket
    connection_id: str
    user_name: str
    user_id: Optional[str]
    session_id: str
    connected_at: datetime
    subscriptions: Set[str]  # Message types this connection subscribes to
    
    def to_info(self) -> SpectatorConnectionInfo:
        """Convert to SpectatorConnectionInfo."""
        return SpectatorConnectionInfo(
            connection_id=self.connection_id,
            user_name=self.user_name,
            user_id=self.user_id,
            connected_at=self.connected_at,
            session_id=self.session_id
        )


class ConnectionManager:
    """
    Manages WebSocket connections for training metrics.
    
    Handles connection lifecycle, message broadcasting, and subscription management.
    """
    
    def __init__(self):
        # Active connections by connection ID
        self._connections: Dict[str, Connection] = {}
        
        # Connections grouped by session ID
        self._session_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Connections grouped by subscription type
        self._subscription_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Message callbacks
        self._message_callbacks: Dict[str, List[Callable[[str, Dict[str, Any]], Awaitable[None]]]] = defaultdict(list)
        
        # Connection event callbacks
        self._connect_callbacks: List[Callable[[Connection], Awaitable[None]]] = []
        self._disconnect_callbacks: List[Callable[[Connection], Awaitable[None]]] = []
    
    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        user_name: str,
        user_id: Optional[str] = None,
        subscriptions: Optional[List[str]] = None
    ) -> str:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket instance
            session_id: Training session ID
            user_name: Display name for the user
            user_id: Optional user ID
            subscriptions: List of message types to subscribe to
            
        Returns:
            Connection ID for the new connection
        """
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        
        if subscriptions is None:
            # Default subscriptions
            subscriptions = [
                MessageType.TRAINING_METRICS,
                MessageType.BOT_DECISION,
                MessageType.GRAPH_UPDATE,
                MessageType.TRAINING_STATUS,
                MessageType.CONNECTION_STATUS
            ]
        
        connection = Connection(
            websocket=websocket,
            connection_id=connection_id,
            user_name=user_name,
            user_id=user_id,
            session_id=session_id,
            connected_at=datetime.now(),
            subscriptions=set(subscriptions)
        )
        
        # Store connection
        self._connections[connection_id] = connection
        self._session_connections[session_id].add(connection_id)
        
        # Add to subscription groups
        for subscription in subscriptions:
            self._subscription_connections[subscription].add(connection_id)
        
        # Notify callbacks
        for callback in self._connect_callbacks:
            try:
                await callback(connection)
            except Exception as e:
                logger.error(f"Error in connect callback: {e}")
        
        logger.info(f"WebSocket connected: {connection_id} ({user_name}) to session {session_id}")
        
        # Send connection confirmation
        await self.send_to_connection(
            connection_id,
            MessageType.CONNECTION_STATUS,
            {
                "status": "connected",
                "connection_id": connection_id,
                "session_id": session_id,
                "subscriptions": list(subscriptions)
            }
        )
        
        return connection_id
    
    async def disconnect(self, connection_id: str) -> None:
        """
        Disconnect a WebSocket connection.
        
        Args:
            connection_id: ID of the connection to disconnect
        """
        if connection_id not in self._connections:
            return
        
        connection = self._connections[connection_id]
        
        # Remove from session connections
        self._session_connections[connection.session_id].discard(connection_id)
        if not self._session_connections[connection.session_id]:
            del self._session_connections[connection.session_id]
        
        # Remove from subscription groups
        for subscription in connection.subscriptions:
            self._subscription_connections[subscription].discard(connection_id)
            if not self._subscription_connections[subscription]:
                del self._subscription_connections[subscription]
        
        # Notify callbacks
        for callback in self._disconnect_callbacks:
            try:
                await callback(connection)
            except Exception as e:
                logger.error(f"Error in disconnect callback: {e}")
        
        # Remove connection
        del self._connections[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id} ({connection.user_name})")
    
    async def send_to_connection(
        self,
        connection_id: str,
        message_type: MessageType,
        data: Any
    ) -> bool:
        """
        Send a message to a specific connection.
        
        Args:
            connection_id: ID of the connection
            message_type: Type of message
            data: Message data
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if connection_id not in self._connections:
            return False
        
        connection = self._connections[connection_id]
        
        # Check if connection subscribes to this message type (except for system messages)
        if message_type not in connection.subscriptions and message_type not in [MessageType.CONNECTION_STATUS, MessageType.ERROR]:
            return True  # Not an error, just not subscribed
        
        message = WebSocketMessage(
            type=message_type,
            data=data
        )
        
        try:
            await connection.websocket.send_text(message.model_dump_json())
            return True
        except Exception as e:
            logger.error(f"Failed to send message to connection {connection_id}: {e}")
            # Disconnect the connection
            await self.disconnect(connection_id)
            return False
    
    async def broadcast_to_session(
        self,
        session_id: str,
        message_type: MessageType,
        data: Any
    ) -> int:
        """
        Broadcast a message to all connections in a session.
        
        Args:
            session_id: Training session ID
            message_type: Type of message
            data: Message data
            
        Returns:
            Number of connections that received the message
        """
        if session_id not in self._session_connections:
            return 0
        
        connection_ids = list(self._session_connections[session_id])
        sent_count = 0
        
        # Send to all connections in parallel
        tasks = []
        for connection_id in connection_ids:
            tasks.append(self.send_to_connection(connection_id, message_type, data))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, bool) and result:
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_subscription(
        self,
        message_type: MessageType,
        data: Any
    ) -> int:
        """
        Broadcast a message to all connections subscribed to a message type.
        
        Args:
            message_type: Type of message
            data: Message data
            
        Returns:
            Number of connections that received the message
        """
        if message_type not in self._subscription_connections:
            return 0
        
        connection_ids = list(self._subscription_connections[message_type])
        sent_count = 0
        
        # Send to all connections in parallel
        tasks = []
        for connection_id in connection_ids:
            tasks.append(self.send_to_connection(connection_id, message_type, data))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, bool) and result:
                sent_count += 1
        
        return sent_count
    
    async def handle_message(
        self,
        connection_id: str,
        message: str
    ) -> None:
        """
        Handle incoming message from a WebSocket connection.
        
        Args:
            connection_id: ID of the connection
            message: Raw message string
        """
        if connection_id not in self._connections:
            return
        
        try:
            message_data = json.loads(message)
            message_type = message_data.get('type')
            
            # Notify message callbacks
            if message_type in self._message_callbacks:
                for callback in self._message_callbacks[message_type]:
                    try:
                        await callback(connection_id, message_data)
                    except Exception as e:
                        logger.error(f"Error in message callback: {e}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message from {connection_id}: {e}")
            await self.send_to_connection(
                connection_id,
                MessageType.ERROR,
                {"error": "invalid_json", "message": "Invalid JSON format"}
            )
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            await self.send_to_connection(
                connection_id,
                MessageType.ERROR,
                {"error": "message_error", "message": str(e)}
            )
    
    def get_connection_info(self, connection_id: str) -> Optional[SpectatorConnectionInfo]:
        """
        Get information about a connection.
        
        Args:
            connection_id: ID of the connection
            
        Returns:
            SpectatorConnectionInfo or None if not found
        """
        if connection_id not in self._connections:
            return None
        
        return self._connections[connection_id].to_info()
    
    def get_session_connections(self, session_id: str) -> List[SpectatorConnectionInfo]:
        """
        Get all connections for a session.
        
        Args:
            session_id: Training session ID
            
        Returns:
            List of SpectatorConnectionInfo objects
        """
        if session_id not in self._session_connections:
            return []
        
        connections = []
        for connection_id in self._session_connections[session_id]:
            if connection_id in self._connections:
                connections.append(self._connections[connection_id].to_info())
        
        return connections
    
    def get_connection_count(self, session_id: Optional[str] = None) -> int:
        """
        Get the number of active connections.
        
        Args:
            session_id: Optional session ID to filter by
            
        Returns:
            Number of active connections
        """
        if session_id:
            return len(self._session_connections.get(session_id, set()))
        else:
            return len(self._connections)
    
    def register_message_callback(
        self,
        message_type: str,
        callback: Callable[[str, Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Register a callback for incoming messages.
        
        Args:
            message_type: Type of message to listen for
            callback: Async function to call with (connection_id, message_data)
        """
        self._message_callbacks[message_type].append(callback)
    
    def register_connect_callback(
        self,
        callback: Callable[[Connection], Awaitable[None]]
    ) -> None:
        """
        Register a callback for new connections.
        
        Args:
            callback: Async function to call with Connection object
        """
        self._connect_callbacks.append(callback)
    
    def register_disconnect_callback(
        self,
        callback: Callable[[Connection], Awaitable[None]]
    ) -> None:
        """
        Register a callback for disconnections.
        
        Args:
            callback: Async function to call with Connection object
        """
        self._disconnect_callbacks.append(callback)
    
    async def cleanup(self) -> None:
        """Clean up all connections and resources."""
        # Disconnect all connections
        connection_ids = list(self._connections.keys())
        for connection_id in connection_ids:
            await self.disconnect(connection_id)
        
        # Clear all data structures
        self._connections.clear()
        self._session_connections.clear()
        self._subscription_connections.clear()
        self._message_callbacks.clear()
        self._connect_callbacks.clear()
        self._disconnect_callbacks.clear()
        
        logger.info("ConnectionManager cleanup completed")


class WebSocketManager:
    """
    High-level WebSocket manager for training metrics.
    
    Provides simplified interface for broadcasting training data.
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def broadcast_training_metrics(
        self,
        session_id: str,
        metrics_data: TrainingMetricsData
    ) -> int:
        """
        Broadcast training metrics to all spectators in a session.
        
        Args:
            session_id: Training session ID
            metrics_data: Training metrics data
            
        Returns:
            Number of connections that received the message
        """
        return await self.connection_manager.broadcast_to_session(
            session_id,
            MessageType.TRAINING_METRICS,
            metrics_data.model_dump()
        )
    
    async def broadcast_bot_decision(
        self,
        session_id: str,
        decision_data: BotDecisionData
    ) -> int:
        """
        Broadcast bot decision data to all spectators in a session.
        
        Args:
            session_id: Training session ID
            decision_data: Bot decision data
            
        Returns:
            Number of connections that received the message
        """
        return await self.connection_manager.broadcast_to_session(
            session_id,
            MessageType.BOT_DECISION,
            decision_data.model_dump()
        )
    
    async def broadcast_graph_update(
        self,
        session_id: str,
        graph_data: PerformanceGraphData
    ) -> int:
        """
        Broadcast performance graph update to all spectators in a session.
        
        Args:
            session_id: Training session ID
            graph_data: Performance graph data
            
        Returns:
            Number of connections that received the message
        """
        return await self.connection_manager.broadcast_to_session(
            session_id,
            MessageType.GRAPH_UPDATE,
            graph_data.model_dump()
        )
    
    async def broadcast_training_status(
        self,
        session_id: str,
        status_data: Dict[str, Any]
    ) -> int:
        """
        Broadcast training status update to all spectators in a session.
        
        Args:
            session_id: Training session ID
            status_data: Status update data
            
        Returns:
            Number of connections that received the message
        """
        return await self.connection_manager.broadcast_to_session(
            session_id,
            MessageType.TRAINING_STATUS,
            status_data
        )