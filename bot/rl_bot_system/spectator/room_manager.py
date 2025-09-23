"""
Room Manager for spectator access control and room code generation.

This module handles room creation, access control, and integration
with the game server for spectator functionality.
"""

import asyncio
import logging
import secrets
import string
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access levels for spectator rooms."""
    PUBLIC = "public"           # Anyone can join
    PASSWORD = "password"       # Requires password
    INVITE_ONLY = "invite_only" # Requires explicit invitation
    PRIVATE = "private"         # Creator and invited users only


@dataclass
class RoomAccessControl:
    """Access control configuration for spectator rooms."""
    access_level: AccessLevel
    password: Optional[str] = None
    max_spectators: int = 10
    allowed_users: Set[str] = None
    blocked_users: Set[str] = None
    require_approval: bool = False
    auto_approve_timeout: int = 30  # seconds
    
    def __post_init__(self):
        if self.allowed_users is None:
            self.allowed_users = set()
        if self.blocked_users is None:
            self.blocked_users = set()


@dataclass
class SpectatorRoomInfo:
    """Information about a spectator room."""
    room_id: str
    room_code: str
    training_session_id: str
    creator_id: str
    created_at: datetime
    expires_at: datetime
    access_control: RoomAccessControl
    current_spectators: int
    spectator_list: List[Dict[str, Any]]
    room_status: str  # 'active', 'paused', 'ended'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'room_id': self.room_id,
            'room_code': self.room_code,
            'training_session_id': self.training_session_id,
            'creator_id': self.creator_id,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'access_level': self.access_control.access_level.value,
            'max_spectators': self.access_control.max_spectators,
            'current_spectators': self.current_spectators,
            'spectator_list': self.spectator_list,
            'room_status': self.room_status,
            'has_password': self.access_control.password is not None,
            'require_approval': self.access_control.require_approval
        }


class SpectatorRoomManager:
    """
    Manages spectator room creation, access control, and room codes.
    
    Handles room lifecycle, access permissions, and integration with
    the game server for spectator functionality.
    """
    
    def __init__(self, game_server_url: str = "http://localhost:4000"):
        self.game_server_url = game_server_url
        self._active_rooms: Dict[str, SpectatorRoomInfo] = {}
        self._room_code_mapping: Dict[str, str] = {}  # room_code -> room_id
        self._pending_approvals: Dict[str, List[Dict[str, Any]]] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_rooms())
    
    async def create_spectator_room(
        self,
        training_session_id: str,
        creator_id: str,
        access_control: RoomAccessControl,
        room_duration_hours: int = 24,
        custom_room_code: Optional[str] = None
    ) -> SpectatorRoomInfo:
        """
        Create a new spectator room for a training session.
        
        Args:
            training_session_id: ID of the training session to observe
            creator_id: ID of the user creating the room
            access_control: Access control configuration
            room_duration_hours: How long the room should remain active
            custom_room_code: Optional custom room code (must be unique)
            
        Returns:
            SpectatorRoomInfo object with room details
            
        Raises:
            ValueError: If custom room code is already in use
        """
        room_id = self._generate_room_id()
        
        # Generate or validate room code
        if custom_room_code:
            if custom_room_code in self._room_code_mapping:
                raise ValueError(f"Room code {custom_room_code} is already in use")
            room_code = custom_room_code.upper()
        else:
            room_code = self._generate_room_code()
        
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=room_duration_hours)
        
        room_info = SpectatorRoomInfo(
            room_id=room_id,
            room_code=room_code,
            training_session_id=training_session_id,
            creator_id=creator_id,
            created_at=created_at,
            expires_at=expires_at,
            access_control=access_control,
            current_spectators=0,
            spectator_list=[],
            room_status='active'
        )
        
        self._active_rooms[room_id] = room_info
        self._room_code_mapping[room_code] = room_id
        self._pending_approvals[room_id] = []
        
        logger.info(f"Created spectator room {room_code} (ID: {room_id}) for training {training_session_id}")
        return room_info
    
    async def join_room_request(
        self,
        room_code: str,
        user_id: str,
        user_name: str,
        password: Optional[str] = None,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Request to join a spectator room.
        
        Args:
            room_code: Room code to join
            user_id: ID of the user requesting access
            user_name: Display name of the user
            password: Password if room is password protected
            user_metadata: Additional user information
            
        Returns:
            Join result with status and room information
            
        Raises:
            ValueError: If room doesn't exist or access is denied
            RuntimeError: If room is full or expired
        """
        room_code = room_code.upper()
        
        if room_code not in self._room_code_mapping:
            raise ValueError(f"Room with code {room_code} not found")
        
        room_id = self._room_code_mapping[room_code]
        room_info = self._active_rooms[room_id]
        
        # Check if room is expired
        if datetime.now() > room_info.expires_at:
            raise RuntimeError(f"Room {room_code} has expired")
        
        # Check if room is active
        if room_info.room_status != 'active':
            raise RuntimeError(f"Room {room_code} is not active")
        
        # Check if room is full
        if room_info.current_spectators >= room_info.access_control.max_spectators:
            raise RuntimeError(f"Room {room_code} is full")
        
        # Check access permissions
        access_result = self._check_access_permission(room_info, user_id, password)
        
        if access_result['status'] == 'denied':
            raise ValueError(f"Access denied: {access_result['reason']}")
        
        if access_result['status'] == 'pending_approval':
            # Add to pending approvals
            approval_request = {
                'user_id': user_id,
                'user_name': user_name,
                'requested_at': datetime.now(),
                'user_metadata': user_metadata or {}
            }
            
            self._pending_approvals[room_id].append(approval_request)
            
            # Set auto-approval timeout if configured
            if room_info.access_control.auto_approve_timeout > 0:
                asyncio.create_task(
                    self._auto_approve_timeout(room_id, user_id, room_info.access_control.auto_approve_timeout)
                )
            
            return {
                'status': 'pending_approval',
                'message': 'Your request to join is pending approval',
                'room_info': room_info.to_dict()
            }
        
        # Access granted - add user to room
        spectator_info = {
            'user_id': user_id,
            'user_name': user_name,
            'joined_at': datetime.now().isoformat(),
            'user_metadata': user_metadata or {}
        }
        
        room_info.spectator_list.append(spectator_info)
        room_info.current_spectators += 1
        
        logger.info(f"User {user_name} joined spectator room {room_code}")
        
        return {
            'status': 'approved',
            'message': 'Successfully joined the room',
            'room_info': room_info.to_dict(),
            'spectator_info': spectator_info
        }
    
    async def leave_room(
        self,
        room_code: str,
        user_id: str
    ) -> None:
        """
        Leave a spectator room.
        
        Args:
            room_code: Room code to leave
            user_id: ID of the user leaving
        """
        room_code = room_code.upper()
        
        if room_code not in self._room_code_mapping:
            return
        
        room_id = self._room_code_mapping[room_code]
        room_info = self._active_rooms[room_id]
        
        # Remove user from spectator list
        room_info.spectator_list = [
            spec for spec in room_info.spectator_list
            if spec['user_id'] != user_id
        ]
        
        room_info.current_spectators = len(room_info.spectator_list)
        
        logger.info(f"User {user_id} left spectator room {room_code}")
    
    async def approve_join_request(
        self,
        room_code: str,
        user_id: str,
        approver_id: str,
        approved: bool
    ) -> Dict[str, Any]:
        """
        Approve or deny a pending join request.
        
        Args:
            room_code: Room code
            user_id: ID of the user requesting access
            approver_id: ID of the user approving/denying
            approved: Whether to approve or deny the request
            
        Returns:
            Result of the approval action
        """
        room_code = room_code.upper()
        
        if room_code not in self._room_code_mapping:
            raise ValueError(f"Room with code {room_code} not found")
        
        room_id = self._room_code_mapping[room_code]
        room_info = self._active_rooms[room_id]
        
        # Check if approver has permission (creator or admin)
        if approver_id != room_info.creator_id:
            # Could add admin check here
            raise ValueError("Only the room creator can approve join requests")
        
        # Find pending request
        pending_request = None
        for request in self._pending_approvals[room_id]:
            if request['user_id'] == user_id:
                pending_request = request
                break
        
        if not pending_request:
            raise ValueError(f"No pending request found for user {user_id}")
        
        # Remove from pending list
        self._pending_approvals[room_id].remove(pending_request)
        
        if approved:
            # Check if room still has space
            if room_info.current_spectators >= room_info.access_control.max_spectators:
                return {
                    'status': 'denied',
                    'reason': 'Room is now full'
                }
            
            # Add user to room
            spectator_info = {
                'user_id': user_id,
                'user_name': pending_request['user_name'],
                'joined_at': datetime.now().isoformat(),
                'user_metadata': pending_request['user_metadata'],
                'approved_by': approver_id
            }
            
            room_info.spectator_list.append(spectator_info)
            room_info.current_spectators += 1
            
            logger.info(f"Approved join request for {user_id} in room {room_code}")
            
            return {
                'status': 'approved',
                'spectator_info': spectator_info,
                'room_info': room_info.to_dict()
            }
        else:
            logger.info(f"Denied join request for {user_id} in room {room_code}")
            
            return {
                'status': 'denied',
                'reason': 'Request denied by room creator'
            }
    
    def get_room_info(self, room_code: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a spectator room.
        
        Args:
            room_code: Room code to look up
            
        Returns:
            Room information dictionary or None if not found
        """
        room_code = room_code.upper()
        
        if room_code not in self._room_code_mapping:
            return None
        
        room_id = self._room_code_mapping[room_code]
        room_info = self._active_rooms[room_id]
        
        return room_info.to_dict()
    
    def get_pending_approvals(self, room_code: str) -> List[Dict[str, Any]]:
        """
        Get pending approval requests for a room.
        
        Args:
            room_code: Room code to check
            
        Returns:
            List of pending approval requests
        """
        room_code = room_code.upper()
        
        if room_code not in self._room_code_mapping:
            return []
        
        room_id = self._room_code_mapping[room_code]
        return self._pending_approvals.get(room_id, [])
    
    def list_user_rooms(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all rooms created by or accessible to a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of room information dictionaries
        """
        user_rooms = []
        
        for room_info in self._active_rooms.values():
            # Include if user is creator
            if room_info.creator_id == user_id:
                room_dict = room_info.to_dict()
                room_dict['user_role'] = 'creator'
                user_rooms.append(room_dict)
                continue
            
            # Include if user is currently in the room
            for spectator in room_info.spectator_list:
                if spectator['user_id'] == user_id:
                    room_dict = room_info.to_dict()
                    room_dict['user_role'] = 'spectator'
                    user_rooms.append(room_dict)
                    break
        
        return user_rooms
    
    async def close_room(self, room_code: str, closer_id: str) -> None:
        """
        Close a spectator room.
        
        Args:
            room_code: Room code to close
            closer_id: ID of the user closing the room
            
        Raises:
            ValueError: If user doesn't have permission to close the room
        """
        room_code = room_code.upper()
        
        if room_code not in self._room_code_mapping:
            return
        
        room_id = self._room_code_mapping[room_code]
        room_info = self._active_rooms[room_id]
        
        # Check permission (only creator can close)
        if closer_id != room_info.creator_id:
            raise ValueError("Only the room creator can close the room")
        
        # Update room status
        room_info.room_status = 'ended'
        
        # Clean up
        del self._room_code_mapping[room_code]
        del self._active_rooms[room_id]
        
        if room_id in self._pending_approvals:
            del self._pending_approvals[room_id]
        
        logger.info(f"Closed spectator room {room_code}")
    
    async def cleanup(self) -> None:
        """Clean up all resources and close all rooms."""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all data
        self._active_rooms.clear()
        self._room_code_mapping.clear()
        self._pending_approvals.clear()
        
        logger.info("SpectatorRoomManager cleanup completed")
    
    def _generate_room_id(self) -> str:
        """Generate a unique room ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _generate_room_code(self) -> str:
        """Generate a unique room code."""
        # Generate a 6-character alphanumeric code
        chars = string.ascii_uppercase + string.digits
        # Exclude confusing characters
        chars = chars.replace('0', '').replace('O', '').replace('1', '').replace('I', '')
        
        while True:
            code = ''.join(secrets.choice(chars) for _ in range(6))
            if code not in self._room_code_mapping:
                return code
    
    def _check_access_permission(
        self,
        room_info: SpectatorRoomInfo,
        user_id: str,
        password: Optional[str]
    ) -> Dict[str, Any]:
        """
        Check if a user has permission to access a room.
        
        Args:
            room_info: Room information
            user_id: ID of the user requesting access
            password: Password provided by user
            
        Returns:
            Dictionary with access result
        """
        access_control = room_info.access_control
        
        # Check if user is blocked
        if user_id in access_control.blocked_users:
            return {'status': 'denied', 'reason': 'User is blocked from this room'}
        
        # Check access level
        if access_control.access_level == AccessLevel.PRIVATE:
            if user_id != room_info.creator_id and user_id not in access_control.allowed_users:
                return {'status': 'denied', 'reason': 'Room is private'}
        
        elif access_control.access_level == AccessLevel.INVITE_ONLY:
            if user_id not in access_control.allowed_users:
                return {'status': 'denied', 'reason': 'Room is invite-only'}
        
        elif access_control.access_level == AccessLevel.PASSWORD:
            if not password or password != access_control.password:
                return {'status': 'denied', 'reason': 'Invalid password'}
        
        # Check if approval is required
        if access_control.require_approval and user_id != room_info.creator_id:
            return {'status': 'pending_approval', 'reason': 'Approval required'}
        
        return {'status': 'approved', 'reason': 'Access granted'}
    
    async def _auto_approve_timeout(
        self,
        room_id: str,
        user_id: str,
        timeout_seconds: int
    ) -> None:
        """Auto-approve a pending request after timeout."""
        await asyncio.sleep(timeout_seconds)
        
        # Check if request is still pending
        if room_id in self._pending_approvals:
            for request in self._pending_approvals[room_id]:
                if request['user_id'] == user_id:
                    # Auto-approve
                    room_info = self._active_rooms[room_id]
                    
                    if room_info.current_spectators < room_info.access_control.max_spectators:
                        # Remove from pending
                        self._pending_approvals[room_id].remove(request)
                        
                        # Add to room
                        spectator_info = {
                            'user_id': user_id,
                            'user_name': request['user_name'],
                            'joined_at': datetime.now().isoformat(),
                            'user_metadata': request['user_metadata'],
                            'auto_approved': True
                        }
                        
                        room_info.spectator_list.append(spectator_info)
                        room_info.current_spectators += 1
                        
                        logger.info(f"Auto-approved join request for {user_id} in room {room_info.room_code}")
                    
                    break
    
    async def _cleanup_expired_rooms(self) -> None:
        """Background task to clean up expired rooms."""
        while True:
            try:
                current_time = datetime.now()
                expired_rooms = []
                
                for room_id, room_info in self._active_rooms.items():
                    if current_time > room_info.expires_at:
                        expired_rooms.append((room_id, room_info.room_code))
                
                for room_id, room_code in expired_rooms:
                    logger.info(f"Cleaning up expired spectator room {room_code}")
                    
                    # Clean up
                    del self._room_code_mapping[room_code]
                    del self._active_rooms[room_id]
                    
                    if room_id in self._pending_approvals:
                        del self._pending_approvals[room_id]
                
                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in room cleanup: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying