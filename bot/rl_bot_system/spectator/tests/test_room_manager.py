"""
Tests for SpectatorRoomManager functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from rl_bot_system.spectator.room_manager import (
    SpectatorRoomManager, RoomAccessControl, AccessLevel, SpectatorRoomInfo
)


class TestSpectatorRoomManager:
    """Test cases for SpectatorRoomManager."""
    
    @pytest.fixture
    async def room_manager(self):
        """Create a SpectatorRoomManager instance for testing."""
        manager = SpectatorRoomManager(game_server_url="http://test-server:4000")
        yield manager
        await manager.cleanup()
    
    @pytest.fixture
    def public_access_control(self):
        """Create public access control configuration."""
        return RoomAccessControl(
            access_level=AccessLevel.PUBLIC,
            max_spectators=10
        )
    
    @pytest.fixture
    def password_access_control(self):
        """Create password-protected access control configuration."""
        return RoomAccessControl(
            access_level=AccessLevel.PASSWORD,
            password="test123",
            max_spectators=5
        )
    
    @pytest.fixture
    def private_access_control(self):
        """Create private access control configuration."""
        return RoomAccessControl(
            access_level=AccessLevel.PRIVATE,
            allowed_users={"user1", "user2"},
            max_spectators=3
        )
    
    @pytest.mark.asyncio
    async def test_create_public_room(self, room_manager, public_access_control):
        """Test creating a public spectator room."""
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_123",
            creator_id="creator_1",
            access_control=public_access_control,
            room_duration_hours=2
        )
        
        assert room_info.training_session_id == "training_123"
        assert room_info.creator_id == "creator_1"
        assert room_info.access_control.access_level == AccessLevel.PUBLIC
        assert room_info.access_control.max_spectators == 10
        assert room_info.current_spectators == 0
        assert room_info.room_status == 'active'
        assert len(room_info.room_code) == 6
        
        # Check that room is stored
        assert room_info.room_id in room_manager._active_rooms
        assert room_info.room_code in room_manager._room_code_mapping
    
    @pytest.mark.asyncio
    async def test_create_room_with_custom_code(self, room_manager, public_access_control):
        """Test creating a room with a custom room code."""
        custom_code = "CUSTOM"
        
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_456",
            creator_id="creator_2",
            access_control=public_access_control,
            custom_room_code=custom_code
        )
        
        assert room_info.room_code == custom_code
        assert custom_code in room_manager._room_code_mapping
    
    @pytest.mark.asyncio
    async def test_create_room_duplicate_code(self, room_manager, public_access_control):
        """Test creating a room with a duplicate custom code."""
        custom_code = "DUPLIC"
        
        # Create first room
        await room_manager.create_spectator_room(
            training_session_id="training_1",
            creator_id="creator_1",
            access_control=public_access_control,
            custom_room_code=custom_code
        )
        
        # Try to create second room with same code
        with pytest.raises(ValueError, match="already in use"):
            await room_manager.create_spectator_room(
                training_session_id="training_2",
                creator_id="creator_2",
                access_control=public_access_control,
                custom_room_code=custom_code
            )
    
    @pytest.mark.asyncio
    async def test_join_public_room_success(self, room_manager, public_access_control):
        """Test successfully joining a public room."""
        # Create room
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_public",
            creator_id="creator_1",
            access_control=public_access_control
        )
        
        # Join room
        join_result = await room_manager.join_room_request(
            room_code=room_info.room_code,
            user_id="user_1",
            user_name="TestUser",
            user_metadata={"role": "tester"}
        )
        
        assert join_result['status'] == 'approved'
        assert 'room_info' in join_result
        assert 'spectator_info' in join_result
        
        # Check that user was added to room
        updated_room = room_manager._active_rooms[room_info.room_id]
        assert updated_room.current_spectators == 1
        assert len(updated_room.spectator_list) == 1
        assert updated_room.spectator_list[0]['user_id'] == "user_1"
        assert updated_room.spectator_list[0]['user_name'] == "TestUser"
    
    @pytest.mark.asyncio
    async def test_join_password_room_success(self, room_manager, password_access_control):
        """Test successfully joining a password-protected room."""
        # Create password-protected room
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_pwd",
            creator_id="creator_1",
            access_control=password_access_control
        )
        
        # Join with correct password
        join_result = await room_manager.join_room_request(
            room_code=room_info.room_code,
            user_id="user_1",
            user_name="TestUser",
            password="test123"
        )
        
        assert join_result['status'] == 'approved'
        
        # Check that user was added
        updated_room = room_manager._active_rooms[room_info.room_id]
        assert updated_room.current_spectators == 1
    
    @pytest.mark.asyncio
    async def test_join_password_room_wrong_password(self, room_manager, password_access_control):
        """Test joining a password-protected room with wrong password."""
        # Create password-protected room
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_pwd_fail",
            creator_id="creator_1",
            access_control=password_access_control
        )
        
        # Try to join with wrong password
        with pytest.raises(ValueError, match="Invalid password"):
            await room_manager.join_room_request(
                room_code=room_info.room_code,
                user_id="user_1",
                user_name="TestUser",
                password="wrong_password"
            )
    
    @pytest.mark.asyncio
    async def test_join_private_room_allowed_user(self, room_manager, private_access_control):
        """Test joining a private room as an allowed user."""
        # Create private room
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_private",
            creator_id="creator_1",
            access_control=private_access_control
        )
        
        # Join as allowed user
        join_result = await room_manager.join_room_request(
            room_code=room_info.room_code,
            user_id="user1",  # In allowed_users
            user_name="AllowedUser"
        )
        
        assert join_result['status'] == 'approved'
    
    @pytest.mark.asyncio
    async def test_join_private_room_denied_user(self, room_manager, private_access_control):
        """Test joining a private room as a non-allowed user."""
        # Create private room
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_private_deny",
            creator_id="creator_1",
            access_control=private_access_control
        )
        
        # Try to join as non-allowed user
        with pytest.raises(ValueError, match="Room is private"):
            await room_manager.join_room_request(
                room_code=room_info.room_code,
                user_id="unauthorized_user",
                user_name="UnauthorizedUser"
            )
    
    @pytest.mark.asyncio
    async def test_join_nonexistent_room(self, room_manager):
        """Test joining a room that doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            await room_manager.join_room_request(
                room_code="NOROOM",
                user_id="user_1",
                user_name="TestUser"
            )
    
    @pytest.mark.asyncio
    async def test_join_full_room(self, room_manager):
        """Test joining a room that is at capacity."""
        # Create room with max 1 spectator
        access_control = RoomAccessControl(
            access_level=AccessLevel.PUBLIC,
            max_spectators=1
        )
        
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_full",
            creator_id="creator_1",
            access_control=access_control
        )
        
        # First user joins successfully
        await room_manager.join_room_request(
            room_code=room_info.room_code,
            user_id="user_1",
            user_name="FirstUser"
        )
        
        # Second user should be denied
        with pytest.raises(RuntimeError, match="is full"):
            await room_manager.join_room_request(
                room_code=room_info.room_code,
                user_id="user_2",
                user_name="SecondUser"
            )
    
    @pytest.mark.asyncio
    async def test_join_expired_room(self, room_manager, public_access_control):
        """Test joining an expired room."""
        # Create room that expires immediately
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_expired",
            creator_id="creator_1",
            access_control=public_access_control,
            room_duration_hours=0
        )
        
        # Manually set expiration to past
        room_manager._active_rooms[room_info.room_id].expires_at = (
            datetime.now() - timedelta(hours=1)
        )
        
        # Try to join expired room
        with pytest.raises(RuntimeError, match="has expired"):
            await room_manager.join_room_request(
                room_code=room_info.room_code,
                user_id="user_1",
                user_name="TestUser"
            )
    
    @pytest.mark.asyncio
    async def test_leave_room(self, room_manager, public_access_control):
        """Test leaving a spectator room."""
        # Create and join room
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_leave",
            creator_id="creator_1",
            access_control=public_access_control
        )
        
        await room_manager.join_room_request(
            room_code=room_info.room_code,
            user_id="user_1",
            user_name="TestUser"
        )
        
        # Verify user is in room
        updated_room = room_manager._active_rooms[room_info.room_id]
        assert updated_room.current_spectators == 1
        
        # Leave room
        await room_manager.leave_room(
            room_code=room_info.room_code,
            user_id="user_1"
        )
        
        # Verify user is removed
        updated_room = room_manager._active_rooms[room_info.room_id]
        assert updated_room.current_spectators == 0
        assert len(updated_room.spectator_list) == 0
    
    @pytest.mark.asyncio
    async def test_approval_required(self, room_manager):
        """Test room with approval requirement."""
        access_control = RoomAccessControl(
            access_level=AccessLevel.PUBLIC,
            require_approval=True,
            max_spectators=10
        )
        
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_approval",
            creator_id="creator_1",
            access_control=access_control
        )
        
        # Join request should be pending
        join_result = await room_manager.join_room_request(
            room_code=room_info.room_code,
            user_id="user_1",
            user_name="TestUser"
        )
        
        assert join_result['status'] == 'pending_approval'
        
        # User should not be in room yet
        updated_room = room_manager._active_rooms[room_info.room_id]
        assert updated_room.current_spectators == 0
        
        # Should be in pending approvals
        pending = room_manager.get_pending_approvals(room_info.room_code)
        assert len(pending) == 1
        assert pending[0]['user_id'] == "user_1"
    
    @pytest.mark.asyncio
    async def test_approve_join_request(self, room_manager):
        """Test approving a join request."""
        access_control = RoomAccessControl(
            access_level=AccessLevel.PUBLIC,
            require_approval=True,
            max_spectators=10
        )
        
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_approve",
            creator_id="creator_1",
            access_control=access_control
        )
        
        # Submit join request
        await room_manager.join_room_request(
            room_code=room_info.room_code,
            user_id="user_1",
            user_name="TestUser"
        )
        
        # Approve request
        approval_result = await room_manager.approve_join_request(
            room_code=room_info.room_code,
            user_id="user_1",
            approver_id="creator_1",
            approved=True
        )
        
        assert approval_result['status'] == 'approved'
        
        # User should now be in room
        updated_room = room_manager._active_rooms[room_info.room_id]
        assert updated_room.current_spectators == 1
        
        # Should not be in pending approvals
        pending = room_manager.get_pending_approvals(room_info.room_code)
        assert len(pending) == 0
    
    @pytest.mark.asyncio
    async def test_deny_join_request(self, room_manager):
        """Test denying a join request."""
        access_control = RoomAccessControl(
            access_level=AccessLevel.PUBLIC,
            require_approval=True,
            max_spectators=10
        )
        
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_deny",
            creator_id="creator_1",
            access_control=access_control
        )
        
        # Submit join request
        await room_manager.join_room_request(
            room_code=room_info.room_code,
            user_id="user_1",
            user_name="TestUser"
        )
        
        # Deny request
        approval_result = await room_manager.approve_join_request(
            room_code=room_info.room_code,
            user_id="user_1",
            approver_id="creator_1",
            approved=False
        )
        
        assert approval_result['status'] == 'denied'
        
        # User should not be in room
        updated_room = room_manager._active_rooms[room_info.room_id]
        assert updated_room.current_spectators == 0
        
        # Should not be in pending approvals
        pending = room_manager.get_pending_approvals(room_info.room_code)
        assert len(pending) == 0
    
    @pytest.mark.asyncio
    async def test_approve_unauthorized(self, room_manager):
        """Test that only room creator can approve requests."""
        access_control = RoomAccessControl(
            access_level=AccessLevel.PUBLIC,
            require_approval=True,
            max_spectators=10
        )
        
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_unauth",
            creator_id="creator_1",
            access_control=access_control
        )
        
        # Submit join request
        await room_manager.join_room_request(
            room_code=room_info.room_code,
            user_id="user_1",
            user_name="TestUser"
        )
        
        # Try to approve as different user
        with pytest.raises(ValueError, match="Only the room creator"):
            await room_manager.approve_join_request(
                room_code=room_info.room_code,
                user_id="user_1",
                approver_id="unauthorized_user",
                approved=True
            )
    
    def test_get_room_info(self, room_manager, public_access_control):
        """Test getting room information."""
        # Test nonexistent room
        info = room_manager.get_room_info("NOROOM")
        assert info is None
        
        # Create room and get info
        async def create_and_test():
            room_info = await room_manager.create_spectator_room(
                training_session_id="training_info",
                creator_id="creator_1",
                access_control=public_access_control
            )
            
            info = room_manager.get_room_info(room_info.room_code)
            assert info is not None
            assert info['room_code'] == room_info.room_code
            assert info['training_session_id'] == "training_info"
            assert info['creator_id'] == "creator_1"
            assert info['access_level'] == AccessLevel.PUBLIC.value
        
        asyncio.run(create_and_test())
    
    def test_list_user_rooms(self, room_manager, public_access_control):
        """Test listing rooms for a user."""
        async def create_and_test():
            # Create room as creator
            room_info1 = await room_manager.create_spectator_room(
                training_session_id="training_creator",
                creator_id="user_1",
                access_control=public_access_control
            )
            
            # Create another room and join as spectator
            room_info2 = await room_manager.create_spectator_room(
                training_session_id="training_spectator",
                creator_id="other_user",
                access_control=public_access_control
            )
            
            await room_manager.join_room_request(
                room_code=room_info2.room_code,
                user_id="user_1",
                user_name="User1"
            )
            
            # List rooms for user_1
            user_rooms = room_manager.list_user_rooms("user_1")
            assert len(user_rooms) == 2
            
            # Check roles
            roles = [room['user_role'] for room in user_rooms]
            assert 'creator' in roles
            assert 'spectator' in roles
        
        asyncio.run(create_and_test())
    
    @pytest.mark.asyncio
    async def test_close_room(self, room_manager, public_access_control):
        """Test closing a spectator room."""
        # Create room
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_close",
            creator_id="creator_1",
            access_control=public_access_control
        )
        
        # Verify room exists
        assert room_info.room_id in room_manager._active_rooms
        assert room_info.room_code in room_manager._room_code_mapping
        
        # Close room
        await room_manager.close_room(
            room_code=room_info.room_code,
            closer_id="creator_1"
        )
        
        # Verify room is removed
        assert room_info.room_id not in room_manager._active_rooms
        assert room_info.room_code not in room_manager._room_code_mapping
    
    @pytest.mark.asyncio
    async def test_close_room_unauthorized(self, room_manager, public_access_control):
        """Test that only room creator can close room."""
        # Create room
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_close_unauth",
            creator_id="creator_1",
            access_control=public_access_control
        )
        
        # Try to close as different user
        with pytest.raises(ValueError, match="Only the room creator"):
            await room_manager.close_room(
                room_code=room_info.room_code,
                closer_id="unauthorized_user"
            )
    
    def test_generate_room_code(self, room_manager):
        """Test room code generation."""
        # Generate multiple codes and verify properties
        codes = set()
        for _ in range(100):
            code = room_manager._generate_room_code()
            assert len(code) == 6
            assert code.isalnum()
            assert code.isupper()
            # Should not contain confusing characters
            assert '0' not in code
            assert 'O' not in code
            assert '1' not in code
            assert 'I' not in code
            codes.add(code)
        
        # Should generate unique codes
        assert len(codes) == 100
    
    def test_check_access_permission_blocked_user(self, room_manager):
        """Test access permission checking for blocked users."""
        access_control = RoomAccessControl(
            access_level=AccessLevel.PUBLIC,
            blocked_users={"blocked_user"}
        )
        
        room_info = SpectatorRoomInfo(
            room_id="test_room",
            room_code="TEST01",
            training_session_id="training_test",
            creator_id="creator_1",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            access_control=access_control,
            current_spectators=0,
            spectator_list=[],
            room_status='active'
        )
        
        # Blocked user should be denied
        result = room_manager._check_access_permission(room_info, "blocked_user", None)
        assert result['status'] == 'denied'
        assert 'blocked' in result['reason']
        
        # Normal user should be approved
        result = room_manager._check_access_permission(room_info, "normal_user", None)
        assert result['status'] == 'approved'
    
    def test_room_access_control_dataclass(self):
        """Test RoomAccessControl dataclass functionality."""
        # Test with defaults
        access_control = RoomAccessControl(access_level=AccessLevel.PUBLIC)
        
        assert access_control.access_level == AccessLevel.PUBLIC
        assert access_control.password is None
        assert access_control.max_spectators == 10
        assert access_control.allowed_users == set()
        assert access_control.blocked_users == set()
        assert access_control.require_approval is False
        
        # Test with custom values
        access_control = RoomAccessControl(
            access_level=AccessLevel.PRIVATE,
            max_spectators=5,
            allowed_users={"user1", "user2"},
            blocked_users={"blocked"},
            require_approval=True
        )
        
        assert access_control.access_level == AccessLevel.PRIVATE
        assert access_control.max_spectators == 5
        assert "user1" in access_control.allowed_users
        assert "blocked" in access_control.blocked_users
        assert access_control.require_approval is True
    
    def test_spectator_room_info_serialization(self):
        """Test SpectatorRoomInfo serialization."""
        access_control = RoomAccessControl(
            access_level=AccessLevel.PASSWORD,
            password="secret123",
            max_spectators=8
        )
        
        room_info = SpectatorRoomInfo(
            room_id="room_123",
            room_code="ABC123",
            training_session_id="training_456",
            creator_id="creator_1",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=2),
            access_control=access_control,
            current_spectators=3,
            spectator_list=[
                {"user_id": "user1", "user_name": "User One"},
                {"user_id": "user2", "user_name": "User Two"}
            ],
            room_status='active'
        )
        
        data_dict = room_info.to_dict()
        
        assert data_dict['room_id'] == "room_123"
        assert data_dict['room_code'] == "ABC123"
        assert data_dict['training_session_id'] == "training_456"
        assert data_dict['creator_id'] == "creator_1"
        assert data_dict['access_level'] == AccessLevel.PASSWORD.value
        assert data_dict['max_spectators'] == 8
        assert data_dict['current_spectators'] == 3
        assert data_dict['room_status'] == 'active'
        assert data_dict['has_password'] is True
        assert len(data_dict['spectator_list']) == 2