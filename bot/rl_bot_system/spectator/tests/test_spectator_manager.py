"""
Tests for SpectatorManager functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from bot.rl_bot_system.spectator.spectator_manager import (
    SpectatorManager, SpectatorSession, SpectatorMode
)
from bot.rl_bot_system.spectator.training_metrics_overlay import MetricsData


@pytest.fixture
async def spectator_manager():
    """Create a SpectatorManager instance for testing."""
    manager = SpectatorManager(game_server_url="http://test-server:4000")
    yield manager
    await manager.cleanup()


class TestSpectatorManager:
    """Test cases for SpectatorManager."""
    
    @pytest.fixture
    def sample_metrics_data(self):
        """Create sample metrics data for testing."""
        return MetricsData(
            timestamp=datetime.now(),
            episode=100,
            total_episodes=1000,
            current_reward=150.5,
            average_reward=120.3,
            best_reward=200.0,
            episode_length=500,
            win_rate=0.75,
            loss_value=0.05,
            learning_rate=0.001,
            epsilon=0.1,
            model_generation=2,
            algorithm="DQN",
            training_time_elapsed=3600.0,
            action_probabilities={"move_left": 0.3, "move_right": 0.4, "shoot": 0.3},
            state_values=0.85,
            selected_action="move_right"
        )
    
    @pytest.mark.asyncio
    async def test_create_spectator_session(self, spectator_manager):
        """Test creating a spectator session."""
        training_session_id = "training_123"
        
        session = await spectator_manager.create_spectator_session(
            training_session_id=training_session_id,
            spectator_mode=SpectatorMode.LIVE_TRAINING,
            max_spectators=5,
            password_protected=True,
            enable_metrics_overlay=True
        )
        
        assert session.training_session_id == training_session_id
        assert session.spectator_mode == SpectatorMode.LIVE_TRAINING
        assert session.max_spectators == 5
        assert session.current_spectators == 0
        assert session.room_password is not None
        assert session.metrics_overlay is True
        assert len(session.room_code) == 6
        
        # Check that session is stored
        assert session.session_id in spectator_manager._active_sessions
        assert session.session_id in spectator_manager._session_clients
    
    @pytest.mark.asyncio
    async def test_create_session_without_password(self, spectator_manager):
        """Test creating a session without password protection."""
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_456",
            password_protected=False
        )
        
        assert session.room_password is None
    
    @pytest.mark.asyncio
    async def test_join_spectator_session_success(self, spectator_manager):
        """Test successfully joining a spectator session."""
        # Create session
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_789",
            password_protected=False
        )
        
        # Mock GameClient
        with patch('bot.rl_bot_system.spectator.spectator_manager.GameClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.connect.return_value = {"success": True}
            
            # Join session
            connection_info = await spectator_manager.join_spectator_session(
                session_id=session.session_id,
                spectator_name="TestSpectator",
                password=None
            )
            
            assert connection_info['session_id'] == session.session_id
            assert connection_info['room_code'] == session.room_code
            assert connection_info['spectator_mode'] == SpectatorMode.LIVE_TRAINING.value
            assert 'client' in connection_info
            
            # Check that spectator count increased
            updated_session = spectator_manager._active_sessions[session.session_id]
            assert updated_session.current_spectators == 1
    
    @pytest.mark.asyncio
    async def test_join_session_with_password(self, spectator_manager):
        """Test joining a password-protected session."""
        # Create password-protected session
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_pwd",
            password_protected=True
        )
        
        # Try to join without password - should fail
        with pytest.raises(ValueError, match="Invalid password"):
            await spectator_manager.join_spectator_session(
                session_id=session.session_id,
                spectator_name="TestSpectator",
                password=None
            )
        
        # Try to join with wrong password - should fail
        with pytest.raises(ValueError, match="Invalid password"):
            await spectator_manager.join_spectator_session(
                session_id=session.session_id,
                spectator_name="TestSpectator",
                password="wrong_password"
            )
        
        # Join with correct password - should succeed
        with patch('bot.rl_bot_system.spectator.spectator_manager.GameClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.connect.return_value = {"success": True}
            
            connection_info = await spectator_manager.join_spectator_session(
                session_id=session.session_id,
                spectator_name="TestSpectator",
                password=session.room_password
            )
            
            assert connection_info['session_id'] == session.session_id
    
    @pytest.mark.asyncio
    async def test_join_nonexistent_session(self, spectator_manager):
        """Test joining a session that doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            await spectator_manager.join_spectator_session(
                session_id="nonexistent_session",
                spectator_name="TestSpectator"
            )
    
    @pytest.mark.asyncio
    async def test_join_full_session(self, spectator_manager):
        """Test joining a session that is at capacity."""
        # Create session with max 1 spectator
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_full",
            max_spectators=1,
            password_protected=False
        )
        
        with patch('bot.rl_bot_system.spectator.spectator_manager.GameClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.connect.return_value = {"success": True}
            
            # First spectator should succeed
            await spectator_manager.join_spectator_session(
                session_id=session.session_id,
                spectator_name="Spectator1"
            )
            
            # Second spectator should fail
            with pytest.raises(RuntimeError, match="is full"):
                await spectator_manager.join_spectator_session(
                    session_id=session.session_id,
                    spectator_name="Spectator2"
                )
    
    @pytest.mark.asyncio
    async def test_leave_spectator_session(self, spectator_manager):
        """Test leaving a spectator session."""
        # Create and join session
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_leave",
            password_protected=False
        )
        
        with patch('bot.rl_bot_system.spectator.spectator_manager.GameClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.connect.return_value = {"success": True}
            
            connection_info = await spectator_manager.join_spectator_session(
                session_id=session.session_id,
                spectator_name="TestSpectator"
            )
            
            client = connection_info['client']
            
            # Verify spectator is in session
            updated_session = spectator_manager._active_sessions[session.session_id]
            assert updated_session.current_spectators == 1
            
            # Leave session
            await spectator_manager.leave_spectator_session(
                session_id=session.session_id,
                client=client
            )
            
            # Verify spectator count decreased
            updated_session = spectator_manager._active_sessions[session.session_id]
            assert updated_session.current_spectators == 0
            
            # Verify client was closed
            mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_training_metrics(self, spectator_manager, sample_metrics_data):
        """Test updating training metrics for a session."""
        # Create session with metrics overlay
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_metrics",
            enable_metrics_overlay=True
        )
        
        # Mock the metrics overlay
        mock_overlay = AsyncMock()
        spectator_manager._metrics_overlays[session.session_id] = mock_overlay
        
        # Update metrics
        await spectator_manager.update_training_metrics(
            session_id=session.session_id,
            metrics_data=sample_metrics_data
        )
        
        # Verify overlay was updated
        mock_overlay.update_metrics.assert_called_once_with(sample_metrics_data)
    
    @pytest.mark.asyncio
    async def test_metrics_callback(self, spectator_manager, sample_metrics_data):
        """Test metrics callback functionality."""
        # Create session
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_callback"
        )
        
        # Register callback
        callback_called = False
        received_metrics = None
        
        async def test_callback(metrics_data):
            nonlocal callback_called, received_metrics
            callback_called = True
            received_metrics = metrics_data
        
        spectator_manager.register_metrics_callback(
            session_id=session.session_id,
            callback=test_callback
        )
        
        # Update metrics
        await spectator_manager.update_training_metrics(
            session_id=session.session_id,
            metrics_data=sample_metrics_data
        )
        
        # Verify callback was called
        assert callback_called
        assert received_metrics == sample_metrics_data
    
    @pytest.mark.asyncio
    async def test_get_session_info(self, spectator_manager):
        """Test getting session information."""
        # Create session
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_info"
        )
        
        # Get session info
        info = spectator_manager.get_session_info(session.session_id)
        
        assert info is not None
        assert info['session_id'] == session.session_id
        assert info['training_session_id'] == "training_info"
        assert info['spectator_mode'] == SpectatorMode.LIVE_TRAINING.value
        
        # Test nonexistent session
        info = spectator_manager.get_session_info("nonexistent")
        assert info is None
    
    @pytest.mark.asyncio
    async def test_list_active_sessions(self, spectator_manager):
        """Test listing active sessions."""
        # Initially no sessions
        sessions = spectator_manager.list_active_sessions()
        assert len(sessions) == 0
        
        # Create sessions
        session1 = await spectator_manager.create_spectator_session(
            training_session_id="training_1"
        )
        session2 = await spectator_manager.create_spectator_session(
            training_session_id="training_2"
        )
        
        # List sessions
        sessions = spectator_manager.list_active_sessions()
        assert len(sessions) == 2
        
        session_ids = [s['session_id'] for s in sessions]
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids
    
    @pytest.mark.asyncio
    async def test_close_session(self, spectator_manager):
        """Test closing a spectator session."""
        # Create session
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_close"
        )
        
        # Verify session exists
        assert session.session_id in spectator_manager._active_sessions
        
        # Close session
        await spectator_manager.close_session(session.session_id)
        
        # Verify session is removed
        assert session.session_id not in spectator_manager._active_sessions
        assert session.session_id not in spectator_manager._session_clients
    
    @pytest.mark.asyncio
    async def test_access_control(self, spectator_manager):
        """Test access control functionality."""
        # Create session with access control
        access_control = {
            'allow_anonymous': False,
            'allowed_users': ['user1', 'user2'],
            'blocked_users': ['blocked_user']
        }
        
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_access",
            access_control=access_control
        )
        
        # Test blocked user
        with pytest.raises(ValueError, match="Access denied"):
            await spectator_manager.join_spectator_session(
                session_id=session.session_id,
                spectator_name="BlockedUser",
                user_id="blocked_user"
            )
        
        # Test anonymous user (should be denied)
        with pytest.raises(ValueError, match="Access denied"):
            await spectator_manager.join_spectator_session(
                session_id=session.session_id,
                spectator_name="Anonymous"
            )
    
    @pytest.mark.asyncio
    async def test_expired_session(self, spectator_manager):
        """Test handling of expired sessions."""
        # Create session that expires immediately
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_expired",
            session_duration_hours=0  # Expires immediately
        )
        
        # Manually set expiration to past
        spectator_manager._active_sessions[session.session_id].expires_at = (
            datetime.now() - timedelta(hours=1)
        )
        
        # Try to join expired session
        with pytest.raises(RuntimeError, match="has expired"):
            await spectator_manager.join_spectator_session(
                session_id=session.session_id,
                spectator_name="TestSpectator"
            )
    
    def test_generate_room_code(self, spectator_manager):
        """Test room code generation."""
        # Generate multiple codes and verify uniqueness
        codes = set()
        for _ in range(100):
            code = spectator_manager._generate_room_code()
            assert len(code) == 6
            assert code.isalnum()
            assert code.isupper()
            codes.add(code)
        
        # Should have generated unique codes
        assert len(codes) == 100
    
    def test_check_access_permission(self, spectator_manager):
        """Test access permission checking."""
        # Create session with various access controls
        session = SpectatorSession(
            session_id="test_session",
            training_session_id="training_test",
            room_code="TEST01",
            room_password=None,
            spectator_mode=SpectatorMode.LIVE_TRAINING,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            max_spectators=10,
            current_spectators=0,
            access_control={
                'allow_anonymous': True,
                'blocked_users': ['blocked_user'],
                'allowed_users': []
            },
            metrics_overlay=True,
            performance_graphs=True,
            decision_visualization=True
        )
        
        # Test normal user
        assert spectator_manager._check_access_permission(session, "normal_user")
        
        # Test blocked user
        assert not spectator_manager._check_access_permission(session, "blocked_user")
        
        # Test anonymous access
        assert spectator_manager._check_access_permission(session, None)