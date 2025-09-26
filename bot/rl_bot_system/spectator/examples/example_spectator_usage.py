"""
Example usage of the spectator system for RL bot training observation.

This example demonstrates how to:
1. Create spectator sessions for training observation
2. Set up room access control and management
3. Handle training metrics overlay and visualization
4. Manage spectator connections and real-time updates
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from rl_bot_system.spectator import (
    SpectatorManager, SpectatorSession, SpectatorMode,
    TrainingMetricsOverlay, MetricsData,
    SpectatorRoomManager, RoomAccessControl, AccessLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_spectator_session():
    """
    Example: Create a basic spectator session for training observation.
    """
    print("\n=== Basic Spectator Session Example ===")
    
    # Create spectator manager
    spectator_manager = SpectatorManager(game_server_url="http://localhost:4000")
    
    try:
        # Create a spectator session for a training session
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_dqn_gen_3",
            spectator_mode=SpectatorMode.LIVE_TRAINING,
            max_spectators=5,
            session_duration_hours=2,
            password_protected=False,
            enable_metrics_overlay=True,
            enable_performance_graphs=True,
            enable_decision_visualization=True
        )
        
        print(f"Created spectator session: {session.session_id}")
        print(f"Room code: {session.room_code}")
        print(f"Training session: {session.training_session_id}")
        print(f"Max spectators: {session.max_spectators}")
        print(f"Expires at: {session.expires_at}")
        
        # Get session info
        session_info = spectator_manager.get_session_info(session.session_id)
        print(f"Session info: {session_info}")
        
        # List active sessions
        active_sessions = spectator_manager.list_active_sessions()
        print(f"Active sessions: {len(active_sessions)}")
        
    finally:
        await spectator_manager.cleanup()


async def example_password_protected_session():
    """
    Example: Create a password-protected spectator session.
    """
    print("\n=== Password-Protected Session Example ===")
    
    spectator_manager = SpectatorManager()
    
    try:
        # Create password-protected session
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_ppo_gen_5",
            spectator_mode=SpectatorMode.LIVE_TRAINING,
            max_spectators=3,
            password_protected=True,  # Will generate a password
            enable_metrics_overlay=True
        )
        
        print(f"Created password-protected session: {session.session_id}")
        print(f"Room code: {session.room_code}")
        print(f"Password: {session.room_password}")
        
        # Simulate joining with correct password
        print("\nSimulating spectator join...")
        # Note: In real usage, this would use actual GameClient
        # connection_info = await spectator_manager.join_spectator_session(
        #     session_id=session.session_id,
        #     spectator_name="TestSpectator",
        #     password=session.room_password
        # )
        
    finally:
        await spectator_manager.cleanup()


async def example_training_metrics_overlay():
    """
    Example: Demonstrate training metrics overlay functionality.
    """
    print("\n=== Training Metrics Overlay Example ===")
    
    # Create metrics overlay
    overlay = TrainingMetricsOverlay(
        session_id="test_session",
        enable_graphs=True,
        enable_decision_viz=True
    )
    
    try:
        # Create sample metrics data
        metrics_data = MetricsData(
            timestamp=datetime.now(),
            episode=150,
            total_episodes=1000,
            current_reward=85.5,
            average_reward=72.3,
            best_reward=120.0,
            episode_length=450,
            win_rate=0.68,
            loss_value=0.045,
            learning_rate=0.0005,
            epsilon=0.15,
            model_generation=3,
            algorithm="DQN",
            training_time_elapsed=2700.0,
            action_probabilities={
                "move_left": 0.25,
                "move_right": 0.35,
                "jump": 0.15,
                "shoot": 0.25
            },
            state_values=0.82,
            selected_action="move_right",
            actions_per_second=12.5,
            frames_per_second=60.0,
            memory_usage_mb=512.0
        )
        
        print("Sample metrics data:")
        print(f"  Episode: {metrics_data.episode}/{metrics_data.total_episodes}")
        print(f"  Current reward: {metrics_data.current_reward}")
        print(f"  Win rate: {metrics_data.win_rate:.2%}")
        print(f"  Algorithm: {metrics_data.algorithm}")
        print(f"  Generation: {metrics_data.model_generation}")
        
        # Update metrics (would normally broadcast to spectators)
        await overlay.update_metrics(metrics_data)
        
        # Send bot decision data
        await overlay.send_bot_decision_data(
            action_probabilities=metrics_data.action_probabilities,
            state_values=metrics_data.state_values,
            selected_action=metrics_data.selected_action
        )
        
        # Get metrics history
        history = overlay.get_metrics_history(max_points=10)
        print(f"Metrics history length: {len(history)}")
        
        # Get current metrics
        current = overlay.get_current_metrics()
        print(f"Current metrics episode: {current.episode if current else 'None'}")
        
    finally:
        await overlay.cleanup()


async def example_room_manager():
    """
    Example: Demonstrate room manager functionality with access control.
    """
    print("\n=== Room Manager Example ===")
    
    room_manager = SpectatorRoomManager()
    
    try:
        # Create public room
        public_access = RoomAccessControl(
            access_level=AccessLevel.PUBLIC,
            max_spectators=10
        )
        
        public_room = await room_manager.create_spectator_room(
            training_session_id="training_public",
            creator_id="creator_123",
            access_control=public_access,
            room_duration_hours=4
        )
        
        print(f"Created public room: {public_room.room_code}")
        
        # Create password-protected room
        password_access = RoomAccessControl(
            access_level=AccessLevel.PASSWORD,
            password="secret123",
            max_spectators=5
        )
        
        password_room = await room_manager.create_spectator_room(
            training_session_id="training_password",
            creator_id="creator_456",
            access_control=password_access,
            custom_room_code="SECURE"
        )
        
        print(f"Created password room: {password_room.room_code}")
        
        # Create private room with allowed users
        private_access = RoomAccessControl(
            access_level=AccessLevel.PRIVATE,
            allowed_users={"user1", "user2", "user3"},
            max_spectators=3
        )
        
        private_room = await room_manager.create_spectator_room(
            training_session_id="training_private",
            creator_id="creator_789",
            access_control=private_access
        )
        
        print(f"Created private room: {private_room.room_code}")
        
        # Simulate joining public room
        join_result = await room_manager.join_room_request(
            room_code=public_room.room_code,
            user_id="spectator_1",
            user_name="Public Spectator",
            user_metadata={"role": "observer"}
        )
        
        print(f"Join public room result: {join_result['status']}")
        
        # Simulate joining password room with correct password
        join_result = await room_manager.join_room_request(
            room_code=password_room.room_code,
            user_id="spectator_2",
            user_name="Password Spectator",
            password="secret123"
        )
        
        print(f"Join password room result: {join_result['status']}")
        
        # Try joining private room as allowed user
        join_result = await room_manager.join_room_request(
            room_code=private_room.room_code,
            user_id="user1",
            user_name="Allowed User"
        )
        
        print(f"Join private room result: {join_result['status']}")
        
        # Get room information
        room_info = room_manager.get_room_info(public_room.room_code)
        print(f"Public room spectators: {room_info['current_spectators']}")
        
        # List rooms for a user
        user_rooms = room_manager.list_user_rooms("creator_123")
        print(f"Rooms for creator_123: {len(user_rooms)}")
        
    finally:
        await room_manager.cleanup()


async def example_approval_workflow():
    """
    Example: Demonstrate approval workflow for spectator access.
    """
    print("\n=== Approval Workflow Example ===")
    
    room_manager = SpectatorRoomManager()
    
    try:
        # Create room requiring approval
        approval_access = RoomAccessControl(
            access_level=AccessLevel.PUBLIC,
            require_approval=True,
            auto_approve_timeout=30,  # Auto-approve after 30 seconds
            max_spectators=5
        )
        
        room_info = await room_manager.create_spectator_room(
            training_session_id="training_approval",
            creator_id="room_creator",
            access_control=approval_access
        )
        
        print(f"Created approval-required room: {room_info.room_code}")
        
        # Submit join request
        join_result = await room_manager.join_room_request(
            room_code=room_info.room_code,
            user_id="pending_user",
            user_name="Pending Spectator",
            user_metadata={"reason": "Want to observe training"}
        )
        
        print(f"Join request status: {join_result['status']}")
        
        # Check pending approvals
        pending = room_manager.get_pending_approvals(room_info.room_code)
        print(f"Pending approvals: {len(pending)}")
        
        if pending:
            print(f"Pending user: {pending[0]['user_name']}")
            print(f"Requested at: {pending[0]['requested_at']}")
        
        # Approve the request
        approval_result = await room_manager.approve_join_request(
            room_code=room_info.room_code,
            user_id="pending_user",
            approver_id="room_creator",
            approved=True
        )
        
        print(f"Approval result: {approval_result['status']}")
        
        # Check room status after approval
        updated_info = room_manager.get_room_info(room_info.room_code)
        print(f"Room spectators after approval: {updated_info['current_spectators']}")
        
    finally:
        await room_manager.cleanup()


async def example_metrics_callback():
    """
    Example: Demonstrate metrics callback functionality.
    """
    print("\n=== Metrics Callback Example ===")
    
    spectator_manager = SpectatorManager()
    
    try:
        # Create session
        session = await spectator_manager.create_spectator_session(
            training_session_id="training_callback",
            enable_metrics_overlay=True
        )
        
        print(f"Created session for callback demo: {session.session_id}")
        
        # Define callback function
        async def training_progress_callback(metrics_data: MetricsData):
            """Callback to handle training progress updates."""
            print(f"Training Progress Update:")
            print(f"  Episode: {metrics_data.episode}")
            print(f"  Reward: {metrics_data.current_reward:.2f}")
            print(f"  Win Rate: {metrics_data.win_rate:.2%}")
            
            # Could save to database, send notifications, etc.
            if metrics_data.episode % 100 == 0:
                print(f"  Milestone reached: Episode {metrics_data.episode}")
        
        # Register callback
        spectator_manager.register_metrics_callback(
            session_id=session.session_id,
            callback=training_progress_callback
        )
        
        # Simulate training progress updates
        for episode in range(1, 6):
            metrics_data = MetricsData(
                timestamp=datetime.now(),
                episode=episode * 50,
                total_episodes=1000,
                current_reward=50.0 + episode * 10,
                average_reward=45.0 + episode * 8,
                best_reward=100.0,
                episode_length=300,
                win_rate=0.5 + episode * 0.05,
                loss_value=0.1 - episode * 0.01,
                learning_rate=0.001,
                epsilon=0.5 - episode * 0.05,
                model_generation=1,
                algorithm="DQN",
                training_time_elapsed=episode * 300.0
            )
            
            # Update metrics (will trigger callback)
            await spectator_manager.update_training_metrics(
                session_id=session.session_id,
                metrics_data=metrics_data
            )
            
            # Small delay to simulate real training
            await asyncio.sleep(0.1)
        
    finally:
        await spectator_manager.cleanup()


async def main():
    """Run all examples."""
    print("Spectator System Examples")
    print("=" * 50)
    
    try:
        await example_basic_spectator_session()
        await example_password_protected_session()
        await example_training_metrics_overlay()
        await example_room_manager()
        await example_approval_workflow()
        await example_metrics_callback()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())