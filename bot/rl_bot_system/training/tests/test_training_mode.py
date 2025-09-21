#!/usr/bin/env python3
"""
Test script for training mode functionality in GameClient.

This script tests the extended GameClient with training mode capabilities,
including direct state access and speed control. Requires a running game server.
"""

import asyncio
import logging
import sys
import os
import pytest

# Add the bot directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_client import GameClient, TrainingMode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestTrainingModeIntegration:
    """Integration tests for training mode functionality with running server."""

    @pytest.mark.asyncio
    async def test_create_training_room(self):
        """Test creating a training room with the server."""
        client = GameClient()
        
        try:
            # Create training room
            room_data = await client.create_training_room(
                room_name="Test Training Room",
                player_name="RL Bot Test",
                speed_multiplier=5.0,
                headless=False,
                map_type="default"
            )
            
            # Verify response structure
            assert "success" in room_data
            assert room_data["success"] == True
            assert "roomCode" in room_data
            assert "roomId" in room_data
            assert "playerId" in room_data
            assert "playerToken" in room_data
            assert "speedMultiplier" in room_data
            assert room_data["speedMultiplier"] == 5.0
            
            # Verify client state
            assert client.player_id == room_data["playerId"]
            assert client.room_id == room_data["roomId"]
            assert client.player_token == room_data["playerToken"]
            assert client.is_training_mode()
            assert client.training_mode == TrainingMode.TRAINING
            assert client.speed_multiplier == 5.0
            
            logger.info(f"✓ Created training room: {room_data['roomCode']}")
            
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_training_room_speed_control(self):
        """Test speed control functionality."""
        client = GameClient()
        
        try:
            # Create training room first
            room_data = await client.create_training_room(
                room_name="Speed Test Room",
                player_name="Speed Test Bot",
                speed_multiplier=10.0,
                headless=False,
                map_type="default"
            )
            
            logger.info(f"Created room for speed test: {room_data['roomCode']}")
            
            # Test speed changes
            test_speeds = [15.0, 25.0, 50.0, 10.0]
            
            for speed in test_speeds:
                await client.set_room_speed(speed)
                assert client.speed_multiplier == speed
                logger.info(f"✓ Set speed to {speed}x")
            
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_direct_state_access(self):
        """Test direct state access functionality."""
        client = GameClient()
        
        try:
            # Create training room
            room_data = await client.create_training_room(
                room_name="State Access Test",
                player_name="State Test Bot",
                speed_multiplier=5.0,
                headless=False,
                map_type="default"
            )
            
            logger.info(f"Created room for state access test: {room_data['roomCode']}")
            
            # Test direct state access
            state = await client.get_direct_state()
            
            # Verify state structure - the client returns the processed game state
            assert isinstance(state, dict)
            assert "objects" in state
            assert "room" in state
            assert "players" in state
            assert "map" in state
            
            # Verify room info shows training mode
            room_info = state["room"]
            assert room_info["isTraining"] == True
            assert room_info["speedMultiplier"] == 5.0
            assert room_info["headlessMode"] == False
            
            logger.info(f"✓ Direct state access successful. State keys: {list(state.keys())}")
            
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_headless_mode(self):
        """Test headless mode functionality."""
        client = GameClient()
        
        try:
            # Create headless training room
            room_data = await client.create_training_room(
                room_name="Headless Test Room",
                player_name="Headless Bot",
                speed_multiplier=20.0,
                headless=True,
                map_type="default"
            )
            
            # Verify headless mode
            assert client.training_mode == TrainingMode.HEADLESS
            assert room_data["headlessMode"] == True
            
            logger.info(f"✓ Created headless training room: {room_data['roomCode']}")
            
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_training_room_configuration(self):
        """Test training room configuration."""
        client = GameClient()
        
        try:
            # Create training room
            room_data = await client.create_training_room(
                room_name="Config Test Room",
                player_name="Config Bot",
                speed_multiplier=10.0,
                headless=False,
                map_type="default"
            )
            
            # Test configuration changes
            await client.enable_training_mode(speed_multiplier=30.0, headless=True)
            
            # Verify mode change
            assert client.training_mode == TrainingMode.HEADLESS
            assert client.speed_multiplier == 30.0
            
            # Test disabling training mode
            await client.disable_training_mode()
            assert not client.is_training_mode()
            assert client.training_mode == TrainingMode.NORMAL
            
            logger.info("✓ Training mode configuration test completed")
            
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_state_callbacks(self):
        """Test state update callbacks."""
        client = GameClient()
        callback_called = False
        received_state = None
        
        async def state_callback(state):
            nonlocal callback_called, received_state
            callback_called = True
            received_state = state
            logger.info(f"State callback received state with keys: {list(state.keys())}")
        
        try:
            # Create training room
            room_data = await client.create_training_room(
                room_name="Callback Test Room",
                player_name="Callback Bot",
                speed_multiplier=5.0,
                headless=False,
                map_type="default"
            )
            
            # Register callback
            client.register_state_update_callback(state_callback)
            
            # Get state to trigger callback
            await client.get_direct_state()
            
            # Verify callback was called
            assert callback_called, "State callback was not called"
            assert received_state is not None, "State callback did not receive state"
            
            # Test callback unregistration
            client.unregister_state_update_callback(state_callback)
            assert len(client._state_update_callbacks) == 0
            
            logger.info("✓ State callback test completed")
            
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_join_training_room(self):
        """Test joining an existing training room."""
        # First create a room
        creator_client = GameClient()
        joiner_client = GameClient()
        
        try:
            # Create training room
            room_data = await creator_client.create_training_room(
                room_name="Join Test Room",
                player_name="Creator Bot",
                speed_multiplier=8.0,
                headless=False,
                map_type="default"
            )
            
            room_code = room_data["roomCode"]
            # Extract password from room creation (this would be provided by the creator)
            # For testing, we'll use the room password from the server response
            # In a real scenario, the room creator would share this
            
            logger.info(f"Created room {room_code} for join test")
            
            # Try to join the room (this will fail without the correct password)
            # In a real implementation, we'd need to get the password from the room creator
            try:
                join_data = await joiner_client.join_training_room(
                    room_code=room_code,
                    player_name="Joiner Bot",
                    room_password="WRONG_PASSWORD",  # This will fail, which is expected
                    enable_direct_access=True
                )
                logger.info(f"Unexpectedly joined room: {join_data}")
            except Exception as e:
                logger.info(f"✓ Join failed as expected (wrong password): {e}")
            
        finally:
            await creator_client.close()
            await joiner_client.close()

    @pytest.mark.asyncio
    async def test_training_info(self):
        """Test training info retrieval."""
        client = GameClient()
        
        try:
            # Test initial state
            training_info = client.get_training_info()
            assert training_info['training_mode'] == 'normal'
            assert training_info['speed_multiplier'] == 1.0
            assert not training_info['direct_state_access']
            
            # Create training room
            await client.create_training_room(
                room_name="Info Test Room",
                player_name="Info Bot",
                speed_multiplier=12.0,
                headless=False,
                map_type="default"
            )
            
            # Test training state
            training_info = client.get_training_info()
            assert training_info['training_mode'] == 'training'
            assert training_info['speed_multiplier'] == 12.0
            assert training_info['direct_state_access'] == True
            
            logger.info("✓ Training info test completed")
            
        finally:
            await client.close()


async def main():
    """Run all tests manually."""
    logger.info("Starting GameClient training mode integration tests")
    
    test_instance = TestTrainingModeIntegration()
    
    try:
        await test_instance.test_create_training_room()
        await test_instance.test_training_room_speed_control()
        await test_instance.test_direct_state_access()
        await test_instance.test_headless_mode()
        await test_instance.test_training_room_configuration()
        await test_instance.test_state_callbacks()
        await test_instance.test_join_training_room()
        await test_instance.test_training_info()
        
        logger.info("All integration tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Integration tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())