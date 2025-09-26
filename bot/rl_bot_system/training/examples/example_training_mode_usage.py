#!/usr/bin/env python3
"""
Example usage of training mode functionality with GameClient.

This example demonstrates how to use the extended GameClient with training mode
capabilities for RL bot training.
"""

import asyncio
import logging
import sys
import os

from core.game_client import GameClient, TrainingMode
from rl_bot_system.environment.game_environment import GameEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def example_training_room_creation():
    """Example: Create a training room with speed control."""
    logger.info("Example: Creating training room")
    
    client = GameClient()
    
    try:
        # Create a training room with 10x speed
        room_data = await client.create_training_room(
            room_name="RL Training Room",
            player_name="RL Bot v1.0",
            speed_multiplier=10.0,
            headless=False,  # Keep visual feedback for this example
            map_type="default"
        )
        
        logger.info(f"Created training room: {room_data['roomCode']}")
        logger.info(f"Speed multiplier: {room_data['speedMultiplier']}x")
        
        # The client is now in training mode
        training_info = client.get_training_info()
        logger.info(f"Training mode: {training_info['training_mode']}")
        logger.info(f"Direct state access: {training_info['direct_state_access']}")
        
        return room_data
        
    except Exception as e:
        logger.error(f"Failed to create training room: {e}")
        return None
    finally:
        await client.close()

async def example_speed_control():
    """Example: Control training room speed dynamically."""
    logger.info("Example: Dynamic speed control")
    
    client = GameClient()
    
    try:
        # Enable training mode manually
        await client.enable_training_mode(speed_multiplier=5.0, headless=False)
        
        # Simulate having a room (in real usage, you'd create or join a room first)
        client.room_id = "example_room"
        client.player_token = "example_token"
        
        # Demonstrate speed changes during training
        speeds = [5.0, 10.0, 25.0, 50.0]
        
        for speed in speeds:
            logger.info(f"Setting speed to {speed}x")
            try:
                await client.set_room_speed(speed)
                logger.info(f"Speed successfully set to {speed}x")
            except Exception as e:
                logger.warning(f"Speed control failed (expected without server): {e}")
        
        # Switch to headless mode for maximum speed
        await client.enable_training_mode(speed_multiplier=100.0, headless=True)
        logger.info("Switched to headless mode with 100x speed")
        
    finally:
        await client.close()

async def example_direct_state_access():
    """Example: Use direct state access for faster training."""
    logger.info("Example: Direct state access")
    
    client = GameClient()
    
    try:
        # Enable training mode with direct state access
        await client.enable_training_mode(speed_multiplier=20.0, headless=False)
        
        # Simulate having a room
        client.room_id = "example_room"
        client.player_token = "example_token"
        
        # Set up state update callback
        async def on_state_update(state):
            logger.info(f"Received state update with {len(state)} keys")
            if 'objects' in state:
                logger.info(f"Game objects: {len(state['objects'])}")
        
        client.register_state_update_callback(on_state_update)
        
        # In a real training loop, you would call this repeatedly
        try:
            state = await client.get_direct_state()
            logger.info("Direct state access successful")
        except Exception as e:
            logger.warning(f"Direct state access failed (expected without server): {e}")
        
    finally:
        await client.close()

async def example_training_with_environment():
    """Example: Use training mode with GameEnvironment."""
    logger.info("Example: Training mode with GameEnvironment")
    
    # Create a mock client for this example
    from unittest.mock import Mock, AsyncMock
    
    mock_client = Mock()
    mock_client.is_training_mode.return_value = True
    mock_client.training_mode = TrainingMode.TRAINING
    mock_client.register_message_handler = Mock()
    mock_client.send_keyboard_input = AsyncMock()
    mock_client.send_mouse_input = AsyncMock()
    
    # Create environment with training mode
    env = GameEnvironment(
        game_client=mock_client,
        training_mode=TrainingMode.TRAINING,
        max_episode_steps=1000
    )
    
    logger.info(f"Environment training mode: {env.training_mode.value}")
    
    # Switch to evaluation mode
    env.set_training_mode(TrainingMode.EVALUATION)
    logger.info(f"Switched to evaluation mode: {env.training_mode.value}")

async def example_training_session_workflow():
    """Example: Complete training session workflow."""
    logger.info("Example: Complete training session workflow")
    
    client = GameClient()
    
    try:
        # Step 1: Create training room
        logger.info("Step 1: Creating training room...")
        try:
            room_data = await client.create_training_room(
                room_name="RL Training Session",
                player_name="RL Bot",
                speed_multiplier=15.0,
                headless=False
            )
            logger.info(f"✓ Training room created: {room_data.get('roomCode', 'N/A')}")
        except Exception as e:
            logger.warning(f"✗ Room creation failed (expected without server): {e}")
            # Continue with mock setup for demonstration
            client.room_id = "mock_room"
            client.player_token = "mock_token"
            await client.enable_training_mode(speed_multiplier=15.0)
        
        # Step 2: Configure training parameters
        logger.info("Step 2: Configuring training parameters...")
        training_info = client.get_training_info()
        logger.info(f"✓ Training mode: {training_info['training_mode']}")
        logger.info(f"✓ Speed multiplier: {training_info['speed_multiplier']}x")
        logger.info(f"✓ Direct state access: {training_info['direct_state_access']}")
        
        # Step 3: Set up state monitoring
        logger.info("Step 3: Setting up state monitoring...")
        
        state_updates = 0
        async def training_state_callback(state):
            nonlocal state_updates
            state_updates += 1
            if state_updates % 100 == 0:  # Log every 100 updates
                logger.info(f"Processed {state_updates} state updates")
        
        client.register_state_update_callback(training_state_callback)
        logger.info("✓ State monitoring configured")
        
        # Step 4: Simulate training episodes
        logger.info("Step 4: Simulating training episodes...")
        
        for episode in range(3):
            logger.info(f"Episode {episode + 1}/3")
            
            # Simulate getting state and taking actions
            try:
                state = await client.get_direct_state()
                logger.info("  ✓ Got game state")
            except Exception:
                logger.info("  ✓ State access simulated (server not running)")
            
            # Simulate some training steps
            await asyncio.sleep(0.1)  # Simulate training time
            
            # Adjust speed based on training progress
            new_speed = 15.0 + (episode * 10.0)  # Increase speed each episode
            try:
                await client.set_room_speed(new_speed)
                logger.info(f"  ✓ Adjusted speed to {new_speed}x")
            except Exception:
                logger.info(f"  ✓ Speed adjustment simulated: {new_speed}x")
        
        # Step 5: Switch to headless mode for final training
        logger.info("Step 5: Switching to headless mode for intensive training...")
        await client.enable_training_mode(speed_multiplier=50.0, headless=True)
        
        training_info = client.get_training_info()
        logger.info(f"✓ Final training mode: {training_info['training_mode']}")
        logger.info(f"✓ Final speed: {training_info['speed_multiplier']}x")
        
        logger.info("Training session workflow completed successfully!")
        
    finally:
        await client.close()

async def main():
    """Run all examples."""
    logger.info("Starting GameClient training mode examples")
    
    try:
        await example_training_room_creation()
        await example_speed_control()
        await example_direct_state_access()
        await example_training_with_environment()
        await example_training_session_workflow()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())