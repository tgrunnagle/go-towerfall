"""
Example usage of the training session management system.

This script demonstrates how to use the TrainingSession, SessionManager,
BatchEpisodeManager, and TrainingAPI components.
"""

import asyncio
import logging
from datetime import datetime

from rl_bot_system.training import (
    TrainingSession,
    TrainingConfig,
    TrainingMode,
    SessionManager,
    ResourceLimits,
    BatchEpisodeManager,
    TrainingAPI
)


async def example_basic_training_session():
    """Example of creating and running a basic training session."""
    print("=== Basic Training Session Example ===")
    
    # Create training configuration
    config = TrainingConfig(
        speed_multiplier=10.0,
        headless_mode=True,
        max_episodes=50,
        parallel_episodes=2,
        training_mode=TrainingMode.TRAINING,
        spectator_enabled=False
    )
    
    # Create training session
    session = TrainingSession(
        session_id="example_session",
        config=config
    )
    
    # Register event handlers
    async def on_episode_complete(result):
        print(f"Episode {result.episode_id} completed: reward={result.total_reward}")
    
    async def on_metrics_update(metrics):
        print(f"Progress: {metrics.episodes_completed} episodes, avg reward: {metrics.average_reward:.2f}")
    
    session.register_episode_handler(on_episode_complete)
    session.register_metrics_handler(on_metrics_update)
    
    try:
        # Initialize session (would create training room in real scenario)
        print("Initializing training session...")
        await session.initialize()
        
        # Get session info
        info = await session.get_session_info()
        print(f"Session created: {info['sessionId']}")
        print(f"Configuration: {info['config']}")
        
        print("Training session example completed successfully!")
        
    except Exception as e:
        print(f"Error in training session: {e}")
    finally:
        await session.stop()


async def example_session_manager():
    """Example of using the SessionManager for multiple sessions."""
    print("\n=== Session Manager Example ===")
    
    # Create resource limits
    limits = ResourceLimits(
        max_concurrent_sessions=2,
        max_parallel_episodes_per_session=2,
        max_total_parallel_episodes=4
    )
    
    # Create session manager
    manager = SessionManager(resource_limits=limits)
    
    # Register event handlers
    async def on_session_start(session_id):
        print(f"Session {session_id} started")
    
    async def on_session_complete(session_id, metrics):
        print(f"Session {session_id} completed with {metrics.episodes_completed} episodes")
    
    manager.register_session_start_handler(on_session_start)
    manager.register_session_complete_handler(on_session_complete)
    
    try:
        # Start session manager
        await manager.start()
        
        # Create multiple sessions
        config1 = TrainingConfig(max_episodes=20, parallel_episodes=1)
        config2 = TrainingConfig(max_episodes=30, parallel_episodes=2)
        
        session_id1 = await manager.create_session(config=config1, priority=1)
        session_id2 = await manager.create_session(config=config2, priority=2)
        
        print(f"Created sessions: {session_id1}, {session_id2}")
        
        # Get system status
        status = await manager.get_all_sessions_info()
        print(f"Active sessions: {len(status['activeSessions'])}")
        print(f"Queued sessions: {status['queuedSessions']}")
        
        # Wait a bit for processing
        await asyncio.sleep(1.0)
        
        print("Session manager example completed successfully!")
        
    except Exception as e:
        print(f"Error in session manager: {e}")
    finally:
        await manager.stop()


async def example_batch_episode_manager():
    """Example of using the BatchEpisodeManager for parallel episodes."""
    print("\n=== Batch Episode Manager Example ===")
    
    # Create batch manager
    batch_manager = BatchEpisodeManager(
        max_parallel_episodes=4,
        max_retries=2,
        episode_timeout=60
    )
    
    # Register event handlers
    async def on_episode_complete(result):
        print(f"Episode {result.episode_id} completed: {result.game_result}")
    
    async def on_batch_complete(batch_id, results):
        print(f"Batch {batch_id} completed with {len(results)} successful episodes")
    
    batch_manager.register_episode_complete_handler(on_episode_complete)
    batch_manager.register_batch_complete_handler(on_batch_complete)
    
    try:
        # Start batch manager
        await batch_manager.start()
        
        # Submit a batch of episodes
        episode_ids = [f"episode_{i:03d}" for i in range(10)]
        await batch_manager.submit_batch(
            batch_id="example_batch",
            episode_ids=episode_ids,
            room_code="TR123456",
            max_parallel=3,
            timeout_seconds=30
        )
        
        print(f"Submitted batch with {len(episode_ids)} episodes")
        
        # Monitor batch status
        for _ in range(5):
            await asyncio.sleep(1.0)
            status = await batch_manager.get_batch_status("example_batch")
            if status:
                print(f"Batch progress: {status['completed']}/{status['totalEpisodes']} completed")
                if status['completed'] == status['totalEpisodes']:
                    break
        
        print("Batch episode manager example completed successfully!")
        
    except Exception as e:
        print(f"Error in batch manager: {e}")
    finally:
        await batch_manager.stop()


async def example_training_api():
    """Example of using the TrainingAPI for HTTP-style interactions."""
    print("\n=== Training API Example ===")
    
    # Create training API
    api = TrainingAPI()
    
    try:
        # Start API
        await api.start()
        
        # Create a training session via API
        request_data = {
            "config": {
                "speedMultiplier": 5.0,
                "headlessMode": False,
                "maxEpisodes": 100,
                "parallelEpisodes": 2,
                "trainingMode": "training"
            },
            "priority": 1,
            "sessionId": "api_session"
        }
        
        result = await api.create_training_session(request_data)
        print(f"Session creation result: {result}")
        
        # Get system status
        status = await api.get_system_status()
        print(f"System status: {status['status']}")
        print(f"Active sessions: {status['sessions']['active']}")
        
        # Validate configuration
        config_to_validate = {
            "speedMultiplier": 15.0,
            "maxEpisodes": 500,
            "parallelEpisodes": 3,
            "trainingMode": "headless"
        }
        
        validation_result = await api.validate_config(config_to_validate)
        print(f"Config validation: {validation_result}")
        
        # Request a training room
        room_request = {
            "speedMultiplier": 20.0,
            "headlessMode": True,
            "maxPlayers": 6
        }
        
        room_result = await api.request_training_room(room_request)
        print(f"Training room: {room_result['roomCode']}")
        
        print("Training API example completed successfully!")
        
    except Exception as e:
        print(f"Error in training API: {e}")
    finally:
        await api.stop()


async def main():
    """Run all examples."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("Training Session Management Examples")
    print("=" * 50)
    
    # Run examples
    await example_basic_training_session()
    await example_session_manager()
    await example_batch_episode_manager()
    await example_training_api()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())