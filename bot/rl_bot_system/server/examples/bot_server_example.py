"""
Example script demonstrating BotServer usage.

This script shows how to create and manage bot instances using the BotServer.
"""

import asyncio
import logging
from rl_bot_system.server.bot_server import (
    BotServer, BotServerConfig, BotConfig, BotType, DifficultyLevel
)


async def main():
    """Main example function."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create bot server configuration
    config = BotServerConfig(
        max_bots_per_room=4,
        max_total_bots=20,
        bot_timeout_seconds=300,
        game_server_url="http://localhost:4000"
    )
    
    # Create and start bot server
    bot_server = BotServer(config)
    await bot_server.start()
    
    logger.info("Bot server started")
    
    try:
        # Example 1: Spawn a rules-based bot
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.INTERMEDIATE,
            name="ExampleBot1"
        )
        
        room_info = {
            'room_code': 'TEST123',
            'room_password': '',
            'room_id': 'example_room'
        }
        
        logger.info("Spawning rules-based bot...")
        bot_id = await bot_server.spawn_bot(bot_config, room_info)
        logger.info(f"Spawned bot with ID: {bot_id}")
        
        # Wait a bit for initialization
        await asyncio.sleep(2)
        
        # Check bot status
        status = bot_server.get_bot_status(bot_id)
        if status:
            logger.info(f"Bot status: {status['status']}")
        
        # Example 2: Configure bot difficulty
        logger.info("Changing bot difficulty to Expert...")
        success = await bot_server.configure_bot_difficulty(bot_id, DifficultyLevel.EXPERT)
        if success:
            logger.info("Difficulty changed successfully")
        
        # Example 3: Get server status
        server_status = bot_server.get_server_status()
        logger.info(f"Server status: {server_status['total_bots']} bots active")
        
        # Example 4: Get available bot types
        bot_types = bot_server.get_available_bot_types()
        logger.info(f"Available bot types: {[bt['type'] for bt in bot_types]}")
        
        # Example 5: Spawn multiple bots
        logger.info("Spawning multiple bots...")
        bot_ids = []
        
        for i in range(3):
            config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.BEGINNER,
                name=f"Bot{i+2}"
            )
            
            room_info = {
                'room_code': 'TEST123',
                'room_password': '',
                'room_id': 'example_room'
            }
            
            bot_id = await bot_server.spawn_bot(config, room_info)
            bot_ids.append(bot_id)
            logger.info(f"Spawned bot {i+2} with ID: {bot_id}")
        
        # Wait a bit
        await asyncio.sleep(3)
        
        # Check room bots
        room_bots = bot_server.get_room_bots('example_room')
        logger.info(f"Room has {len(room_bots)} bots")
        
        # Example 6: Terminate some bots
        logger.info("Terminating first bot...")
        success = await bot_server.terminate_bot(bot_ids[0])
        if success:
            logger.info("Bot terminated successfully")
        
        # Wait a bit more to see the system in action
        logger.info("Letting bots run for 10 seconds...")
        await asyncio.sleep(10)
        
        # Final status check
        final_status = bot_server.get_server_status()
        logger.info(f"Final server status: {final_status}")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
    
    finally:
        # Clean up
        logger.info("Stopping bot server...")
        await bot_server.stop()
        logger.info("Bot server stopped")


if __name__ == "__main__":
    asyncio.run(main())