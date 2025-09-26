"""
Example script demonstrating BotServer usage.

This script shows how to create and manage bot instances using the BotServer.

Prerequisites:
- Go game server running on localhost:4000
- Run from the bot/ directory: python -m rl_bot_system.server.examples.bot_server_example
"""

import asyncio
import logging
import aiohttp
from rl_bot_system.server.bot_server import (
    BotServer, BotServerConfig, BotConfig, BotType, DifficultyLevel
)


async def check_game_server():
    """Check if the game server is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:4000/api/maps", timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status == 200
    except Exception:
        return False


async def create_test_room():
    """Create a test room for the bots."""
    try:
        async with aiohttp.ClientSession() as session:
            create_data = {
                "playerName": "BotServerExample",
                "roomName": "Bot Server Example Room",
                "mapType": "default"
            }
            
            async with session.post("http://localhost:4000/api/createGame", json=create_data) as response:
                response.raise_for_status()
                data = await response.json()
                
                if not data.get("success"):
                    raise RuntimeError(f"Failed to create room: {data.get('error')}")
                
                room_id = data["roomId"]
                room_code = data["roomCode"]
                
                # Get room password
                async with session.get(f"http://localhost:4000/api/rooms/{room_id}/details") as response:
                    response.raise_for_status()
                    details_data = await response.json()
                    room_password = details_data["roomPassword"]
                
                return room_id, room_code, room_password
                
    except Exception as e:
        raise RuntimeError(f"Failed to create test room: {e}")


async def main():
    """Main example function."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Check if game server is running
    logger.info("Checking if game server is running...")
    if not await check_game_server():
        logger.error("Game server is not running on localhost:4000")
        logger.error("Please start the Go game server before running this example")
        return
    
    logger.info("Game server is running ✓")
    
    # Create a test room
    logger.info("Creating test room...")
    try:
        room_id, room_code, room_password = await create_test_room()
        logger.info(f"Created test room: {room_code} (ID: {room_id})")
    except Exception as e:
        logger.error(f"Failed to create test room: {e}")
        return
    
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
    
    logger.info("Bot server started ✓")
    
    try:
        # Example 1: Spawn a rules-based bot
        bot_config = BotConfig(
            bot_type=BotType.RULES_BASED,
            difficulty=DifficultyLevel.INTERMEDIATE,
            name="ExampleBot1"
        )
        
        room_info = {
            'room_code': room_code,
            'room_password': room_password,
            'room_id': room_id
        }
        
        logger.info("Spawning rules-based bot...")
        bot_id = await bot_server.spawn_bot(bot_config, room_info)
        logger.info(f"Spawned bot with ID: {bot_id}")
        
        # Wait for initialization
        logger.info("Waiting for bot initialization...")
        await asyncio.sleep(3)
        
        # Check bot status
        status = bot_server.get_bot_status(bot_id)
        if status:
            logger.info(f"Bot status: {status['status']}")
            logger.info(f"Bot name: {status['config']['name']}")
        
        # Example 2: Configure bot difficulty
        logger.info("Changing bot difficulty to Expert...")
        success = await bot_server.configure_bot_difficulty(bot_id, DifficultyLevel.EXPERT)
        if success:
            logger.info("Difficulty changed successfully ✓")
            
            # Check updated status
            updated_status = bot_server.get_bot_status(bot_id)
            if updated_status:
                logger.info(f"New difficulty: {updated_status['config']['difficulty']}")
        
        # Example 3: Get server status
        server_status = bot_server.get_server_status()
        logger.info(f"Server status: {server_status['total_bots']} bots active")
        logger.info(f"Server running: {server_status['running']}")
        
        # Example 4: Get available bot types
        bot_types = bot_server.get_available_bot_types()
        logger.info(f"Available bot types: {[bt['type'] for bt in bot_types]}")
        
        # Example 5: Spawn multiple bots
        logger.info("Spawning multiple bots...")
        bot_ids = [bot_id]  # Include the first bot
        
        for i in range(2):  # Spawn 2 more bots
            config = BotConfig(
                bot_type=BotType.RULES_BASED,
                difficulty=DifficultyLevel.BEGINNER,
                name=f"Bot{i+2}"
            )
            
            bot_id_new = await bot_server.spawn_bot(config, room_info)
            bot_ids.append(bot_id_new)
            logger.info(f"Spawned bot {i+2} with ID: {bot_id_new}")
        
        # Wait for all bots to initialize
        logger.info("Waiting for all bots to initialize...")
        await asyncio.sleep(3)
        
        # Check room bots
        room_bots = bot_server.get_room_bots(room_id)
        logger.info(f"Room has {len(room_bots)} bots")
        
        # Example 6: Monitor bot health
        logger.info("Checking bot health...")
        for bot_id_check in bot_ids:
            health = await bot_server.monitor_bot_health(bot_id_check)
            logger.info(f"Bot {bot_id_check}: healthy={health.get('healthy', 'unknown')}")
        
        # Example 7: Get all bot health
        all_health = await bot_server.get_all_bot_health()
        logger.info(f"Health check completed for {len(all_health)} bots")
        
        # Let bots run for a while
        logger.info("Letting bots run for 10 seconds...")
        await asyncio.sleep(10)
        
        # Example 8: Terminate some bots
        logger.info("Terminating first bot...")
        success = await bot_server.terminate_bot(bot_ids[0])
        if success:
            logger.info("Bot terminated successfully ✓")
        
        # Final status check
        final_status = bot_server.get_server_status()
        logger.info(f"Final server status: {final_status['total_bots']} bots active")
        
        # Clean up remaining bots
        logger.info("Cleaning up remaining bots...")
        for bot_id_cleanup in bot_ids[1:]:
            await bot_server.terminate_bot(bot_id_cleanup)
        
        logger.info("Example completed successfully! ✓")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        logger.info("Stopping bot server...")
        await bot_server.stop()
        logger.info("Bot server stopped ✓")


if __name__ == "__main__":
    asyncio.run(main())