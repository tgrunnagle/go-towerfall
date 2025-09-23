"""
Bot API Server for handling HTTP requests from the Go game server.

This module provides a FastAPI server that exposes bot management endpoints
for integration with the Go game server.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from bot.rl_bot_system.server.bot_server import BotServer, BotServerConfig, BotConfig, BotType, DifficultyLevel
from bot.rl_bot_system.rules_based.rules_based_bot import DifficultyLevel as RulesDifficultyLevel


# Pydantic models for API requests/responses

class BotTypeInfo(BaseModel):
    type: str
    name: str
    description: str
    difficulties: List[str]
    available_generations: Optional[List[int]] = None
    supports_training_mode: bool

class AvailableBotsResponse(BaseModel):
    success: bool
    bot_types: List[BotTypeInfo]
    error: Optional[str] = None

class SpawnBotRequest(BaseModel):
    bot_type: str
    difficulty: str
    bot_name: str
    room_code: str
    room_password: str = ""
    generation: Optional[int] = None
    training_mode: bool = False

class SpawnBotResponse(BaseModel):
    success: bool
    bot_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

class TerminateBotRequest(BaseModel):
    bot_id: str

class TerminateBotResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class ConfigureBotRequest(BaseModel):
    difficulty: str

class ConfigureBotResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class BotStatusInfo(BaseModel):
    bot_id: str
    bot_type: str
    difficulty: str
    name: str
    generation: Optional[int] = None
    status: str
    room_id: Optional[str] = None
    created_at: str
    performance: Dict[str, Any]

class RoomBotsResponse(BaseModel):
    success: bool
    bots: List[BotStatusInfo]
    error: Optional[str] = None

class ServerStatusResponse(BaseModel):
    success: bool
    server_running: bool
    total_bots: int
    active_rooms: int
    server_stats: Dict[str, Any]
    error: Optional[str] = None


class BotAPIServer:
    """
    FastAPI server for bot management integration with Go game server.
    """
    
    def __init__(self, bot_server: BotServer, host: str = "localhost", port: int = 8001):
        """
        Initialize the Bot API Server.
        
        Args:
            bot_server: The BotServer instance to manage
            host: Host to bind the server to
            port: Port to bind the server to
        """
        self.bot_server = bot_server
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Bot Management API",
            description="API for managing AI bots in game rooms",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/api/bots/available", response_model=AvailableBotsResponse)
        async def get_available_bots():
            """Get available bot types and configurations."""
            try:
                bot_types = self.bot_server.get_available_bot_types()
                
                # Convert to API format
                api_bot_types = []
                for bot_type in bot_types:
                    api_bot_types.append(BotTypeInfo(
                        type=bot_type['type'],
                        name=bot_type['name'],
                        description=bot_type['description'],
                        difficulties=bot_type['difficulties'],
                        available_generations=bot_type.get('available_generations'),
                        supports_training_mode=bot_type['supports_training_mode']
                    ))
                
                return AvailableBotsResponse(
                    success=True,
                    bot_types=api_bot_types
                )
                
            except Exception as e:
                self.logger.error(f"Error getting available bots: {e}")
                return AvailableBotsResponse(
                    success=False,
                    bot_types=[],
                    error=str(e)
                )
        
        @self.app.post("/api/bots/spawn", response_model=SpawnBotResponse)
        async def spawn_bot(request: SpawnBotRequest, background_tasks: BackgroundTasks):
            """Spawn a new bot instance."""
            try:
                # Convert request to bot config
                bot_type = BotType(request.bot_type)
                difficulty = DifficultyLevel(request.difficulty)
                
                bot_config = BotConfig(
                    bot_type=bot_type,
                    difficulty=difficulty,
                    name=request.bot_name,
                    generation=request.generation,
                    training_mode=request.training_mode
                )
                
                room_info = {
                    'room_code': request.room_code,
                    'room_password': request.room_password
                }
                
                # Spawn bot
                bot_id = await self.bot_server.spawn_bot(bot_config, room_info)
                
                return SpawnBotResponse(
                    success=True,
                    bot_id=bot_id,
                    message="Bot spawned successfully"
                )
                
            except Exception as e:
                self.logger.error(f"Error spawning bot: {e}")
                return SpawnBotResponse(
                    success=False,
                    error=str(e)
                )
        
        @self.app.post("/api/bots/{bot_id}/terminate", response_model=TerminateBotResponse)
        async def terminate_bot(bot_id: str):
            """Terminate a bot instance."""
            try:
                success = await self.bot_server.terminate_bot(bot_id)
                
                if success:
                    return TerminateBotResponse(
                        success=True,
                        message="Bot terminated successfully"
                    )
                else:
                    return TerminateBotResponse(
                        success=False,
                        error="Failed to terminate bot"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error terminating bot {bot_id}: {e}")
                return TerminateBotResponse(
                    success=False,
                    error=str(e)
                )
        
        @self.app.put("/api/bots/{bot_id}/configure", response_model=ConfigureBotResponse)
        async def configure_bot(bot_id: str, request: ConfigureBotRequest):
            """Configure bot difficulty."""
            try:
                difficulty = DifficultyLevel(request.difficulty)
                success = await self.bot_server.configure_bot_difficulty(bot_id, difficulty)
                
                if success:
                    return ConfigureBotResponse(
                        success=True,
                        message="Bot configured successfully"
                    )
                else:
                    return ConfigureBotResponse(
                        success=False,
                        error="Failed to configure bot"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error configuring bot {bot_id}: {e}")
                return ConfigureBotResponse(
                    success=False,
                    error=str(e)
                )
        
        @self.app.get("/api/rooms/{room_id}/bots", response_model=RoomBotsResponse)
        async def get_room_bots(room_id: str):
            """Get bots in a specific room."""
            try:
                bots = self.bot_server.get_room_bots(room_id)
                
                # Convert to API format
                api_bots = []
                for bot in bots:
                    api_bots.append(BotStatusInfo(
                        bot_id=bot['bot_id'],
                        bot_type=bot['config']['bot_type'],
                        difficulty=bot['config']['difficulty'],
                        name=bot['config']['name'],
                        generation=bot['config'].get('generation'),
                        status=bot['status'],
                        room_id=bot.get('room_id'),
                        created_at=bot['created_at'],
                        performance=bot['performance_stats']
                    ))
                
                return RoomBotsResponse(
                    success=True,
                    bots=api_bots
                )
                
            except Exception as e:
                self.logger.error(f"Error getting room bots for {room_id}: {e}")
                return RoomBotsResponse(
                    success=False,
                    bots=[],
                    error=str(e)
                )
        
        @self.app.get("/api/bots/status", response_model=ServerStatusResponse)
        async def get_server_status():
            """Get bot server status."""
            try:
                status = self.bot_server.get_server_status()
                
                return ServerStatusResponse(
                    success=True,
                    server_running=status['running'],
                    total_bots=status['total_bots'],
                    active_rooms=status['active_rooms'],
                    server_stats=status
                )
                
            except Exception as e:
                self.logger.error(f"Error getting server status: {e}")
                return ServerStatusResponse(
                    success=False,
                    server_running=False,
                    total_bots=0,
                    active_rooms=0,
                    server_stats={},
                    error=str(e)
                )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    async def start(self):
        """Start the API server."""
        self.logger.info(f"Starting Bot API Server on {self.host}:{self.port}")
        
        # Start the bot server first
        await self.bot_server.start()
        
        # Configure uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self):
        """Stop the API server."""
        self.logger.info("Stopping Bot API Server")
        await self.bot_server.stop()


async def main():
    """Main entry point for running the bot API server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create bot server configuration
    bot_config = BotServerConfig(
        max_bots_per_room=8,
        max_total_bots=50,
        game_server_url="http://localhost:4000",
        models_dir="bot/data/models"
    )
    
    # Create bot server
    bot_server = BotServer(bot_config)
    
    # Create API server
    api_server = BotAPIServer(bot_server, host="localhost", port=8001)
    
    try:
        # Start the server
        await api_server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await api_server.stop()


if __name__ == "__main__":
    asyncio.run(main())