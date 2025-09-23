"""
Bot Server API integration for the unified server.

This module provides bot management endpoints that integrate with the unified server.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from rl_bot_system.server.bot_server import BotServer, BotServerConfig, BotConfig, BotType
from rl_bot_system.rules_based.rules_based_bot import DifficultyLevel


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
    room_code: Optional[str] = None
    room_id: Optional[str] = None
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


class BotServerApi:
    """
    Bot Server API integration for the unified server.
    """
    
    def __init__(self, config: BotServerConfig):
        """
        Initialize the Bot Server API.
        
        Args:
            config: Bot server configuration
        """
        self.config = config
        self.bot_server = BotServer(config)
        self.logger = logging.getLogger(__name__)
        
        # Create router
        self.router = APIRouter(prefix="/api/bots", tags=["bots"])
        self._setup_routes()
    
    async def initialize(self):
        """Initialize the bot server."""
        await self.bot_server.start()
        self.logger.info("Bot server API initialized")
    
    async def cleanup(self):
        """Clean up the bot server."""
        await self.bot_server.stop()
        self.logger.info("Bot server API cleaned up")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.router.get("/available", response_model=AvailableBotsResponse)
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
        
        @self.router.post("/spawn", response_model=SpawnBotResponse)
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
        
        @self.router.post("/{bot_id}/terminate", response_model=TerminateBotResponse)
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
        
        @self.router.put("/{bot_id}/configure", response_model=ConfigureBotResponse)
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
        
        @self.router.get("/rooms/{room_id}/bots", response_model=RoomBotsResponse)
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
        
        @self.router.get("/status", response_model=ServerStatusResponse)
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


# Alias for backward compatibility
BotServerConfig = BotServerConfig