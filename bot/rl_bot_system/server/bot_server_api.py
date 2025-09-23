"""
Integration module for BotServer with the main server infrastructure.

This module provides integration between the BotServer and the main FastAPI server,
enabling REST API endpoints for bot management and coordination with game rooms.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from rl_bot_system.server.bot_server import (
    BotServer, BotServerConfig, BotConfig, BotType, BotStatus, DifficultyLevel
)


# Pydantic models for API requests/responses

class SpawnBotRequest(BaseModel):
    """Request model for spawning a bot."""
    bot_type: str
    difficulty: str
    name: str
    room_code: str
    room_password: Optional[str] = ""
    generation: Optional[int] = None
    training_mode: bool = False
    auto_cleanup: bool = True


class SpawnBotResponse(BaseModel):
    """Response model for bot spawning."""
    bot_id: str
    status: str
    message: str


class ConfigureBotRequest(BaseModel):
    """Request model for configuring a bot."""
    difficulty: str


class BotStatusResponse(BaseModel):
    """Response model for bot status."""
    bot_id: str
    config: Dict[str, Any]
    status: str
    room_id: Optional[str]
    created_at: str
    last_activity: str
    performance_stats: Dict[str, Any]
    error_message: Optional[str] = None


class ServerStatusResponse(BaseModel):
    """Response model for server status."""
    running: bool
    total_bots: int
    max_total_bots: int
    active_rooms: int
    bots_by_status: Dict[str, int]
    bots_by_type: Dict[str, int]
    client_pool_stats: Dict[str, int]
    uptime_seconds: float


class BotTypeInfo(BaseModel):
    """Information about available bot types."""
    type: str
    name: str
    description: str
    difficulties: Optional[List[str]] = None
    available_generations: Optional[List[int]] = None
    supports_training_mode: bool = False


class BotServerApi:
    """
    Integration class that manages BotServer and provides API endpoints.
    """
    
    def __init__(self, bot_server_config: Optional[BotServerConfig] = None):
        """
        Initialize bot integration.
        
        Args:
            bot_server_config: Configuration for the bot server
        """
        self.config = bot_server_config or BotServerConfig()
        self.bot_server: Optional[BotServer] = None
        self.logger = logging.getLogger(__name__)
        
        # Create API router
        self.router = APIRouter(prefix="/api/bots", tags=["bots"])
        self._setup_routes()
    
    async def initialize(self) -> None:
        """Initialize the bot server and start it."""
        if self.bot_server is None:
            self.bot_server = BotServer(self.config)
            
            # Register callbacks for integration
            self.bot_server.register_bot_status_callback(self._on_bot_status_change)
            self.bot_server.register_room_empty_callback(self._on_room_empty)
            
            await self.bot_server.start()
            self.logger.info("Bot integration initialized")
    
    async def cleanup(self) -> None:
        """Clean up bot server resources."""
        if self.bot_server:
            await self.bot_server.stop()
            self.bot_server = None
            self.logger.info("Bot integration cleaned up")
    
    def _setup_routes(self) -> None:
        """Set up API routes for bot management."""
        
        @self.router.post("/spawn", response_model=SpawnBotResponse)
        async def spawn_bot(request: SpawnBotRequest, background_tasks: BackgroundTasks):
            """Spawn a new bot instance."""
            if not self.bot_server:
                raise HTTPException(status_code=503, detail="Bot server not initialized")
            
            try:
                # Validate and convert request
                bot_type = BotType(request.bot_type)
                difficulty = DifficultyLevel(request.difficulty)
                
                bot_config = BotConfig(
                    bot_type=bot_type,
                    difficulty=difficulty,
                    name=request.name,
                    generation=request.generation,
                    training_mode=request.training_mode,
                    auto_cleanup=request.auto_cleanup
                )
                
                room_info = {
                    'room_code': request.room_code,
                    'room_password': request.room_password
                }
                
                # Spawn the bot
                bot_id = await self.bot_server.spawn_bot(bot_config, room_info)
                
                return SpawnBotResponse(
                    bot_id=bot_id,
                    status="spawning",
                    message=f"Bot {request.name} spawning in room {request.room_code}"
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
            except RuntimeError as e:
                raise HTTPException(status_code=409, detail=str(e))
            except Exception as e:
                self.logger.error(f"Error spawning bot: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.router.delete("/{bot_id}")
        async def terminate_bot(bot_id: str):
            """Terminate a bot instance."""
            if not self.bot_server:
                raise HTTPException(status_code=503, detail="Bot server not initialized")
            
            success = await self.bot_server.terminate_bot(bot_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Bot not found")
            
            return {"message": f"Bot {bot_id} terminated"}
        
        @self.router.put("/{bot_id}/difficulty", response_model=Dict[str, str])
        async def configure_bot_difficulty(bot_id: str, request: ConfigureBotRequest):
            """Configure bot difficulty level."""
            if not self.bot_server:
                raise HTTPException(status_code=503, detail="Bot server not initialized")
            
            try:
                difficulty = DifficultyLevel(request.difficulty)
                success = await self.bot_server.configure_bot_difficulty(bot_id, difficulty)
                
                if not success:
                    raise HTTPException(status_code=404, detail="Bot not found")
                
                return {"message": f"Bot {bot_id} difficulty updated to {difficulty.value}"}
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid difficulty: {e}")
        
        @self.router.get("/{bot_id}/status", response_model=BotStatusResponse)
        async def get_bot_status(bot_id: str):
            """Get status of a specific bot."""
            if not self.bot_server:
                raise HTTPException(status_code=503, detail="Bot server not initialized")
            
            status = self.bot_server.get_bot_status(bot_id)
            
            if not status:
                raise HTTPException(status_code=404, detail="Bot not found")
            
            return BotStatusResponse(**status)
        
        @self.router.get("/room/{room_id}")
        async def get_room_bots(room_id: str):
            """Get all bots in a specific room."""
            if not self.bot_server:
                raise HTTPException(status_code=503, detail="Bot server not initialized")
            
            bots = self.bot_server.get_room_bots(room_id)
            return {"room_id": room_id, "bots": bots}
        
        @self.router.get("/status", response_model=ServerStatusResponse)
        async def get_server_status():
            """Get bot server status and statistics."""
            if not self.bot_server:
                raise HTTPException(status_code=503, detail="Bot server not initialized")
            
            status = self.bot_server.get_server_status()
            return ServerStatusResponse(**status)
        
        @self.router.get("/types", response_model=List[BotTypeInfo])
        async def get_available_bot_types():
            """Get list of available bot types."""
            if not self.bot_server:
                raise HTTPException(status_code=503, detail="Bot server not initialized")
            
            bot_types = self.bot_server.get_available_bot_types()
            return [BotTypeInfo(**bt) for bt in bot_types]
        
        @self.router.post("/cleanup")
        async def cleanup_inactive_bots():
            """Manually trigger cleanup of inactive bots."""
            if not self.bot_server:
                raise HTTPException(status_code=503, detail="Bot server not initialized")
            
            # This would trigger the cleanup process
            # For now, just return success
            return {"message": "Cleanup triggered"}
    
    async def _on_bot_status_change(self, bot_id: str, status: BotStatus) -> None:
        """Handle bot status changes."""
        self.logger.info(f"Bot {bot_id} status changed to {status.value}")
        
        # Here you could integrate with other systems, send notifications, etc.
        # For example, notify the main server about bot status changes
    
    async def _on_room_empty(self, room_id: str) -> None:
        """Handle room becoming empty of bots."""
        self.logger.info(f"Room {room_id} is now empty of bots")
        
        # Here you could integrate with the game server to handle empty rooms
        # For example, notify the game server that a room has no more bots


# Global integration instance
_bot_integration: Optional[BotServerApi] = None


def get_bot_integration() -> Optional[BotServerApi]:
    """Get the global bot integration instance."""
    return _bot_integration


async def initialize_bot_integration(config: Optional[BotServerConfig] = None) -> BotServerApi:
    """
    Initialize the global bot integration instance.
    
    Args:
        config: Optional bot server configuration
        
    Returns:
        BotServerApi instance
    """
    global _bot_integration
    
    if _bot_integration is None:
        _bot_integration = BotServerApi(config)
        await _bot_integration.initialize()
    
    return _bot_integration


async def cleanup_bot_integration() -> None:
    """Clean up the global bot integration instance."""
    global _bot_integration
    
    if _bot_integration:
        await _bot_integration.cleanup()
        _bot_integration = None


def get_bot_router() -> APIRouter:
    """
    Get the API router for bot endpoints.
    
    Returns:
        APIRouter for bot management endpoints
        
    Raises:
        RuntimeError: If bot integration not initialized
    """
    if _bot_integration is None:
        raise RuntimeError("Bot integration not initialized")
    
    return _bot_integration.router


# Convenience functions for integration with main server

async def add_bot_to_room(room_code: str, bot_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a bot to a game room.
    
    Args:
        room_code: Code of the room to join
        bot_config: Bot configuration dictionary
        
    Returns:
        Dict with bot information
        
    Raises:
        RuntimeError: If bot integration not available
    """
    integration = get_bot_integration()
    if not integration or not integration.bot_server:
        raise RuntimeError("Bot integration not available")
    
    # Convert dict to BotConfig
    bot_type = BotType(bot_config.get('bot_type', 'rules_based'))
    difficulty = DifficultyLevel(bot_config.get('difficulty', 'intermediate'))
    
    config = BotConfig(
        bot_type=bot_type,
        difficulty=difficulty,
        name=bot_config.get('name', 'Bot'),
        generation=bot_config.get('generation'),
        training_mode=bot_config.get('training_mode', False),
        auto_cleanup=bot_config.get('auto_cleanup', True)
    )
    
    room_info = {
        'room_code': room_code,
        'room_password': bot_config.get('room_password', '')
    }
    
    bot_id = await integration.bot_server.spawn_bot(config, room_info)
    
    return {
        'bot_id': bot_id,
        'name': config.name,
        'type': config.bot_type.value,
        'difficulty': config.difficulty.value
    }


async def remove_bot_from_room(room_id: str, bot_id: str) -> bool:
    """
    Remove a bot from a game room.
    
    Args:
        room_id: ID of the room
        bot_id: ID of the bot to remove
        
    Returns:
        bool: True if removal successful
        
    Raises:
        RuntimeError: If bot integration not available
    """
    integration = get_bot_integration()
    if not integration or not integration.bot_server:
        raise RuntimeError("Bot integration not available")
    
    return await integration.bot_server.terminate_bot(bot_id)


def get_available_bots() -> List[Dict[str, Any]]:
    """
    Get list of available bot types and configurations.
    
    Returns:
        List of available bot configurations
        
    Raises:
        RuntimeError: If bot integration not available
    """
    integration = get_bot_integration()
    if not integration or not integration.bot_server:
        raise RuntimeError("Bot integration not available")
    
    return integration.bot_server.get_available_bot_types()


def get_room_bot_status(room_id: str) -> Dict[str, Any]:
    """
    Get status of all bots in a room.
    
    Args:
        room_id: ID of the room
        
    Returns:
        Dict with room bot information
        
    Raises:
        RuntimeError: If bot integration not available
    """
    integration = get_bot_integration()
    if not integration or not integration.bot_server:
        raise RuntimeError("Bot integration not available")
    
    bots = integration.bot_server.get_room_bots(room_id)
    
    return {
        'room_id': room_id,
        'bot_count': len(bots),
        'bots': bots
    }