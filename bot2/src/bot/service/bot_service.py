"""FastAPI Bot Service for managing game bots.

This module provides the REST API layer for the Bot Service, exposing endpoints
for spawning bots, managing active bots, listing available models, and health checks.
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

from bot.service.bot_manager import (
    BotInfo,
    BotManager,
    SpawnBotRequest,
    SpawnBotResponse,
)
from bot.training.registry import ModelMetadata

logger = logging.getLogger(__name__)

# Global bot manager instance (initialized in __main__.py)
_bot_manager: BotManager | None = None


def get_bot_manager() -> BotManager:
    """Dependency that provides the BotManager instance."""
    if _bot_manager is None:
        raise RuntimeError("BotManager not initialized")
    return _bot_manager


def set_bot_manager(manager: BotManager) -> None:
    """Set the global BotManager instance (called from __main__.py)."""
    global _bot_manager
    _bot_manager = manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    yield
    # Shutdown: cleanup bot manager
    if _bot_manager is not None:
        logger.info("Shutting down bot manager...")
        await _bot_manager.shutdown()


app = FastAPI(
    title="Bot Service",
    description="Service for spawning and managing game bots",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/bots/models", response_model=list[ModelMetadata])
def list_models(
    manager: Annotated[BotManager, Depends(get_bot_manager)],
) -> list[ModelMetadata]:
    """List available trained models."""
    return manager.list_models()


@app.post("/bots/spawn", response_model=SpawnBotResponse)
async def spawn_bot(
    request: SpawnBotRequest,
    manager: Annotated[BotManager, Depends(get_bot_manager)],
) -> SpawnBotResponse:
    """Spawn a bot to join a game room.

    Returns immediately with bot_id. Bot connects in background.
    """
    return await manager.spawn_bot(request)


@app.delete("/bots/{bot_id}")
async def delete_bot(
    bot_id: str,
    manager: Annotated[BotManager, Depends(get_bot_manager)],
) -> dict[str, bool]:
    """Remove a bot from its game."""
    success = await manager.destroy_bot(bot_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot {bot_id} not found",
        )
    return {"success": True}


@app.get("/bots", response_model=list[BotInfo])
def list_bots(
    manager: Annotated[BotManager, Depends(get_bot_manager)],
) -> list[BotInfo]:
    """List all active bots."""
    return manager.list_bots()


@app.get("/bots/{bot_id}", response_model=BotInfo)
def get_bot(
    bot_id: str,
    manager: Annotated[BotManager, Depends(get_bot_manager)],
) -> BotInfo:
    """Get information about a specific bot."""
    bot = manager.get_bot(bot_id)
    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot {bot_id} not found",
        )
    return bot
