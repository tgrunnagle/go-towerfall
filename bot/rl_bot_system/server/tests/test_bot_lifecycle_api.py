"""
Tests for bot lifecycle management API integration.

This module tests the integration between the API layer and bot lifecycle management.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.rl_bot_system.server.bot_server_api import BotServerApi
from bot.rl_bot_system.server.bot_server import (
    BotServerConfig, BotConfig, BotType, BotStatus, DifficultyLevel
)


@pytest.mark.asyncio
async def test_bot_server_api_initialization():
    """Test BotServerApi initialization and setup."""
    config = BotServerConfig(max_total_bots=10)
    api = BotServerApi(config)
    
    assert api.config == config
    assert api.bot_server is None
    assert api.router is not None
    
    # Test initialization
    with patch('bot.rl_bot_system.server.bot_server_api.BotServer') as mock_bot_server_class:
        mock_bot_server = AsyncMock()
        mock_bot_server_class.return_value = mock_bot_server
        
        await api.initialize()
        
        assert api.bot_server == mock_bot_server
        mock_bot_server.start.assert_called_once()
        mock_bot_server.register_bot_status_callback.assert_called_once()
        mock_bot_server.register_room_empty_callback.assert_called_once()


@pytest.mark.asyncio
async def test_bot_server_api_cleanup():
    """Test BotServerApi cleanup functionality."""
    config = BotServerConfig(max_total_bots=10)
    api = BotServerApi(config)
    
    # Mock the bot server
    mock_bot_server = AsyncMock()
    api.bot_server = mock_bot_server
    
    # Test cleanup
    await api.cleanup()
    
    mock_bot_server.stop.assert_called_once()
    assert api.bot_server is None


@pytest.mark.asyncio
async def test_api_integration_methods():
    """Test the API integration methods."""
    config = BotServerConfig(max_total_bots=10)
    api = BotServerApi(config)
    
    # Test that the API has the expected router and configuration
    assert api.router is not None
    assert api.config.max_total_bots == 10
    
    # Test that cleanup works when bot_server is None
    await api.cleanup()
    assert api.bot_server is None


@pytest.mark.asyncio
async def test_callback_handlers():
    """Test the callback handlers for bot status and room events."""
    config = BotServerConfig(max_total_bots=10)
    api = BotServerApi(config)
    
    # Test bot status change callback
    await api._on_bot_status_change('test_bot_123', BotStatus.ACTIVE)
    # Should not raise any exceptions
    
    # Test room empty callback
    await api._on_room_empty('test_room_123')
    # Should not raise any exceptions


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])