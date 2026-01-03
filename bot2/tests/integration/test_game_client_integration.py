"""Integration tests for GameClient with real backend.

Tests cover:
- Creating games in REST mode
- Fetching game state after creation
- Submitting bot actions
- Resetting games
- Getting kill/death statistics
"""

import pytest

from bot.client import ClientMode, GameClient
from bot.models import GameState
from tests.conftest import requires_server, unique_room_name


@pytest.mark.integration
class TestGameClientIntegration:
    """Integration tests for GameClient with real server."""

    @requires_server
    @pytest.mark.asyncio
    async def test_create_game_rest_mode(self, server_url: str) -> None:
        """Create a game in REST mode."""
        room_name = unique_room_name("CreateGame")
        client = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client:
            response = await client.create_game(
                player_name="IntegrationTestBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            assert response.success is True
            assert response.room_id is not None
            assert response.player_id is not None
            assert client.room_id == response.room_id
            assert client.player_id == response.player_id

    @requires_server
    @pytest.mark.asyncio
    async def test_fetch_game_state_after_creation(self, server_url: str) -> None:
        """Fetch game state after creating a game."""
        room_name = unique_room_name("FetchState")
        client = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client:
            await client.create_game(
                player_name="IntegrationTestBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            state = await client.get_game_state()

            assert isinstance(state, GameState)
            assert state.canvas_size_x > 0
            assert state.canvas_size_y > 0
            # Our player should be in the game
            assert client.player_id is not None
            assert client.player_id in state.players

    @requires_server
    @pytest.mark.asyncio
    async def test_submit_bot_actions(self, server_url: str) -> None:
        """Submit bot actions to the server."""
        room_name = unique_room_name("SubmitActions")
        client = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client:
            await client.create_game(
                player_name="IntegrationTestBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            # Submit movement action (press right key)
            await client.send_keyboard_input("d", True)

            # Get state after action
            state = await client.get_game_state()

            # Should still have valid state
            assert isinstance(state, GameState)
            assert client.player_id in state.players

    @requires_server
    @pytest.mark.asyncio
    async def test_reset_game(self, server_url: str) -> None:
        """Reset a game."""
        room_name = unique_room_name("ResetGame")
        client = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client:
            await client.create_game(
                player_name="IntegrationTestBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            # Get initial state
            initial_state = await client.get_game_state()
            assert client.player_id is not None
            assert client.player_id in initial_state.players

            # Reset the game
            await client.reset_game(map_type="default")

            # Get state after reset
            reset_state = await client.get_game_state()

            # Player should still exist after reset
            assert client.player_id in reset_state.players
            reset_player = reset_state.players[client.player_id]

            # Player should be alive after reset
            assert reset_player.dead is False

    @requires_server
    @pytest.mark.asyncio
    async def test_get_kill_death_statistics(self, server_url: str) -> None:
        """Get kill/death statistics from the server."""
        room_name = unique_room_name("GetStats")
        client = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client:
            await client.create_game(
                player_name="IntegrationTestBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            stats = await client.get_stats()

            # Should have stats for our player
            assert client.player_id is not None
            assert client.player_id in stats

            player_stats = stats[client.player_id]
            assert player_stats.kills >= 0
            assert player_stats.deaths >= 0
