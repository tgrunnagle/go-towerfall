"""Integration tests for error handling and edge cases.

Tests cover:
- Invalid action values outside valid range
- Stepping a closed environment
- Resetting a closed environment
- Invalid/missing room names
- Duplicate room creation
- Server timeout handling
"""

import pytest

from bot.actions import ACTION_SPACE_SIZE
from bot.client import ClientMode, GameClient, GameClientError
from bot.gym import TowerfallEnv
from tests.conftest import requires_server, unique_room_name

# Default map type for integration tests
DEFAULT_MAP_TYPE = "default"


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling edge cases."""

    @requires_server
    def test_invalid_action_value(self, server_url: str) -> None:
        """Verify graceful handling when action outside [0, 26] is submitted."""
        room_name = unique_room_name("ErrInvalidAction")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            env.reset()

            # Submit action outside valid range (valid range is 0 to ACTION_SPACE_SIZE-1)
            invalid_action = ACTION_SPACE_SIZE + 10  # Well outside valid range

            # The Action enum should raise ValueError for invalid actions
            with pytest.raises(ValueError):
                env.step(invalid_action)
        finally:
            env.close()

    @requires_server
    def test_negative_action_value(self, server_url: str) -> None:
        """Verify graceful handling when negative action is submitted."""
        room_name = unique_room_name("ErrNegAction")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            env.reset()

            # Submit negative action
            with pytest.raises(ValueError):
                env.step(-1)
        finally:
            env.close()

    @requires_server
    def test_step_after_close(self, server_url: str) -> None:
        """Verify appropriate error when stepping a closed environment."""
        room_name = unique_room_name("ErrStepAfterClose")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        env.reset()
        env.close()

        # Step after close should raise RuntimeError
        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(0)

    @requires_server
    def test_step_before_reset(self, server_url: str) -> None:
        """Verify appropriate error when stepping before reset."""
        room_name = unique_room_name("ErrStepBeforeReset")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            # Step without reset should raise RuntimeError
            with pytest.raises(RuntimeError, match="not initialized"):
                env.step(0)
        finally:
            env.close()

    @requires_server
    def test_reset_after_close_creates_new_game(self, server_url: str) -> None:
        """Verify reset after close creates a new game successfully."""
        room_name = unique_room_name("ErrResetAfterClose")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        # First reset and close
        obs1, info1 = env.reset()
        env.close()

        # Reset after close should work by creating a new connection
        # Use a new room name to avoid conflicts
        env.room_name = unique_room_name("ErrResetAfterClose2")
        obs2, info2 = env.reset()

        try:
            assert obs2.shape == env.observation_space.shape
            assert "room_id" in info2
        finally:
            env.close()

    @requires_server
    @pytest.mark.asyncio
    async def test_get_state_before_create_game(self, server_url: str) -> None:
        """Verify error when getting state before creating a game."""
        client = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client:
            # Getting state before creating a game should raise error
            with pytest.raises(GameClientError, match="Not connected"):
                await client.get_game_state()

    @requires_server
    @pytest.mark.asyncio
    async def test_get_stats_before_create_game(self, server_url: str) -> None:
        """Verify error when getting stats before creating a game."""
        client = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client:
            # Getting stats before creating a game should raise error
            with pytest.raises(GameClientError, match="Not connected"):
                await client.get_stats()

    @requires_server
    @pytest.mark.asyncio
    async def test_reset_game_before_create(self, server_url: str) -> None:
        """Verify error when resetting before creating a game."""
        client = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client:
            # Resetting before creating a game should raise error
            with pytest.raises(GameClientError, match="Not connected"):
                await client.reset_game()

    @requires_server
    @pytest.mark.asyncio
    async def test_send_action_before_create_game(self, server_url: str) -> None:
        """Verify error when sending action before creating a game."""
        client = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client:
            # Sending action before creating a game should raise error
            with pytest.raises(GameClientError, match="Not connected"):
                await client.send_keyboard_input("d", True)


@pytest.mark.integration
class TestDuplicateRoomHandling:
    """Tests for duplicate room creation scenarios."""

    @requires_server
    @pytest.mark.asyncio
    async def test_duplicate_room_name_same_client(self, server_url: str) -> None:
        """Verify behavior when creating room with same name twice from same client."""
        room_name = unique_room_name("DupRoomSame")
        client = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client:
            # Create first game
            response1 = await client.create_game(
                player_name="TestBot1",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            assert response1.success is True
            assert response1.room_id is not None

            # Creating another game replaces the client's connection
            # The server may reuse or create a new room
            response2 = await client.create_game(
                player_name="TestBot2",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            assert response2.success is True
            # Client should be connected to the new room
            assert client.room_id == response2.room_id

    @requires_server
    @pytest.mark.asyncio
    async def test_duplicate_room_name_different_clients(self, server_url: str) -> None:
        """Verify behavior when two clients create rooms with the same name."""
        room_name = unique_room_name("DupRoomDiff")

        client1 = GameClient(http_url=server_url, mode=ClientMode.REST)
        client2 = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client1:
            # Create first game
            response1 = await client1.create_game(
                player_name="TestBot1",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            assert response1.success is True

            async with client2:
                # Create second game with same room name
                # Server should create a separate room (different room_id)
                response2 = await client2.create_game(
                    player_name="TestBot2",
                    room_name=room_name,
                    map_type="default",
                    training_mode=True,
                    tick_rate_multiplier=10.0,
                )

                assert response2.success is True
                # Both clients should be functional
                state1 = await client1.get_game_state()
                state2 = await client2.get_game_state()

                assert state1 is not None
                assert state2 is not None


@pytest.mark.integration
class TestClientCloseBehavior:
    """Tests for client close behavior."""

    @requires_server
    @pytest.mark.asyncio
    async def test_double_close(self, server_url: str) -> None:
        """Verify double close doesn't raise errors."""
        room_name = unique_room_name("DoubleClose")
        client = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with client:
            await client.create_game(
                player_name="TestBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

        # Client is already closed by context manager
        # Second close should not raise
        await client.close()

    @requires_server
    def test_env_double_close(self, server_url: str) -> None:
        """Verify environment double close doesn't raise errors."""
        room_name = unique_room_name("EnvDoubleClose")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        env.reset()
        env.close()

        # Second close should not raise
        env.close()


@pytest.mark.integration
class TestTimeoutHandling:
    """Tests for timeout scenarios."""

    @requires_server
    @pytest.mark.asyncio
    async def test_short_timeout_on_valid_server(self, server_url: str) -> None:
        """Verify operations complete within reasonable timeout."""
        room_name = unique_room_name("TimeoutTest")
        # Use default timeout - operations should complete quickly
        client = GameClient(http_url=server_url, mode=ClientMode.REST, timeout=30.0)

        async with client:
            # These operations should complete without timeout
            response = await client.create_game(
                player_name="TestBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )
            assert response.success is True

            state = await client.get_game_state()
            assert state is not None
