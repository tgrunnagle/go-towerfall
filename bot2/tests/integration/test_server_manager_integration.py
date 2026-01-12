"""Integration tests for GameServerManager.

Tests require a running go-towerfall server.

Run with: pytest tests/integration/test_server_manager_integration.py -v
"""

import pytest

from bot.training import (
    GameInstance,
    GameServerManager,
    MaxGamesExceededError,
    TrainingGameConfig,
)
from tests.conftest import requires_server, unique_room_name


@requires_server
@pytest.mark.integration
class TestGameServerManagerIntegration:
    """Integration tests for GameServerManager with a real server."""

    @pytest.mark.asyncio
    async def test_health_check(self, server_url: str) -> None:
        """Test health check against real server."""
        async with GameServerManager(http_url=server_url) as manager:
            result = await manager.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_create_training_game(self, server_url: str) -> None:
        """Test creating a training game instance."""
        config = TrainingGameConfig(
            room_name=unique_room_name("ServerManagerTest"),
            map_type="basic",
            tick_multiplier=10.0,
            max_game_duration_sec=60,
            disable_respawn_timer=True,
            max_kills=20,
        )

        async with GameServerManager(http_url=server_url) as manager:
            instance = await manager.create_game(config, "TestBot")

            assert isinstance(instance, GameInstance)
            assert instance.room_id is not None
            assert instance.room_code is not None
            assert instance.player_id is not None
            assert instance.player_token is not None
            assert instance.canvas_size[0] > 0
            assert instance.canvas_size[1] > 0
            assert instance.is_active is True

            # Verify game is tracked
            assert manager.get_game_status(instance.room_id) == instance
            assert instance in manager.get_active_games()

    @pytest.mark.asyncio
    async def test_create_multiple_games(self, server_url: str) -> None:
        """Test creating multiple concurrent training games."""
        async with GameServerManager(
            http_url=server_url,
            max_concurrent_games=3,
        ) as manager:
            configs = [
                TrainingGameConfig(room_name=unique_room_name(f"MultiTest{i}"))
                for i in range(3)
            ]

            instances = []
            for config in configs:
                instance = await manager.create_game(config, "TestBot")
                instances.append(instance)

            assert len(manager.get_active_games()) == 3

            # All instances should have unique room IDs
            room_ids = [inst.room_id for inst in instances]
            assert len(set(room_ids)) == 3

    @pytest.mark.asyncio
    async def test_max_concurrent_games_enforcement(self, server_url: str) -> None:
        """Test that max concurrent games limit is enforced."""
        async with GameServerManager(
            http_url=server_url,
            max_concurrent_games=1,
        ) as manager:
            # Create first game
            config1 = TrainingGameConfig(room_name=unique_room_name("MaxTest1"))
            await manager.create_game(config1, "TestBot")

            # Second game should fail
            config2 = TrainingGameConfig(room_name=unique_room_name("MaxTest2"))
            with pytest.raises(MaxGamesExceededError):
                await manager.create_game(config2, "TestBot")

    @pytest.mark.asyncio
    async def test_terminate_game(self, server_url: str) -> None:
        """Test terminating a game instance."""
        async with GameServerManager(http_url=server_url) as manager:
            config = TrainingGameConfig(room_name=unique_room_name("TerminateTest"))
            instance = await manager.create_game(config, "TestBot")

            assert manager.get_game_status(instance.room_id) is not None

            await manager.terminate_game(instance.room_id)

            assert manager.get_game_status(instance.room_id) is None
            assert len(manager.get_active_games()) == 0

    @pytest.mark.asyncio
    async def test_terminate_all_games(self, server_url: str) -> None:
        """Test terminating all game instances."""
        async with GameServerManager(http_url=server_url) as manager:
            # Create multiple games
            for i in range(3):
                config = TrainingGameConfig(
                    room_name=unique_room_name(f"TermAllTest{i}")
                )
                await manager.create_game(config, f"TestBot{i}")

            assert len(manager.get_active_games()) == 3

            await manager.terminate_all()

            assert len(manager.get_active_games()) == 0

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, server_url: str) -> None:
        """Test that context manager cleans up games on exit."""
        manager = GameServerManager(http_url=server_url)

        async with manager:
            config = TrainingGameConfig(room_name=unique_room_name("CleanupTest"))
            await manager.create_game(config, "TestBot")
            assert len(manager.get_active_games()) == 1

        # After exiting context, games should be terminated
        assert len(manager._games) == 0

    @pytest.mark.asyncio
    async def test_reset_game(self, server_url: str) -> None:
        """Test resetting a game for a new episode."""
        async with GameServerManager(http_url=server_url) as manager:
            config = TrainingGameConfig(room_name=unique_room_name("ResetTest"))
            instance = await manager.create_game(config, "TestBot")

            # Reset the game
            state = await manager.reset_game(instance.room_id)

            # Should return a valid game state
            assert state is not None
            # The game should still be active
            assert manager.get_game_status(instance.room_id) is not None

    @pytest.mark.asyncio
    async def test_training_mode_options_applied(self, server_url: str) -> None:
        """Test that training mode options are applied to created games."""
        config = TrainingGameConfig(
            room_name=unique_room_name("TrainingOptionsTest"),
            map_type="basic",
            tick_multiplier=5.0,
            max_game_duration_sec=120,
            disable_respawn_timer=True,
            max_kills=30,
        )

        async with GameServerManager(http_url=server_url) as manager:
            instance = await manager.create_game(config, "TestBot")

            # Verify the config is stored in the instance
            assert instance.config.tick_multiplier == 5.0
            assert instance.config.max_game_duration_sec == 120
            assert instance.config.disable_respawn_timer is True
            assert instance.config.max_kills == 30
