"""Integration tests for BotManager with real backend.

Tests cover:
- Spawning rule-based bots and verifying connection to game server
- Bot lifecycle management (spawn, query status, destroy)
- Multiple bots in the same or different rooms
- Error handling for invalid room codes and configurations
- Graceful shutdown of BotManager with active bots
"""

import asyncio

import pytest

from bot.client import ClientMode, GameClient
from bot.models import GameState
from bot.service.bot_manager import (
    BotConfig,
    BotManager,
    SpawnBotRequest,
)
from tests.conftest import requires_server, unique_room_name


@pytest.mark.integration
class TestBotManagerSpawnRuleBasedBot:
    """Integration tests for spawning rule-based bots."""

    @requires_server
    @pytest.mark.asyncio
    async def test_spawn_rule_based_bot_connects_to_game(self, server_url: str) -> None:
        """Spawn a rule-based bot and verify it connects to an existing game."""
        room_name = unique_room_name("BotMgrSpawn")
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        # First, create a game room to join
        host_client = GameClient(http_url=server_url, mode=ClientMode.REST)
        async with host_client:
            create_response = await host_client.create_game(
                player_name="HostPlayer",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )
            assert create_response.success is True
            room_code = create_response.room_code
            room_password = create_response.room_password
            assert room_code is not None

            # Create BotManager (no registry needed for rule-based bots)
            manager = BotManager(
                registry=None,
                http_url=server_url,
                ws_url=ws_url,
            )

            try:
                # Spawn a rule-based bot
                request = SpawnBotRequest(
                    room_code=room_code,
                    room_password=room_password,
                    bot_config=BotConfig(
                        bot_type="rule_based",
                        player_name="IntegrationTestBot",
                    ),
                )
                response = await manager.spawn_bot(request)

                assert response.success is True
                assert response.bot_id is not None
                assert response.bot_id.startswith("bot_")

                # Wait for background connection to complete
                await manager.await_pending_tasks()

                # Verify bot is tracked and connected
                bot_info = manager.get_bot(response.bot_id)
                assert bot_info is not None
                assert bot_info.bot_id == response.bot_id
                assert bot_info.bot_type == "rule_based"
                assert bot_info.player_name == "IntegrationTestBot"
                assert bot_info.room_code == room_code
                assert bot_info.is_connected is True

                # Verify the bot appears in the game state
                state = await host_client.get_game_state()
                assert isinstance(state, GameState)
                # Should have at least 2 players (host + bot)
                assert len(state.players) >= 2

            finally:
                await manager.shutdown()

    @requires_server
    @pytest.mark.asyncio
    async def test_spawn_rule_based_bot_receives_game_state(
        self, server_url: str
    ) -> None:
        """Verify spawned bot receives game state updates."""
        room_name = unique_room_name("BotMgrState")
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        # Create a game room
        host_client = GameClient(http_url=server_url, mode=ClientMode.REST)
        async with host_client:
            create_response = await host_client.create_game(
                player_name="HostPlayer",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )
            room_code = create_response.room_code
            room_password = create_response.room_password
            assert room_code is not None

            manager = BotManager(
                registry=None,
                http_url=server_url,
                ws_url=ws_url,
            )

            try:
                # Spawn bot
                request = SpawnBotRequest(
                    room_code=room_code,
                    room_password=room_password,
                    bot_config=BotConfig(
                        bot_type="rule_based",
                        player_name="StateTestBot",
                    ),
                )
                response = await manager.spawn_bot(request)
                assert response.success is True

                # Wait for connection
                await manager.await_pending_tasks()

                # Give bot time to receive game state and potentially send actions
                await asyncio.sleep(0.5)

                # Bot should still be connected and active
                assert response.bot_id is not None
                bot_info = manager.get_bot(response.bot_id)
                assert bot_info is not None
                assert bot_info.is_connected is True

            finally:
                await manager.shutdown()


@pytest.mark.integration
class TestBotManagerLifecycle:
    """Integration tests for bot lifecycle management."""

    @requires_server
    @pytest.mark.asyncio
    async def test_destroy_bot_disconnects_from_game(self, server_url: str) -> None:
        """Verify destroying a bot properly disconnects it from the game."""
        room_name = unique_room_name("BotMgrDestroy")
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        host_client = GameClient(http_url=server_url, mode=ClientMode.REST)
        async with host_client:
            create_response = await host_client.create_game(
                player_name="HostPlayer",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )
            room_code = create_response.room_code
            room_password = create_response.room_password
            assert room_code is not None

            manager = BotManager(
                registry=None,
                http_url=server_url,
                ws_url=ws_url,
            )

            try:
                # Spawn bot
                request = SpawnBotRequest(
                    room_code=room_code,
                    room_password=room_password,
                    bot_config=BotConfig(
                        bot_type="rule_based",
                        player_name="DestroyTestBot",
                    ),
                )
                response = await manager.spawn_bot(request)
                assert response.success is True
                bot_id = response.bot_id
                assert bot_id is not None

                # Wait for connection
                await manager.await_pending_tasks()

                # Verify bot exists
                assert manager.get_bot(bot_id) is not None

                # Destroy the bot
                destroyed = await manager.destroy_bot(bot_id)
                assert destroyed is True

                # Bot should no longer be tracked
                assert manager.get_bot(bot_id) is None

                # Destroying again should return False
                destroyed_again = await manager.destroy_bot(bot_id)
                assert destroyed_again is False

            finally:
                await manager.shutdown()

    @requires_server
    @pytest.mark.asyncio
    async def test_list_bots_returns_active_bots(self, server_url: str) -> None:
        """Verify list_bots returns all active bots."""
        room_name = unique_room_name("BotMgrList")
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        host_client = GameClient(http_url=server_url, mode=ClientMode.REST)
        async with host_client:
            create_response = await host_client.create_game(
                player_name="HostPlayer",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )
            room_code = create_response.room_code
            room_password = create_response.room_password
            assert room_code is not None

            manager = BotManager(
                registry=None,
                http_url=server_url,
                ws_url=ws_url,
            )

            try:
                # Initially no bots
                assert manager.list_bots() == []

                # Spawn first bot
                response1 = await manager.spawn_bot(
                    SpawnBotRequest(
                        room_code=room_code,
                        room_password=room_password,
                        bot_config=BotConfig(
                            bot_type="rule_based",
                            player_name="Bot1",
                        ),
                    )
                )
                assert response1.success is True

                # Spawn second bot
                response2 = await manager.spawn_bot(
                    SpawnBotRequest(
                        room_code=room_code,
                        room_password=room_password,
                        bot_config=BotConfig(
                            bot_type="rule_based",
                            player_name="Bot2",
                        ),
                    )
                )
                assert response2.success is True

                # Wait for connections
                await manager.await_pending_tasks()

                # List should have 2 bots
                bots = manager.list_bots()
                assert len(bots) == 2

                bot_ids = {b.bot_id for b in bots}
                assert response1.bot_id in bot_ids
                assert response2.bot_id in bot_ids

                # Destroy one bot
                assert response1.bot_id is not None
                await manager.destroy_bot(response1.bot_id)

                # List should have 1 bot
                bots = manager.list_bots()
                assert len(bots) == 1
                assert bots[0].bot_id == response2.bot_id

            finally:
                await manager.shutdown()


@pytest.mark.integration
class TestBotManagerMultipleBots:
    """Integration tests for multiple bot scenarios."""

    @requires_server
    @pytest.mark.asyncio
    async def test_multiple_bots_same_room(self, server_url: str) -> None:
        """Spawn multiple bots in the same game room."""
        room_name = unique_room_name("BotMgrMulti")
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        host_client = GameClient(http_url=server_url, mode=ClientMode.REST)
        async with host_client:
            create_response = await host_client.create_game(
                player_name="HostPlayer",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )
            room_code = create_response.room_code
            room_password = create_response.room_password
            assert room_code is not None

            manager = BotManager(
                registry=None,
                http_url=server_url,
                ws_url=ws_url,
            )

            try:
                # Spawn 3 bots
                bot_ids = []
                for i in range(3):
                    response = await manager.spawn_bot(
                        SpawnBotRequest(
                            room_code=room_code,
                            room_password=room_password,
                            bot_config=BotConfig(
                                bot_type="rule_based",
                                player_name=f"MultiBot{i}",
                            ),
                        )
                    )
                    assert response.success is True
                    assert response.bot_id is not None
                    bot_ids.append(response.bot_id)

                # All bot IDs should be unique
                assert len(set(bot_ids)) == 3

                # Wait for all connections
                await manager.await_pending_tasks()

                # All bots should be connected
                for bot_id in bot_ids:
                    bot_info = manager.get_bot(bot_id)
                    assert bot_info is not None
                    assert bot_info.is_connected is True
                    assert bot_info.room_code == room_code

                # Game should have multiple players
                state = await host_client.get_game_state()
                # Host + 3 bots = 4 players
                assert len(state.players) >= 4

            finally:
                await manager.shutdown()

    @requires_server
    @pytest.mark.asyncio
    async def test_bots_in_different_rooms(self, server_url: str) -> None:
        """Spawn bots in different game rooms."""
        room_name1 = unique_room_name("BotMgrRoom1")
        room_name2 = unique_room_name("BotMgrRoom2")
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        host_client1 = GameClient(http_url=server_url, mode=ClientMode.REST)
        host_client2 = GameClient(http_url=server_url, mode=ClientMode.REST)

        async with host_client1, host_client2:
            # Create two separate game rooms
            response1 = await host_client1.create_game(
                player_name="Host1",
                room_name=room_name1,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )
            response2 = await host_client2.create_game(
                player_name="Host2",
                room_name=room_name2,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            room_code1 = response1.room_code
            room_password1 = response1.room_password
            room_code2 = response2.room_code
            room_password2 = response2.room_password
            assert room_code1 is not None
            assert room_code2 is not None

            manager = BotManager(
                registry=None,
                http_url=server_url,
                ws_url=ws_url,
            )

            try:
                # Spawn bot in room 1
                spawn1 = await manager.spawn_bot(
                    SpawnBotRequest(
                        room_code=room_code1,
                        room_password=room_password1,
                        bot_config=BotConfig(
                            bot_type="rule_based",
                            player_name="BotRoom1",
                        ),
                    )
                )
                assert spawn1.success is True

                # Spawn bot in room 2
                spawn2 = await manager.spawn_bot(
                    SpawnBotRequest(
                        room_code=room_code2,
                        room_password=room_password2,
                        bot_config=BotConfig(
                            bot_type="rule_based",
                            player_name="BotRoom2",
                        ),
                    )
                )
                assert spawn2.success is True

                # Wait for connections
                await manager.await_pending_tasks()

                # Verify bots are in different rooms
                assert spawn1.bot_id is not None
                assert spawn2.bot_id is not None
                bot1_info = manager.get_bot(spawn1.bot_id)
                bot2_info = manager.get_bot(spawn2.bot_id)

                assert bot1_info is not None
                assert bot2_info is not None
                assert bot1_info.room_code == room_code1
                assert bot2_info.room_code == room_code2
                assert bot1_info.is_connected is True
                assert bot2_info.is_connected is True

            finally:
                await manager.shutdown()


@pytest.mark.integration
class TestBotManagerErrorHandling:
    """Integration tests for error handling scenarios."""

    @requires_server
    @pytest.mark.asyncio
    async def test_spawn_bot_invalid_room_code(self, server_url: str) -> None:
        """Spawning a bot with invalid room code should fail gracefully."""
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        manager = BotManager(
            registry=None,
            http_url=server_url,
            ws_url=ws_url,
        )

        try:
            # Spawn with a non-existent room code
            response = await manager.spawn_bot(
                SpawnBotRequest(
                    room_code="INVALID_ROOM_CODE_12345",
                    room_password="FAKE",  # Need non-empty password for server
                    bot_config=BotConfig(
                        bot_type="rule_based",
                        player_name="InvalidRoomBot",
                    ),
                )
            )

            # Spawn should initially succeed (background connection)
            assert response.success is True
            bot_id = response.bot_id
            assert bot_id is not None

            # Wait for background connection to fail
            await manager.await_pending_tasks()

            # Bot should be removed after failed connection
            bot_info = manager.get_bot(bot_id)
            assert bot_info is None

            # List should be empty
            assert manager.list_bots() == []

        finally:
            await manager.shutdown()

    @requires_server
    @pytest.mark.asyncio
    async def test_spawn_neural_network_bot_without_registry(
        self, server_url: str
    ) -> None:
        """Spawning neural network bot without registry should fail."""
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        # Create manager without registry
        manager = BotManager(
            registry=None,
            http_url=server_url,
            ws_url=ws_url,
        )

        try:
            # Try to spawn neural network bot
            response = await manager.spawn_bot(
                SpawnBotRequest(
                    room_code="ANYROOM",
                    room_password="FAKE",  # Need non-empty password for server
                    bot_config=BotConfig(
                        bot_type="neural_network",
                        model_id="some_model",
                        player_name="NeuralBot",
                    ),
                )
            )

            # Should fail because no registry
            assert response.success is False
            assert response.error is not None
            assert "registry" in response.error.lower()

        finally:
            await manager.shutdown()


@pytest.mark.integration
class TestBotManagerShutdown:
    """Integration tests for graceful shutdown."""

    @requires_server
    @pytest.mark.asyncio
    async def test_shutdown_stops_all_bots(self, server_url: str) -> None:
        """Verify shutdown stops all active bots."""
        room_name = unique_room_name("BotMgrShutdown")
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        host_client = GameClient(http_url=server_url, mode=ClientMode.REST)
        async with host_client:
            create_response = await host_client.create_game(
                player_name="HostPlayer",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )
            room_code = create_response.room_code
            room_password = create_response.room_password
            assert room_code is not None

            manager = BotManager(
                registry=None,
                http_url=server_url,
                ws_url=ws_url,
            )

            # Spawn multiple bots
            for i in range(3):
                response = await manager.spawn_bot(
                    SpawnBotRequest(
                        room_code=room_code,
                        room_password=room_password,
                        bot_config=BotConfig(
                            bot_type="rule_based",
                            player_name=f"ShutdownBot{i}",
                        ),
                    )
                )
                assert response.success is True

            # Wait for connections
            await manager.await_pending_tasks()

            # Verify bots are active
            assert len(manager.list_bots()) == 3

            # Shutdown
            await manager.shutdown()

            # All bots should be removed
            assert len(manager.list_bots()) == 0

    @requires_server
    @pytest.mark.asyncio
    async def test_shutdown_with_no_bots(self, server_url: str) -> None:
        """Verify shutdown works correctly with no active bots."""
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        manager = BotManager(
            registry=None,
            http_url=server_url,
            ws_url=ws_url,
        )

        # Shutdown with no bots should not raise
        await manager.shutdown()

        # Should still be empty
        assert len(manager.list_bots()) == 0


@pytest.mark.integration
@pytest.mark.slow
class TestBotManagerConcurrency:
    """Integration tests for concurrent bot operations."""

    @requires_server
    @pytest.mark.asyncio
    async def test_concurrent_spawn_operations(self, server_url: str) -> None:
        """Spawn multiple bots concurrently."""
        room_name = unique_room_name("BotMgrConcurrent")
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        host_client = GameClient(http_url=server_url, mode=ClientMode.REST)
        async with host_client:
            create_response = await host_client.create_game(
                player_name="HostPlayer",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )
            room_code = create_response.room_code
            room_password = create_response.room_password
            assert room_code is not None

            manager = BotManager(
                registry=None,
                http_url=server_url,
                ws_url=ws_url,
            )

            try:
                # Spawn 5 bots concurrently
                spawn_tasks = [
                    manager.spawn_bot(
                        SpawnBotRequest(
                            room_code=room_code,
                            room_password=room_password,
                            bot_config=BotConfig(
                                bot_type="rule_based",
                                player_name=f"ConcurrentBot{i}",
                            ),
                        )
                    )
                    for i in range(5)
                ]

                responses = await asyncio.gather(*spawn_tasks)

                # All spawns should succeed
                for response in responses:
                    assert response.success is True
                    assert response.bot_id is not None

                # All bot IDs should be unique
                bot_ids = [r.bot_id for r in responses]
                assert len(set(bot_ids)) == 5

                # Wait for all connections
                await manager.await_pending_tasks()

                # All bots should be connected
                bots = manager.list_bots()
                assert len(bots) == 5
                for bot in bots:
                    assert bot.is_connected is True

            finally:
                await manager.shutdown()

    @requires_server
    @pytest.mark.asyncio
    async def test_spawn_and_destroy_interleaved(self, server_url: str) -> None:
        """Test interleaved spawn and destroy operations."""
        room_name = unique_room_name("BotMgrInterleaved")
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        host_client = GameClient(http_url=server_url, mode=ClientMode.REST)
        async with host_client:
            create_response = await host_client.create_game(
                player_name="HostPlayer",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )
            room_code = create_response.room_code
            room_password = create_response.room_password
            assert room_code is not None

            manager = BotManager(
                registry=None,
                http_url=server_url,
                ws_url=ws_url,
            )

            try:
                # Spawn first bot
                response1 = await manager.spawn_bot(
                    SpawnBotRequest(
                        room_code=room_code,
                        room_password=room_password,
                        bot_config=BotConfig(
                            bot_type="rule_based",
                            player_name="InterleavedBot1",
                        ),
                    )
                )
                assert response1.success is True
                await manager.await_pending_tasks()

                # Spawn second bot
                response2 = await manager.spawn_bot(
                    SpawnBotRequest(
                        room_code=room_code,
                        room_password=room_password,
                        bot_config=BotConfig(
                            bot_type="rule_based",
                            player_name="InterleavedBot2",
                        ),
                    )
                )
                assert response2.success is True
                await manager.await_pending_tasks()

                # Destroy first bot
                assert response1.bot_id is not None
                await manager.destroy_bot(response1.bot_id)

                # Spawn third bot
                response3 = await manager.spawn_bot(
                    SpawnBotRequest(
                        room_code=room_code,
                        room_password=room_password,
                        bot_config=BotConfig(
                            bot_type="rule_based",
                            player_name="InterleavedBot3",
                        ),
                    )
                )
                assert response3.success is True
                await manager.await_pending_tasks()

                # Should have 2 bots (bot2 and bot3)
                bots = manager.list_bots()
                assert len(bots) == 2

                bot_ids = {b.bot_id for b in bots}
                assert response1.bot_id not in bot_ids
                assert response2.bot_id in bot_ids
                assert response3.bot_id in bot_ids

            finally:
                await manager.shutdown()
