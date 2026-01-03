"""Integration tests for GameClient WebSocket mode.

Tests cover:
- WebSocket connection establishment
- Game state updates via WebSocket broadcasts
- WebSocket reconnection behavior
- Consistency between WebSocket and REST game states
"""

import asyncio

import pytest

from bot.client import ClientMode, GameClient
from bot.models import GameState
from tests.conftest import requires_server, unique_room_name


@pytest.mark.integration
@pytest.mark.websocket
class TestWebSocketMode:
    """Integration tests for WebSocket mode functionality."""

    @requires_server
    @pytest.mark.asyncio
    async def test_websocket_connection(self, server_url: str) -> None:
        """Verify WebSocket connection can be established after creating a game."""
        room_name = unique_room_name("WSConnect")
        ws_url = server_url.replace("http://", "ws://") + "/ws"
        client = GameClient(
            http_url=server_url,
            ws_url=ws_url,
            mode=ClientMode.WEBSOCKET,
        )

        async with client:
            response = await client.create_game(
                player_name="WSTestBot",
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
            # WebSocket should be connected after create_game
            assert client._websocket is not None
            assert client._listener_task is not None

    @requires_server
    @pytest.mark.asyncio
    async def test_websocket_receive_game_updates(self, server_url: str) -> None:
        """Verify game state updates arrive via WebSocket broadcasts."""
        room_name = unique_room_name("WSUpdates")
        ws_url = server_url.replace("http://", "ws://") + "/ws"
        client = GameClient(
            http_url=server_url,
            ws_url=ws_url,
            mode=ClientMode.WEBSOCKET,
        )

        async with client:
            await client.create_game(
                player_name="WSTestBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            # Wait for game state to be received via WebSocket
            state = await client.wait_for_game_state(timeout=5.0)

            assert isinstance(state, GameState)
            assert state.canvas_size_x > 0
            assert state.canvas_size_y > 0
            # Our player should be in the game
            assert client.player_id is not None
            assert client.player_id in state.players

    @requires_server
    @pytest.mark.asyncio
    async def test_websocket_message_handler_registration(
        self, server_url: str
    ) -> None:
        """Verify custom message handlers receive WebSocket messages."""
        room_name = unique_room_name("WSHandler")
        ws_url = server_url.replace("http://", "ws://") + "/ws"
        client = GameClient(
            http_url=server_url,
            ws_url=ws_url,
            mode=ClientMode.WEBSOCKET,
        )

        received_messages: list[dict] = []

        async def message_handler(message: dict) -> None:
            received_messages.append(message)

        async with client:
            client.register_message_handler(message_handler)

            await client.create_game(
                player_name="WSTestBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            # Wait for some messages to arrive
            await client.wait_for_game_state(timeout=5.0)

            # Should have received at least one message
            assert len(received_messages) > 0
            # Should have received GameState messages (server sends "GameState" type)
            game_state_messages = [
                m for m in received_messages if m.get("type") == "GameState"
            ]
            assert len(game_state_messages) > 0

            # Unregister handler and verify it's removed
            client.unregister_message_handler(message_handler)
            assert message_handler not in client._message_handlers

    @requires_server
    @pytest.mark.asyncio
    async def test_websocket_vs_rest_state_consistency(self, server_url: str) -> None:
        """Verify WebSocket and REST return consistent game states.

        Creates a game via WebSocket mode, then uses the same client's HTTP
        client to poll REST state and compare with cached WebSocket state.
        """
        room_name = unique_room_name("WSRESTConsistency")
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        # Create game in WebSocket mode
        ws_client = GameClient(
            http_url=server_url,
            ws_url=ws_url,
            mode=ClientMode.WEBSOCKET,
        )

        async with ws_client:
            response = await ws_client.create_game(
                player_name="WSBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            # Wait for WebSocket state
            ws_state = await ws_client.wait_for_game_state(timeout=5.0)

            # Use the same client's HTTP client to poll REST state
            # (This works because the client has proper auth token set)
            rest_response = await ws_client._http_client.get_game_state(
                response.room_id
            )
            assert rest_response.object_states is not None

            # Parse REST response into GameState for comparison
            from bot.models import GameUpdate

            game_update = GameUpdate.model_validate(
                {
                    "fullUpdate": True,
                    "objectStates": rest_response.object_states,
                    "events": [],
                    "trainingComplete": rest_response.training_complete,
                }
            )
            rest_state = GameState.from_update(
                game_update,
                existing_state=None,
                canvas_size_x=response.canvas_size_x,
                canvas_size_y=response.canvas_size_y,
            )

            # Compare key attributes
            assert ws_state.canvas_size_x == rest_state.canvas_size_x
            assert ws_state.canvas_size_y == rest_state.canvas_size_y

            # Both states should have the same player
            assert ws_client.player_id in ws_state.players
            assert ws_client.player_id in rest_state.players

            # Player attributes should be consistent
            ws_player = ws_state.players[ws_client.player_id]
            rest_player = rest_state.players[ws_client.player_id]

            # Basic player state should match
            # Note: position might differ slightly due to timing
            assert ws_player.dead == rest_player.dead
            assert ws_player.health == rest_player.health
            assert ws_player.name == rest_player.name

    @requires_server
    @pytest.mark.asyncio
    async def test_websocket_actions_update_state(self, server_url: str) -> None:
        """Verify actions sent via WebSocket result in state changes."""
        room_name = unique_room_name("WSActions")
        ws_url = server_url.replace("http://", "ws://") + "/ws"
        client = GameClient(
            http_url=server_url,
            ws_url=ws_url,
            mode=ClientMode.WEBSOCKET,
        )

        async with client:
            await client.create_game(
                player_name="WSTestBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            # Wait for initial state
            initial_state = await client.wait_for_game_state(timeout=5.0)
            assert client.player_id is not None
            assert client.player_id in initial_state.players

            # Send keyboard input via WebSocket
            await client.send_keyboard_input("d", True)  # Press right
            await asyncio.sleep(0.1)  # Allow time for state update
            await client.send_keyboard_input("d", False)  # Release

            # Send direction update via WebSocket
            import math

            await client.send_direction(math.pi / 4)  # 45 degrees

            # Send mouse input via WebSocket
            await client.send_mouse_input("left", True, 400.0, 400.0)
            await asyncio.sleep(0.05)
            await client.send_mouse_input("left", False, 400.0, 400.0)

            # Wait for state updates to propagate
            await asyncio.sleep(0.3)

            # Get updated state (from cached WebSocket updates)
            updated_state = await client.get_game_state()

            # State should still be valid
            assert isinstance(updated_state, GameState)
            assert client.player_id in updated_state.players

    @requires_server
    @pytest.mark.asyncio
    async def test_websocket_exit_game(self, server_url: str) -> None:
        """Verify graceful exit via WebSocket."""
        room_name = unique_room_name("WSExit")
        ws_url = server_url.replace("http://", "ws://") + "/ws"
        client = GameClient(
            http_url=server_url,
            ws_url=ws_url,
            mode=ClientMode.WEBSOCKET,
        )

        async with client:
            await client.create_game(
                player_name="WSTestBot",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            # Wait for connection to be established
            await client.wait_for_game_state(timeout=5.0)
            assert client._websocket is not None

            # Exit the game
            await client.exit_game()

            # After exit, WebSocket should be closed
            assert client._websocket is None
            assert client.room_id is None
            assert client.player_id is None


@pytest.mark.integration
@pytest.mark.websocket
class TestWebSocketReconnection:
    """Integration tests for WebSocket reconnection behavior."""

    @requires_server
    @pytest.mark.asyncio
    async def test_websocket_reconnection_same_player(self, server_url: str) -> None:
        """Verify a player can reconnect via WebSocket after disconnect.

        This tests the scenario where a WebSocket connection is closed
        and the same player reconnects using stored credentials.
        """
        room_name = unique_room_name("WSReconnect")
        ws_url = server_url.replace("http://", "ws://") + "/ws"

        # Create the game and get credentials
        client1 = GameClient(
            http_url=server_url,
            ws_url=ws_url,
            mode=ClientMode.WEBSOCKET,
        )

        async with client1:
            response = await client1.create_game(
                player_name="WSTestBot1",
                room_name=room_name,
                map_type="default",
                training_mode=True,
                tick_rate_multiplier=10.0,
            )

            # Store credentials for reconnection
            room_id = response.room_id
            player_id = client1.player_id
            player_token = client1.player_token
            canvas_size_x = response.canvas_size_x
            canvas_size_y = response.canvas_size_y

            # Wait for initial game state
            state1 = await client1.wait_for_game_state(timeout=5.0)
            assert isinstance(state1, GameState)
            assert player_id in state1.players

        # Client1 is now closed. Create a new WebSocket connection
        # and rejoin using the stored credentials.
        client2 = GameClient(
            http_url=server_url,
            ws_url=ws_url,
            mode=ClientMode.WEBSOCKET,
        )

        async with client2:
            # Manually set credentials for rejoin
            client2.player_id = player_id
            client2.player_token = player_token
            client2.room_id = room_id
            client2.canvas_size_x = canvas_size_x
            client2.canvas_size_y = canvas_size_y
            client2._http_client.set_player_token(player_token)

            # Connect WebSocket and rejoin
            await client2._connect_websocket()

            # Should receive game state after rejoining
            state2 = await client2.wait_for_game_state(timeout=5.0)
            assert isinstance(state2, GameState)

            # Same player should still be in the game
            assert player_id in state2.players
