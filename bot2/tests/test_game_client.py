"""Unit tests for GameClient.

Tests cover:
- Initialization with different modes
- Context manager behavior
- WebSocket mode functionality
- REST mode functionality
- Action routing based on mode
- Game state management
- Error handling
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from bot.client import (
    ClientMode,
    GameClient,
    GameClientError,
)
from bot.models import (
    CreateGameResponse,
    GameState,
    GetGameStateResponse,
    JoinGameResponse,
    PlayerStatsDTO,
    ResetGameResponse,
)


def make_create_game_response(**overrides: Any) -> CreateGameResponse:
    """Create a CreateGameResponse for testing."""
    data: dict[str, Any] = {
        "success": True,
        "roomId": "room-123",
        "roomCode": "ABC123",
        "roomName": "Test Room",
        "playerId": "player-456",
        "playerToken": "token-789",
        "canvasSizeX": 800,
        "canvasSizeY": 600,
    }
    data.update(overrides)
    return CreateGameResponse.model_validate(data)


def make_join_game_response(**overrides: Any) -> JoinGameResponse:
    """Create a JoinGameResponse for testing."""
    data: dict[str, Any] = {
        "success": True,
        "roomId": "room-123",
        "roomCode": "ABC123",
        "playerId": "player-789",
        "playerToken": "token-012",
        "isSpectator": False,
        "canvasSizeX": 800,
        "canvasSizeY": 600,
    }
    data.update(overrides)
    return JoinGameResponse.model_validate(data)


def make_reset_game_response(**overrides: Any) -> ResetGameResponse:
    """Create a ResetGameResponse for testing."""
    data: dict[str, Any] = {
        "success": True,
        "roomId": "room-123",
        "mapType": "arena1",
        "canvasSizeX": 800,
        "canvasSizeY": 600,
    }
    data.update(overrides)
    return ResetGameResponse.model_validate(data)


def make_get_game_state_response(**overrides: Any) -> GetGameStateResponse:
    """Create a GetGameStateResponse for testing."""
    data: dict[str, Any] = {
        "success": True,
        "roomId": "room-123",
        "gameUpdate": {
            "fullUpdate": True,
            "objectStates": {},
        },
    }
    data.update(overrides)
    return GetGameStateResponse.model_validate(data)


def make_player_stats_dto(**overrides: Any) -> PlayerStatsDTO:
    """Create a PlayerStatsDTO for testing."""
    data: dict[str, Any] = {
        "playerId": "player-1",
        "playerName": "Bot1",
        "kills": 5,
        "deaths": 2,
    }
    data.update(overrides)
    return PlayerStatsDTO.model_validate(data)


class TestGameClientInit:
    """Tests for GameClient initialization."""

    def test_default_configuration(self) -> None:
        """Test client initializes with default configuration."""
        client = GameClient()
        assert client.http_url == "http://localhost:4000"
        assert client.ws_url == "ws://localhost:4000/ws"
        assert client.mode == ClientMode.WEBSOCKET
        assert client.timeout == 30.0
        assert client._http_client is not None
        assert client._websocket is None
        assert client.player_id is None
        assert client.room_id is None

    def test_websocket_mode_configuration(self) -> None:
        """Test client with explicit WebSocket mode."""
        client = GameClient(mode=ClientMode.WEBSOCKET)
        assert client.mode == ClientMode.WEBSOCKET

    def test_rest_mode_configuration(self) -> None:
        """Test client with REST mode."""
        client = GameClient(mode=ClientMode.REST)
        assert client.mode == ClientMode.REST

    def test_custom_urls(self) -> None:
        """Test client with custom URLs."""
        client = GameClient(
            http_url="http://example.com:8080",
            ws_url="ws://example.com:8080/game",
        )
        assert client.http_url == "http://example.com:8080"
        assert client.ws_url == "ws://example.com:8080/game"


class TestGameClientContextManager:
    """Tests for async context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_http_client(self) -> None:
        """Test context manager connects HTTP client."""
        client = GameClient(mode=ClientMode.REST)

        with patch.object(
            client._http_client, "connect", new_callable=AsyncMock
        ) as mock_connect:
            with patch.object(client._http_client, "close", new_callable=AsyncMock):
                async with client:
                    mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exit(self) -> None:
        """Test context manager closes client on exit."""
        client = GameClient(mode=ClientMode.REST)

        with patch.object(client._http_client, "connect", new_callable=AsyncMock):
            with patch.object(
                client._http_client, "close", new_callable=AsyncMock
            ) as mock_close:
                async with client:
                    pass
                mock_close.assert_called_once()


class TestGameClientCreateGame:
    """Tests for create_game() method."""

    @pytest.mark.asyncio
    async def test_create_game_rest_mode(self) -> None:
        """Test create_game in REST mode does not connect WebSocket."""
        client = GameClient(mode=ClientMode.REST)

        mock_response = make_create_game_response()

        with patch.object(client._http_client, "connect", new_callable=AsyncMock):
            with patch.object(client._http_client, "close", new_callable=AsyncMock):
                with patch.object(
                    client._http_client, "create_game", new_callable=AsyncMock
                ) as mock_create:
                    mock_create.return_value = mock_response

                    async with client:
                        result = await client.create_game(
                            player_name="TestBot",
                            room_name="Test Room",
                            map_type="arena1",
                        )

                        assert result.room_id == "room-123"
                        assert client.player_id == "player-456"
                        assert client.room_id == "room-123"
                        assert client.canvas_size_x == 800
                        assert client._websocket is None  # No WebSocket in REST mode

    @pytest.mark.asyncio
    async def test_create_game_websocket_mode_connects_ws(self) -> None:
        """Test create_game in WebSocket mode connects WebSocket."""
        client = GameClient(mode=ClientMode.WEBSOCKET)

        mock_response = make_create_game_response()

        mock_ws = AsyncMock()

        with patch.object(client._http_client, "connect", new_callable=AsyncMock):
            with patch.object(client._http_client, "close", new_callable=AsyncMock):
                with patch.object(
                    client._http_client, "create_game", new_callable=AsyncMock
                ) as mock_create:
                    mock_create.return_value = mock_response

                    with patch(
                        "bot.client.game_client.websockets.connect",
                        new_callable=AsyncMock,
                    ) as mock_connect:
                        mock_connect.return_value = mock_ws

                        async with client:
                            await client.create_game(
                                player_name="TestBot",
                                room_name="Test Room",
                                map_type="arena1",
                            )

                            # WebSocket should be connected
                            mock_connect.assert_called_once_with(
                                "ws://localhost:4000/ws"
                            )

    @pytest.mark.asyncio
    async def test_create_game_with_training_options(self) -> None:
        """Test create_game passes training mode options."""
        client = GameClient(mode=ClientMode.REST)

        mock_response = make_create_game_response(
            trainingMode=True,
            tickMultiplier=2.0,
        )

        with patch.object(client._http_client, "connect", new_callable=AsyncMock):
            with patch.object(client._http_client, "close", new_callable=AsyncMock):
                with patch.object(
                    client._http_client, "create_game", new_callable=AsyncMock
                ) as mock_create:
                    mock_create.return_value = mock_response

                    async with client:
                        await client.create_game(
                            player_name="MLBot",
                            room_name="Training",
                            map_type="arena1",
                            training_mode=True,
                            tick_rate_multiplier=2.0,
                        )

                        # Verify training options passed
                        mock_create.assert_called_once()
                        call_kwargs = mock_create.call_args[1]
                        assert call_kwargs["training_mode"] is True
                        assert call_kwargs["tick_multiplier"] == 2.0


class TestGameClientJoinGame:
    """Tests for join_game() method."""

    @pytest.mark.asyncio
    async def test_join_game_stores_credentials(self) -> None:
        """Test join_game stores player credentials."""
        client = GameClient(mode=ClientMode.REST)

        mock_response = make_join_game_response()

        with patch.object(client._http_client, "connect", new_callable=AsyncMock):
            with patch.object(client._http_client, "close", new_callable=AsyncMock):
                with patch.object(
                    client._http_client, "join_game", new_callable=AsyncMock
                ) as mock_join:
                    mock_join.return_value = mock_response

                    async with client:
                        result = await client.join_game(
                            player_name="JoiningBot",
                            room_code="ABC123",
                        )

                        assert result.player_id == "player-789"
                        assert client.player_id == "player-789"
                        assert client.player_token == "token-012"
                        assert client.room_id == "room-123"
                        assert client.room_code == "ABC123"


class TestGameClientKeyboardInput:
    """Tests for send_keyboard_input() method."""

    @pytest.mark.asyncio
    async def test_keyboard_input_rest_mode_submits_action(self) -> None:
        """Test keyboard input in REST mode submits via HTTP."""
        client = GameClient(mode=ClientMode.REST)
        client.room_id = "room-123"
        client.player_id = "player-456"

        with patch.object(client._http_client, "connect", new_callable=AsyncMock):
            with patch.object(client._http_client, "close", new_callable=AsyncMock):
                with patch.object(
                    client._http_client, "submit_action", new_callable=AsyncMock
                ) as mock_submit:
                    async with client:
                        await client.send_keyboard_input("d", True)

                        mock_submit.assert_called_once()
                        call_args = mock_submit.call_args
                        assert call_args[1]["room_id"] == "room-123"
                        assert call_args[1]["player_id"] == "player-456"
                        actions = call_args[1]["actions"]
                        assert len(actions) == 1
                        assert actions[0].type == "key"
                        assert actions[0].key == "d"
                        assert actions[0].is_down is True

    @pytest.mark.asyncio
    async def test_keyboard_input_websocket_mode_sends_ws(self) -> None:
        """Test keyboard input in WebSocket mode sends via WebSocket."""
        client = GameClient(mode=ClientMode.WEBSOCKET)
        mock_ws = AsyncMock()
        client._websocket = mock_ws

        await client._send_ws_keyboard("w", True)

        mock_ws.send.assert_called_once()
        import json

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "Key"
        assert sent_data["payload"]["key"] == "w"
        assert sent_data["payload"]["isDown"] is True

    @pytest.mark.asyncio
    async def test_keyboard_input_websocket_not_connected_raises(self) -> None:
        """Test keyboard input raises error if WebSocket not connected."""
        client = GameClient(mode=ClientMode.WEBSOCKET)
        client._websocket = None

        with pytest.raises(GameClientError, match="WebSocket not connected"):
            await client._send_ws_keyboard("w", True)


class TestGameClientMouseInput:
    """Tests for send_mouse_input() method."""

    @pytest.mark.asyncio
    async def test_mouse_input_rest_mode(self) -> None:
        """Test mouse input in REST mode submits via HTTP."""
        client = GameClient(mode=ClientMode.REST)
        client.room_id = "room-123"
        client.player_id = "player-456"

        with patch.object(client._http_client, "connect", new_callable=AsyncMock):
            with patch.object(client._http_client, "close", new_callable=AsyncMock):
                with patch.object(
                    client._http_client, "submit_action", new_callable=AsyncMock
                ) as mock_submit:
                    async with client:
                        await client.send_mouse_input("left", True, 100.0, 200.0)

                        mock_submit.assert_called_once()
                        actions = mock_submit.call_args[1]["actions"]
                        assert len(actions) == 1
                        assert actions[0].type == "click"
                        assert actions[0].button == 0  # left = 0
                        assert actions[0].x == 100.0
                        assert actions[0].y == 200.0

    @pytest.mark.asyncio
    async def test_mouse_input_right_button(self) -> None:
        """Test mouse input with right button."""
        client = GameClient(mode=ClientMode.REST)
        client.room_id = "room-123"
        client.player_id = "player-456"

        with patch.object(client._http_client, "connect", new_callable=AsyncMock):
            with patch.object(client._http_client, "close", new_callable=AsyncMock):
                with patch.object(
                    client._http_client, "submit_action", new_callable=AsyncMock
                ) as mock_submit:
                    async with client:
                        await client.send_mouse_input("right", False, 50.0, 75.0)

                        actions = mock_submit.call_args[1]["actions"]
                        assert actions[0].button == 2  # right = 2
                        assert actions[0].is_down is False


class TestGameClientGameState:
    """Tests for get_game_state() method."""

    @pytest.mark.asyncio
    async def test_get_game_state_rest_mode_polls_server(self) -> None:
        """Test get_game_state in REST mode polls the server."""
        client = GameClient(mode=ClientMode.REST)
        client.room_id = "room-123"
        client.canvas_size_x = 800
        client.canvas_size_y = 600

        mock_response = make_get_game_state_response(
            gameUpdate={
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "TestPlayer",
                        "x": 100.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 2,
                        "ac": 3,
                    }
                },
            },
        )

        with patch.object(client._http_client, "connect", new_callable=AsyncMock):
            with patch.object(client._http_client, "close", new_callable=AsyncMock):
                with patch.object(
                    client._http_client, "get_game_state", new_callable=AsyncMock
                ) as mock_get_state:
                    mock_get_state.return_value = mock_response

                    async with client:
                        state = await client.get_game_state()

                        assert isinstance(state, GameState)
                        mock_get_state.assert_called_once_with("room-123")

    @pytest.mark.asyncio
    async def test_get_game_state_websocket_mode_returns_cached(self) -> None:
        """Test get_game_state in WebSocket mode returns cached state."""
        client = GameClient(mode=ClientMode.WEBSOCKET)

        # Pre-populate cached state with empty players dict
        client._game_state = GameState(
            players={},
            canvas_size_x=800,
            canvas_size_y=600,
        )

        state = await client.get_game_state()
        assert state is client._game_state

    @pytest.mark.asyncio
    async def test_get_game_state_websocket_no_state_raises(self) -> None:
        """Test get_game_state raises if no cached state in WebSocket mode."""
        client = GameClient(mode=ClientMode.WEBSOCKET)
        client._game_state = None

        with pytest.raises(GameClientError, match="No game state received yet"):
            await client.get_game_state()

    @pytest.mark.asyncio
    async def test_get_game_state_rest_not_connected_raises(self) -> None:
        """Test get_game_state raises if not connected in REST mode."""
        client = GameClient(mode=ClientMode.REST)
        client.room_id = None

        with pytest.raises(GameClientError, match="Not connected to a game"):
            await client.get_game_state()


class TestGameClientResetGame:
    """Tests for reset_game() method."""

    @pytest.mark.asyncio
    async def test_reset_game_rest_mode(self) -> None:
        """Test reset_game in REST mode."""
        client = GameClient(mode=ClientMode.REST)
        client.room_id = "room-123"

        mock_response = make_reset_game_response(
            mapType="arena2",
            canvasSizeX=1024,
            canvasSizeY=768,
        )

        with patch.object(client._http_client, "connect", new_callable=AsyncMock):
            with patch.object(client._http_client, "close", new_callable=AsyncMock):
                with patch.object(
                    client._http_client, "reset_game", new_callable=AsyncMock
                ) as mock_reset:
                    mock_reset.return_value = mock_response

                    async with client:
                        await client.reset_game(map_type="arena2")

                        mock_reset.assert_called_once_with("room-123", "arena2")
                        assert client.canvas_size_x == 1024
                        assert client.canvas_size_y == 768
                        assert client._game_state is None  # State cleared

    @pytest.mark.asyncio
    async def test_reset_game_websocket_mode_raises(self) -> None:
        """Test reset_game raises in WebSocket mode."""
        client = GameClient(mode=ClientMode.WEBSOCKET)

        with pytest.raises(GameClientError, match="only available in REST mode"):
            await client.reset_game()

    @pytest.mark.asyncio
    async def test_reset_game_not_connected_raises(self) -> None:
        """Test reset_game raises if not connected."""
        client = GameClient(mode=ClientMode.REST)
        client.room_id = None

        with pytest.raises(GameClientError, match="Not connected to a game"):
            await client.reset_game()


class TestGameClientGetStats:
    """Tests for get_stats() method."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self) -> None:
        """Test get_stats retrieves player statistics."""
        client = GameClient(mode=ClientMode.REST)
        client.room_id = "room-123"

        mock_stats = {
            "player-1": make_player_stats_dto(),
        }

        with patch.object(client._http_client, "connect", new_callable=AsyncMock):
            with patch.object(client._http_client, "close", new_callable=AsyncMock):
                with patch.object(
                    client._http_client, "get_game_stats", new_callable=AsyncMock
                ) as mock_get_stats:
                    mock_get_stats.return_value = mock_stats

                    async with client:
                        stats = await client.get_stats()

                        assert "player-1" in stats
                        assert stats["player-1"].kills == 5

    @pytest.mark.asyncio
    async def test_get_stats_not_connected_raises(self) -> None:
        """Test get_stats raises if not connected."""
        client = GameClient(mode=ClientMode.REST)
        client.room_id = None

        with pytest.raises(GameClientError, match="Not connected to a game"):
            await client.get_stats()


class TestGameClientMessageHandlers:
    """Tests for message handler registration."""

    def test_register_message_handler(self) -> None:
        """Test registering a message handler."""
        client = GameClient()

        async def handler(msg: dict) -> None:  # type: ignore[type-arg]
            pass

        client.register_message_handler(handler)
        assert handler in client._message_handlers

    def test_unregister_message_handler(self) -> None:
        """Test unregistering a message handler."""
        client = GameClient()

        async def handler(msg: dict) -> None:  # type: ignore[type-arg]
            pass

        client.register_message_handler(handler)
        client.unregister_message_handler(handler)
        assert handler not in client._message_handlers

    def test_unregister_nonexistent_handler_safe(self) -> None:
        """Test unregistering a non-existent handler is safe."""
        client = GameClient()

        async def handler(msg: dict) -> None:  # type: ignore[type-arg]
            pass

        # Should not raise
        client.unregister_message_handler(handler)


class TestGameClientHandleMessage:
    """Tests for WebSocket message handling."""

    @pytest.mark.asyncio
    async def test_handle_game_update_message(self) -> None:
        """Test handling GameUpdate message updates cached state."""
        client = GameClient(mode=ClientMode.WEBSOCKET)
        client.canvas_size_x = 800
        client.canvas_size_y = 600

        message = """{
            "type": "GameUpdate",
            "payload": {
                "fullUpdate": true,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "TestPlayer",
                        "x": 100.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": false,
                        "sht": false,
                        "jc": 2,
                        "ac": 3
                    }
                }
            }
        }"""

        await client._handle_message(message)

        assert client._game_state is not None
        assert "player-1" in client._game_state.players

    @pytest.mark.asyncio
    async def test_handle_message_calls_handlers(self) -> None:
        """Test message handling calls registered handlers."""
        client = GameClient(mode=ClientMode.WEBSOCKET)

        received_messages: list[dict] = []  # type: ignore[type-arg]

        async def handler(msg: dict) -> None:  # type: ignore[type-arg]
            received_messages.append(msg)

        client.register_message_handler(handler)

        message = '{"type": "CustomMessage", "data": "test"}'
        await client._handle_message(message)

        assert len(received_messages) == 1
        assert received_messages[0]["type"] == "CustomMessage"

    @pytest.mark.asyncio
    async def test_handle_message_invalid_json(self) -> None:
        """Test handling invalid JSON does not crash."""
        client = GameClient()

        # Should not raise, just log error
        await client._handle_message("not valid json")

    @pytest.mark.asyncio
    async def test_handle_message_handler_error_isolated(self) -> None:
        """Test handler error does not affect other handlers."""
        client = GameClient()

        calls: list[str] = []

        async def good_handler(msg: dict) -> None:  # type: ignore[type-arg]
            calls.append("good")

        async def bad_handler(msg: dict) -> None:  # type: ignore[type-arg]
            raise ValueError("Handler error")

        client.register_message_handler(bad_handler)
        client.register_message_handler(good_handler)

        await client._handle_message('{"type": "Test"}')

        # Good handler should still be called despite bad handler error
        assert "good" in calls


class TestGameClientClose:
    """Tests for close() method."""

    @pytest.mark.asyncio
    async def test_close_clears_state(self) -> None:
        """Test close clears all state."""
        client = GameClient(mode=ClientMode.REST)
        client.player_id = "player-123"
        client.room_id = "room-456"
        client._game_state = GameState()

        async def mock_handler(msg: dict) -> None:  # type: ignore[type-arg]
            pass

        client.register_message_handler(mock_handler)

        with patch.object(client._http_client, "close", new_callable=AsyncMock):
            await client.close()

        assert client.player_id is None
        assert client.room_id is None
        assert client._game_state is None
        assert len(client._message_handlers) == 0

    @pytest.mark.asyncio
    async def test_close_cancels_listener_task(self) -> None:
        """Test close cancels WebSocket listener task."""
        import asyncio

        client = GameClient(mode=ClientMode.WEBSOCKET)

        # Create a mock task
        async def mock_listener() -> None:
            await asyncio.sleep(100)

        client._listener_task = asyncio.create_task(mock_listener())

        with patch.object(client._http_client, "close", new_callable=AsyncMock):
            await client.close()

        assert client._listener_task is None


class TestGameClientExitGame:
    """Tests for exit_game() method."""

    @pytest.mark.asyncio
    async def test_exit_game_websocket_sends_message(self) -> None:
        """Test exit_game sends exit message via WebSocket."""
        client = GameClient(mode=ClientMode.WEBSOCKET)
        mock_ws = AsyncMock()
        client._websocket = mock_ws

        with patch.object(client._http_client, "close", new_callable=AsyncMock):
            await client.exit_game()

        # Should have sent exit message
        assert mock_ws.send.called
        import json

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "ExitGame"


class TestGameClientModeImmutability:
    """Tests for mode immutability."""

    def test_mode_set_at_init(self) -> None:
        """Test mode is set at initialization."""
        ws_client = GameClient(mode=ClientMode.WEBSOCKET)
        rest_client = GameClient(mode=ClientMode.REST)

        assert ws_client.mode == ClientMode.WEBSOCKET
        assert rest_client.mode == ClientMode.REST


class TestClientModeEnum:
    """Tests for ClientMode enum."""

    def test_websocket_value(self) -> None:
        """Test WEBSOCKET enum value."""
        assert ClientMode.WEBSOCKET.value == "websocket"

    def test_rest_value(self) -> None:
        """Test REST enum value."""
        assert ClientMode.REST.value == "rest"


class TestImports:
    """Tests that all imports work correctly."""

    def test_imports_from_bot_client(self) -> None:
        """Test importing from bot.client works."""
        from bot.client import (
            ClientMode,
            GameClient,
            GameClientError,
        )

        assert GameClient is not None
        assert ClientMode is not None
        assert GameClientError is not None

    def test_gameclient_error_inherits_from_http_error(self) -> None:
        """Test GameClientError inherits from GameHTTPClientError."""
        from bot.client import GameClientError, GameHTTPClientError

        error = GameClientError("Test error")
        assert isinstance(error, GameHTTPClientError)
        assert isinstance(error, Exception)
