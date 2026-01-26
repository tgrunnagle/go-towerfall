"""Unit tests for WebSocketBotClient.

Tests cover:
- Initialization with config and default config
- start() creates GameClient and joins room
- start() registers message handler
- stop() closes client connection
- stop() is safe to call without start
- stop() is safe to call multiple times
- reset() delegates to runner.reset()
- reset() is safe before start
- Game state forwarded to runner.on_game_state()
- Runner errors are caught and logged (don't crash client)
- player_id property returns client's player_id
- is_connected property reflects connection state
- Message handler only processes GameState messages
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.client import ClientMode, GameClient
from bot.models import GameState
from bot.service.websocket_bot import (
    WebSocketBotClient,
    WebSocketBotClientConfig,
)


class MockBotRunner:
    """Mock bot runner for testing."""

    def __init__(self) -> None:
        self.on_game_state = AsyncMock()
        self.reset = MagicMock()
        self.client: GameClient | None = None


class TestWebSocketBotClientInit:
    """Tests for WebSocketBotClient initialization."""

    def test_init_with_config(self) -> None:
        """Initialize with custom config."""
        runner = MockBotRunner()
        config = WebSocketBotClientConfig(
            http_url="http://example.com:8000",
            ws_url="ws://example.com:8000/ws",
            player_name="TestBot",
            timeout=60.0,
        )
        client = WebSocketBotClient(runner=runner, config=config)  # type: ignore[arg-type]

        assert client.runner is runner
        assert client.config is config
        assert client.config.http_url == "http://example.com:8000"
        assert client.config.ws_url == "ws://example.com:8000/ws"
        assert client.config.player_name == "TestBot"
        assert client.config.timeout == 60.0

    def test_init_with_default_config(self) -> None:
        """Initialize without config uses defaults."""
        runner = MockBotRunner()
        client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        assert client.runner is runner
        assert client.config.http_url == "http://localhost:4000"
        assert client.config.ws_url == "ws://localhost:4000/ws"
        assert client.config.player_name == "Bot"
        assert client.config.timeout == 30.0

    def test_init_state(self) -> None:
        """Initialize with correct internal state."""
        runner = MockBotRunner()
        client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        assert client._client is None
        assert client._running is False


class TestWebSocketBotClientStart:
    """Tests for WebSocketBotClient.start() method."""

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_start_creates_client_websocket_mode(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """start() creates GameClient in WebSocket mode."""
        mock_client = AsyncMock(spec=GameClient)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        config = WebSocketBotClientConfig(
            http_url="http://example.com",
            ws_url="ws://example.com/ws",
        )
        bot_client = WebSocketBotClient(runner=runner, config=config)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123", room_password="secret")

        # Verify GameClient created with correct parameters
        mock_game_client_class.assert_called_once_with(
            http_url="http://example.com",
            ws_url="ws://example.com/ws",
            mode=ClientMode.WEBSOCKET,
        )

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_start_connects_and_joins(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """start() calls connect() and join_game()."""
        mock_client = AsyncMock(spec=GameClient)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123", room_password="secret")

        # Verify connection and join
        mock_client.connect.assert_awaited_once()
        mock_client.join_game.assert_awaited_once_with(
            room_code="ROOM123",
            player_name="Bot",
            room_password="secret",
            is_spectator=False,
        )

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_start_registers_message_handler(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """start() registers message handler."""
        mock_client = AsyncMock(spec=GameClient)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")

        # Verify message handler registered
        mock_client.register_message_handler.assert_called_once()

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_start_sets_running_flag(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """start() sets _running flag to True."""
        mock_client = AsyncMock(spec=GameClient)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        assert bot_client._running is False
        await bot_client.start(room_code="ROOM123")
        assert bot_client._running is True

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_start_sets_runner_client(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """start() sets runner's client attribute."""
        mock_client = AsyncMock(spec=GameClient)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        assert runner.client is None
        await bot_client.start(room_code="ROOM123")
        assert runner.client is mock_client


class TestWebSocketBotClientStop:
    """Tests for WebSocketBotClient.stop() method."""

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_stop_closes_client(self, mock_game_client_class: MagicMock) -> None:
        """stop() closes the GameClient connection."""
        mock_client = AsyncMock(spec=GameClient)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")
        await bot_client.stop()

        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_stop_clears_running_flag(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """stop() sets _running flag to False."""
        mock_client = AsyncMock(spec=GameClient)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")
        assert bot_client._running is True

        await bot_client.stop()
        assert bot_client._running is False

    @pytest.mark.asyncio
    async def test_stop_safe_without_start(self) -> None:
        """stop() is safe to call without start()."""
        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        # Should not raise
        await bot_client.stop()

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_stop_safe_multiple_times(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """stop() is safe to call multiple times."""
        mock_client = AsyncMock(spec=GameClient)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")
        await bot_client.stop()
        await bot_client.stop()  # Should not raise

        # Client close should only be called once
        assert mock_client.close.await_count == 1


class TestWebSocketBotClientReset:
    """Tests for WebSocketBotClient.reset() method."""

    def test_reset_delegates_to_runner(self) -> None:
        """reset() calls runner.reset()."""
        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        bot_client.reset()

        runner.reset.assert_called_once()

    def test_reset_safe_before_start(self) -> None:
        """reset() is safe to call before start()."""
        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        # Should not raise
        bot_client.reset()


class TestWebSocketBotClientMessageHandling:
    """Tests for WebSocketBotClient message handling."""

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_game_state_forwarded_to_runner(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """Game state messages are forwarded to runner.on_game_state()."""
        mock_client = AsyncMock(spec=GameClient)
        mock_game_state = MagicMock(spec=GameState)
        mock_client._game_state = mock_game_state
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")

        # Get the registered message handler
        handler = mock_client.register_message_handler.call_args[0][0]

        # Simulate GameState message
        message = {"type": "GameState", "payload": {}}
        await handler(message)

        # Verify runner.on_game_state() called with cached state
        runner.on_game_state.assert_awaited_once_with(mock_game_state)

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_non_game_state_messages_ignored(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """Non-GameState messages are ignored."""
        mock_client = AsyncMock(spec=GameClient)
        mock_client._game_state = MagicMock(spec=GameState)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")

        # Get the registered message handler
        handler = mock_client.register_message_handler.call_args[0][0]

        # Simulate non-GameState message
        message = {"type": "PlayerJoined", "payload": {}}
        await handler(message)

        # Verify runner.on_game_state() NOT called
        runner.on_game_state.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_messages_ignored_after_stop(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """Messages are ignored after stop() is called."""
        mock_client = AsyncMock(spec=GameClient)
        mock_client._game_state = MagicMock(spec=GameState)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")

        # Get the registered message handler
        handler = mock_client.register_message_handler.call_args[0][0]

        await bot_client.stop()

        # Simulate GameState message after stop
        message = {"type": "GameState", "payload": {}}
        await handler(message)

        # Verify runner.on_game_state() NOT called
        runner.on_game_state.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_runner_errors_caught_and_logged(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """Errors in runner.on_game_state() are caught and logged."""
        mock_client = AsyncMock(spec=GameClient)
        mock_client._game_state = MagicMock(spec=GameState)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        # Make runner raise an error
        runner.on_game_state.side_effect = ValueError("Test error")

        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")

        # Get the registered message handler
        handler = mock_client.register_message_handler.call_args[0][0]

        # Simulate GameState message (should not raise)
        message = {"type": "GameState", "payload": {}}
        await handler(message)

        # Verify error was caught (test passes if no exception raised)

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_messages_ignored_when_client_none(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """Messages are ignored if client is None."""
        mock_client = AsyncMock(spec=GameClient)
        mock_client._game_state = MagicMock(spec=GameState)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")

        # Get the registered message handler
        handler = mock_client.register_message_handler.call_args[0][0]

        # Set client to None (simulating edge case)
        bot_client._client = None

        # Simulate GameState message
        message = {"type": "GameState", "payload": {}}
        await handler(message)

        # Verify runner.on_game_state() NOT called
        runner.on_game_state.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_messages_ignored_when_game_state_none(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """Messages are ignored if client._game_state is None."""
        mock_client = AsyncMock(spec=GameClient)
        mock_client._game_state = None  # No game state cached yet
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")

        # Get the registered message handler
        handler = mock_client.register_message_handler.call_args[0][0]

        # Simulate GameState message
        message = {"type": "GameState", "payload": {}}
        await handler(message)

        # Verify runner.on_game_state() NOT called
        runner.on_game_state.assert_not_awaited()


class TestWebSocketBotClientProperties:
    """Tests for WebSocketBotClient properties."""

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_player_id_property_returns_client_player_id(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """player_id property returns client's player_id."""
        mock_client = AsyncMock(spec=GameClient)
        mock_client.player_id = "player-123"
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")

        assert bot_client.player_id == "player-123"

    def test_player_id_property_none_before_start(self) -> None:
        """player_id property returns None before start()."""
        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        assert bot_client.player_id is None

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_is_connected_true_when_running(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """is_connected is True when client is running."""
        mock_client = AsyncMock(spec=GameClient)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")

        assert bot_client.is_connected is True

    def test_is_connected_false_before_start(self) -> None:
        """is_connected is False before start()."""
        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        assert bot_client.is_connected is False

    @pytest.mark.asyncio
    @patch("bot.service.websocket_bot.GameClient")
    async def test_is_connected_false_after_stop(
        self, mock_game_client_class: MagicMock
    ) -> None:
        """is_connected is False after stop()."""
        mock_client = AsyncMock(spec=GameClient)
        mock_game_client_class.return_value = mock_client

        runner = MockBotRunner()
        bot_client = WebSocketBotClient(runner=runner)  # type: ignore[arg-type]

        await bot_client.start(room_code="ROOM123")
        await bot_client.stop()

        assert bot_client.is_connected is False
