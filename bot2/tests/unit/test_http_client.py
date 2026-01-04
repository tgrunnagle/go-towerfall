"""Unit tests for GameHTTPClient.

Tests cover:
- Successful API calls with mocked responses
- Error handling for various failure modes
- Retry logic with backoff
- Context manager behavior
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import HTTPError, Response, TimeoutException

from bot.client import (
    GameAPIError,
    GameConnectionError,
    GameHTTPClient,
    GameHTTPClientError,
)
from bot.models import (
    BotAction,
    CreateGameResponse,
    JoinGameResponse,
    MapInfo,
)


class TestGameHTTPClientInit:
    """Tests for GameHTTPClient initialization."""

    def test_default_configuration(self) -> None:
        """Test client initializes with default configuration."""
        client = GameHTTPClient()
        assert client.base_url == "http://localhost:4000"
        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client._client is None

    def test_custom_configuration(self) -> None:
        """Test client accepts custom configuration."""
        client = GameHTTPClient(
            base_url="http://example.com:8080/",
            timeout=60.0,
            max_retries=5,
        )
        assert client.base_url == "http://example.com:8080"  # trailing slash stripped
        assert client.timeout == 60.0
        assert client.max_retries == 5


class TestGameHTTPClientContextManager:
    """Tests for async context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_closes(self) -> None:
        """Test context manager properly connects and closes client."""
        client = GameHTTPClient()

        async with client:
            assert client._client is not None

        assert client._client is None

    @pytest.mark.asyncio
    async def test_connect_creates_client(self) -> None:
        """Test connect() creates httpx AsyncClient."""
        client = GameHTTPClient()
        assert client._client is None

        await client.connect()
        assert client._client is not None

        await client.close()

    @pytest.mark.asyncio
    async def test_connect_is_idempotent(self) -> None:
        """Test calling connect() multiple times is safe."""
        client = GameHTTPClient()
        await client.connect()
        first_client = client._client

        await client.connect()
        assert client._client is first_client  # Same instance

        await client.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        """Test calling close() multiple times is safe."""
        client = GameHTTPClient()
        await client.connect()
        await client.close()
        await client.close()  # Should not raise
        assert client._client is None


class TestGetMaps:
    """Tests for get_maps() method."""

    @pytest.mark.asyncio
    async def test_get_maps_success(self) -> None:
        """Test successful maps retrieval."""
        mock_response_data = {
            "maps": [
                {
                    "type": "arena1",
                    "name": "Arena 1",
                    "canvas_size_x": 800,
                    "canvas_size_y": 600,
                },
                {
                    "type": "arena2",
                    "name": "Arena 2",
                    "canvas_size_x": 1024,
                    "canvas_size_y": 768,
                },
            ]
        }

        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_request.return_value = mock_response

            maps = await client.get_maps()

            assert len(maps) == 2
            assert isinstance(maps[0], MapInfo)
            assert maps[0].type == "arena1"
            assert maps[0].name == "Arena 1"
            assert maps[1].type == "arena2"

        await client.close()


class TestCreateGame:
    """Tests for create_game() method."""

    @pytest.mark.asyncio
    async def test_create_game_success(self) -> None:
        """Test successful game creation."""
        mock_response_data = {
            "success": True,
            "roomId": "room-123",
            "roomCode": "ABC123",
            "roomName": "Test Room",
            "roomPassword": "test-pass",
            "playerId": "player-456",
            "playerToken": "token-789",
            "canvasSizeX": 800,
            "canvasSizeY": 600,
        }

        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_request.return_value = mock_response

            result = await client.create_game(
                player_name="TestBot",
                room_name="Test Room",
                map_type="arena1",
            )

            assert isinstance(result, CreateGameResponse)
            assert result.success is True
            assert result.room_id == "room-123"
            assert result.room_code == "ABC123"
            assert result.player_id == "player-456"
            assert result.player_token == "token-789"

            # Verify request was made with correct data
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/api/createGame"
            assert call_args[1]["json"]["playerName"] == "TestBot"
            assert call_args[1]["json"]["roomName"] == "Test Room"
            assert call_args[1]["json"]["mapType"] == "arena1"

        await client.close()

    @pytest.mark.asyncio
    async def test_create_game_with_training_mode(self) -> None:
        """Test game creation with training mode options."""
        mock_response_data = {
            "success": True,
            "roomId": "room-123",
            "roomCode": "ABC123",
            "roomName": "Training Room",
            "roomPassword": "test-pass",
            "playerId": "player-456",
            "playerToken": "token-789",
            "canvasSizeX": 800,
            "canvasSizeY": 600,
            "trainingMode": True,
            "tickMultiplier": 2.0,
        }

        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_request.return_value = mock_response

            result = await client.create_game(
                player_name="TrainingBot",
                room_name="Training Room",
                map_type="arena1",
                training_mode=True,
                tick_multiplier=2.0,
            )

            assert result.training_mode is True
            assert result.tick_multiplier == 2.0

            # Verify training mode options sent in request
            call_args = mock_request.call_args
            assert call_args[1]["json"]["trainingMode"] is True
            assert call_args[1]["json"]["tickMultiplier"] == 2.0

        await client.close()


class TestJoinGame:
    """Tests for join_game() method."""

    @pytest.mark.asyncio
    async def test_join_game_success(self) -> None:
        """Test successful game joining."""
        mock_response_data = {
            "success": True,
            "roomId": "room-123",
            "roomCode": "ABC123",
            "roomName": "Test Room",
            "playerId": "player-789",
            "playerToken": "token-012",
            "isSpectator": False,
            "canvasSizeX": 800,
            "canvasSizeY": 600,
        }

        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_request.return_value = mock_response

            result = await client.join_game(
                player_name="JoiningBot",
                room_code="ABC123",
            )

            assert isinstance(result, JoinGameResponse)
            assert result.success is True
            assert result.room_code == "ABC123"
            assert result.player_id == "player-789"
            assert result.is_spectator is False

        await client.close()

    @pytest.mark.asyncio
    async def test_join_game_as_spectator(self) -> None:
        """Test joining game as spectator."""
        mock_response_data = {
            "success": True,
            "roomId": "room-123",
            "roomCode": "ABC123",
            "playerId": "spectator-789",
            "playerToken": "token-012",
            "isSpectator": True,
            "canvasSizeX": 800,
            "canvasSizeY": 600,
        }

        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_request.return_value = mock_response

            result = await client.join_game(
                player_name="Spectator",
                room_code="ABC123",
                is_spectator=True,
            )

            assert result.is_spectator is True

            # Verify spectator flag sent in request
            call_args = mock_request.call_args
            assert call_args[1]["json"]["isSpectator"] is True

        await client.close()


class TestTrainingModeEndpoints:
    """Tests for training mode API methods."""

    @pytest.mark.asyncio
    async def test_get_game_state_success(self) -> None:
        """Test successful game state retrieval."""
        mock_response_data = {
            "success": True,
            "roomId": "room-123",
            "objectStates": {},
            "trainingComplete": False,
        }

        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_request.return_value = mock_response

            result = await client.get_game_state("room-123")

            assert result.success is True
            assert result.room_id == "room-123"
            # Verify endpoint is correct (headers may be added by _request)
            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"
            assert "/api/rooms/room-123/state" in call_args[0][1]

        await client.close()

    @pytest.mark.asyncio
    async def test_submit_action_success(self) -> None:
        """Test successful action submission."""
        mock_response_data = {
            "success": True,
            "actionsProcessed": 2,
            "timestamp": 1234567890,
        }

        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_request.return_value = mock_response

            actions = [
                BotAction(type="key", key="W", is_down=True),
                BotAction(type="click", x=100.0, y=200.0, button=0, is_down=True),
            ]

            result = await client.submit_action("room-123", "player-456", actions)

            assert result.success is True
            assert result.actions_processed == 2

            call_args = mock_request.call_args
            # Note: endpoint is /action (singular), not /actions
            assert "/api/rooms/room-123/players/player-456/action" in call_args[0][1]

        await client.close()

    @pytest.mark.asyncio
    async def test_reset_game_success(self) -> None:
        """Test successful game reset."""
        mock_response_data = {
            "success": True,
            "roomId": "room-123",
            "mapType": "arena2",
            "canvasSizeX": 1024,
            "canvasSizeY": 768,
        }

        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_request.return_value = mock_response

            result = await client.reset_game("room-123", map_type="arena2")

            assert result.success is True
            assert result.map_type == "arena2"

        await client.close()

    @pytest.mark.asyncio
    async def test_get_game_stats_success(self) -> None:
        """Test successful stats retrieval."""
        mock_response_data = {
            "success": True,
            "roomId": "room-123",
            "playerStats": {
                "player-1": {
                    "playerId": "player-1",
                    "playerName": "Bot1",
                    "kills": 5,
                    "deaths": 2,
                },
                "player-2": {
                    "playerId": "player-2",
                    "playerName": "Bot2",
                    "kills": 2,
                    "deaths": 5,
                },
            },
        }

        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_request.return_value = mock_response

            stats = await client.get_game_stats("room-123")

            assert len(stats) == 2
            assert stats["player-1"].kills == 5
            assert stats["player-2"].deaths == 5

        await client.close()


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_not_connected_raises_error(self) -> None:
        """Test that making request without connecting raises error."""
        client = GameHTTPClient()
        # Don't call connect()

        with pytest.raises(GameConnectionError, match="Client not connected"):
            await client.get_maps()

    @pytest.mark.asyncio
    async def test_http_400_raises_api_error(self) -> None:
        """Test that HTTP 400 raises GameAPIError."""
        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 400
            mock_response.text = "Bad Request: Invalid room code"
            mock_request.return_value = mock_response

            with pytest.raises(GameAPIError) as exc_info:
                await client.join_game("Bot", "INVALID")

            assert exc_info.value.status_code == 400
            assert "Bad Request" in str(exc_info.value)

        await client.close()

    @pytest.mark.asyncio
    async def test_http_404_raises_api_error(self) -> None:
        """Test that HTTP 404 raises GameAPIError."""
        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 404
            mock_response.text = "Room not found"
            mock_request.return_value = mock_response

            with pytest.raises(GameAPIError) as exc_info:
                await client.join_game("Bot", "NOTFOUND")

            assert exc_info.value.status_code == 404

        await client.close()

    @pytest.mark.asyncio
    async def test_http_500_raises_api_error(self) -> None:
        """Test that HTTP 500 raises GameAPIError."""
        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_request.return_value = mock_response

            with pytest.raises(GameAPIError) as exc_info:
                await client.get_maps()

            assert exc_info.value.status_code == 500

        await client.close()

    @pytest.mark.asyncio
    async def test_application_error_raises_api_error(self) -> None:
        """Test that success=false in response raises GameAPIError."""
        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "success": False,
                "error": "Room is full",
            }
            mock_request.return_value = mock_response

            with pytest.raises(GameAPIError, match="Room is full"):
                await client.join_game("Bot", "ABC123")

        await client.close()


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self) -> None:
        """Test that timeout triggers retry."""
        client = GameHTTPClient(max_retries=3)
        await client.connect()

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutException("Request timed out")
            # Success on third attempt
            response = MagicMock(spec=Response)
            response.status_code = 200
            response.json.return_value = {"maps": []}
            return response

        with patch.object(client._client, "request", side_effect=mock_request):
            with patch("bot.client.http_client.asyncio.sleep", new_callable=AsyncMock):
                result = await client.get_maps()
                assert result == []
                assert call_count == 3

        await client.close()

    @pytest.mark.asyncio
    async def test_http_error_triggers_retry(self) -> None:
        """Test that HTTP errors trigger retry."""
        client = GameHTTPClient(max_retries=2)
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = HTTPError("Connection failed")

            with patch(
                "bot.client.http_client.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep:
                with pytest.raises(GameConnectionError, match="HTTP error"):
                    await client.get_maps()

                # Should have retried once (2 total attempts)
                assert mock_request.call_count == 2
                # Should have slept once between attempts
                assert mock_sleep.call_count == 1

        await client.close()

    @pytest.mark.asyncio
    async def test_api_error_not_retried(self) -> None:
        """Test that API errors are not retried."""
        client = GameHTTPClient(max_retries=3)
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 400
            mock_response.text = "Bad Request"
            mock_request.return_value = mock_response

            with pytest.raises(GameAPIError):
                await client.get_maps()

            # Should only be called once (no retries for API errors)
            assert mock_request.call_count == 1

        await client.close()

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self) -> None:
        """Test that after all retries exhausted, error is raised."""
        client = GameHTTPClient(max_retries=3)
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = TimeoutException("Request timed out")

            with patch("bot.client.http_client.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(GameConnectionError, match="Request timeout"):
                    await client.get_maps()

                # Should have tried 3 times
                assert mock_request.call_count == 3

        await client.close()

    @pytest.mark.asyncio
    async def test_exponential_backoff(self) -> None:
        """Test that exponential backoff is applied between retries."""
        client = GameHTTPClient(max_retries=4)
        await client.connect()

        sleep_times: list[float] = []

        async def capture_sleep(delay: float) -> None:
            sleep_times.append(delay)

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = TimeoutException("Request timed out")

            with patch(
                "bot.client.http_client.asyncio.sleep", side_effect=capture_sleep
            ):
                with pytest.raises(GameConnectionError):
                    await client.get_maps()

                # Should have 3 sleeps (between 4 attempts)
                assert len(sleep_times) == 3
                # Check exponential pattern: 0.1, 0.2, 0.4
                assert sleep_times[0] == pytest.approx(0.1)
                assert sleep_times[1] == pytest.approx(0.2)
                assert sleep_times[2] == pytest.approx(0.4)

        await client.close()


class TestResponseValidation:
    """Tests for response validation."""

    @pytest.mark.asyncio
    async def test_invalid_response_raises_error(self) -> None:
        """Test that invalid response data raises validation error."""
        client = GameHTTPClient()
        await client.connect()

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 200
            # Missing required fields
            mock_response.json.return_value = {"invalid": "data"}
            mock_request.return_value = mock_response

            with pytest.raises(GameHTTPClientError, match="validation error"):
                await client.get_maps()

        await client.close()


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_api_error_inherits_from_base(self) -> None:
        """Test GameAPIError inherits from GameHTTPClientError."""
        error = GameAPIError("Test error", status_code=400)
        assert isinstance(error, GameHTTPClientError)
        assert isinstance(error, Exception)
        assert error.status_code == 400

    def test_connection_error_inherits_from_base(self) -> None:
        """Test GameConnectionError inherits from GameHTTPClientError."""
        error = GameConnectionError("Connection failed")
        assert isinstance(error, GameHTTPClientError)
        assert isinstance(error, Exception)

    def test_api_error_without_status_code(self) -> None:
        """Test GameAPIError can be created without status code."""
        error = GameAPIError("Application error")
        assert error.status_code is None
        assert str(error) == "Application error"


class TestImports:
    """Tests that all imports work correctly."""

    def test_imports_from_bot_client(self) -> None:
        """Test importing from bot.client works."""
        from bot.client import (
            GameAPIError,
            GameConnectionError,
            GameHTTPClient,
            GameHTTPClientError,
        )

        assert GameHTTPClient is not None
        assert GameHTTPClientError is not None
        assert GameAPIError is not None
        assert GameConnectionError is not None
