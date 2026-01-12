"""Unit tests for GameServerManager.

Tests cover:
- TrainingGameConfig and GameInstance dataclasses
- GameServerManager lifecycle management
- Game creation with mocked HTTP responses
- Error handling for various failure modes
- Context manager behavior
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import HTTPStatusError, RequestError, Response

from bot.training import (
    GameCreationError,
    GameInstance,
    GameNotFoundError,
    GameServerError,
    GameServerManager,
    MaxGamesExceededError,
    TrainingGameConfig,
)


class TestTrainingGameConfig:
    """Tests for TrainingGameConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TrainingGameConfig(room_name="Test Room")
        assert config.room_name == "Test Room"
        assert config.map_type == "basic"
        assert config.tick_multiplier == 10.0
        assert config.max_game_duration_sec == 60
        assert config.disable_respawn_timer is True
        assert config.max_kills == 20

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = TrainingGameConfig(
            room_name="Custom Room",
            map_type="arena1",
            tick_multiplier=5.0,
            max_game_duration_sec=120,
            disable_respawn_timer=False,
            max_kills=50,
        )
        assert config.room_name == "Custom Room"
        assert config.map_type == "arena1"
        assert config.tick_multiplier == 5.0
        assert config.max_game_duration_sec == 120
        assert config.disable_respawn_timer is False
        assert config.max_kills == 50

    def test_equality(self) -> None:
        """Test config equality comparison."""
        config1 = TrainingGameConfig(room_name="Room A", tick_multiplier=5.0)
        config2 = TrainingGameConfig(room_name="Room A", tick_multiplier=5.0)
        config3 = TrainingGameConfig(room_name="Room B", tick_multiplier=5.0)

        assert config1 == config2
        assert config1 != config3


class TestGameInstance:
    """Tests for GameInstance dataclass."""

    def test_creation(self) -> None:
        """Test creating a game instance."""
        config = TrainingGameConfig(room_name="Test Room")
        instance = GameInstance(
            room_id="room-123",
            room_code="ABC123",
            room_name="Test Room",
            config=config,
            player_id="player-456",
            player_token="token-789",
            canvas_size=(800, 600),
            created_at=1234567890.0,
        )

        assert instance.room_id == "room-123"
        assert instance.room_code == "ABC123"
        assert instance.room_name == "Test Room"
        assert instance.config == config
        assert instance.player_id == "player-456"
        assert instance.player_token == "token-789"
        assert instance.canvas_size == (800, 600)
        assert instance.created_at == 1234567890.0
        assert instance.is_active is True

    def test_is_active_default(self) -> None:
        """Test is_active defaults to True."""
        config = TrainingGameConfig(room_name="Test")
        instance = GameInstance(
            room_id="room-1",
            room_code="CODE1",
            room_name="Test",
            config=config,
            player_id="player-1",
            player_token="token-1",
            canvas_size=(800, 600),
            created_at=time.time(),
        )
        assert instance.is_active is True

    def test_is_active_can_be_set(self) -> None:
        """Test is_active can be explicitly set."""
        config = TrainingGameConfig(room_name="Test")
        instance = GameInstance(
            room_id="room-1",
            room_code="CODE1",
            room_name="Test",
            config=config,
            player_id="player-1",
            player_token="token-1",
            canvas_size=(800, 600),
            created_at=time.time(),
            is_active=False,
        )
        assert instance.is_active is False


class TestGameServerManagerInit:
    """Tests for GameServerManager initialization."""

    def test_default_configuration(self) -> None:
        """Test manager initializes with default configuration."""
        manager = GameServerManager()
        assert manager._http_url == "http://localhost:4000"
        assert manager._max_concurrent_games == 10
        assert manager._client is None
        assert manager._games == {}

    def test_custom_configuration(self) -> None:
        """Test manager accepts custom configuration."""
        manager = GameServerManager(
            http_url="http://example.com:8080/",
            max_concurrent_games=20,
        )
        assert manager._http_url == "http://example.com:8080"  # trailing slash stripped
        assert manager._max_concurrent_games == 20


class TestGameServerManagerContextManager:
    """Tests for async context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_creates_and_closes_client(self) -> None:
        """Test context manager properly creates and closes client."""
        manager = GameServerManager()

        async with manager:
            assert manager._client is not None

        assert manager._client is None

    @pytest.mark.asyncio
    async def test_context_manager_terminates_games_on_exit(self) -> None:
        """Test context manager terminates all games on exit."""
        mock_response_data = {
            "success": True,
            "roomId": "room-123",
            "roomCode": "ABC123",
            "roomName": "Test Room",
            "playerId": "player-456",
            "playerToken": "token-789",
            "canvasSizeX": 800,
            "canvasSizeY": 600,
        }

        manager = GameServerManager()

        async with manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                config = TrainingGameConfig(room_name="Test")
                await manager.create_game(config)
                assert len(manager.get_active_games()) == 1

        # After exit, games should be terminated and cleared
        assert len(manager._games) == 0


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
            "playerId": "player-456",
            "playerToken": "token-789",
            "canvasSizeX": 800,
            "canvasSizeY": 600,
        }

        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                config = TrainingGameConfig(room_name="Test Room", map_type="arena1")
                instance = await manager.create_game(config, "TestBot")

                assert isinstance(instance, GameInstance)
                assert instance.room_id == "room-123"
                assert instance.room_code == "ABC123"
                assert instance.player_id == "player-456"
                assert instance.player_token == "token-789"
                assert instance.canvas_size == (800, 600)
                assert instance.is_active is True

                # Verify request payload
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert call_args[0][0] == "/api/createGame"
                payload = call_args[1]["json"]
                assert payload["playerName"] == "TestBot"
                assert payload["roomName"] == "Test Room"
                assert payload["mapType"] == "arena1"
                assert payload["trainingMode"] is True

    @pytest.mark.asyncio
    async def test_create_game_with_training_options(self) -> None:
        """Test game creation sends training mode options."""
        mock_response_data = {
            "success": True,
            "roomId": "room-123",
            "roomCode": "ABC123",
            "roomName": "Training",
            "playerId": "player-456",
            "playerToken": "token-789",
            "canvasSizeX": 800,
            "canvasSizeY": 600,
        }

        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                config = TrainingGameConfig(
                    room_name="Training",
                    tick_multiplier=5.0,
                    max_game_duration_sec=120,
                    disable_respawn_timer=False,
                    max_kills=30,
                )
                await manager.create_game(config)

                payload = mock_post.call_args[1]["json"]
                assert payload["tickMultiplier"] == 5.0
                assert payload["maxGameDurationSec"] == 120
                assert payload["disableRespawnTimer"] is False
                assert payload["maxKills"] == 30

    @pytest.mark.asyncio
    async def test_create_game_tracks_instance(self) -> None:
        """Test that created games are tracked."""
        mock_response_data = {
            "success": True,
            "roomId": "room-123",
            "roomCode": "ABC123",
            "roomName": "Test",
            "playerId": "player-456",
            "playerToken": "token-789",
            "canvasSizeX": 800,
            "canvasSizeY": 600,
        }

        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                config = TrainingGameConfig(room_name="Test")
                instance = await manager.create_game(config)

                assert manager.get_game_status("room-123") == instance
                assert instance in manager.get_active_games()

    @pytest.mark.asyncio
    async def test_create_game_without_context_manager_raises(self) -> None:
        """Test that creating game without context manager raises error."""
        manager = GameServerManager()
        config = TrainingGameConfig(room_name="Test")

        with pytest.raises(GameServerError, match="Client not connected"):
            await manager.create_game(config)

    @pytest.mark.asyncio
    async def test_create_game_api_error(self) -> None:
        """Test game creation handles API error response."""
        mock_response_data = {
            "success": False,
            "error": "Map not found",
        }

        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                config = TrainingGameConfig(room_name="Test")

                with pytest.raises(GameCreationError, match="Map not found"):
                    await manager.create_game(config)

    @pytest.mark.asyncio
    async def test_create_game_http_error(self) -> None:
        """Test game creation handles HTTP errors."""
        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 500
                mock_post.return_value = mock_response
                mock_response.raise_for_status.side_effect = HTTPStatusError(
                    "Server Error",
                    request=MagicMock(),
                    response=mock_response,
                )

                config = TrainingGameConfig(room_name="Test")

                with pytest.raises(GameCreationError, match="HTTP error"):
                    await manager.create_game(config)

    @pytest.mark.asyncio
    async def test_create_game_network_error(self) -> None:
        """Test game creation handles network errors."""
        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_post.side_effect = RequestError("Connection refused")

                config = TrainingGameConfig(room_name="Test")

                with pytest.raises(GameCreationError, match="Network error"):
                    await manager.create_game(config)


class TestMaxConcurrentGames:
    """Tests for max concurrent games enforcement."""

    @pytest.mark.asyncio
    async def test_max_games_exceeded(self) -> None:
        """Test that exceeding max games raises error."""
        mock_response_data = {
            "success": True,
            "roomId": "room-{i}",
            "roomCode": "CODE{i}",
            "roomName": "Test",
            "playerId": "player-{i}",
            "playerToken": "token-{i}",
            "canvasSizeX": 800,
            "canvasSizeY": 600,
        }

        async with GameServerManager(max_concurrent_games=2) as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:

                def make_response(i: int) -> MagicMock:
                    data = mock_response_data.copy()
                    data["roomId"] = f"room-{i}"
                    data["roomCode"] = f"CODE{i}"
                    data["playerId"] = f"player-{i}"
                    data["playerToken"] = f"token-{i}"
                    response = MagicMock(spec=Response)
                    response.status_code = 200
                    response.json.return_value = data
                    response.raise_for_status = MagicMock()
                    return response

                # Create first two games successfully
                mock_post.return_value = make_response(1)
                await manager.create_game(TrainingGameConfig(room_name="Game 1"))

                mock_post.return_value = make_response(2)
                await manager.create_game(TrainingGameConfig(room_name="Game 2"))

                assert len(manager.get_active_games()) == 2

                # Third game should fail
                with pytest.raises(MaxGamesExceededError, match="Maximum of 2"):
                    await manager.create_game(TrainingGameConfig(room_name="Game 3"))

    @pytest.mark.asyncio
    async def test_can_create_after_termination(self) -> None:
        """Test that games can be created after termination frees slots."""
        async with GameServerManager(max_concurrent_games=1) as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "success": True,
                    "roomId": "room-1",
                    "roomCode": "CODE1",
                    "roomName": "Test",
                    "playerId": "player-1",
                    "playerToken": "token-1",
                    "canvasSizeX": 800,
                    "canvasSizeY": 600,
                }
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                # Create first game
                game1 = await manager.create_game(
                    TrainingGameConfig(room_name="Game 1")
                )
                assert len(manager.get_active_games()) == 1

                # Terminate it
                await manager.terminate_game(game1.room_id)
                assert len(manager.get_active_games()) == 0

                # Now we can create another
                mock_response.json.return_value = {
                    "success": True,
                    "roomId": "room-2",
                    "roomCode": "CODE2",
                    "roomName": "Game 2",
                    "playerId": "player-2",
                    "playerToken": "token-2",
                    "canvasSizeX": 800,
                    "canvasSizeY": 600,
                }
                game2 = await manager.create_game(
                    TrainingGameConfig(room_name="Game 2")
                )
                assert len(manager.get_active_games()) == 1
                assert game2.room_id == "room-2"


class TestTerminateGame:
    """Tests for terminate_game() method."""

    @pytest.mark.asyncio
    async def test_terminate_game_success(self) -> None:
        """Test successful game termination."""
        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "success": True,
                    "roomId": "room-123",
                    "roomCode": "ABC123",
                    "roomName": "Test",
                    "playerId": "player-456",
                    "playerToken": "token-789",
                    "canvasSizeX": 800,
                    "canvasSizeY": 600,
                }
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                config = TrainingGameConfig(room_name="Test")
                instance = await manager.create_game(config)

                assert manager.get_game_status("room-123") is not None

                await manager.terminate_game("room-123")

                assert manager.get_game_status("room-123") is None
                assert instance not in manager.get_active_games()

    @pytest.mark.asyncio
    async def test_terminate_nonexistent_game(self) -> None:
        """Test terminating a non-existent game raises error."""
        async with GameServerManager() as manager:
            with pytest.raises(GameNotFoundError, match="Game nonexistent not found"):
                await manager.terminate_game("nonexistent")


class TestTerminateAll:
    """Tests for terminate_all() method."""

    @pytest.mark.asyncio
    async def test_terminate_all(self) -> None:
        """Test terminating all games."""
        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:

                def make_response(room_id: str) -> MagicMock:
                    response = MagicMock(spec=Response)
                    response.status_code = 200
                    response.json.return_value = {
                        "success": True,
                        "roomId": room_id,
                        "roomCode": f"CODE-{room_id}",
                        "roomName": "Test",
                        "playerId": f"player-{room_id}",
                        "playerToken": f"token-{room_id}",
                        "canvasSizeX": 800,
                        "canvasSizeY": 600,
                    }
                    response.raise_for_status = MagicMock()
                    return response

                # Create multiple games
                mock_post.return_value = make_response("room-1")
                await manager.create_game(TrainingGameConfig(room_name="Game 1"))

                mock_post.return_value = make_response("room-2")
                await manager.create_game(TrainingGameConfig(room_name="Game 2"))

                mock_post.return_value = make_response("room-3")
                await manager.create_game(TrainingGameConfig(room_name="Game 3"))

                assert len(manager.get_active_games()) == 3

                await manager.terminate_all()

                assert len(manager.get_active_games()) == 0
                assert len(manager._games) == 0


class TestGetGameStatus:
    """Tests for get_game_status() method."""

    @pytest.mark.asyncio
    async def test_get_game_status_found(self) -> None:
        """Test getting status of existing game."""
        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "success": True,
                    "roomId": "room-123",
                    "roomCode": "ABC123",
                    "roomName": "Test",
                    "playerId": "player-456",
                    "playerToken": "token-789",
                    "canvasSizeX": 800,
                    "canvasSizeY": 600,
                }
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                config = TrainingGameConfig(room_name="Test")
                instance = await manager.create_game(config)

                status = manager.get_game_status("room-123")
                assert status == instance
                assert status is not None
                assert status.room_id == "room-123"

    def test_get_game_status_not_found(self) -> None:
        """Test getting status of non-existent game returns None."""
        manager = GameServerManager()
        assert manager.get_game_status("nonexistent") is None


class TestGetActiveGames:
    """Tests for get_active_games() method."""

    @pytest.mark.asyncio
    async def test_get_active_games(self) -> None:
        """Test getting list of active games."""
        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:

                def make_response(room_id: str) -> MagicMock:
                    response = MagicMock(spec=Response)
                    response.status_code = 200
                    response.json.return_value = {
                        "success": True,
                        "roomId": room_id,
                        "roomCode": f"CODE-{room_id}",
                        "roomName": "Test",
                        "playerId": f"player-{room_id}",
                        "playerToken": f"token-{room_id}",
                        "canvasSizeX": 800,
                        "canvasSizeY": 600,
                    }
                    response.raise_for_status = MagicMock()
                    return response

                mock_post.return_value = make_response("room-1")
                game1 = await manager.create_game(
                    TrainingGameConfig(room_name="Game 1")
                )

                mock_post.return_value = make_response("room-2")
                game2 = await manager.create_game(
                    TrainingGameConfig(room_name="Game 2")
                )

                active = manager.get_active_games()
                assert len(active) == 2
                assert game1 in active
                assert game2 in active

    def test_get_active_games_empty(self) -> None:
        """Test getting active games when none exist."""
        manager = GameServerManager()
        assert manager.get_active_games() == []


class TestHealthCheck:
    """Tests for health_check() method."""

    @pytest.mark.asyncio
    async def test_health_check_success(self) -> None:
        """Test successful health check."""
        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "get", new_callable=AsyncMock
            ) as mock_get:
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 200
                mock_get.return_value = mock_response

                result = await manager.health_check()
                assert result is True
                mock_get.assert_called_once_with("/api/maps")

    @pytest.mark.asyncio
    async def test_health_check_failure_status(self) -> None:
        """Test health check with non-200 status."""
        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "get", new_callable=AsyncMock
            ) as mock_get:
                mock_response = MagicMock(spec=Response)
                mock_response.status_code = 500
                mock_get.return_value = mock_response

                result = await manager.health_check()
                assert result is False

    @pytest.mark.asyncio
    async def test_health_check_network_error(self) -> None:
        """Test health check with network error."""
        async with GameServerManager() as manager:
            with patch.object(
                manager._client, "get", new_callable=AsyncMock
            ) as mock_get:
                mock_get.side_effect = RequestError("Connection refused")

                result = await manager.health_check()
                assert result is False

    @pytest.mark.asyncio
    async def test_health_check_no_client(self) -> None:
        """Test health check without client connection."""
        manager = GameServerManager()
        result = await manager.health_check()
        assert result is False


class TestResetGame:
    """Tests for reset_game() method."""

    @pytest.mark.asyncio
    async def test_reset_game_success(self) -> None:
        """Test successful game reset."""
        async with GameServerManager() as manager:
            # Setup: create a game first
            with patch.object(
                manager._client, "post", new_callable=AsyncMock
            ) as mock_post:
                mock_create_response = MagicMock(spec=Response)
                mock_create_response.status_code = 200
                mock_create_response.json.return_value = {
                    "success": True,
                    "roomId": "room-123",
                    "roomCode": "ABC123",
                    "roomName": "Test",
                    "playerId": "player-456",
                    "playerToken": "token-789",
                    "canvasSizeX": 800,
                    "canvasSizeY": 600,
                }
                mock_create_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_create_response

                config = TrainingGameConfig(room_name="Test")
                await manager.create_game(config)

            # Test reset
            with (
                patch.object(
                    manager._client, "post", new_callable=AsyncMock
                ) as mock_post,
                patch.object(
                    manager._client, "get", new_callable=AsyncMock
                ) as mock_get,
            ):
                # Mock reset response
                mock_reset_response = MagicMock(spec=Response)
                mock_reset_response.status_code = 200
                mock_reset_response.json.return_value = {"success": True}
                mock_reset_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_reset_response

                # Mock state response with complete player state
                mock_state_response = MagicMock(spec=Response)
                mock_state_response.status_code = 200
                mock_state_response.json.return_value = {
                    "success": True,
                    "objectStates": {
                        "player-456": {
                            "id": "player-456",
                            "objectType": "player",
                            "name": "TrainingBot",
                            "x": 400.0,
                            "y": 300.0,
                            "dx": 0.0,
                            "dy": 0.0,
                            "dir": 0.0,
                            "rad": 16.0,
                            "h": 3,
                            "dead": False,
                            "sht": False,
                            "jc": 2,
                            "ac": 3,
                        }
                    },
                    "trainingComplete": False,
                }
                mock_state_response.raise_for_status = MagicMock()
                mock_get.return_value = mock_state_response

                state = await manager.reset_game("room-123")

                mock_post.assert_called_once()
                assert "/api/rooms/room-123/reset" in mock_post.call_args[0][0]
                assert state is not None

    @pytest.mark.asyncio
    async def test_reset_game_not_found(self) -> None:
        """Test reset raises error for non-existent game."""
        async with GameServerManager() as manager:
            with pytest.raises(GameNotFoundError, match="Game nonexistent not found"):
                await manager.reset_game("nonexistent")

    @pytest.mark.asyncio
    async def test_reset_game_without_client(self) -> None:
        """Test reset raises error without client connection."""
        manager = GameServerManager()
        # Manually add a game to the tracker to test the client check
        config = TrainingGameConfig(room_name="Test")
        manager._games["room-123"] = GameInstance(
            room_id="room-123",
            room_code="ABC123",
            room_name="Test",
            config=config,
            player_id="player-456",
            player_token="token-789",
            canvas_size=(800, 600),
            created_at=0,
        )

        with pytest.raises(GameServerError, match="Client not connected"):
            await manager.reset_game("room-123")


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_game_creation_error_inherits_from_base(self) -> None:
        """Test GameCreationError inherits from GameServerError."""
        error = GameCreationError("Creation failed")
        assert isinstance(error, GameServerError)
        assert isinstance(error, Exception)

    def test_max_games_exceeded_inherits_from_base(self) -> None:
        """Test MaxGamesExceededError inherits from GameServerError."""
        error = MaxGamesExceededError("Max reached")
        assert isinstance(error, GameServerError)
        assert isinstance(error, Exception)

    def test_game_not_found_inherits_from_base(self) -> None:
        """Test GameNotFoundError inherits from GameServerError."""
        error = GameNotFoundError("Not found")
        assert isinstance(error, GameServerError)
        assert isinstance(error, Exception)


class TestImports:
    """Tests that all imports work correctly."""

    def test_imports_from_bot_training(self) -> None:
        """Test importing from bot.training works."""
        from bot.training import (
            GameCreationError,
            GameInstance,
            GameNotFoundError,
            GameServerError,
            GameServerManager,
            MaxGamesExceededError,
            TrainingGameConfig,
        )

        assert GameServerManager is not None
        assert TrainingGameConfig is not None
        assert GameInstance is not None
        assert GameServerError is not None
        assert GameCreationError is not None
        assert MaxGamesExceededError is not None
        assert GameNotFoundError is not None
