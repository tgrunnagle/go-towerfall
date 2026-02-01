"""Unit tests for FastAPI Bot Service.

Tests cover:
- POST /bots/spawn endpoint
- DELETE /bots/{bot_id} endpoint
- GET /bots endpoint
- GET /bots/{bot_id} endpoint
- GET /bots/models endpoint
- GET /health endpoint
- Route order (models vs bot_id)
- BotManager not initialized error
"""

from collections.abc import Generator
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from bot.service.bot_manager import BotInfo, SpawnBotResponse
from bot.service.bot_service import app, get_bot_manager, set_bot_manager
from bot.training.registry import (
    ModelMetadata,
    NetworkArchitecture,
    TrainingMetrics,
)


@pytest.fixture
def mock_manager() -> MagicMock:
    """Create a mock BotManager for testing."""
    manager = MagicMock()
    manager.spawn_bot = AsyncMock()
    manager.destroy_bot = AsyncMock()
    manager.get_bot = MagicMock()
    manager.list_bots = MagicMock()
    manager.list_models = MagicMock()
    manager.shutdown = AsyncMock()
    return manager


@pytest.fixture
def client(mock_manager: MagicMock) -> Generator[TestClient, None, None]:
    """Create a TestClient with mocked BotManager."""
    app.dependency_overrides[get_bot_manager] = lambda: mock_manager
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_bot_info() -> BotInfo:
    """Create a sample BotInfo for testing."""
    return BotInfo(
        bot_id="bot_abc123",
        bot_type="rule_based",
        model_id=None,
        player_name="TestBot",
        room_code="ABC123",
        is_connected=True,
    )


@pytest.fixture
def sample_model_metadata() -> ModelMetadata:
    """Create a sample ModelMetadata for testing."""
    return ModelMetadata(
        model_id="ppo_gen_005",
        generation=5,
        created_at=datetime(2024, 1, 1),
        training_duration_seconds=3600.0,
        training_metrics=TrainingMetrics(
            total_episodes=1000,
            total_timesteps=100000,
            average_reward=50.0,
            average_episode_length=100.0,
            win_rate=0.5,
            average_kills=2.0,
            average_deaths=2.0,
            kills_deaths_ratio=1.0,
        ),
        architecture=NetworkArchitecture(
            observation_size=128,
            action_size=32,
        ),
        checkpoint_path="test.pth",
    )


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_check_returns_healthy(self, client: TestClient) -> None:
        """GET /health returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestSpawnBotEndpoint:
    """Tests for POST /bots/spawn endpoint."""

    def test_spawn_bot_success(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """POST /bots/spawn returns bot_id on success."""
        mock_manager.spawn_bot.return_value = SpawnBotResponse(
            success=True, bot_id="bot_123"
        )

        response = client.post(
            "/bots/spawn",
            json={
                "room_code": "ABC123",
                "room_password": "",
                "bot_config": {"bot_type": "rule_based", "player_name": "TestBot"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["bot_id"] == "bot_123"
        assert data["error"] is None

    def test_spawn_bot_with_password(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """POST /bots/spawn accepts room password."""
        mock_manager.spawn_bot.return_value = SpawnBotResponse(
            success=True, bot_id="bot_456"
        )

        response = client.post(
            "/bots/spawn",
            json={
                "room_code": "XYZ789",
                "room_password": "secret123",
                "bot_config": {"bot_type": "rule_based"},
            },
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

        # Verify password was passed to manager
        call_args = mock_manager.spawn_bot.call_args
        request = call_args[0][0]
        assert request.room_password == "secret123"

    def test_spawn_neural_network_bot(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """POST /bots/spawn works with neural network bot."""
        mock_manager.spawn_bot.return_value = SpawnBotResponse(
            success=True, bot_id="bot_nn_001"
        )

        response = client.post(
            "/bots/spawn",
            json={
                "room_code": "ABC123",
                "bot_config": {
                    "bot_type": "neural_network",
                    "model_id": "ppo_gen_005",
                    "player_name": "NeuralBot",
                },
            },
        )

        assert response.status_code == 200
        assert response.json()["success"] is True
        assert response.json()["bot_id"] == "bot_nn_001"

    def test_spawn_bot_failure_returns_error(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """POST /bots/spawn returns error message on failure."""
        mock_manager.spawn_bot.return_value = SpawnBotResponse(
            success=False, error="Model not found"
        )

        response = client.post(
            "/bots/spawn",
            json={
                "room_code": "ABC123",
                "bot_config": {
                    "bot_type": "neural_network",
                    "model_id": "invalid_model",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["bot_id"] is None
        assert data["error"] == "Model not found"

    def test_spawn_bot_invalid_request_returns_422(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """POST /bots/spawn with invalid request returns 422."""
        response = client.post(
            "/bots/spawn",
            json={
                "room_code": "ABC123",
                # Missing required bot_config
            },
        )

        assert response.status_code == 422

    def test_spawn_bot_invalid_bot_type_returns_422(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """POST /bots/spawn with invalid bot_type returns 422."""
        response = client.post(
            "/bots/spawn",
            json={
                "room_code": "ABC123",
                "bot_config": {"bot_type": "invalid_type"},
            },
        )

        assert response.status_code == 422


class TestDeleteBotEndpoint:
    """Tests for DELETE /bots/{bot_id} endpoint."""

    def test_delete_bot_success(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """DELETE /bots/{bot_id} returns success for existing bot."""
        mock_manager.destroy_bot.return_value = True

        response = client.delete("/bots/bot_abc123")

        assert response.status_code == 200
        assert response.json() == {"success": True}
        mock_manager.destroy_bot.assert_called_once_with("bot_abc123")

    def test_delete_bot_not_found(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """DELETE /bots/{bot_id} returns 404 for unknown bot."""
        mock_manager.destroy_bot.return_value = False

        response = client.delete("/bots/unknown_bot")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestListBotsEndpoint:
    """Tests for GET /bots endpoint."""

    def test_list_bots_returns_all(
        self,
        client: TestClient,
        mock_manager: MagicMock,
        sample_bot_info: BotInfo,
    ) -> None:
        """GET /bots returns list of active bots."""
        bot2 = BotInfo(
            bot_id="bot_def456",
            bot_type="neural_network",
            model_id="ppo_gen_003",
            player_name="NeuralBot",
            room_code="XYZ789",
            is_connected=False,
        )
        mock_manager.list_bots.return_value = [sample_bot_info, bot2]

        response = client.get("/bots")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["bot_id"] == "bot_abc123"
        assert data[1]["bot_id"] == "bot_def456"

    def test_list_bots_empty(self, client: TestClient, mock_manager: MagicMock) -> None:
        """GET /bots returns empty list when no bots."""
        mock_manager.list_bots.return_value = []

        response = client.get("/bots")

        assert response.status_code == 200
        assert response.json() == []


class TestGetBotEndpoint:
    """Tests for GET /bots/{bot_id} endpoint."""

    def test_get_bot_success(
        self,
        client: TestClient,
        mock_manager: MagicMock,
        sample_bot_info: BotInfo,
    ) -> None:
        """GET /bots/{bot_id} returns BotInfo for existing bot."""
        mock_manager.get_bot.return_value = sample_bot_info

        response = client.get("/bots/bot_abc123")

        assert response.status_code == 200
        data = response.json()
        assert data["bot_id"] == "bot_abc123"
        assert data["bot_type"] == "rule_based"
        assert data["player_name"] == "TestBot"
        assert data["room_code"] == "ABC123"
        assert data["is_connected"] is True
        mock_manager.get_bot.assert_called_once_with("bot_abc123")

    def test_get_bot_not_found(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """GET /bots/{bot_id} returns 404 for unknown bot."""
        mock_manager.get_bot.return_value = None

        response = client.get("/bots/unknown_bot")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestListModelsEndpoint:
    """Tests for GET /bots/models endpoint."""

    def test_list_models_returns_models(
        self,
        client: TestClient,
        mock_manager: MagicMock,
        sample_model_metadata: ModelMetadata,
    ) -> None:
        """GET /bots/models returns list of available models."""
        mock_manager.list_models.return_value = [sample_model_metadata]

        response = client.get("/bots/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["model_id"] == "ppo_gen_005"
        assert data[0]["generation"] == 5

    def test_list_models_empty(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """GET /bots/models returns empty list when no models."""
        mock_manager.list_models.return_value = []

        response = client.get("/bots/models")

        assert response.status_code == 200
        assert response.json() == []


class TestRouteOrder:
    """Tests to verify correct route order (models vs bot_id)."""

    def test_models_route_not_interpreted_as_bot_id(
        self,
        client: TestClient,
        mock_manager: MagicMock,
    ) -> None:
        """GET /bots/models is not interpreted as GET /bots/{bot_id='models'}."""
        mock_manager.list_models.return_value = []
        mock_manager.get_bot.return_value = None

        response = client.get("/bots/models")

        # Should call list_models, not get_bot
        mock_manager.list_models.assert_called_once()
        mock_manager.get_bot.assert_not_called()
        assert response.status_code == 200


class TestBotManagerNotInitialized:
    """Tests for BotManager not initialized error."""

    @pytest.fixture(autouse=True)
    def clear_manager_state(self) -> Generator[None, None, None]:
        """Clear and restore manager state for each test."""
        # Import module to access internal state
        import bot.service.bot_service as bot_service_module

        # Save original state
        original_manager = bot_service_module._bot_manager
        original_overrides = app.dependency_overrides.copy()

        # Clear for test
        set_bot_manager(None)  # type: ignore[arg-type]
        app.dependency_overrides.clear()

        yield

        # Restore original state
        bot_service_module._bot_manager = original_manager
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    def test_spawn_bot_manager_not_initialized(self) -> None:
        """POST /bots/spawn raises RuntimeError when BotManager not initialized."""
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/bots/spawn",
                json={
                    "room_code": "ABC123",
                    "bot_config": {"bot_type": "rule_based"},
                },
            )

        assert response.status_code == 500

    def test_list_bots_manager_not_initialized(self) -> None:
        """GET /bots raises RuntimeError when BotManager not initialized."""
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/bots")

        assert response.status_code == 500

    def test_health_works_without_manager(self) -> None:
        """GET /health works even when BotManager not initialized."""
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_long_bot_id(self, client: TestClient, mock_manager: MagicMock) -> None:
        """Handles very long bot_id values."""
        long_id = "bot_" + "x" * 1000
        mock_manager.get_bot.return_value = None

        response = client.get(f"/bots/{long_id}")

        assert response.status_code == 404
        mock_manager.get_bot.assert_called_once_with(long_id)

    def test_special_characters_in_bot_id(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """Handles special characters in bot_id."""
        mock_manager.get_bot.return_value = None

        response = client.get("/bots/bot_abc-123_456")

        assert response.status_code == 404
        mock_manager.get_bot.assert_called_once_with("bot_abc-123_456")

    def test_url_encoded_space_in_bot_id(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """Handles URL-encoded space in bot_id."""
        mock_manager.get_bot.return_value = None

        # URL-encoded space (%20) decodes to a space character
        response = client.get("/bots/bot%20abc")

        assert response.status_code == 404
        mock_manager.get_bot.assert_called_once_with("bot abc")

    def test_url_encoded_slash_returns_not_found(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """URL-encoded slash in bot_id is treated as path separator and returns 404."""
        mock_manager.get_bot.return_value = None

        # URL-encoded forward slash (%2F) - Starlette/ASGI decodes this as a path separator
        # which means /bots/bot%2F123 becomes /bots/bot/123 and doesn't match our route
        response = client.get("/bots/bot%2F123")

        # This returns 404 because /bots/bot/123 doesn't match any route
        assert response.status_code == 404
        # get_bot is NOT called because the route doesn't match /bots/{bot_id}
        mock_manager.get_bot.assert_not_called()

    def test_path_traversal_attempt_returns_not_found(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """Path traversal attempts don't match route and return 404."""
        mock_manager.get_bot.return_value = None

        # URL-encoded path traversal - this doesn't match our routes
        response = client.get("/bots/..%2Fetc%2Fpasswd")

        # This returns 404 because the decoded path doesn't match any route
        assert response.status_code == 404
        # get_bot is NOT called because the route doesn't match
        mock_manager.get_bot.assert_not_called()


class TestMainEntry:
    """Tests for the CLI entry point (__main__.py)."""

    @pytest.fixture
    def mock_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear all bot service environment variables to test defaults."""
        env_vars = [
            "BOT_SERVICE_PORT",
            "BOT_SERVICE_HOST",
            "GAME_SERVER_HTTP_URL",
            "GAME_SERVER_WS_URL",
            "MODEL_REGISTRY_PATH",
            "DEFAULT_DEVICE",
        ]
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)

    def test_main_uses_default_environment_values(
        self, mock_env_defaults: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() uses default values when env vars not set."""
        from bot.service import __main__ as main_module

        captured_args: dict = {}

        def mock_uvicorn_run(app, host: str, port: int) -> None:
            captured_args["host"] = host
            captured_args["port"] = port

        monkeypatch.setattr("bot.service.__main__.uvicorn.run", mock_uvicorn_run)
        monkeypatch.setattr(
            "bot.service.__main__.BotManager",
            lambda **kwargs: MagicMock(),
        )
        monkeypatch.setattr("bot.service.__main__.set_bot_manager", lambda m: None)

        main_module.main()

        assert captured_args["host"] == "0.0.0.0"
        assert captured_args["port"] == 8080

    def test_main_uses_custom_environment_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() uses custom values from environment variables."""
        from bot.service import __main__ as main_module

        monkeypatch.setenv("BOT_SERVICE_PORT", "9000")
        monkeypatch.setenv("BOT_SERVICE_HOST", "127.0.0.1")
        monkeypatch.setenv("GAME_SERVER_HTTP_URL", "http://game:5000")
        monkeypatch.setenv("GAME_SERVER_WS_URL", "ws://game:5000/ws")
        monkeypatch.setenv("DEFAULT_DEVICE", "cuda")

        captured_args: dict = {}
        captured_manager_args: dict = {}

        def mock_uvicorn_run(app, host: str, port: int) -> None:
            captured_args["host"] = host
            captured_args["port"] = port

        def mock_bot_manager(**kwargs) -> MagicMock:
            captured_manager_args.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("bot.service.__main__.uvicorn.run", mock_uvicorn_run)
        monkeypatch.setattr("bot.service.__main__.BotManager", mock_bot_manager)
        monkeypatch.setattr("bot.service.__main__.set_bot_manager", lambda m: None)

        main_module.main()

        assert captured_args["host"] == "127.0.0.1"
        assert captured_args["port"] == 9000
        assert captured_manager_args["http_url"] == "http://game:5000"
        assert captured_manager_args["ws_url"] == "ws://game:5000/ws"
        assert captured_manager_args["default_device"] == "cuda"

    def test_main_initializes_model_registry_when_path_provided(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: str
    ) -> None:
        """main() initializes ModelRegistry when MODEL_REGISTRY_PATH is set."""
        from bot.service import __main__ as main_module

        monkeypatch.setenv("MODEL_REGISTRY_PATH", str(tmp_path))

        mock_registry = MagicMock()
        mock_registry_class = MagicMock(return_value=mock_registry)
        captured_manager_args: dict = {}

        def mock_bot_manager(**kwargs) -> MagicMock:
            captured_manager_args.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("bot.service.__main__.uvicorn.run", lambda *a, **kw: None)
        monkeypatch.setattr("bot.service.__main__.ModelRegistry", mock_registry_class)
        monkeypatch.setattr("bot.service.__main__.BotManager", mock_bot_manager)
        monkeypatch.setattr("bot.service.__main__.set_bot_manager", lambda m: None)

        main_module.main()

        mock_registry_class.assert_called_once_with(str(tmp_path))
        assert captured_manager_args["registry"] is mock_registry

    def test_main_logs_warning_when_registry_path_not_set(
        self, mock_env_defaults: None, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        """main() logs warning when MODEL_REGISTRY_PATH is not set."""
        from bot.service import __main__ as main_module

        monkeypatch.setattr("bot.service.__main__.uvicorn.run", lambda *a, **kw: None)
        monkeypatch.setattr(
            "bot.service.__main__.BotManager",
            lambda **kwargs: MagicMock(),
        )
        monkeypatch.setattr("bot.service.__main__.set_bot_manager", lambda m: None)

        with caplog.at_level("WARNING"):
            main_module.main()

        assert any(
            "MODEL_REGISTRY_PATH not set" in record.message for record in caplog.records
        )

    def test_main_passes_none_registry_when_path_not_set(
        self, mock_env_defaults: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() passes None registry when MODEL_REGISTRY_PATH is not set."""
        from bot.service import __main__ as main_module

        captured_manager_args: dict = {}

        def mock_bot_manager(**kwargs) -> MagicMock:
            captured_manager_args.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("bot.service.__main__.uvicorn.run", lambda *a, **kw: None)
        monkeypatch.setattr("bot.service.__main__.BotManager", mock_bot_manager)
        monkeypatch.setattr("bot.service.__main__.set_bot_manager", lambda m: None)

        main_module.main()

        assert captured_manager_args["registry"] is None

    def test_main_invalid_port_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        """main() raises ValueError for invalid port value and logs error."""
        from bot.service import __main__ as main_module

        monkeypatch.setenv("BOT_SERVICE_PORT", "invalid_port")

        with caplog.at_level("ERROR"):
            with pytest.raises(ValueError, match="invalid literal"):
                main_module.main()

        assert any(
            "Invalid BOT_SERVICE_PORT value" in record.message
            for record in caplog.records
        )
