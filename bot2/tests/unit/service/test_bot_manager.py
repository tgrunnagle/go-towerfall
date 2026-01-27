"""Unit tests for BotManager.

Tests cover:
- BotManager initialization with and without registry
- spawn_bot() with rule-based bots
- spawn_bot() with neural network bots (by model_id and generation)
- spawn_bot() error handling (invalid model, no registry, etc.)
- spawn_bot() generates unique bot_ids
- destroy_bot() stops and removes bots
- destroy_bot() returns False for unknown bot_id
- get_bot() returns BotInfo for active bots
- get_bot() returns None for unknown bot_id
- list_bots() returns all active bots
- list_models() returns models from registry
- shutdown() stops all active bots
- Background connection failure handling
- Pydantic model validation
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.service.bot_manager import (
    BotConfig,
    BotInfo,
    BotManager,
    SpawnBotRequest,
    SpawnBotResponse,
)
from bot.training.registry import (
    ModelMetadata,
    ModelNotFoundError,
    NetworkArchitecture,
    TrainingMetrics,
)


class MockActorCriticNetwork:
    """Mock ActorCriticNetwork for testing."""

    def to(self, device):
        return self

    def eval(self):
        return self


class MockModelRegistry:
    """Mock ModelRegistry for testing."""

    def __init__(self) -> None:
        self.get_model = MagicMock()
        self.get_model_by_generation = MagicMock()
        self.list_models = MagicMock(return_value=[])

    def setup_model(
        self, model_id: str, generation: int = 0
    ) -> tuple[MockActorCriticNetwork, ModelMetadata]:
        """Setup a mock model with metadata."""
        network = MockActorCriticNetwork()
        metadata = ModelMetadata(
            model_id=model_id,
            generation=generation,
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
        return network, metadata


class TestBotManagerInit:
    """Tests for BotManager initialization."""

    def test_init_with_registry(self) -> None:
        """Initialize with ModelRegistry."""
        registry = MockModelRegistry()
        manager = BotManager(
            registry=registry,  # type: ignore[arg-type]
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        assert manager._registry is registry
        assert manager._http_url == "http://localhost:4000"
        assert manager._ws_url == "ws://localhost:4000/ws"
        assert manager._device == "cpu"
        assert manager._bots == {}

    def test_init_without_registry(self) -> None:
        """Initialize without registry (rule-based only mode)."""
        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        assert manager._registry is None
        assert manager._http_url == "http://localhost:4000"
        assert manager._ws_url == "ws://localhost:4000/ws"

    def test_init_custom_device(self) -> None:
        """Initialize with custom device."""
        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
            default_device="cuda",
        )

        assert manager._device == "cuda"


class TestBotManagerSpawnBot:
    """Tests for BotManager.spawn_bot() method."""

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.RuleBasedBotRunner")
    async def test_spawn_rule_based_bot(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """spawn_bot() with rule-based bot returns success."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            room_password="secret",
            bot_config=BotConfig(
                bot_type="rule_based",
                player_name="RuleBot",
            ),
        )

        response = await manager.spawn_bot(request)

        assert response.success is True
        assert response.bot_id is not None
        assert response.bot_id.startswith("bot_")
        assert response.error is None
        assert len(manager._bots) == 1

        # Verify runner was created correctly
        mock_runner_class.assert_called_once()

        # Verify client was created correctly
        mock_client_class.assert_called_once()

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.NeuralNetBotRunner")
    async def test_spawn_neural_network_bot_by_model_id(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """spawn_bot() with neural network bot by model_id returns success."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        registry = MockModelRegistry()
        network, metadata = registry.setup_model("ppo_gen_005", generation=5)
        registry.get_model.return_value = (network, metadata)

        manager = BotManager(
            registry=registry,  # type: ignore[arg-type]
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            room_password="",
            bot_config=BotConfig(
                bot_type="neural_network",
                model_id="ppo_gen_005",
                player_name="NeuralBot",
            ),
        )

        response = await manager.spawn_bot(request)

        assert response.success is True
        assert response.bot_id is not None
        assert response.error is None
        assert len(manager._bots) == 1

        # Verify model was loaded
        registry.get_model.assert_called_once_with("ppo_gen_005", device="cpu")

        # Verify runner was created with network
        mock_runner_class.assert_called_once()
        call_kwargs = mock_runner_class.call_args.kwargs
        assert call_kwargs["network"] is network

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.NeuralNetBotRunner")
    async def test_spawn_neural_network_bot_by_generation(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """spawn_bot() with neural network bot by generation returns success."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        registry = MockModelRegistry()
        network, metadata = registry.setup_model("ppo_gen_003", generation=3)
        registry.get_model_by_generation.return_value = (network, metadata)

        manager = BotManager(
            registry=registry,  # type: ignore[arg-type]
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="XYZ789",
            bot_config=BotConfig(
                bot_type="neural_network",
                generation=3,
                player_name="GenBot",
            ),
        )

        response = await manager.spawn_bot(request)

        assert response.success is True
        assert response.bot_id is not None
        assert response.error is None

        # Verify model was loaded by generation
        registry.get_model_by_generation.assert_called_once_with(3, device="cpu")

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.NeuralNetBotRunner")
    async def test_spawn_neural_network_bot_with_both_model_id_and_generation(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """spawn_bot() with both model_id and generation uses model_id (precedence)."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        registry = MockModelRegistry()
        network, metadata = registry.setup_model("ppo_gen_005", generation=5)
        registry.get_model.return_value = (network, metadata)

        manager = BotManager(
            registry=registry,  # type: ignore[arg-type]
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            room_password="",
            bot_config=BotConfig(
                bot_type="neural_network",
                model_id="ppo_gen_005",
                generation=3,  # Both provided, model_id should take precedence
                player_name="BothBot",
            ),
        )

        response = await manager.spawn_bot(request)

        assert response.success is True
        assert response.bot_id is not None

        # Verify model was loaded by model_id (not generation)
        registry.get_model.assert_called_once_with("ppo_gen_005", device="cpu")
        # get_model_by_generation should NOT have been called
        registry.get_model_by_generation.assert_not_called()

    @pytest.mark.asyncio
    async def test_spawn_neural_network_bot_no_registry(self) -> None:
        """spawn_bot() with neural network but no registry returns error."""
        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            bot_config=BotConfig(
                bot_type="neural_network",
                model_id="ppo_gen_001",
            ),
        )

        response = await manager.spawn_bot(request)

        assert response.success is False
        assert response.bot_id is None
        assert response.error is not None
        assert "ModelRegistry" in response.error
        assert len(manager._bots) == 0

    @pytest.mark.asyncio
    async def test_spawn_neural_network_bot_invalid_model_id(self) -> None:
        """spawn_bot() with invalid model_id returns error."""
        registry = MockModelRegistry()
        registry.get_model.side_effect = ModelNotFoundError("Model 'invalid' not found")

        manager = BotManager(
            registry=registry,  # type: ignore[arg-type]
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            bot_config=BotConfig(
                bot_type="neural_network",
                model_id="invalid",
            ),
        )

        response = await manager.spawn_bot(request)

        assert response.success is False
        assert response.bot_id is None
        assert response.error is not None
        assert "not found" in response.error
        assert len(manager._bots) == 0

    @pytest.mark.asyncio
    async def test_spawn_neural_network_bot_invalid_generation(self) -> None:
        """spawn_bot() with invalid generation returns error."""
        registry = MockModelRegistry()
        registry.get_model_by_generation.return_value = None

        manager = BotManager(
            registry=registry,  # type: ignore[arg-type]
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            bot_config=BotConfig(
                bot_type="neural_network",
                generation=999,
            ),
        )

        response = await manager.spawn_bot(request)

        assert response.success is False
        assert response.bot_id is None
        assert response.error is not None
        assert "generation 999" in response.error
        assert len(manager._bots) == 0

    @pytest.mark.asyncio
    async def test_spawn_neural_network_bot_no_model_specified(self) -> None:
        """BotConfig with neural network but no model_id or generation raises ValidationError."""
        from pydantic import ValidationError

        # Creating BotConfig should raise ValidationError due to validator
        with pytest.raises(ValidationError, match="model_id or generation"):
            BotConfig(
                bot_type="neural_network",
            )

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.RuleBasedBotRunner")
    async def test_spawn_bot_generates_unique_ids(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """spawn_bot() generates unique bot_ids."""
        mock_client_class.return_value = AsyncMock()
        mock_runner_class.return_value = MagicMock()

        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            bot_config=BotConfig(bot_type="rule_based"),
        )

        response1 = await manager.spawn_bot(request)
        response2 = await manager.spawn_bot(request)
        response3 = await manager.spawn_bot(request)

        assert response1.bot_id != response2.bot_id
        assert response2.bot_id != response3.bot_id
        assert response1.bot_id != response3.bot_id
        assert len(manager._bots) == 3


class TestBotManagerDestroyBot:
    """Tests for BotManager.destroy_bot() method."""

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.RuleBasedBotRunner")
    async def test_destroy_bot_stops_and_removes(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """destroy_bot() stops and removes bot."""
        mock_client = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_runner_class.return_value = MagicMock()

        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            bot_config=BotConfig(bot_type="rule_based"),
        )

        response = await manager.spawn_bot(request)
        assert len(manager._bots) == 1
        assert response.bot_id is not None

        result = await manager.destroy_bot(response.bot_id)

        assert result is True
        assert len(manager._bots) == 0
        mock_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_destroy_bot_unknown_id_returns_false(self) -> None:
        """destroy_bot() returns False for unknown bot_id."""
        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        result = await manager.destroy_bot("unknown_bot_id")

        assert result is False

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.RuleBasedBotRunner")
    async def test_destroy_bot_handles_stop_error(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """destroy_bot() handles errors from client.stop()."""
        mock_client = AsyncMock()
        mock_client.stop = AsyncMock(side_effect=Exception("Stop failed"))
        mock_client_class.return_value = mock_client
        mock_runner_class.return_value = MagicMock()

        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            bot_config=BotConfig(bot_type="rule_based"),
        )

        response = await manager.spawn_bot(request)
        assert response.bot_id is not None

        # Should return True even if stop() fails
        result = await manager.destroy_bot(response.bot_id)

        assert result is True
        assert len(manager._bots) == 0


class TestBotManagerGetBot:
    """Tests for BotManager.get_bot() method."""

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.RuleBasedBotRunner")
    async def test_get_bot_returns_info(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """get_bot() returns BotInfo for active bot."""
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client_class.return_value = mock_client
        mock_runner_class.return_value = MagicMock()

        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            room_password="secret",
            bot_config=BotConfig(
                bot_type="rule_based",
                player_name="TestBot",
            ),
        )

        response = await manager.spawn_bot(request)
        assert response.bot_id is not None

        info = manager.get_bot(response.bot_id)

        assert info is not None
        assert info.bot_id == response.bot_id
        assert info.bot_type == "rule_based"
        assert info.model_id is None
        assert info.player_name == "TestBot"
        assert info.room_code == "ABC123"
        assert info.is_connected is True

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.NeuralNetBotRunner")
    async def test_get_bot_includes_model_id(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """get_bot() includes model_id for neural network bots."""
        mock_client = AsyncMock()
        mock_client.is_connected = False
        mock_client_class.return_value = mock_client
        mock_runner_class.return_value = MagicMock()

        registry = MockModelRegistry()
        network, metadata = registry.setup_model("ppo_gen_002", generation=2)
        registry.get_model.return_value = (network, metadata)

        manager = BotManager(
            registry=registry,  # type: ignore[arg-type]
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="XYZ789",
            bot_config=BotConfig(
                bot_type="neural_network",
                model_id="ppo_gen_002",
            ),
        )

        response = await manager.spawn_bot(request)
        assert response.bot_id is not None

        info = manager.get_bot(response.bot_id)

        assert info is not None
        assert info.model_id == "ppo_gen_002"

    def test_get_bot_unknown_id_returns_none(self) -> None:
        """get_bot() returns None for unknown bot_id."""
        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        info = manager.get_bot("unknown_id")

        assert info is None


class TestBotManagerListBots:
    """Tests for BotManager.list_bots() method."""

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.RuleBasedBotRunner")
    async def test_list_bots_returns_all_active(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """list_bots() returns all active bots."""
        mock_client_class.return_value = AsyncMock()
        mock_runner_class.return_value = MagicMock()

        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request1 = SpawnBotRequest(
            room_code="ROOM1",
            bot_config=BotConfig(bot_type="rule_based", player_name="Bot1"),
        )
        request2 = SpawnBotRequest(
            room_code="ROOM2",
            bot_config=BotConfig(bot_type="rule_based", player_name="Bot2"),
        )

        await manager.spawn_bot(request1)
        await manager.spawn_bot(request2)

        bots = manager.list_bots()

        assert len(bots) == 2
        assert all(isinstance(bot, BotInfo) for bot in bots)
        player_names = {bot.player_name for bot in bots}
        assert player_names == {"Bot1", "Bot2"}

    def test_list_bots_returns_empty_when_no_bots(self) -> None:
        """list_bots() returns empty list when no bots."""
        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        bots = manager.list_bots()

        assert bots == []


class TestBotManagerListModels:
    """Tests for BotManager.list_models() method."""

    def test_list_models_returns_from_registry(self) -> None:
        """list_models() returns models from registry."""
        registry = MockModelRegistry()
        _, metadata1 = registry.setup_model("ppo_gen_001", generation=1)
        _, metadata2 = registry.setup_model("ppo_gen_002", generation=2)
        registry.list_models.return_value = [metadata1, metadata2]

        manager = BotManager(
            registry=registry,  # type: ignore[arg-type]
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        models = manager.list_models()

        assert len(models) == 2
        assert models[0].model_id == "ppo_gen_001"
        assert models[1].model_id == "ppo_gen_002"

    def test_list_models_returns_empty_when_no_registry(self) -> None:
        """list_models() returns empty list when no registry."""
        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        models = manager.list_models()

        assert models == []


class TestBotManagerShutdown:
    """Tests for BotManager.shutdown() method."""

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.RuleBasedBotRunner")
    async def test_shutdown_stops_all_bots(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """shutdown() stops all active bots."""
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        mock_client1.start = AsyncMock()
        mock_client2.start = AsyncMock()
        mock_client_class.side_effect = [mock_client1, mock_client2]
        mock_runner_class.return_value = MagicMock()

        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            bot_config=BotConfig(bot_type="rule_based"),
        )

        await manager.spawn_bot(request)
        await manager.spawn_bot(request)

        assert len(manager._bots) == 2

        await manager.shutdown()

        assert len(manager._bots) == 0
        mock_client1.stop.assert_called_once()
        mock_client2.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_no_bots(self) -> None:
        """shutdown() is safe when no bots active."""
        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        await manager.shutdown()

        assert len(manager._bots) == 0


class TestBotManagerBackgroundConnection:
    """Tests for background bot connection handling."""

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.RuleBasedBotRunner")
    async def test_background_connection_failure_removes_bot(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """Background connection failure removes bot from tracking."""
        mock_client = AsyncMock()
        mock_client.start = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client_class.return_value = mock_client
        mock_runner_class.return_value = MagicMock()

        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            bot_config=BotConfig(bot_type="rule_based"),
        )

        response = await manager.spawn_bot(request)

        assert response.success is True
        assert len(manager._bots) == 1

        # Wait for background task to complete
        await manager.await_pending_tasks()

        # Bot should be removed from tracking
        assert len(manager._bots) == 0

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.RuleBasedBotRunner")
    async def test_concurrent_spawn_bot_calls(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """Multiple concurrent spawn_bot() calls produce unique bot IDs."""
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_runner_class.return_value = MagicMock()

        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        # Spawn multiple bots concurrently
        requests = [
            SpawnBotRequest(
                room_code=f"ROOM{i}",
                bot_config=BotConfig(bot_type="rule_based"),
            )
            for i in range(5)
        ]

        responses = await asyncio.gather(*[manager.spawn_bot(req) for req in requests])

        # All should succeed
        assert all(r.success for r in responses)

        # All bot IDs should be unique
        bot_ids = [r.bot_id for r in responses]
        assert len(bot_ids) == len(set(bot_ids))

        # All bots should be tracked
        assert len(manager._bots) == 5

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.RuleBasedBotRunner")
    async def test_destroy_bot_twice_idempotent(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """destroy_bot() called twice for same bot_id is idempotent."""
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_runner_class.return_value = MagicMock()

        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        request = SpawnBotRequest(
            room_code="ABC123",
            bot_config=BotConfig(bot_type="rule_based"),
        )

        response = await manager.spawn_bot(request)
        bot_id = response.bot_id
        assert bot_id is not None

        # First destroy should succeed
        result1 = await manager.destroy_bot(bot_id)
        assert result1 is True

        # Second destroy should return False (not found)
        result2 = await manager.destroy_bot(bot_id)
        assert result2 is False

        # Bot should not be in tracking
        assert len(manager._bots) == 0

    @pytest.mark.asyncio
    @patch("bot.service.bot_manager.WebSocketBotClient")
    @patch("bot.service.bot_manager.RuleBasedBotRunner")
    async def test_shutdown_with_pending_connections(
        self,
        mock_runner_class: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """shutdown() waits for pending background connection tasks."""
        # Create a mock client that takes time to start
        mock_client = AsyncMock()
        connection_started = asyncio.Event()

        async def slow_start(room_code: str, room_password: str) -> None:
            await asyncio.sleep(0.05)  # Simulate slow connection
            connection_started.set()

        mock_client.start = AsyncMock(side_effect=slow_start)
        mock_client.stop = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_runner_class.return_value = MagicMock()

        manager = BotManager(
            registry=None,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        # Spawn a bot (connection starts in background)
        request = SpawnBotRequest(
            room_code="ABC123",
            bot_config=BotConfig(bot_type="rule_based"),
        )
        await manager.spawn_bot(request)

        # Immediately call shutdown (while connection is pending)
        assert not connection_started.is_set()
        await manager.shutdown()

        # Connection should have completed during shutdown
        assert connection_started.is_set()

        # All tasks should be complete
        assert len(manager._tasks) == 0


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_bot_config_rule_based(self) -> None:
        """BotConfig validates rule_based type."""
        config = BotConfig(bot_type="rule_based")

        assert config.bot_type == "rule_based"
        assert config.model_id is None
        assert config.generation is None
        assert config.player_name == "Bot"

    def test_bot_config_neural_network_with_model_id(self) -> None:
        """BotConfig validates neural_network with model_id."""
        config = BotConfig(
            bot_type="neural_network",
            model_id="ppo_gen_005",
            player_name="NeuralBot",
        )

        assert config.bot_type == "neural_network"
        assert config.model_id == "ppo_gen_005"
        assert config.generation is None
        assert config.player_name == "NeuralBot"

    def test_bot_config_neural_network_with_generation(self) -> None:
        """BotConfig validates neural_network with generation."""
        config = BotConfig(
            bot_type="neural_network",
            generation=10,
        )

        assert config.bot_type == "neural_network"
        assert config.model_id is None
        assert config.generation == 10

    def test_bot_config_neural_network_with_both_model_id_and_generation(self) -> None:
        """BotConfig accepts both model_id and generation (model_id takes precedence in _load_model)."""
        config = BotConfig(
            bot_type="neural_network",
            model_id="ppo_gen_005",
            generation=3,
        )

        # Both values should be stored
        assert config.bot_type == "neural_network"
        assert config.model_id == "ppo_gen_005"
        assert config.generation == 3

    def test_bot_config_invalid_type_raises(self) -> None:
        """BotConfig raises error for invalid bot_type."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BotConfig(bot_type="invalid_type")  # type: ignore[arg-type]

    def test_bot_config_rule_based_with_model_id_raises(self) -> None:
        """BotConfig raises error for rule_based with model_id."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError,
            match="Rule-based bots cannot specify model_id or generation",
        ):
            BotConfig(bot_type="rule_based", model_id="ppo_gen_005")

    def test_bot_config_rule_based_with_generation_raises(self) -> None:
        """BotConfig raises error for rule_based with generation."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError,
            match="Rule-based bots cannot specify model_id or generation",
        ):
            BotConfig(bot_type="rule_based", generation=5)

    def test_bot_config_neural_network_without_model_or_gen_raises(self) -> None:
        """BotConfig raises error for neural_network without model_id or generation."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError,
            match="Neural network bots require either model_id or generation",
        ):
            BotConfig(bot_type="neural_network")

    def test_spawn_bot_request_defaults(self) -> None:
        """SpawnBotRequest has correct defaults."""
        request = SpawnBotRequest(
            room_code="ABC123",
            bot_config=BotConfig(bot_type="rule_based"),
        )

        assert request.room_code == "ABC123"
        assert request.room_password == ""
        assert request.bot_config.bot_type == "rule_based"

    def test_spawn_bot_response_success(self) -> None:
        """SpawnBotResponse success state."""
        response = SpawnBotResponse(success=True, bot_id="bot_123")

        assert response.success is True
        assert response.bot_id == "bot_123"
        assert response.error is None

    def test_spawn_bot_response_error(self) -> None:
        """SpawnBotResponse error state."""
        response = SpawnBotResponse(success=False, error="Model not found")

        assert response.success is False
        assert response.bot_id is None
        assert response.error == "Model not found"

    def test_bot_info_serialization(self) -> None:
        """BotInfo serialization."""
        info = BotInfo(
            bot_id="bot_abc123",
            bot_type="neural_network",
            model_id="ppo_gen_005",
            player_name="NeuralBot",
            room_code="XYZ789",
            is_connected=True,
        )

        data = info.model_dump()

        assert data["bot_id"] == "bot_abc123"
        assert data["bot_type"] == "neural_network"
        assert data["model_id"] == "ppo_gen_005"
        assert data["player_name"] == "NeuralBot"
        assert data["room_code"] == "XYZ789"
        assert data["is_connected"] is True
