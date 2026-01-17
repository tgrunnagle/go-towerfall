"""Unit tests for opponent manager module.

Tests cover:
- Factory function with valid/invalid opponent types
- RuleBasedOpponent start/stop lifecycle
- ModelOpponent for self-play training
- NoOpponent no-op behavior
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from bot.gym.model_opponent import ModelOpponent
from bot.gym.opponent_manager import (
    NoOpponent,
    RuleBasedOpponent,
    create_opponent,
)


class TestCreateOpponent:
    """Tests for the create_opponent factory function."""

    def test_create_rule_based_opponent(self) -> None:
        """Factory creates RuleBasedOpponent for 'rule_based' type."""
        opponent = create_opponent(
            opponent_type="rule_based",
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )
        assert isinstance(opponent, RuleBasedOpponent)

    def test_create_rule_based_opponent_with_custom_name(self) -> None:
        """Factory passes player_name to RuleBasedOpponent."""
        opponent = create_opponent(
            opponent_type="rule_based",
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
            player_name="CustomBot",
        )
        assert isinstance(opponent, RuleBasedOpponent)
        assert opponent.player_name == "CustomBot"

    def test_create_no_opponent(self) -> None:
        """Factory creates NoOpponent for 'none' type."""
        opponent = create_opponent(
            opponent_type="none",
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )
        assert isinstance(opponent, NoOpponent)

    def test_create_opponent_invalid_type(self) -> None:
        """Factory raises ValueError for unknown type."""
        with pytest.raises(ValueError, match="Unknown opponent type"):
            create_opponent(
                opponent_type="invalid",
                http_url="http://localhost:4000",
                ws_url="ws://localhost:4000/ws",
            )

    def test_create_opponent_invalid_type_message(self) -> None:
        """Factory error includes the invalid type in message."""
        with pytest.raises(ValueError, match="invalid_type"):
            create_opponent(
                opponent_type="invalid_type",
                http_url="http://localhost:4000",
                ws_url="ws://localhost:4000/ws",
            )


class TestNoOpponent:
    """Tests for NoOpponent no-op behavior."""

    @pytest.mark.asyncio
    async def test_start_is_noop(self) -> None:
        """NoOpponent.start() does nothing and doesn't raise."""
        opponent = NoOpponent()
        await opponent.start("ROOM123")  # Should not raise

    @pytest.mark.asyncio
    async def test_stop_is_noop(self) -> None:
        """NoOpponent.stop() does nothing and doesn't raise."""
        opponent = NoOpponent()
        await opponent.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_on_game_state_is_noop(self) -> None:
        """NoOpponent.on_game_state() does nothing and doesn't raise."""
        opponent = NoOpponent()
        mock_state = MagicMock()
        await opponent.on_game_state(mock_state)  # Should not raise

    def test_reset_is_noop(self) -> None:
        """NoOpponent.reset() does nothing and doesn't raise."""
        opponent = NoOpponent()
        opponent.reset()  # Should not raise

    @pytest.mark.asyncio
    async def test_full_lifecycle(self) -> None:
        """NoOpponent can go through full lifecycle without issues."""
        opponent = NoOpponent()
        await opponent.start("ROOM", "password")
        mock_state = MagicMock()
        await opponent.on_game_state(mock_state)
        opponent.reset()
        await opponent.on_game_state(mock_state)
        await opponent.stop()


class TestRuleBasedOpponent:
    """Tests for RuleBasedOpponent lifecycle."""

    def test_init_stores_config(self) -> None:
        """RuleBasedOpponent stores configuration on init."""
        opponent = RuleBasedOpponent(
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
            player_name="TestBot",
        )
        assert opponent.http_url == "http://localhost:4000"
        assert opponent.ws_url == "ws://localhost:4000/ws"
        assert opponent.player_name == "TestBot"

    def test_init_default_name(self) -> None:
        """RuleBasedOpponent uses default name if not provided."""
        opponent = RuleBasedOpponent(
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )
        assert opponent.player_name == "RuleBot"

    @pytest.mark.asyncio
    async def test_start_creates_client_and_joins(self) -> None:
        """RuleBasedOpponent.start() creates client and joins room."""
        with patch("bot.gym.opponent_manager.GameClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            opponent = RuleBasedOpponent(
                http_url="http://localhost:4000",
                ws_url="ws://localhost:4000/ws",
                player_name="TestBot",
            )

            await opponent.start(room_code="TEST123", room_password="secret")

            # Verify client was created with correct parameters
            MockClient.assert_called_once()
            call_kwargs = MockClient.call_args.kwargs
            assert call_kwargs["http_url"] == "http://localhost:4000"
            assert call_kwargs["ws_url"] == "ws://localhost:4000/ws"

            # Verify connect was called
            mock_client.connect.assert_called_once()

            # Verify join_game was called with correct parameters
            mock_client.join_game.assert_called_once_with(
                room_code="TEST123",
                player_name="TestBot",
                room_password="secret",
                is_spectator=False,
            )

    @pytest.mark.asyncio
    async def test_stop_closes_client(self) -> None:
        """RuleBasedOpponent.stop() closes the client."""
        with patch("bot.gym.opponent_manager.GameClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            opponent = RuleBasedOpponent(
                http_url="http://localhost:4000",
                ws_url="ws://localhost:4000/ws",
            )

            await opponent.start(room_code="TEST")
            await opponent.stop()

            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_without_start(self) -> None:
        """RuleBasedOpponent.stop() is safe to call without start."""
        opponent = RuleBasedOpponent(
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )
        await opponent.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_on_game_state_calls_runner(self) -> None:
        """RuleBasedOpponent.on_game_state() calls runner.on_game_state()."""
        with patch("bot.gym.opponent_manager.GameClient") as MockClient:
            with patch("bot.gym.opponent_manager.RuleBasedBotRunner") as MockRunner:
                mock_client = AsyncMock()
                mock_client.player_id = "player-1"
                MockClient.return_value = mock_client

                mock_runner = AsyncMock()
                MockRunner.return_value = mock_runner

                opponent = RuleBasedOpponent(
                    http_url="http://localhost:4000",
                    ws_url="ws://localhost:4000/ws",
                )

                await opponent.start(room_code="TEST")

                # Create a mock game state
                mock_state = MagicMock()
                await opponent.on_game_state(mock_state)

                mock_runner.on_game_state.assert_called_once_with(mock_state)

    @pytest.mark.asyncio
    async def test_on_game_state_before_start(self) -> None:
        """RuleBasedOpponent.on_game_state() is safe before start."""
        opponent = RuleBasedOpponent(
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )
        mock_state = MagicMock()
        await opponent.on_game_state(mock_state)  # Should not raise

    def test_reset_calls_runner_reset(self) -> None:
        """RuleBasedOpponent.reset() calls runner.reset()."""
        opponent = RuleBasedOpponent(
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        # Create a mock runner
        mock_runner = MagicMock()
        opponent._runner = mock_runner

        opponent.reset()

        mock_runner.reset.assert_called_once()

    def test_reset_before_start(self) -> None:
        """RuleBasedOpponent.reset() is safe before start."""
        opponent = RuleBasedOpponent(
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )
        opponent.reset()  # Should not raise

    @pytest.mark.asyncio
    async def test_on_game_state_handles_errors(self) -> None:
        """RuleBasedOpponent.on_game_state() logs errors but doesn't raise."""
        with patch("bot.gym.opponent_manager.GameClient") as MockClient:
            with patch("bot.gym.opponent_manager.RuleBasedBotRunner") as MockRunner:
                mock_client = AsyncMock()
                mock_client.player_id = "player-1"
                MockClient.return_value = mock_client

                mock_runner = AsyncMock()
                mock_runner.on_game_state.side_effect = RuntimeError("Test error")
                MockRunner.return_value = mock_runner

                opponent = RuleBasedOpponent(
                    http_url="http://localhost:4000",
                    ws_url="ws://localhost:4000/ws",
                )

                await opponent.start(room_code="TEST")

                # Should not raise, just log the error
                mock_state = MagicMock()
                await opponent.on_game_state(mock_state)


class TestCreateModelOpponent:
    """Tests for creating model opponents via factory."""

    def test_create_model_opponent_requires_model(self) -> None:
        """Factory raises ValueError when model is None for 'model' type."""
        with pytest.raises(ValueError, match="model parameter required"):
            create_opponent(
                opponent_type="model",
                http_url="http://localhost:4000",
                ws_url="ws://localhost:4000/ws",
            )

    def test_create_model_opponent_with_model(self) -> None:
        """Factory creates ModelOpponent when model is provided."""
        from bot.agent.network import ActorCriticNetwork

        mock_model = ActorCriticNetwork(observation_size=10, action_size=5)

        opponent = create_opponent(
            opponent_type="model",
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
            player_name="ModelBot",
            model=mock_model,
        )

        assert isinstance(opponent, ModelOpponent)
        assert opponent.player_name == "ModelBot"

    def test_create_model_opponent_with_device(self) -> None:
        """Factory passes device to ModelOpponent."""
        from bot.agent.network import ActorCriticNetwork

        mock_model = ActorCriticNetwork(observation_size=10, action_size=5)
        device = torch.device("cpu")

        opponent = create_opponent(
            opponent_type="model",
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
            model=mock_model,
            device=device,
        )

        assert isinstance(opponent, ModelOpponent)
        assert opponent.device == device


class TestModelOpponent:
    """Tests for ModelOpponent class."""

    def test_init_stores_config(self) -> None:
        """ModelOpponent stores configuration on init."""
        from bot.agent.network import ActorCriticNetwork

        mock_model = ActorCriticNetwork(observation_size=10, action_size=5)

        opponent = ModelOpponent(
            model=mock_model,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
            player_name="TestModel",
        )

        assert opponent.http_url == "http://localhost:4000"
        assert opponent.ws_url == "ws://localhost:4000/ws"
        assert opponent.player_name == "TestModel"
        assert opponent.model is mock_model

    def test_init_default_name(self) -> None:
        """ModelOpponent uses default name if not provided."""
        from bot.agent.network import ActorCriticNetwork

        mock_model = ActorCriticNetwork(observation_size=10, action_size=5)

        opponent = ModelOpponent(
            model=mock_model,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        assert opponent.player_name == "ModelBot"

    def test_init_sets_eval_mode(self) -> None:
        """ModelOpponent sets model to eval mode on init."""
        from bot.agent.network import ActorCriticNetwork

        mock_model = ActorCriticNetwork(observation_size=10, action_size=5)

        opponent = ModelOpponent(
            model=mock_model,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        assert not opponent.model.training

    @pytest.mark.asyncio
    async def test_start_creates_client_and_joins(self) -> None:
        """ModelOpponent.start() creates client and joins room."""
        from bot.agent.network import ActorCriticNetwork

        mock_model = ActorCriticNetwork(observation_size=10, action_size=5)

        with patch("bot.gym.model_opponent.GameClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            opponent = ModelOpponent(
                model=mock_model,
                http_url="http://localhost:4000",
                ws_url="ws://localhost:4000/ws",
                player_name="TestModel",
            )

            await opponent.start(room_code="TEST123", room_password="secret")

            # Verify client was created
            MockClient.assert_called_once()

            # Verify connect was called
            mock_client.connect.assert_called_once()

            # Verify join_game was called with correct parameters
            mock_client.join_game.assert_called_once_with(
                room_code="TEST123",
                player_name="TestModel",
                room_password="secret",
                is_spectator=False,
            )

    @pytest.mark.asyncio
    async def test_stop_closes_client(self) -> None:
        """ModelOpponent.stop() closes the client."""
        from bot.agent.network import ActorCriticNetwork

        mock_model = ActorCriticNetwork(observation_size=10, action_size=5)

        with patch("bot.gym.model_opponent.GameClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            opponent = ModelOpponent(
                model=mock_model,
                http_url="http://localhost:4000",
                ws_url="ws://localhost:4000/ws",
            )

            await opponent.start(room_code="TEST")
            await opponent.stop()

            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_without_start(self) -> None:
        """ModelOpponent.stop() is safe to call without start."""
        from bot.agent.network import ActorCriticNetwork

        mock_model = ActorCriticNetwork(observation_size=10, action_size=5)

        opponent = ModelOpponent(
            model=mock_model,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        await opponent.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_on_game_state_before_start(self) -> None:
        """ModelOpponent.on_game_state() is safe before start."""
        from bot.agent.network import ActorCriticNetwork

        mock_model = ActorCriticNetwork(observation_size=10, action_size=5)

        opponent = ModelOpponent(
            model=mock_model,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        mock_state = MagicMock()
        await opponent.on_game_state(mock_state)  # Should not raise

    def test_reset_is_noop(self) -> None:
        """ModelOpponent.reset() is a no-op but doesn't raise."""
        from bot.agent.network import ActorCriticNetwork

        mock_model = ActorCriticNetwork(observation_size=10, action_size=5)

        opponent = ModelOpponent(
            model=mock_model,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        opponent.reset()  # Should not raise
