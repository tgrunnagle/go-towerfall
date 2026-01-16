"""Unit tests for TrainingOrchestrator.

Tests cover:
- Initialization
- Callback registration and invocation
- Stop signal handling
- Checkpoint save/load
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from bot.agent.network import ActorCriticNetwork
from bot.agent.ppo_trainer import PPOTrainer
from bot.training.orchestrator import TrainingOrchestrator
from bot.training.orchestrator_config import OrchestratorConfig


class TestTrainingOrchestratorInit:
    """Tests for TrainingOrchestrator initialization."""

    def test_default_initialization(self) -> None:
        """Test orchestrator initializes with default config."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        assert orchestrator.config is config
        assert orchestrator.total_timesteps == 0
        assert orchestrator.num_updates == 0
        assert orchestrator.current_generation == 0
        assert orchestrator.is_running is False

    def test_components_uninitialized(self) -> None:
        """Test components are None before setup."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        assert orchestrator.env is None
        assert orchestrator.network is None
        assert orchestrator.trainer is None
        assert orchestrator.registry is None

    def test_callbacks_list_empty(self) -> None:
        """Test callbacks list is initially empty."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        assert len(orchestrator._callbacks) == 0

    def test_device_auto_detected(self) -> None:
        """Test device is auto-detected."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        # Should be either cuda or cpu depending on environment
        assert orchestrator.device.type in ["cuda", "cpu"]


class TestCallbackRegistration:
    """Tests for callback registration and invocation."""

    def test_register_callback(self) -> None:
        """Test registering a callback."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        callback = MagicMock()
        orchestrator.register_callback(callback)

        assert len(orchestrator._callbacks) == 1
        assert orchestrator._callbacks[0] is callback

    def test_register_multiple_callbacks(self) -> None:
        """Test registering multiple callbacks."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        callback1 = MagicMock()
        callback2 = MagicMock()
        callback3 = MagicMock()

        orchestrator.register_callback(callback1)
        orchestrator.register_callback(callback2)
        orchestrator.register_callback(callback3)

        assert len(orchestrator._callbacks) == 3

    def test_invoke_callbacks(self) -> None:
        """Test callbacks are invoked with correct event."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        callback = MagicMock()
        orchestrator.register_callback(callback)

        event = {"type": "progress", "metrics": {"timesteps": 1000}}
        orchestrator._invoke_callbacks(event)

        callback.assert_called_once_with(event)

    def test_invoke_multiple_callbacks(self) -> None:
        """Test all callbacks are invoked."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        callback1 = MagicMock()
        callback2 = MagicMock()

        orchestrator.register_callback(callback1)
        orchestrator.register_callback(callback2)

        event = {"type": "evaluation", "metrics": {"avg_reward": 10.0}}
        orchestrator._invoke_callbacks(event)

        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)

    def test_callback_exception_handled(self) -> None:
        """Test callback exceptions don't crash orchestrator."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        failing_callback = MagicMock(side_effect=Exception("Test error"))
        good_callback = MagicMock()

        orchestrator.register_callback(failing_callback)
        orchestrator.register_callback(good_callback)

        event = {"type": "progress", "metrics": {}}

        # Should not raise
        orchestrator._invoke_callbacks(event)

        # Good callback should still be called
        good_callback.assert_called_once_with(event)


class TestStopSignal:
    """Tests for stop signal handling."""

    def test_stop_sets_flag(self) -> None:
        """Test stop() sets is_running to False."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        orchestrator.is_running = True
        orchestrator.stop()

        assert orchestrator.is_running is False

    def test_stop_when_not_running(self) -> None:
        """Test stop() works when not running."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        assert orchestrator.is_running is False
        orchestrator.stop()  # Should not raise
        assert orchestrator.is_running is False


class TestCheckpointSaveLoad:
    """Tests for checkpoint save and load functionality."""

    @pytest.fixture
    def orchestrator_with_network(self, tmp_path: Path) -> TrainingOrchestrator:
        """Create orchestrator with initialized network for testing."""
        config = OrchestratorConfig(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            registry_path=str(tmp_path / "registry"),
        )
        orchestrator = TrainingOrchestrator(config)

        # Manually initialize network and trainer for testing
        orchestrator.network = ActorCriticNetwork(
            observation_size=114,
            action_size=27,
        ).to(orchestrator.device)

        orchestrator.trainer = PPOTrainer(
            network=orchestrator.network,
            config=config.ppo_config,
            device=orchestrator.device,
        )

        # Set some state
        orchestrator.total_timesteps = 5000
        orchestrator.num_updates = 10
        orchestrator.current_generation = 2
        orchestrator._training_start_time = 1000.0

        return orchestrator

    def test_save_checkpoint_creates_file(
        self, orchestrator_with_network: TrainingOrchestrator, tmp_path: Path
    ) -> None:
        """Test _save_checkpoint creates a file."""
        orchestrator = orchestrator_with_network
        Path(orchestrator.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        orchestrator._save_checkpoint()

        checkpoint_path = (
            Path(orchestrator.config.checkpoint_dir) / "checkpoint_5000.pt"
        )
        assert checkpoint_path.exists()

    def test_save_checkpoint_contains_state(
        self, orchestrator_with_network: TrainingOrchestrator, tmp_path: Path
    ) -> None:
        """Test checkpoint contains expected state."""
        orchestrator = orchestrator_with_network
        Path(orchestrator.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        orchestrator._save_checkpoint()

        checkpoint_path = (
            Path(orchestrator.config.checkpoint_dir) / "checkpoint_5000.pt"
        )
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert "network_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["total_timesteps"] == 5000
        assert checkpoint["num_updates"] == 10
        assert checkpoint["generation"] == 2

    def test_load_checkpoint_restores_state(
        self, orchestrator_with_network: TrainingOrchestrator, tmp_path: Path
    ) -> None:
        """Test load_checkpoint restores training state."""
        orchestrator = orchestrator_with_network
        Path(orchestrator.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        orchestrator._save_checkpoint()

        # Reset state
        orchestrator.total_timesteps = 0
        orchestrator.num_updates = 0
        orchestrator.current_generation = 0

        checkpoint_path = str(
            Path(orchestrator.config.checkpoint_dir) / "checkpoint_5000.pt"
        )
        orchestrator.load_checkpoint(checkpoint_path)

        assert orchestrator.total_timesteps == 5000
        assert orchestrator.num_updates == 10
        assert orchestrator.current_generation == 2

    def test_load_checkpoint_updates_trainer(
        self, orchestrator_with_network: TrainingOrchestrator, tmp_path: Path
    ) -> None:
        """Test load_checkpoint updates trainer counters."""
        orchestrator = orchestrator_with_network
        Path(orchestrator.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        orchestrator._save_checkpoint()

        # Reset trainer state
        assert orchestrator.trainer is not None
        orchestrator.trainer.total_timesteps = 0
        orchestrator.trainer.num_updates = 0

        checkpoint_path = str(
            Path(orchestrator.config.checkpoint_dir) / "checkpoint_5000.pt"
        )
        orchestrator.load_checkpoint(checkpoint_path)

        assert orchestrator.trainer.total_timesteps == 5000
        assert orchestrator.trainer.num_updates == 10

    def test_load_checkpoint_without_setup_raises(self, tmp_path: Path) -> None:
        """Test load_checkpoint raises if setup not called."""
        config = OrchestratorConfig(checkpoint_dir=str(tmp_path))
        orchestrator = TrainingOrchestrator(config)

        with pytest.raises(RuntimeError, match="Must call setup"):
            orchestrator.load_checkpoint("nonexistent.pt")


class TestLoggingProgress:
    """Tests for logging and progress tracking."""

    @pytest.fixture
    def orchestrator_for_logging(self) -> TrainingOrchestrator:
        """Create orchestrator for logging tests."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)
        orchestrator.total_timesteps = 10000
        orchestrator.num_updates = 20
        orchestrator._training_start_time = 0.0
        return orchestrator

    def test_log_progress_invokes_callback(
        self, orchestrator_for_logging: TrainingOrchestrator
    ) -> None:
        """Test _log_progress invokes callbacks with progress event."""
        orchestrator = orchestrator_for_logging
        callback = MagicMock()
        orchestrator.register_callback(callback)

        update_metrics = {"policy_loss": 0.5, "value_loss": 0.2, "entropy": 0.1}
        orchestrator._log_progress(update_metrics, [], [], [], [])

        # Check callback was called with progress event
        assert callback.called
        call_args = callback.call_args[0][0]
        assert call_args["type"] == "progress"
        assert "metrics" in call_args

    def test_log_progress_includes_metrics(
        self, orchestrator_for_logging: TrainingOrchestrator
    ) -> None:
        """Test _log_progress includes update metrics in callback."""
        orchestrator = orchestrator_for_logging
        received_events: list[dict[str, Any]] = []

        def capture_callback(event: dict[str, Any]) -> None:
            received_events.append(event)

        orchestrator.register_callback(capture_callback)

        update_metrics = {"policy_loss": 0.5, "value_loss": 0.2, "entropy": 0.1}
        orchestrator._log_progress(update_metrics, [], [], [], [])

        assert len(received_events) == 1
        metrics = received_events[0]["metrics"]
        assert metrics["timesteps"] == 10000
        assert metrics["updates"] == 20
        assert metrics["policy_loss"] == 0.5

    def test_log_progress_computes_averages(
        self, orchestrator_for_logging: TrainingOrchestrator
    ) -> None:
        """Test _log_progress computes average statistics."""
        orchestrator = orchestrator_for_logging
        received_events: list[dict[str, Any]] = []

        def capture_callback(event: dict[str, Any]) -> None:
            received_events.append(event)

        orchestrator.register_callback(capture_callback)

        update_metrics = {"policy_loss": 0.5}
        episode_rewards = [10.0, 20.0, 30.0]
        episode_lengths = [100, 200, 300]
        episode_kills = [1.0, 2.0, 3.0]
        episode_deaths = [0.5, 1.0, 1.5]

        orchestrator._log_progress(
            update_metrics,
            episode_rewards,
            episode_lengths,
            episode_kills,
            episode_deaths,
        )

        metrics = received_events[0]["metrics"]
        assert metrics["avg_episode_reward"] == pytest.approx(20.0)
        assert metrics["avg_episode_length"] == pytest.approx(200.0)
        assert metrics["num_episodes"] == 3


class TestTrainErrorHandling:
    """Tests for error handling in train method."""

    def test_train_without_setup_raises(self) -> None:
        """Test train raises if setup not called."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        with pytest.raises(RuntimeError, match="Must call setup"):
            import asyncio

            asyncio.run(orchestrator.train())


class TestContextManager:
    """Tests for async context manager protocol."""

    @pytest.mark.asyncio
    async def test_cleanup_with_no_env(self) -> None:
        """Test cleanup works when env is None."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        # Should not raise even with no env
        await orchestrator.cleanup()

        assert orchestrator.env is None

    @pytest.mark.asyncio
    async def test_cleanup_closes_env(self) -> None:
        """Test cleanup closes environment."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        # Create mock env
        mock_env = MagicMock()
        orchestrator.env = mock_env

        await orchestrator.cleanup()

        mock_env.close.assert_called_once()
        assert orchestrator.env is None

    @pytest.mark.asyncio
    async def test_cleanup_closes_eval_env(self) -> None:
        """Test cleanup closes evaluation environment."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        # Create mock envs
        mock_env = MagicMock()
        mock_eval_env = MagicMock()
        orchestrator.env = mock_env
        orchestrator._eval_env = mock_eval_env

        await orchestrator.cleanup()

        mock_env.close.assert_called_once()
        mock_eval_env.close.assert_called_once()
        assert orchestrator.env is None
        assert orchestrator._eval_env is None


class TestExtractEpisodeStats:
    """Tests for episode statistics extraction."""

    def test_extract_episode_stats_empty_lists(self) -> None:
        """Test _extract_episode_stats handles empty lists."""
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        rewards: list[float] = []
        lengths: list[int] = []
        kills: list[float] = []
        deaths: list[float] = []

        # Should not raise
        orchestrator._extract_episode_stats(rewards, lengths, kills, deaths)

        # Lists should remain empty (placeholder implementation)
        assert len(rewards) == 0
