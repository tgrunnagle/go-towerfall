"""Integration tests for TrainingOrchestrator with real go-towerfall backend.

Tests cover:
- Setup and initialization
- Short training runs
- Checkpoint save and load
- Callback invocation
- Graceful stop
- Model registration

All tests require a running go-towerfall server.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from bot.agent.ppo_trainer import PPOConfig
from bot.training import (
    ModelRegistry,
    OrchestratorConfig,
    TrainingGameConfig,
    TrainingOrchestrator,
)
from tests.conftest import requires_server

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def orchestrator_config_smoke(tmp_path: Path) -> OrchestratorConfig:
    """Minimal config for smoke tests."""
    return OrchestratorConfig(
        num_envs=1,
        total_timesteps=128,  # Very short training
        ppo_config=PPOConfig(
            num_steps=16,
            num_epochs=1,
            minibatch_size=16,
        ),
        game_config=TrainingGameConfig(
            room_name="OrchestratorSmokeTest",
            tick_multiplier=20.0,  # Fast simulation
            max_game_duration_sec=30,
        ),
        checkpoint_interval=64,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        registry_path=str(tmp_path / "registry"),
        log_interval=32,
        eval_interval=256,  # Don't run eval during smoke tests
        eval_episodes=1,
    )


@pytest.fixture
def orchestrator_config_short(tmp_path: Path) -> OrchestratorConfig:
    """Config for short training runs."""
    return OrchestratorConfig(
        num_envs=2,
        total_timesteps=512,
        ppo_config=PPOConfig(
            num_steps=32,
            num_epochs=2,
            minibatch_size=32,
        ),
        game_config=TrainingGameConfig(
            room_name="OrchestratorShortTest",
            tick_multiplier=20.0,
            max_game_duration_sec=30,
        ),
        checkpoint_interval=256,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        registry_path=str(tmp_path / "registry"),
        log_interval=64,
        eval_interval=512,
        eval_episodes=1,
    )


# ============================================================================
# Smoke Tests
# ============================================================================


@pytest.mark.integration
class TestOrchestratorSmoke:
    """Quick sanity checks for orchestrator with real backend."""

    @requires_server
    @pytest.mark.asyncio
    async def test_setup_initializes_components(
        self, orchestrator_config_smoke: OrchestratorConfig
    ) -> None:
        """Test setup initializes all components."""
        orchestrator = TrainingOrchestrator(orchestrator_config_smoke)

        try:
            await orchestrator.setup()

            # Verify all components are initialized
            assert orchestrator.env is not None
            assert orchestrator.network is not None
            assert orchestrator.trainer is not None
            assert orchestrator.registry is not None

            # Verify network has correct architecture
            assert orchestrator.network.observation_size == 414  # Default obs size
            assert orchestrator.network.action_size == 27  # Default action size
        finally:
            await orchestrator.cleanup()

    @requires_server
    @pytest.mark.asyncio
    async def test_context_manager_setup_and_cleanup(
        self, orchestrator_config_smoke: OrchestratorConfig
    ) -> None:
        """Test context manager handles setup and cleanup."""
        async with TrainingOrchestrator(orchestrator_config_smoke) as orchestrator:
            assert orchestrator.env is not None
            assert orchestrator.network is not None

        # After exit, env should be None
        assert orchestrator.env is None

    @requires_server
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_short_training_completes(
        self, orchestrator_config_smoke: OrchestratorConfig
    ) -> None:
        """Test very short training run completes."""
        async with TrainingOrchestrator(orchestrator_config_smoke) as orchestrator:
            metadata = await orchestrator.train()

            # Verify training completed
            assert (
                orchestrator.total_timesteps
                >= orchestrator_config_smoke.total_timesteps
            )
            assert orchestrator.num_updates > 0

            # Verify model was registered
            assert metadata is not None
            assert metadata.model_id is not None
            assert (
                metadata.training_metrics.total_timesteps
                == orchestrator.total_timesteps
            )


# ============================================================================
# Short Training Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestOrchestratorShortTraining:
    """Short training runs verifying orchestrator behavior."""

    @requires_server
    @pytest.mark.asyncio
    async def test_callbacks_invoked_during_training(
        self, orchestrator_config_short: OrchestratorConfig
    ) -> None:
        """Test callbacks are invoked during training."""
        events_received: list[dict[str, Any]] = []

        def callback(event: dict[str, Any]) -> None:
            events_received.append(event)

        async with TrainingOrchestrator(orchestrator_config_short) as orchestrator:
            orchestrator.register_callback(callback)
            await orchestrator.train()

        # Should have received progress events
        progress_events = [e for e in events_received if e["type"] == "progress"]
        assert len(progress_events) > 0

        # Check progress events have expected metrics
        for event in progress_events:
            assert "metrics" in event
            assert "timesteps" in event["metrics"]
            assert "updates" in event["metrics"]

    @requires_server
    @pytest.mark.asyncio
    async def test_checkpoint_saved_during_training(
        self, orchestrator_config_short: OrchestratorConfig
    ) -> None:
        """Test checkpoints are saved during training."""
        async with TrainingOrchestrator(orchestrator_config_short) as orchestrator:
            await orchestrator.train()

        # Check checkpoint directory has files
        checkpoint_dir = Path(orchestrator_config_short.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) > 0

        # Verify checkpoint can be loaded
        checkpoint = torch.load(checkpoints[0], weights_only=False)
        assert "network_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "total_timesteps" in checkpoint

    @requires_server
    @pytest.mark.asyncio
    async def test_model_registered_after_training(
        self, orchestrator_config_short: OrchestratorConfig
    ) -> None:
        """Test model is registered in registry after training."""
        async with TrainingOrchestrator(orchestrator_config_short) as orchestrator:
            metadata = await orchestrator.train()

        # Verify we can load the model from registry
        registry = ModelRegistry(orchestrator_config_short.registry_path)
        network, loaded_metadata = registry.get_model(metadata.model_id)

        assert network is not None
        assert loaded_metadata.model_id == metadata.model_id
        assert loaded_metadata.generation == orchestrator.current_generation

    @requires_server
    @pytest.mark.asyncio
    async def test_stop_halts_training_early(
        self, orchestrator_config_short: OrchestratorConfig
    ) -> None:
        """Test stop() halts training before completion."""
        orchestrator_config_short.total_timesteps = 10_000  # Long training

        async with TrainingOrchestrator(orchestrator_config_short) as orchestrator:
            # Register callback to stop after a few updates
            stop_after_updates = 3

            def stop_callback(event: dict[str, Any]) -> None:
                if event["type"] == "progress":
                    if event["metrics"]["updates"] >= stop_after_updates:
                        orchestrator.stop()

            orchestrator.register_callback(stop_callback)
            await orchestrator.train()

            # Should have stopped early
            assert (
                orchestrator.num_updates < 100
            )  # Well before 10k timesteps would require

    @requires_server
    @pytest.mark.asyncio
    async def test_checkpoint_load_and_resume(
        self, orchestrator_config_short: OrchestratorConfig
    ) -> None:
        """Test loading checkpoint and resuming training."""
        checkpoint_path = None

        # Phase 1: Train for a bit and save checkpoint
        orchestrator_config_short.total_timesteps = 256
        async with TrainingOrchestrator(orchestrator_config_short) as orchestrator:
            await orchestrator.train()
            checkpoint_path = next(
                Path(orchestrator_config_short.checkpoint_dir).glob("checkpoint_*.pt")
            )
            timesteps_at_checkpoint = int(checkpoint_path.stem.split("_")[1])

        # Phase 2: Create new orchestrator and load checkpoint
        orchestrator_config_short.total_timesteps = 512
        async with TrainingOrchestrator(orchestrator_config_short) as orchestrator:
            orchestrator.load_checkpoint(str(checkpoint_path))

            # Verify state was restored
            assert orchestrator.total_timesteps == timesteps_at_checkpoint

            # Continue training
            await orchestrator.train()

            # Should have trained beyond checkpoint
            assert (
                orchestrator.total_timesteps
                >= orchestrator_config_short.total_timesteps
            )


# ============================================================================
# Registry Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestOrchestratorRegistry:
    """Tests for model registry integration."""

    @requires_server
    @pytest.mark.asyncio
    async def test_generation_increments(
        self, orchestrator_config_smoke: OrchestratorConfig
    ) -> None:
        """Test generation number increments with successive training."""
        # First training run
        async with TrainingOrchestrator(orchestrator_config_smoke) as orchestrator:
            metadata1 = await orchestrator.train()

        assert metadata1.generation == 0

        # Second training run uses first model as opponent (if supported)
        # For now, just verify generation increments when starting fresh
        async with TrainingOrchestrator(orchestrator_config_smoke) as orchestrator:
            # Check that it detects existing models
            assert orchestrator.current_generation == 1

    @requires_server
    @pytest.mark.asyncio
    async def test_training_metrics_recorded(
        self, orchestrator_config_smoke: OrchestratorConfig
    ) -> None:
        """Test training metrics are recorded in model metadata."""
        async with TrainingOrchestrator(orchestrator_config_smoke) as orchestrator:
            metadata = await orchestrator.train()

        # Verify training metrics are present
        assert metadata.training_metrics is not None
        assert metadata.training_metrics.total_timesteps > 0
        assert metadata.training_metrics.total_episodes > 0

    @requires_server
    @pytest.mark.asyncio
    async def test_hyperparameters_recorded(
        self, orchestrator_config_smoke: OrchestratorConfig
    ) -> None:
        """Test PPO hyperparameters are recorded in model metadata."""
        async with TrainingOrchestrator(orchestrator_config_smoke) as orchestrator:
            metadata = await orchestrator.train()

        # Verify hyperparameters are present
        assert metadata.hyperparameters is not None
        assert "num_steps" in metadata.hyperparameters
        assert "learning_rate" in metadata.hyperparameters
        assert (
            metadata.hyperparameters["num_steps"]
            == orchestrator_config_smoke.ppo_config.num_steps
        )
