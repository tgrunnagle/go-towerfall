"""Integration tests for SuccessiveTrainer with real go-towerfall backend.

Tests cover:
- 2-generation successive training run
- Model promotion between generations
- Generation lineage tracking
- Early stopping on stagnation

All tests require a running go-towerfall server.
"""

from pathlib import Path
from typing import Any

import pytest

from bot.agent.ppo_trainer import PPOConfig
from bot.training import (
    GenerationResult,
    OrchestratorConfig,
    PromotionCriteria,
    SuccessiveTrainer,
    SuccessiveTrainingConfig,
    TrainingGameConfig,
)
from tests.conftest import requires_server

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def successive_config_smoke(tmp_path: Path) -> SuccessiveTrainingConfig:
    """Minimal config for smoke tests - very short training."""
    return SuccessiveTrainingConfig(
        base_config=OrchestratorConfig(
            num_envs=1,
            ppo_config=PPOConfig(
                num_steps=16,
                num_epochs=1,
                minibatch_size=16,
            ),
            game_config=TrainingGameConfig(
                room_name="SuccessiveSmokeTest",
                tick_multiplier=20.0,
                max_game_duration_sec=30,
            ),
            log_interval=32,
        ),
        max_generations=2,
        timesteps_per_generation=128,
        evaluation_interval=256,  # Don't run eval during smoke tests
        evaluation_episodes=1,
        promotion_criteria=PromotionCriteria(
            min_kd_ratio=0.0,  # Very permissive - always promote
            kd_improvement=0.0,
            min_eval_episodes=1,
            consecutive_passes=1,
            confidence_threshold=0.5,  # Very permissive
        ),
        max_stagnant_evaluations=5,
        output_dir=str(tmp_path / "successive"),
    )


@pytest.fixture
def successive_config_short(tmp_path: Path) -> SuccessiveTrainingConfig:
    """Config for short training runs with actual evaluation."""
    return SuccessiveTrainingConfig(
        base_config=OrchestratorConfig(
            num_envs=1,
            ppo_config=PPOConfig(
                num_steps=32,
                num_epochs=2,
                minibatch_size=32,
            ),
            game_config=TrainingGameConfig(
                room_name="SuccessiveShortTest",
                tick_multiplier=20.0,
                max_game_duration_sec=30,
            ),
            log_interval=64,
        ),
        max_generations=2,
        timesteps_per_generation=256,
        evaluation_interval=128,
        evaluation_episodes=2,
        promotion_criteria=PromotionCriteria(
            min_kd_ratio=0.0,  # Permissive for testing
            kd_improvement=-2.0,  # Always pass: agent KD 0 vs opponent KD 1 = -1.0 improvement
            min_eval_episodes=1,
            consecutive_passes=1,
            confidence_threshold=0.0,  # No statistical significance required
        ),
        max_stagnant_evaluations=3,
        output_dir=str(tmp_path / "successive"),
    )


# ============================================================================
# Smoke Tests
# ============================================================================


@pytest.mark.integration
class TestSuccessiveTrainerSmoke:
    """Quick sanity checks for successive trainer with real backend."""

    @requires_server
    @pytest.mark.asyncio
    async def test_setup_initializes_components(
        self, successive_config_smoke: SuccessiveTrainingConfig
    ) -> None:
        """Test setup initializes all components."""
        trainer = SuccessiveTrainer(successive_config_smoke)

        try:
            await trainer.setup()

            assert trainer.registry is not None
            assert trainer.eval_manager is not None
            assert trainer.current_generation == 0
            assert trainer.current_opponent_id is None
        finally:
            await trainer.cleanup()

    @requires_server
    @pytest.mark.asyncio
    async def test_context_manager_setup_and_cleanup(
        self, successive_config_smoke: SuccessiveTrainingConfig
    ) -> None:
        """Test context manager handles setup and cleanup."""
        async with SuccessiveTrainer(successive_config_smoke) as trainer:
            assert trainer.registry is not None
            assert trainer.eval_manager is not None

    @requires_server
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_single_generation_completes(
        self, successive_config_smoke: SuccessiveTrainingConfig
    ) -> None:
        """Test single generation training completes."""
        successive_config_smoke.max_generations = 1

        async with SuccessiveTrainer(successive_config_smoke) as trainer:
            results = await trainer.train()

            assert len(results) == 1
            assert results[0].generation == 0
            assert results[0].model_id is not None
            assert results[0].timesteps_trained > 0


# ============================================================================
# Two-Generation Training Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestSuccessiveTrainerTwoGenerations:
    """Tests for 2-generation successive training runs.

    These tests verify the core functionality of successive training:
    training a model, promoting it, then training a second model against it.
    """

    @requires_server
    @pytest.mark.asyncio
    async def test_two_generation_training_completes(
        self, successive_config_short: SuccessiveTrainingConfig
    ) -> None:
        """Test 2-generation successive training run completes.

        This is the primary acceptance criteria test for issue #52.
        """
        async with SuccessiveTrainer(successive_config_short) as trainer:
            results = await trainer.train()

            # Should have trained 2 generations
            assert len(results) == 2

            # Verify generation 0
            gen0 = results[0]
            assert gen0.generation == 0
            assert gen0.model_id is not None
            assert gen0.final_evaluation is not None
            assert gen0.timesteps_trained > 0

            # Verify generation 1
            gen1 = results[1]
            assert gen1.generation == 1
            assert gen1.model_id is not None
            assert gen1.final_evaluation is not None
            assert gen1.timesteps_trained > 0

            # Generation 1 should have trained against generation 0
            assert gen1.model_metadata.opponent_model_id == gen0.model_id

    @requires_server
    @pytest.mark.asyncio
    async def test_generation_lineage_tracked(
        self, successive_config_short: SuccessiveTrainingConfig
    ) -> None:
        """Test generation lineage is correctly tracked."""
        async with SuccessiveTrainer(successive_config_short) as trainer:
            await trainer.train()

            lineage = trainer.get_generation_lineage()

            # Should have 2 entries
            assert len(lineage) == 2

            # Gen 0 trained against rule-based (None)
            assert lineage[0][1] is None

            # Gen 1 trained against gen 0's model
            assert lineage[1][1] == lineage[0][0]

    @requires_server
    @pytest.mark.asyncio
    async def test_models_registered_in_registry(
        self, successive_config_short: SuccessiveTrainingConfig
    ) -> None:
        """Test all trained models are registered in registry."""
        async with SuccessiveTrainer(successive_config_short) as trainer:
            results = await trainer.train()

            # Verify both models can be loaded from registry
            assert trainer.registry is not None
            for result in results:
                network, metadata = trainer.registry.get_model(result.model_id)
                assert network is not None
                assert metadata.model_id == result.model_id
                assert metadata.generation == result.generation

    @requires_server
    @pytest.mark.asyncio
    async def test_callbacks_invoked_per_generation(
        self, successive_config_short: SuccessiveTrainingConfig
    ) -> None:
        """Test callbacks are invoked for each generation completion."""
        events_received: list[dict[str, Any]] = []

        def callback(event: dict[str, Any]) -> None:
            events_received.append(event)

        async with SuccessiveTrainer(successive_config_short) as trainer:
            trainer.register_callback(callback)
            await trainer.train()

        # Should have generation_complete events
        gen_events = [e for e in events_received if e["type"] == "generation_complete"]
        assert len(gen_events) == 2

        # Check generation numbers
        generations = [e["generation"] for e in gen_events]
        assert 0 in generations
        assert 1 in generations

    @requires_server
    @pytest.mark.asyncio
    async def test_best_model_identified(
        self, successive_config_short: SuccessiveTrainingConfig
    ) -> None:
        """Test best model can be identified after training."""
        async with SuccessiveTrainer(successive_config_short) as trainer:
            results = await trainer.train()

            best_model_id = trainer.get_best_model_id()

            assert best_model_id is not None
            # Best model should be one of the trained models
            trained_ids = [r.model_id for r in results]
            assert best_model_id in trained_ids


# ============================================================================
# Control Flow Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestSuccessiveTrainerControlFlow:
    """Tests for successive trainer control flow."""

    @requires_server
    @pytest.mark.asyncio
    async def test_stop_halts_between_generations(
        self, successive_config_short: SuccessiveTrainingConfig
    ) -> None:
        """Test stop() halts training between generations."""
        successive_config_short.max_generations = 5

        async with SuccessiveTrainer(successive_config_short) as trainer:
            # Register callback to stop after first generation
            def stop_callback(event: dict[str, Any]) -> None:
                if event["type"] == "generation_complete":
                    if event["generation"] == 0:
                        trainer.stop()

            trainer.register_callback(stop_callback)
            results = await trainer.train()

            # Should have stopped after 1 generation
            assert len(results) == 1
            assert results[0].generation == 0

    @requires_server
    @pytest.mark.asyncio
    async def test_output_directory_structure(
        self, successive_config_short: SuccessiveTrainingConfig
    ) -> None:
        """Test output directory structure is created correctly."""
        async with SuccessiveTrainer(successive_config_short) as trainer:
            await trainer.train()

        output_dir = Path(successive_config_short.output_dir)

        # Verify directory structure
        assert output_dir.exists()
        assert (output_dir / "checkpoints").exists()
        assert (output_dir / "logs").exists()
        assert (output_dir / "model_registry").exists()

        # Verify generation-specific checkpoint directories
        assert (output_dir / "generation_000" / "checkpoints").exists()
        assert (output_dir / "generation_001" / "checkpoints").exists()


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestSuccessiveTrainerEdgeCases:
    """Tests for edge cases in successive training."""

    @requires_server
    @pytest.mark.asyncio
    async def test_single_generation_max(
        self, successive_config_smoke: SuccessiveTrainingConfig
    ) -> None:
        """Test training with max_generations=1."""
        successive_config_smoke.max_generations = 1

        async with SuccessiveTrainer(successive_config_smoke) as trainer:
            results = await trainer.train()

            assert len(results) == 1
            # No opponent for gen 0 (rule-based)
            assert results[0].model_metadata.opponent_model_id is None

    @requires_server
    @pytest.mark.asyncio
    async def test_generation_result_fields(
        self, successive_config_smoke: SuccessiveTrainingConfig
    ) -> None:
        """Test GenerationResult has all expected fields."""
        successive_config_smoke.max_generations = 1

        async with SuccessiveTrainer(successive_config_smoke) as trainer:
            results = await trainer.train()

            result = results[0]

            # Verify all fields are populated
            assert isinstance(result, GenerationResult)
            assert result.generation == 0
            assert isinstance(result.model_id, str)
            assert result.model_metadata is not None
            assert result.timesteps_trained > 0
            assert isinstance(result.was_promoted, bool)
            assert isinstance(result.promotion_reason, str)
