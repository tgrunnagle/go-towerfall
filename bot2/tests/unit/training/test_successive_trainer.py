"""Unit tests for SuccessiveTrainer.

Tests cover:
- SuccessiveTrainer initialization
- GenerationResult dataclass
- Setup and cleanup lifecycle
- Callback registration
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bot.training.evaluation import EvaluationResult
from bot.training.registry import ModelMetadata
from bot.training.successive_config import PromotionCriteria, SuccessiveTrainingConfig
from bot.training.successive_trainer import GenerationResult, SuccessiveTrainer


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_all_fields_accessible(self) -> None:
        """Test that all fields are accessible."""
        mock_metadata = MagicMock(spec=ModelMetadata)
        mock_metadata.model_id = "ppo_gen_000"
        mock_metadata.opponent_model_id = None

        mock_eval = MagicMock(spec=EvaluationResult)
        mock_eval.kd_ratio = 1.5

        result = GenerationResult(
            generation=0,
            model_id="ppo_gen_000",
            model_metadata=mock_metadata,
            final_evaluation=mock_eval,
            comparison=None,
            timesteps_trained=100_000,
            was_promoted=True,
            promotion_reason="Met all criteria",
        )

        assert result.generation == 0
        assert result.model_id == "ppo_gen_000"
        assert result.timesteps_trained == 100_000
        assert result.was_promoted is True
        assert result.promotion_reason == "Met all criteria"

    def test_none_final_evaluation(self) -> None:
        """Test that final_evaluation can be None."""
        mock_metadata = MagicMock(spec=ModelMetadata)

        result = GenerationResult(
            generation=0,
            model_id="ppo_gen_000",
            model_metadata=mock_metadata,
            final_evaluation=None,
            comparison=None,
            timesteps_trained=100_000,
            was_promoted=False,
            promotion_reason="No evaluation",
        )

        assert result.final_evaluation is None


class TestSuccessiveTrainerInit:
    """Tests for SuccessiveTrainer initialization."""

    def test_init_stores_config(self) -> None:
        """Test that config is stored correctly."""
        config = SuccessiveTrainingConfig(
            max_generations=5,
            output_dir="/test/output",
        )

        trainer = SuccessiveTrainer(config)

        assert trainer.config is config
        assert trainer.output_dir == Path("/test/output")

    def test_init_default_state(self) -> None:
        """Test initial state is correctly set."""
        config = SuccessiveTrainingConfig()
        trainer = SuccessiveTrainer(config)

        assert trainer.current_generation == 0
        assert trainer.current_opponent_id is None
        assert trainer.generation_results == []
        assert trainer.is_running is False

    def test_init_components_none(self) -> None:
        """Test components are None before setup."""
        config = SuccessiveTrainingConfig()
        trainer = SuccessiveTrainer(config)

        assert trainer.registry is None
        assert trainer.eval_manager is None


class TestSuccessiveTrainerSetup:
    """Tests for SuccessiveTrainer.setup method."""

    @pytest.mark.asyncio
    async def test_setup_creates_directories(self, tmp_path: Path) -> None:
        """Test that setup creates output directories."""
        config = SuccessiveTrainingConfig(output_dir=str(tmp_path / "successive"))

        trainer = SuccessiveTrainer(config)
        await trainer.setup()

        assert (tmp_path / "successive").exists()
        assert (tmp_path / "successive" / "checkpoints").exists()
        assert (tmp_path / "successive" / "logs").exists()

    @pytest.mark.asyncio
    async def test_setup_initializes_registry(self, tmp_path: Path) -> None:
        """Test that setup initializes model registry."""
        config = SuccessiveTrainingConfig(output_dir=str(tmp_path / "successive"))

        trainer = SuccessiveTrainer(config)
        await trainer.setup()

        assert trainer.registry is not None

    @pytest.mark.asyncio
    async def test_setup_initializes_eval_manager(self, tmp_path: Path) -> None:
        """Test that setup initializes evaluation manager."""
        config = SuccessiveTrainingConfig(
            output_dir=str(tmp_path / "successive"),
            promotion_criteria=PromotionCriteria(min_kd_ratio=1.5),
        )

        trainer = SuccessiveTrainer(config)
        await trainer.setup()

        assert trainer.eval_manager is not None
        assert trainer.eval_manager.criteria.min_kd_ratio == 1.5


class TestSuccessiveTrainerContextManager:
    """Tests for SuccessiveTrainer context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_calls_setup_and_cleanup(
        self, tmp_path: Path
    ) -> None:
        """Test context manager calls setup and cleanup."""
        config = SuccessiveTrainingConfig(output_dir=str(tmp_path / "successive"))

        async with SuccessiveTrainer(config) as trainer:
            assert trainer.registry is not None
            assert trainer.eval_manager is not None


class TestSuccessiveTrainerCallbacks:
    """Tests for SuccessiveTrainer callbacks."""

    @pytest.mark.asyncio
    async def test_register_callback(self, tmp_path: Path) -> None:
        """Test callback registration."""
        config = SuccessiveTrainingConfig(output_dir=str(tmp_path / "successive"))

        trainer = SuccessiveTrainer(config)
        await trainer.setup()

        callback = MagicMock()
        trainer.register_callback(callback)

        assert callback in trainer._callbacks


class TestSuccessiveTrainerStop:
    """Tests for SuccessiveTrainer.stop method."""

    @pytest.mark.asyncio
    async def test_stop_sets_is_running_false(self, tmp_path: Path) -> None:
        """Test that stop sets is_running to False."""
        config = SuccessiveTrainingConfig(output_dir=str(tmp_path / "successive"))

        trainer = SuccessiveTrainer(config)
        await trainer.setup()
        trainer.is_running = True

        trainer.stop()

        assert trainer.is_running is False


class TestSuccessiveTrainerHelperMethods:
    """Tests for SuccessiveTrainer helper methods."""

    @pytest.mark.asyncio
    async def test_get_best_model_id_empty(self, tmp_path: Path) -> None:
        """Test get_best_model_id returns None when no results."""
        config = SuccessiveTrainingConfig(output_dir=str(tmp_path / "successive"))

        trainer = SuccessiveTrainer(config)

        assert trainer.get_best_model_id() is None

    @pytest.mark.asyncio
    async def test_get_best_model_id_with_results(self, tmp_path: Path) -> None:
        """Test get_best_model_id returns correct model."""
        config = SuccessiveTrainingConfig(output_dir=str(tmp_path / "successive"))

        trainer = SuccessiveTrainer(config)

        # Create mock results
        mock_metadata = MagicMock(spec=ModelMetadata)
        mock_metadata.opponent_model_id = None

        mock_eval_1 = MagicMock(spec=EvaluationResult)
        mock_eval_1.kd_ratio = 1.2

        mock_eval_2 = MagicMock(spec=EvaluationResult)
        mock_eval_2.kd_ratio = 1.8  # Best

        result_1 = GenerationResult(
            generation=0,
            model_id="ppo_gen_000",
            model_metadata=mock_metadata,
            final_evaluation=mock_eval_1,
            comparison=None,
            timesteps_trained=100_000,
            was_promoted=True,
            promotion_reason="Promoted",
        )

        result_2 = GenerationResult(
            generation=1,
            model_id="ppo_gen_001",
            model_metadata=mock_metadata,
            final_evaluation=mock_eval_2,
            comparison=None,
            timesteps_trained=100_000,
            was_promoted=True,
            promotion_reason="Promoted",
        )

        trainer.generation_results = [result_1, result_2]

        assert trainer.get_best_model_id() == "ppo_gen_001"

    @pytest.mark.asyncio
    async def test_get_generation_lineage(self, tmp_path: Path) -> None:
        """Test get_generation_lineage returns correct pairs."""
        config = SuccessiveTrainingConfig(output_dir=str(tmp_path / "successive"))

        trainer = SuccessiveTrainer(config)

        # Create mock results
        mock_metadata_0 = MagicMock(spec=ModelMetadata)
        mock_metadata_0.opponent_model_id = None

        mock_metadata_1 = MagicMock(spec=ModelMetadata)
        mock_metadata_1.opponent_model_id = "ppo_gen_000"

        mock_eval = MagicMock(spec=EvaluationResult)
        mock_eval.kd_ratio = 1.5

        result_0 = GenerationResult(
            generation=0,
            model_id="ppo_gen_000",
            model_metadata=mock_metadata_0,
            final_evaluation=mock_eval,
            comparison=None,
            timesteps_trained=100_000,
            was_promoted=True,
            promotion_reason="Promoted",
        )

        result_1 = GenerationResult(
            generation=1,
            model_id="ppo_gen_001",
            model_metadata=mock_metadata_1,
            final_evaluation=mock_eval,
            comparison=None,
            timesteps_trained=100_000,
            was_promoted=True,
            promotion_reason="Promoted",
        )

        trainer.generation_results = [result_0, result_1]

        lineage = trainer.get_generation_lineage()

        assert lineage == [
            ("ppo_gen_000", None),
            ("ppo_gen_001", "ppo_gen_000"),
        ]


class TestSuccessiveTrainerTrainValidation:
    """Tests for SuccessiveTrainer.train validation."""

    @pytest.mark.asyncio
    async def test_train_requires_setup(self, tmp_path: Path) -> None:
        """Test that train raises if setup not called."""
        config = SuccessiveTrainingConfig(output_dir=str(tmp_path / "successive"))

        trainer = SuccessiveTrainer(config)

        with pytest.raises(RuntimeError, match="Must call setup"):
            await trainer.train()
