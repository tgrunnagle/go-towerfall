"""Unit tests for SuccessiveTrainingConfig and PromotionCriteria.

Tests cover:
- Default values
- Custom values
- Validation
- Serialization (to_dict, from_dict, to_yaml, from_yaml)
- Generation config creation
"""

from pathlib import Path

import pytest

from bot.agent.ppo_trainer import PPOConfig
from bot.training.orchestrator_config import OrchestratorConfig
from bot.training.server_manager import TrainingGameConfig
from bot.training.successive_config import PromotionCriteria, SuccessiveTrainingConfig


class TestPromotionCriteriaDefaults:
    """Tests for PromotionCriteria default values."""

    def test_default_values(self) -> None:
        """Test PromotionCriteria has correct defaults."""
        criteria = PromotionCriteria()

        assert criteria.min_kd_ratio == 1.0
        assert criteria.kd_improvement == 0.1
        assert criteria.min_win_rate is None
        assert criteria.min_eval_episodes == 50
        assert criteria.confidence_threshold == 0.95
        assert criteria.consecutive_passes == 3


class TestPromotionCriteriaCustomValues:
    """Tests for PromotionCriteria with custom values."""

    def test_custom_kd_ratio(self) -> None:
        """Test custom min_kd_ratio value."""
        criteria = PromotionCriteria(min_kd_ratio=1.5)
        assert criteria.min_kd_ratio == 1.5

    def test_custom_kd_improvement(self) -> None:
        """Test custom kd_improvement value."""
        criteria = PromotionCriteria(kd_improvement=0.2)
        assert criteria.kd_improvement == 0.2

    def test_custom_min_win_rate(self) -> None:
        """Test custom min_win_rate value."""
        criteria = PromotionCriteria(min_win_rate=0.55)
        assert criteria.min_win_rate == 0.55

    def test_custom_min_eval_episodes(self) -> None:
        """Test custom min_eval_episodes value."""
        criteria = PromotionCriteria(min_eval_episodes=100)
        assert criteria.min_eval_episodes == 100

    def test_custom_confidence_threshold(self) -> None:
        """Test custom confidence_threshold value."""
        criteria = PromotionCriteria(confidence_threshold=0.99)
        assert criteria.confidence_threshold == 0.99

    def test_custom_consecutive_passes(self) -> None:
        """Test custom consecutive_passes value."""
        criteria = PromotionCriteria(consecutive_passes=5)
        assert criteria.consecutive_passes == 5


class TestPromotionCriteriaValidation:
    """Tests for PromotionCriteria validation."""

    def test_invalid_min_kd_ratio_negative(self) -> None:
        """Test that negative min_kd_ratio raises ValueError."""
        with pytest.raises(ValueError, match="min_kd_ratio must be non-negative"):
            PromotionCriteria(min_kd_ratio=-1.0)

    def test_invalid_kd_improvement_negative(self) -> None:
        """Test that negative kd_improvement raises ValueError."""
        with pytest.raises(ValueError, match="kd_improvement must be non-negative"):
            PromotionCriteria(kd_improvement=-0.1)

    def test_invalid_min_win_rate_below_zero(self) -> None:
        """Test that min_win_rate below 0 raises ValueError."""
        with pytest.raises(ValueError, match="min_win_rate must be between 0 and 1"):
            PromotionCriteria(min_win_rate=-0.1)

    def test_invalid_min_win_rate_above_one(self) -> None:
        """Test that min_win_rate above 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_win_rate must be between 0 and 1"):
            PromotionCriteria(min_win_rate=1.5)

    def test_invalid_min_eval_episodes_zero(self) -> None:
        """Test that min_eval_episodes=0 raises ValueError."""
        with pytest.raises(ValueError, match="min_eval_episodes must be at least 1"):
            PromotionCriteria(min_eval_episodes=0)

    def test_invalid_confidence_threshold_zero(self) -> None:
        """Test that confidence_threshold=0 raises ValueError."""
        with pytest.raises(
            ValueError, match="confidence_threshold must be between 0 and 1"
        ):
            PromotionCriteria(confidence_threshold=0.0)

    def test_invalid_confidence_threshold_one(self) -> None:
        """Test that confidence_threshold=1 raises ValueError."""
        with pytest.raises(
            ValueError, match="confidence_threshold must be between 0 and 1"
        ):
            PromotionCriteria(confidence_threshold=1.0)

    def test_invalid_consecutive_passes_zero(self) -> None:
        """Test that consecutive_passes=0 raises ValueError."""
        with pytest.raises(ValueError, match="consecutive_passes must be at least 1"):
            PromotionCriteria(consecutive_passes=0)


class TestSuccessiveTrainingConfigDefaults:
    """Tests for SuccessiveTrainingConfig default values."""

    def test_default_values(self) -> None:
        """Test SuccessiveTrainingConfig has correct defaults."""
        config = SuccessiveTrainingConfig()

        assert config.max_generations == 10
        assert config.initial_opponent == "rule_based"
        assert config.timesteps_per_generation == 500_000
        assert config.evaluation_interval == 50_000
        assert config.evaluation_episodes == 100
        assert config.max_stagnant_evaluations == 10
        assert config.min_improvement_rate == 0.01
        assert config.output_dir == "./successive_training"
        assert config.base_seed is None

    def test_default_base_config(self) -> None:
        """Test default base_config is OrchestratorConfig."""
        config = SuccessiveTrainingConfig()

        assert isinstance(config.base_config, OrchestratorConfig)

    def test_default_promotion_criteria(self) -> None:
        """Test default promotion_criteria is PromotionCriteria."""
        config = SuccessiveTrainingConfig()

        assert isinstance(config.promotion_criteria, PromotionCriteria)


class TestSuccessiveTrainingConfigCustomValues:
    """Tests for SuccessiveTrainingConfig with custom values."""

    def test_custom_max_generations(self) -> None:
        """Test custom max_generations value."""
        config = SuccessiveTrainingConfig(max_generations=5)
        assert config.max_generations == 5

    def test_custom_initial_opponent_none(self) -> None:
        """Test initial_opponent='none' is valid."""
        config = SuccessiveTrainingConfig(initial_opponent="none")
        assert config.initial_opponent == "none"

    def test_custom_timesteps_per_generation(self) -> None:
        """Test custom timesteps_per_generation value."""
        config = SuccessiveTrainingConfig(timesteps_per_generation=1_000_000)
        assert config.timesteps_per_generation == 1_000_000

    def test_custom_evaluation_settings(self) -> None:
        """Test custom evaluation settings."""
        config = SuccessiveTrainingConfig(
            evaluation_interval=25_000,
            evaluation_episodes=50,
        )
        assert config.evaluation_interval == 25_000
        assert config.evaluation_episodes == 50

    def test_custom_base_config(self) -> None:
        """Test custom base_config value."""
        base = OrchestratorConfig(num_envs=8)
        config = SuccessiveTrainingConfig(base_config=base)
        assert config.base_config.num_envs == 8

    def test_custom_promotion_criteria(self) -> None:
        """Test custom promotion_criteria value."""
        criteria = PromotionCriteria(min_kd_ratio=1.5)
        config = SuccessiveTrainingConfig(promotion_criteria=criteria)
        assert config.promotion_criteria.min_kd_ratio == 1.5

    def test_custom_base_seed(self) -> None:
        """Test custom base_seed value."""
        config = SuccessiveTrainingConfig(base_seed=42)
        assert config.base_seed == 42


class TestSuccessiveTrainingConfigValidation:
    """Tests for SuccessiveTrainingConfig validation."""

    def test_invalid_max_generations_zero(self) -> None:
        """Test that max_generations=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_generations must be at least 1"):
            SuccessiveTrainingConfig(max_generations=0)

    def test_invalid_initial_opponent(self) -> None:
        """Test that invalid initial_opponent raises ValueError."""
        with pytest.raises(
            ValueError, match="initial_opponent must be 'rule_based' or 'none'"
        ):
            SuccessiveTrainingConfig(initial_opponent="invalid")

    def test_invalid_timesteps_per_generation_zero(self) -> None:
        """Test that timesteps_per_generation=0 raises ValueError."""
        with pytest.raises(
            ValueError, match="timesteps_per_generation must be at least 1"
        ):
            SuccessiveTrainingConfig(timesteps_per_generation=0)

    def test_invalid_evaluation_interval_zero(self) -> None:
        """Test that evaluation_interval=0 raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_interval must be at least 1"):
            SuccessiveTrainingConfig(evaluation_interval=0)

    def test_invalid_evaluation_episodes_zero(self) -> None:
        """Test that evaluation_episodes=0 raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_episodes must be at least 1"):
            SuccessiveTrainingConfig(evaluation_episodes=0)

    def test_invalid_max_stagnant_evaluations_zero(self) -> None:
        """Test that max_stagnant_evaluations=0 raises ValueError."""
        with pytest.raises(
            ValueError, match="max_stagnant_evaluations must be at least 1"
        ):
            SuccessiveTrainingConfig(max_stagnant_evaluations=0)

    def test_invalid_min_improvement_rate_negative(self) -> None:
        """Test that negative min_improvement_rate raises ValueError."""
        with pytest.raises(
            ValueError, match="min_improvement_rate must be non-negative"
        ):
            SuccessiveTrainingConfig(min_improvement_rate=-0.01)


class TestSuccessiveTrainingConfigGenerationConfig:
    """Tests for create_generation_config method."""

    def test_generation_config_basic(self) -> None:
        """Test generation config creation with basic settings."""
        config = SuccessiveTrainingConfig(
            base_config=OrchestratorConfig(num_envs=4),
            timesteps_per_generation=100_000,
            output_dir="/test/output",
        )

        gen_config = config.create_generation_config(generation=0, opponent_model_id=None)

        assert gen_config.num_envs == 4
        assert gen_config.total_timesteps == 100_000
        assert gen_config.opponent_model_id is None
        assert "generation_000" in gen_config.checkpoint_dir
        # Use Path for cross-platform comparison
        assert Path(gen_config.registry_path) == Path("/test/output/model_registry")

    def test_generation_config_with_opponent(self) -> None:
        """Test generation config with opponent model."""
        config = SuccessiveTrainingConfig(output_dir="/test/output")

        gen_config = config.create_generation_config(
            generation=1, opponent_model_id="ppo_gen_000"
        )

        assert gen_config.opponent_model_id == "ppo_gen_000"

    def test_generation_config_with_seed(self) -> None:
        """Test generation config with base seed."""
        config = SuccessiveTrainingConfig(base_seed=42)

        gen_config_0 = config.create_generation_config(generation=0, opponent_model_id=None)
        gen_config_1 = config.create_generation_config(generation=1, opponent_model_id=None)

        assert gen_config_0.seed == 42
        assert gen_config_1.seed == 43

    def test_generation_config_no_seed(self) -> None:
        """Test generation config without base seed."""
        config = SuccessiveTrainingConfig(base_seed=None)

        gen_config = config.create_generation_config(generation=0, opponent_model_id=None)

        assert gen_config.seed is None

    def test_generation_config_preserves_ppo_config(self) -> None:
        """Test that PPO config is preserved in generation config."""
        ppo = PPOConfig(num_steps=1024, learning_rate=1e-4)
        config = SuccessiveTrainingConfig(base_config=OrchestratorConfig(ppo_config=ppo))

        gen_config = config.create_generation_config(generation=0, opponent_model_id=None)

        assert gen_config.ppo_config.num_steps == 1024
        assert gen_config.ppo_config.learning_rate == 1e-4

    def test_generation_config_preserves_game_config(self) -> None:
        """Test that game config is preserved in generation config."""
        game = TrainingGameConfig(room_name="Test", tick_multiplier=20.0)
        config = SuccessiveTrainingConfig(
            base_config=OrchestratorConfig(game_config=game)
        )

        gen_config = config.create_generation_config(generation=0, opponent_model_id=None)

        assert gen_config.game_config.room_name == "Test"
        assert gen_config.game_config.tick_multiplier == 20.0


class TestSuccessiveTrainingConfigSerialization:
    """Tests for serialization methods."""

    def test_to_dict_default(self) -> None:
        """Test to_dict with default configuration."""
        config = SuccessiveTrainingConfig()
        data = config.to_dict()

        assert data["max_generations"] == 10
        assert data["initial_opponent"] == "rule_based"
        assert data["timesteps_per_generation"] == 500_000
        assert data["base_seed"] is None
        assert isinstance(data["base_config"], dict)
        assert isinstance(data["promotion_criteria"], dict)

    def test_to_dict_custom(self) -> None:
        """Test to_dict with custom configuration."""
        config = SuccessiveTrainingConfig(
            max_generations=5,
            timesteps_per_generation=200_000,
            promotion_criteria=PromotionCriteria(min_kd_ratio=1.5),
            base_seed=123,
        )
        data = config.to_dict()

        assert data["max_generations"] == 5
        assert data["timesteps_per_generation"] == 200_000
        assert data["promotion_criteria"]["min_kd_ratio"] == 1.5
        assert data["base_seed"] == 123

    def test_from_dict_default(self) -> None:
        """Test from_dict with minimal data."""
        data = {"max_generations": 3, "timesteps_per_generation": 100_000}
        config = SuccessiveTrainingConfig.from_dict(data)

        assert config.max_generations == 3
        assert config.timesteps_per_generation == 100_000
        # Defaults should be applied
        assert isinstance(config.base_config, OrchestratorConfig)
        assert isinstance(config.promotion_criteria, PromotionCriteria)

    def test_from_dict_nested(self) -> None:
        """Test from_dict with nested config data."""
        data = {
            "max_generations": 5,
            "base_config": {"num_envs": 8},
            "promotion_criteria": {"min_kd_ratio": 2.0, "consecutive_passes": 5},
        }
        config = SuccessiveTrainingConfig.from_dict(data)

        assert config.max_generations == 5
        assert config.base_config.num_envs == 8
        assert config.promotion_criteria.min_kd_ratio == 2.0
        assert config.promotion_criteria.consecutive_passes == 5

    def test_roundtrip_to_dict_from_dict(self) -> None:
        """Test that to_dict -> from_dict preserves all values."""
        original = SuccessiveTrainingConfig(
            base_config=OrchestratorConfig(num_envs=8),
            max_generations=7,
            timesteps_per_generation=300_000,
            promotion_criteria=PromotionCriteria(min_kd_ratio=1.3, min_win_rate=0.6),
            base_seed=999,
        )

        data = original.to_dict()
        restored = SuccessiveTrainingConfig.from_dict(data)

        assert restored.max_generations == original.max_generations
        assert (
            restored.timesteps_per_generation == original.timesteps_per_generation
        )
        assert restored.base_seed == original.base_seed
        assert restored.base_config.num_envs == original.base_config.num_envs
        assert (
            restored.promotion_criteria.min_kd_ratio
            == original.promotion_criteria.min_kd_ratio
        )
        assert (
            restored.promotion_criteria.min_win_rate
            == original.promotion_criteria.min_win_rate
        )


class TestSuccessiveTrainingConfigYamlSerialization:
    """Tests for YAML serialization methods."""

    def test_to_yaml_creates_file(self, tmp_path: Path) -> None:
        """Test to_yaml creates a YAML file."""
        config = SuccessiveTrainingConfig(max_generations=5)
        yaml_path = tmp_path / "config.yaml"

        config.to_yaml(yaml_path)

        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "max_generations: 5" in content

    def test_from_yaml_loads_file(self, tmp_path: Path) -> None:
        """Test from_yaml loads configuration from file."""
        yaml_content = """
max_generations: 8
timesteps_per_generation: 250000
base_seed: 42
promotion_criteria:
  min_kd_ratio: 1.2
  consecutive_passes: 4
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        config = SuccessiveTrainingConfig.from_yaml(yaml_path)

        assert config.max_generations == 8
        assert config.timesteps_per_generation == 250_000
        assert config.base_seed == 42
        assert config.promotion_criteria.min_kd_ratio == 1.2
        assert config.promotion_criteria.consecutive_passes == 4

    def test_roundtrip_yaml(self, tmp_path: Path) -> None:
        """Test roundtrip through YAML file preserves all values."""
        original = SuccessiveTrainingConfig(
            max_generations=6,
            timesteps_per_generation=400_000,
            base_config=OrchestratorConfig(num_envs=16),
            promotion_criteria=PromotionCriteria(min_kd_ratio=1.8),
            base_seed=555,
        )

        yaml_path = tmp_path / "roundtrip.yaml"
        original.to_yaml(yaml_path)
        restored = SuccessiveTrainingConfig.from_yaml(yaml_path)

        assert restored.max_generations == original.max_generations
        assert (
            restored.timesteps_per_generation == original.timesteps_per_generation
        )
        assert restored.base_seed == original.base_seed
        assert restored.base_config.num_envs == original.base_config.num_envs
        assert (
            restored.promotion_criteria.min_kd_ratio
            == original.promotion_criteria.min_kd_ratio
        )
