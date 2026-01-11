"""Unit tests for PPO configuration system.

Tests cover:
- Default configuration values
- Field validation with appropriate constraints
- YAML serialization and deserialization roundtrip
- Invalid value rejection
- Edge cases and boundary conditions
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from bot.config import (
    LoggingConfig,
    NetworkConfig,
    PPOConfig,
    PPOCoreConfig,
    TrainingConfig,
)


class TestPPOCoreConfig:
    """Tests for PPOCoreConfig validation."""

    def test_default_values(self) -> None:
        """Test that default config has expected values."""
        config = PPOCoreConfig()
        assert config.learning_rate == 3e-4
        assert config.clip_range == 0.2
        assert config.clip_range_vf is None
        assert config.n_epochs == 10
        assert config.batch_size == 64
        assert config.n_steps == 2048
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.ent_coef == 0.01
        assert config.vf_coef == 0.5
        assert config.max_grad_norm == 0.5

    def test_learning_rate_must_be_positive(self) -> None:
        """Test that learning_rate must be positive."""
        with pytest.raises(ValidationError):
            PPOCoreConfig(learning_rate=-0.001)
        with pytest.raises(ValidationError):
            PPOCoreConfig(learning_rate=0)

    def test_clip_range_bounds(self) -> None:
        """Test clip_range must be in (0, 1]."""
        with pytest.raises(ValidationError):
            PPOCoreConfig(clip_range=-0.1)
        with pytest.raises(ValidationError):
            PPOCoreConfig(clip_range=0)
        with pytest.raises(ValidationError):
            PPOCoreConfig(clip_range=1.5)
        # Edge case: exactly 1.0 should be valid
        config = PPOCoreConfig(clip_range=1.0)
        assert config.clip_range == 1.0

    def test_clip_range_vf_optional(self) -> None:
        """Test clip_range_vf can be None or non-negative."""
        config = PPOCoreConfig(clip_range_vf=None)
        assert config.clip_range_vf is None
        config = PPOCoreConfig(clip_range_vf=0.0)
        assert config.clip_range_vf == 0.0
        config = PPOCoreConfig(clip_range_vf=0.5)
        assert config.clip_range_vf == 0.5
        with pytest.raises(ValidationError):
            PPOCoreConfig(clip_range_vf=-0.1)

    def test_n_epochs_must_be_positive(self) -> None:
        """Test n_epochs must be at least 1."""
        with pytest.raises(ValidationError):
            PPOCoreConfig(n_epochs=0)
        config = PPOCoreConfig(n_epochs=1)
        assert config.n_epochs == 1

    def test_gamma_bounds(self) -> None:
        """Test gamma must be in [0, 1]."""
        with pytest.raises(ValidationError):
            PPOCoreConfig(gamma=-0.1)
        with pytest.raises(ValidationError):
            PPOCoreConfig(gamma=1.5)
        # Boundary values should work
        config = PPOCoreConfig(gamma=0.0)
        assert config.gamma == 0.0
        config = PPOCoreConfig(gamma=1.0)
        assert config.gamma == 1.0

    def test_gae_lambda_bounds(self) -> None:
        """Test gae_lambda must be in [0, 1]."""
        with pytest.raises(ValidationError):
            PPOCoreConfig(gae_lambda=-0.1)
        with pytest.raises(ValidationError):
            PPOCoreConfig(gae_lambda=1.5)
        config = PPOCoreConfig(gae_lambda=0.0)
        assert config.gae_lambda == 0.0
        config = PPOCoreConfig(gae_lambda=1.0)
        assert config.gae_lambda == 1.0

    def test_coefficients_must_be_non_negative(self) -> None:
        """Test that loss coefficients must be >= 0."""
        with pytest.raises(ValidationError):
            PPOCoreConfig(ent_coef=-0.01)
        with pytest.raises(ValidationError):
            PPOCoreConfig(vf_coef=-0.5)
        with pytest.raises(ValidationError):
            PPOCoreConfig(max_grad_norm=-0.1)
        # Zero should be valid
        config = PPOCoreConfig(ent_coef=0.0, vf_coef=0.0, max_grad_norm=0.0)
        assert config.ent_coef == 0.0
        assert config.vf_coef == 0.0
        assert config.max_grad_norm == 0.0

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable after creation."""
        config = PPOCoreConfig()
        with pytest.raises(ValidationError):
            config.learning_rate = 0.001  # type: ignore[misc]


class TestNetworkConfig:
    """Tests for NetworkConfig validation."""

    def test_default_values(self) -> None:
        """Test that default config has expected values."""
        config = NetworkConfig()
        assert config.hidden_sizes == [64, 64]
        assert config.activation == "tanh"
        assert config.share_features is True
        assert config.ortho_init is True

    def test_hidden_sizes_must_have_at_least_one_layer(self) -> None:
        """Test hidden_sizes requires at least one element."""
        with pytest.raises(ValidationError):
            NetworkConfig(hidden_sizes=[])
        config = NetworkConfig(hidden_sizes=[32])
        assert config.hidden_sizes == [32]

    def test_activation_must_be_valid(self) -> None:
        """Test activation must be tanh, relu, or elu."""
        for valid_act in ["tanh", "relu", "elu"]:
            config = NetworkConfig(activation=valid_act)  # type: ignore[arg-type]
            assert config.activation == valid_act
        with pytest.raises(ValidationError):
            NetworkConfig(activation="invalid")  # type: ignore[arg-type]

    def test_custom_hidden_sizes(self) -> None:
        """Test custom hidden layer sizes."""
        config = NetworkConfig(hidden_sizes=[128, 64, 32])
        assert config.hidden_sizes == [128, 64, 32]

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable after creation."""
        config = NetworkConfig()
        with pytest.raises(ValidationError):
            config.activation = "relu"  # type: ignore[misc]


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""

    def test_default_values(self) -> None:
        """Test that default config has expected values."""
        config = TrainingConfig()
        assert config.total_timesteps == 1_000_000
        assert config.seed is None
        assert config.device == "auto"
        assert config.normalize_advantage is True
        assert config.target_kl is None

    def test_total_timesteps_must_be_positive(self) -> None:
        """Test total_timesteps must be at least 1."""
        with pytest.raises(ValidationError):
            TrainingConfig(total_timesteps=0)
        config = TrainingConfig(total_timesteps=1)
        assert config.total_timesteps == 1

    def test_device_must_be_valid(self) -> None:
        """Test device must be cpu, cuda, or auto."""
        for valid_device in ["cpu", "cuda", "auto"]:
            config = TrainingConfig(device=valid_device)  # type: ignore[arg-type]
            assert config.device == valid_device
        with pytest.raises(ValidationError):
            TrainingConfig(device="gpu")  # type: ignore[arg-type]

    def test_seed_optional(self) -> None:
        """Test seed can be None or an integer."""
        config = TrainingConfig(seed=None)
        assert config.seed is None
        config = TrainingConfig(seed=42)
        assert config.seed == 42

    def test_target_kl_optional_and_positive(self) -> None:
        """Test target_kl must be positive if set."""
        config = TrainingConfig(target_kl=None)
        assert config.target_kl is None
        config = TrainingConfig(target_kl=0.01)
        assert config.target_kl == 0.01
        with pytest.raises(ValidationError):
            TrainingConfig(target_kl=0)
        with pytest.raises(ValidationError):
            TrainingConfig(target_kl=-0.01)

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable after creation."""
        config = TrainingConfig()
        with pytest.raises(ValidationError):
            config.device = "cpu"  # type: ignore[misc]


class TestLoggingConfig:
    """Tests for LoggingConfig validation."""

    def test_default_values(self) -> None:
        """Test that default config has expected values."""
        config = LoggingConfig()
        assert config.log_interval == 10
        assert config.save_interval == 50
        assert config.tensorboard is True
        assert config.log_dir == "logs/"

    def test_intervals_must_be_positive(self) -> None:
        """Test intervals must be at least 1."""
        with pytest.raises(ValidationError):
            LoggingConfig(log_interval=0)
        with pytest.raises(ValidationError):
            LoggingConfig(save_interval=0)
        config = LoggingConfig(log_interval=1, save_interval=1)
        assert config.log_interval == 1
        assert config.save_interval == 1

    def test_custom_log_dir(self) -> None:
        """Test custom log directory."""
        config = LoggingConfig(log_dir="/custom/path/")
        assert config.log_dir == "/custom/path/"

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable after creation."""
        config = LoggingConfig()
        with pytest.raises(ValidationError):
            config.tensorboard = False  # type: ignore[misc]


class TestPPOConfig:
    """Tests for the complete PPOConfig."""

    def test_default_config(self) -> None:
        """Test that default config is valid."""
        config = PPOConfig()
        assert config.core.learning_rate == 3e-4
        assert config.core.clip_range == 0.2
        assert config.network.hidden_sizes == [64, 64]
        assert config.training.total_timesteps == 1_000_000
        assert config.logging.log_interval == 10

    def test_nested_config_customization(self) -> None:
        """Test customizing nested configurations."""
        config = PPOConfig(
            core=PPOCoreConfig(learning_rate=1e-3, n_epochs=5),
            network=NetworkConfig(hidden_sizes=[128, 128]),
            training=TrainingConfig(seed=123),
        )
        assert config.core.learning_rate == 1e-3
        assert config.core.n_epochs == 5
        assert config.network.hidden_sizes == [128, 128]
        assert config.training.seed == 123

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable after creation."""
        config = PPOConfig()
        with pytest.raises(ValidationError):
            config.core = PPOCoreConfig()  # type: ignore[misc]

    def test_yaml_roundtrip(self, tmp_path: Path) -> None:
        """Test save/load from YAML maintains values."""
        config = PPOConfig(
            core=PPOCoreConfig(learning_rate=1e-3, gamma=0.95),
            network=NetworkConfig(hidden_sizes=[128, 64], activation="relu"),
            training=TrainingConfig(seed=42, device="cpu"),
            logging=LoggingConfig(log_dir="custom_logs/"),
        )
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(str(yaml_path))
        loaded = PPOConfig.from_yaml(str(yaml_path))
        assert config == loaded

    def test_yaml_roundtrip_defaults(self, tmp_path: Path) -> None:
        """Test that default config survives roundtrip."""
        config = PPOConfig()
        yaml_path = tmp_path / "default_config.yaml"
        config.to_yaml(str(yaml_path))
        loaded = PPOConfig.from_yaml(str(yaml_path))
        assert config == loaded

    def test_from_yaml_with_partial_config(self, tmp_path: Path) -> None:
        """Test loading YAML with only some values specified."""
        yaml_content = """
core:
  learning_rate: 0.001
network:
  activation: relu
"""
        yaml_path = tmp_path / "partial.yaml"
        yaml_path.write_text(yaml_content)
        config = PPOConfig.from_yaml(str(yaml_path))
        # Specified values
        assert config.core.learning_rate == 0.001
        assert config.network.activation == "relu"
        # Default values
        assert config.core.clip_range == 0.2
        assert config.network.hidden_sizes == [64, 64]
        assert config.training.total_timesteps == 1_000_000

    def test_from_yaml_empty_file(self, tmp_path: Path) -> None:
        """Test loading empty YAML file returns defaults."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        config = PPOConfig.from_yaml(str(yaml_path))
        assert config == PPOConfig()

    def test_from_yaml_file_not_found(self) -> None:
        """Test loading non-existent YAML raises error."""
        with pytest.raises(FileNotFoundError):
            PPOConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_invalid_values(self, tmp_path: Path) -> None:
        """Test that invalid values in YAML raise ValidationError."""
        yaml_content = """
core:
  learning_rate: -0.001
"""
        yaml_path = tmp_path / "invalid.yaml"
        yaml_path.write_text(yaml_content)
        with pytest.raises(ValidationError):
            PPOConfig.from_yaml(str(yaml_path))

    def test_model_dump(self) -> None:
        """Test that config can be dumped to dict."""
        config = PPOConfig()
        data = config.model_dump()
        assert isinstance(data, dict)
        assert "core" in data
        assert "network" in data
        assert "training" in data
        assert "logging" in data
        assert data["core"]["learning_rate"] == 3e-4


class TestDefaultYAMLFiles:
    """Tests for the bundled default YAML configuration files."""

    @pytest.fixture
    def defaults_dir(self) -> Path:
        """Get the path to the defaults directory."""
        return (
            Path(__file__).parent.parent.parent / "src" / "bot" / "config" / "defaults"
        )

    def test_ppo_default_yaml_loads(self, defaults_dir: Path) -> None:
        """Test that ppo_default.yaml loads successfully."""
        config_path = defaults_dir / "ppo_default.yaml"
        if config_path.exists():
            config = PPOConfig.from_yaml(str(config_path))
            assert config.core.learning_rate == 3e-4
            assert config.core.clip_range == 0.2

    def test_ppo_fast_yaml_loads(self, defaults_dir: Path) -> None:
        """Test that ppo_fast.yaml loads successfully."""
        config_path = defaults_dir / "ppo_fast.yaml"
        if config_path.exists():
            config = PPOConfig.from_yaml(str(config_path))
            # Fast config should have smaller values for quick experimentation
            assert config.core.n_steps < 2048
            assert config.training.total_timesteps < 1_000_000
