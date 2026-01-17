"""Unit tests for OrchestratorConfig.

Tests cover:
- Default values
- Custom values
- Validation
- Properties
- Serialization (to_dict, from_dict, to_yaml, from_yaml)
- MetricsLoggerConfig
"""

from pathlib import Path

import pytest

from bot.agent.ppo_trainer import PPOConfig
from bot.training.orchestrator_config import MetricsLoggerConfig, OrchestratorConfig
from bot.training.server_manager import TrainingGameConfig


class TestOrchestratorConfigDefaults:
    """Tests for OrchestratorConfig default values."""

    def test_default_values(self) -> None:
        """Test OrchestratorConfig has correct defaults."""
        config = OrchestratorConfig()

        assert config.num_envs == 4
        assert config.game_server_url == "http://localhost:4000"
        assert config.total_timesteps == 1_000_000
        assert config.checkpoint_interval == 10_000
        assert config.checkpoint_dir == "./checkpoints"
        assert config.registry_path == "./model_registry"
        assert config.opponent_model_id is None
        assert config.log_interval == 2048
        assert config.eval_interval == 50_000
        assert config.eval_episodes == 10
        assert config.seed is None

    def test_default_game_config(self) -> None:
        """Test default game_config is TrainingGameConfig."""
        config = OrchestratorConfig()

        assert isinstance(config.game_config, TrainingGameConfig)

    def test_default_ppo_config(self) -> None:
        """Test default ppo_config is PPOConfig."""
        config = OrchestratorConfig()

        assert isinstance(config.ppo_config, PPOConfig)
        assert config.ppo_config.num_steps == 2048

    def test_default_metrics_config(self) -> None:
        """Test default metrics_config is MetricsLoggerConfig with defaults."""
        config = OrchestratorConfig()

        assert isinstance(config.metrics_config, MetricsLoggerConfig)
        assert config.metrics_config.enabled is True
        assert config.metrics_config.log_dir == "./metrics"
        assert config.metrics_config.enable_tensorboard is True
        assert config.metrics_config.enable_file is True
        assert config.metrics_config.file_format == "json"
        assert config.metrics_config.window_size == 100


class TestMetricsLoggerConfig:
    """Tests for MetricsLoggerConfig."""

    def test_default_values(self) -> None:
        """Test MetricsLoggerConfig has correct defaults."""
        config = MetricsLoggerConfig()

        assert config.enabled is True
        assert config.log_dir == "./metrics"
        assert config.enable_tensorboard is True
        assert config.enable_file is True
        assert config.file_format == "json"
        assert config.window_size == 100

    def test_custom_values(self) -> None:
        """Test MetricsLoggerConfig with custom values."""
        config = MetricsLoggerConfig(
            enabled=False,
            log_dir="/custom/logs",
            enable_tensorboard=False,
            enable_file=True,
            file_format="csv",
            window_size=50,
        )

        assert config.enabled is False
        assert config.log_dir == "/custom/logs"
        assert config.enable_tensorboard is False
        assert config.enable_file is True
        assert config.file_format == "csv"
        assert config.window_size == 50

    def test_invalid_window_size_zero(self) -> None:
        """Test that window_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be at least 1"):
            MetricsLoggerConfig(window_size=0)

    def test_invalid_window_size_negative(self) -> None:
        """Test that negative window_size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be at least 1"):
            MetricsLoggerConfig(window_size=-1)

    def test_invalid_file_format(self) -> None:
        """Test that invalid file_format raises ValueError."""
        with pytest.raises(ValueError, match="file_format must be 'json' or 'csv'"):
            MetricsLoggerConfig(file_format="xml")  # type: ignore[arg-type]


class TestOrchestratorConfigCustomValues:
    """Tests for OrchestratorConfig with custom values."""

    def test_custom_num_envs(self) -> None:
        """Test custom num_envs value."""
        config = OrchestratorConfig(num_envs=8)
        assert config.num_envs == 8

    def test_custom_total_timesteps(self) -> None:
        """Test custom total_timesteps value."""
        config = OrchestratorConfig(total_timesteps=500_000)
        assert config.total_timesteps == 500_000

    def test_custom_game_server_url(self) -> None:
        """Test custom game_server_url value."""
        config = OrchestratorConfig(game_server_url="http://192.168.1.100:4000")
        assert config.game_server_url == "http://192.168.1.100:4000"

    def test_custom_ppo_config(self) -> None:
        """Test custom ppo_config value."""
        ppo = PPOConfig(num_steps=1024, learning_rate=1e-4)
        config = OrchestratorConfig(ppo_config=ppo)

        assert config.ppo_config.num_steps == 1024
        assert config.ppo_config.learning_rate == 1e-4

    def test_custom_game_config(self) -> None:
        """Test custom game_config value."""
        game_cfg = TrainingGameConfig(
            room_name="CustomRoom",
            tick_multiplier=20.0,
            max_game_duration_sec=120,
        )
        config = OrchestratorConfig(game_config=game_cfg)

        assert config.game_config.room_name == "CustomRoom"
        assert config.game_config.tick_multiplier == 20.0
        assert config.game_config.max_game_duration_sec == 120

    def test_custom_opponent_model_id(self) -> None:
        """Test custom opponent_model_id value."""
        config = OrchestratorConfig(opponent_model_id="ppo_gen_003")
        assert config.opponent_model_id == "ppo_gen_003"

    def test_custom_seed(self) -> None:
        """Test custom seed value."""
        config = OrchestratorConfig(seed=42)
        assert config.seed == 42

    def test_custom_checkpoint_settings(self) -> None:
        """Test custom checkpoint settings."""
        config = OrchestratorConfig(
            checkpoint_interval=5_000,
            checkpoint_dir="/custom/checkpoints",
        )
        assert config.checkpoint_interval == 5_000
        assert config.checkpoint_dir == "/custom/checkpoints"

    def test_custom_eval_settings(self) -> None:
        """Test custom evaluation settings."""
        config = OrchestratorConfig(
            eval_interval=25_000,
            eval_episodes=20,
        )
        assert config.eval_interval == 25_000
        assert config.eval_episodes == 20

    def test_custom_metrics_config(self) -> None:
        """Test custom metrics_config value."""
        metrics_cfg = MetricsLoggerConfig(
            enabled=True,
            log_dir="/custom/metrics",
            enable_tensorboard=True,
            enable_file=False,
            file_format="csv",
            window_size=200,
        )
        config = OrchestratorConfig(metrics_config=metrics_cfg)

        assert config.metrics_config.enabled is True
        assert config.metrics_config.log_dir == "/custom/metrics"
        assert config.metrics_config.enable_tensorboard is True
        assert config.metrics_config.enable_file is False
        assert config.metrics_config.file_format == "csv"
        assert config.metrics_config.window_size == 200


class TestOrchestratorConfigValidation:
    """Tests for OrchestratorConfig validation."""

    def test_invalid_num_envs_zero(self) -> None:
        """Test that num_envs=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_envs must be at least 1"):
            OrchestratorConfig(num_envs=0)

    def test_invalid_num_envs_negative(self) -> None:
        """Test that negative num_envs raises ValueError."""
        with pytest.raises(ValueError, match="num_envs must be at least 1"):
            OrchestratorConfig(num_envs=-1)

    def test_invalid_total_timesteps_zero(self) -> None:
        """Test that total_timesteps=0 raises ValueError."""
        with pytest.raises(ValueError, match="total_timesteps must be at least 1"):
            OrchestratorConfig(total_timesteps=0)

    def test_invalid_total_timesteps_negative(self) -> None:
        """Test that negative total_timesteps raises ValueError."""
        with pytest.raises(ValueError, match="total_timesteps must be at least 1"):
            OrchestratorConfig(total_timesteps=-1)

    def test_invalid_checkpoint_interval_zero(self) -> None:
        """Test that checkpoint_interval=0 raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_interval must be at least 1"):
            OrchestratorConfig(checkpoint_interval=0)

    def test_invalid_log_interval_zero(self) -> None:
        """Test that log_interval=0 raises ValueError."""
        with pytest.raises(ValueError, match="log_interval must be at least 1"):
            OrchestratorConfig(log_interval=0)

    def test_invalid_eval_interval_zero(self) -> None:
        """Test that eval_interval=0 raises ValueError."""
        with pytest.raises(ValueError, match="eval_interval must be at least 1"):
            OrchestratorConfig(eval_interval=0)

    def test_invalid_eval_episodes_zero(self) -> None:
        """Test that eval_episodes=0 raises ValueError."""
        with pytest.raises(ValueError, match="eval_episodes must be at least 1"):
            OrchestratorConfig(eval_episodes=0)

    def test_valid_single_env(self) -> None:
        """Test that num_envs=1 is valid."""
        config = OrchestratorConfig(num_envs=1)
        assert config.num_envs == 1


class TestOrchestratorConfigProperties:
    """Tests for OrchestratorConfig computed properties."""

    def test_steps_per_rollout_default(self) -> None:
        """Test steps_per_rollout with default config."""
        config = OrchestratorConfig()

        # Default: 4 envs * 2048 steps = 8192
        assert config.steps_per_rollout == 4 * 2048

    def test_steps_per_rollout_custom(self) -> None:
        """Test steps_per_rollout with custom config."""
        ppo = PPOConfig(num_steps=512)
        config = OrchestratorConfig(num_envs=8, ppo_config=ppo)

        # 8 envs * 512 steps = 4096
        assert config.steps_per_rollout == 8 * 512

    def test_steps_per_rollout_single_env(self) -> None:
        """Test steps_per_rollout with single environment."""
        ppo = PPOConfig(num_steps=1024)
        config = OrchestratorConfig(num_envs=1, ppo_config=ppo)

        assert config.steps_per_rollout == 1024


class TestOrchestratorConfigSerialization:
    """Tests for OrchestratorConfig serialization methods."""

    def test_to_dict_default(self) -> None:
        """Test to_dict with default configuration."""
        config = OrchestratorConfig()
        data = config.to_dict()

        assert data["num_envs"] == 4
        assert data["total_timesteps"] == 1_000_000
        assert data["game_server_url"] == "http://localhost:4000"
        assert data["seed"] is None
        assert isinstance(data["ppo_config"], dict)
        assert isinstance(data["game_config"], dict)
        assert isinstance(data["metrics_config"], dict)
        assert data["metrics_config"]["enabled"] is True
        assert data["metrics_config"]["file_format"] == "json"

    def test_to_dict_custom(self) -> None:
        """Test to_dict with custom configuration."""
        ppo = PPOConfig(num_steps=512, learning_rate=1e-4)
        game = TrainingGameConfig(room_name="Test", tick_multiplier=20.0)
        config = OrchestratorConfig(
            num_envs=8,
            total_timesteps=500_000,
            ppo_config=ppo,
            game_config=game,
            seed=42,
        )
        data = config.to_dict()

        assert data["num_envs"] == 8
        assert data["total_timesteps"] == 500_000
        assert data["seed"] == 42
        assert data["ppo_config"]["num_steps"] == 512
        assert data["ppo_config"]["learning_rate"] == 1e-4
        assert data["game_config"]["room_name"] == "Test"
        assert data["game_config"]["tick_multiplier"] == 20.0

    def test_from_dict_default(self) -> None:
        """Test from_dict with minimal data."""
        data = {"num_envs": 8, "total_timesteps": 100_000}
        config = OrchestratorConfig.from_dict(data)

        assert config.num_envs == 8
        assert config.total_timesteps == 100_000
        # Defaults should be applied
        assert isinstance(config.ppo_config, PPOConfig)
        assert isinstance(config.game_config, TrainingGameConfig)
        assert isinstance(config.metrics_config, MetricsLoggerConfig)

    def test_from_dict_nested(self) -> None:
        """Test from_dict with nested config data."""
        data = {
            "num_envs": 4,
            "total_timesteps": 200_000,
            "ppo_config": {
                "num_steps": 1024,
                "learning_rate": 5e-4,
            },
            "game_config": {
                "room_name": "FromDict",
                "tick_multiplier": 15.0,
            },
            "metrics_config": {
                "enabled": False,
                "log_dir": "/metrics/custom",
                "file_format": "csv",
                "window_size": 50,
            },
        }
        config = OrchestratorConfig.from_dict(data)

        assert config.num_envs == 4
        assert config.total_timesteps == 200_000
        assert config.ppo_config.num_steps == 1024
        assert config.ppo_config.learning_rate == 5e-4
        assert config.game_config.room_name == "FromDict"
        assert config.game_config.tick_multiplier == 15.0
        assert config.metrics_config.enabled is False
        assert config.metrics_config.log_dir == "/metrics/custom"
        assert config.metrics_config.file_format == "csv"
        assert config.metrics_config.window_size == 50

    def test_roundtrip_to_dict_from_dict(self) -> None:
        """Test that to_dict -> from_dict preserves all values."""
        original = OrchestratorConfig(
            num_envs=16,
            total_timesteps=750_000,
            ppo_config=PPOConfig(num_steps=4096, gamma=0.98),
            game_config=TrainingGameConfig(room_name="Roundtrip", max_kills=30),
            metrics_config=MetricsLoggerConfig(
                enabled=True,
                log_dir="/custom/metrics",
                file_format="csv",
                window_size=150,
            ),
            seed=123,
            opponent_model_id="ppo_gen_005",
        )

        data = original.to_dict()
        restored = OrchestratorConfig.from_dict(data)

        assert restored.num_envs == original.num_envs
        assert restored.total_timesteps == original.total_timesteps
        assert restored.seed == original.seed
        assert restored.opponent_model_id == original.opponent_model_id
        assert restored.ppo_config.num_steps == original.ppo_config.num_steps
        assert restored.ppo_config.gamma == original.ppo_config.gamma
        assert restored.game_config.room_name == original.game_config.room_name
        assert restored.game_config.max_kills == original.game_config.max_kills
        assert restored.metrics_config.enabled == original.metrics_config.enabled
        assert restored.metrics_config.log_dir == original.metrics_config.log_dir
        assert restored.metrics_config.file_format == original.metrics_config.file_format
        assert restored.metrics_config.window_size == original.metrics_config.window_size


class TestOrchestratorConfigYamlSerialization:
    """Tests for YAML serialization methods."""

    def test_to_yaml_creates_file(self, tmp_path: Path) -> None:
        """Test to_yaml creates a YAML file."""
        config = OrchestratorConfig(num_envs=8)
        yaml_path = tmp_path / "config.yaml"

        config.to_yaml(yaml_path)

        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "num_envs: 8" in content

    def test_to_yaml_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test to_yaml creates parent directories if needed."""
        config = OrchestratorConfig()
        yaml_path = tmp_path / "subdir" / "nested" / "config.yaml"

        config.to_yaml(yaml_path)

        assert yaml_path.exists()

    def test_from_yaml_loads_file(self, tmp_path: Path) -> None:
        """Test from_yaml loads configuration from file."""
        yaml_content = """
num_envs: 12
total_timesteps: 300000
seed: 456
ppo_config:
  num_steps: 2048
  learning_rate: 0.0001
game_config:
  room_name: YamlTest
  tick_multiplier: 5.0
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        config = OrchestratorConfig.from_yaml(yaml_path)

        assert config.num_envs == 12
        assert config.total_timesteps == 300_000
        assert config.seed == 456
        assert config.ppo_config.num_steps == 2048
        assert config.ppo_config.learning_rate == 0.0001
        assert config.game_config.room_name == "YamlTest"
        assert config.game_config.tick_multiplier == 5.0

    def test_from_yaml_file_not_found(self, tmp_path: Path) -> None:
        """Test from_yaml raises FileNotFoundError for missing file."""
        yaml_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            OrchestratorConfig.from_yaml(yaml_path)

    def test_roundtrip_yaml(self, tmp_path: Path) -> None:
        """Test roundtrip through YAML file preserves all values."""
        original = OrchestratorConfig(
            num_envs=6,
            total_timesteps=400_000,
            checkpoint_interval=5000,
            log_interval=1024,
            ppo_config=PPOConfig(
                num_steps=512,
                num_epochs=5,
                clip_range=0.1,
            ),
            game_config=TrainingGameConfig(
                room_name="YamlRoundtrip",
                tick_multiplier=8.0,
                max_game_duration_sec=90,
            ),
            seed=789,
        )

        yaml_path = tmp_path / "roundtrip.yaml"
        original.to_yaml(yaml_path)
        restored = OrchestratorConfig.from_yaml(yaml_path)

        assert restored.num_envs == original.num_envs
        assert restored.total_timesteps == original.total_timesteps
        assert restored.checkpoint_interval == original.checkpoint_interval
        assert restored.log_interval == original.log_interval
        assert restored.seed == original.seed
        assert restored.ppo_config.num_steps == original.ppo_config.num_steps
        assert restored.ppo_config.num_epochs == original.ppo_config.num_epochs
        assert restored.ppo_config.clip_range == original.ppo_config.clip_range
        assert restored.game_config.room_name == original.game_config.room_name
        assert (
            restored.game_config.tick_multiplier == original.game_config.tick_multiplier
        )
        assert (
            restored.game_config.max_game_duration_sec
            == original.game_config.max_game_duration_sec
        )
