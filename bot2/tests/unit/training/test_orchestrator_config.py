"""Unit tests for OrchestratorConfig.

Tests cover:
- Default values
- Custom values
- Validation
- Properties
"""

import pytest

from bot.agent.ppo_trainer import PPOConfig
from bot.training.orchestrator_config import OrchestratorConfig
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
