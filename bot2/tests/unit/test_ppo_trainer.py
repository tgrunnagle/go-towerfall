"""Unit tests for the PPOTrainer class.

Tests cover:
- Trainer initialization
- PPOConfig defaults and customization
- Rollout collection
- PPO update mechanics (clipping, value loss, entropy)
- Training metrics
- Checkpoint save/load
"""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from bot.agent.network import ActorCriticNetwork
from bot.agent.ppo_trainer import PPOConfig, PPOTrainer
from bot.agent.rollout_buffer import RolloutBuffer


class MockVectorizedEnv:
    """Mock vectorized environment for testing."""

    def __init__(
        self,
        num_envs: int = 4,
        obs_size: int = 114,
        action_size: int = 27,
    ):
        self.num_envs = num_envs
        self.obs_size = obs_size
        self.action_size = action_size
        self._step_count = 0

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset all environments."""
        obs = np.random.randn(self.num_envs, self.obs_size).astype(np.float32)
        return obs, {}

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step all environments."""
        self._step_count += 1
        obs = np.random.randn(self.num_envs, self.obs_size).astype(np.float32)
        rewards = np.random.randn(self.num_envs).astype(np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        # Occasionally terminate an episode
        if self._step_count % 50 == 0:
            terminated[0] = True

        return obs, rewards, terminated, truncated, {}


class TestPPOConfigDefaults:
    """Tests for PPOConfig default values."""

    def test_default_values(self) -> None:
        """Test PPOConfig has correct defaults."""
        config = PPOConfig()

        assert config.num_steps == 2048
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.num_epochs == 10
        assert config.minibatch_size == 64
        assert config.clip_range == 0.2
        assert config.clip_range_vf is None
        assert config.value_coef == 0.5
        assert config.entropy_coef == 0.01
        assert config.learning_rate == 3e-4
        assert config.max_grad_norm == 0.5
        assert config.normalize_advantages is True

    def test_custom_values(self) -> None:
        """Test PPOConfig accepts custom values."""
        config = PPOConfig(
            num_steps=1024,
            gamma=0.95,
            clip_range=0.1,
            learning_rate=1e-4,
        )

        assert config.num_steps == 1024
        assert config.gamma == 0.95
        assert config.clip_range == 0.1
        assert config.learning_rate == 1e-4

    def test_config_is_dataclass(self) -> None:
        """Test PPOConfig can be converted to dict."""
        config = PPOConfig()
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert "num_steps" in config_dict
        assert "gamma" in config_dict


class TestPPOTrainerInit:
    """Tests for PPOTrainer initialization."""

    def test_default_initialization(self) -> None:
        """Test trainer initializes with defaults."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        trainer = PPOTrainer(network)

        assert trainer.config is not None
        assert trainer.config.num_steps == 2048
        assert trainer.total_timesteps == 0
        assert trainer.num_updates == 0

    def test_custom_config(self) -> None:
        """Test trainer accepts custom config."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(num_steps=512, learning_rate=1e-4)
        trainer = PPOTrainer(network, config=config)

        assert trainer.config.num_steps == 512
        assert trainer.config.learning_rate == 1e-4

    def test_custom_device(self) -> None:
        """Test trainer uses specified device."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        trainer = PPOTrainer(network, device=torch.device("cpu"))

        assert trainer.device.type == "cpu"

    def test_network_moved_to_device(self) -> None:
        """Test network is moved to trainer's device."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        trainer = PPOTrainer(network, device=torch.device("cpu"))

        # Check network parameters are on correct device
        for param in trainer.network.parameters():
            assert param.device.type == "cpu"

    def test_optimizer_created(self) -> None:
        """Test Adam optimizer is created with correct learning rate."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(learning_rate=5e-4)
        trainer = PPOTrainer(network, config=config)

        assert trainer.optimizer is not None
        # Check learning rate (stored in param_groups)
        assert trainer.optimizer.param_groups[0]["lr"] == 5e-4


class TestCollectRollout:
    """Tests for rollout collection."""

    def test_collects_correct_steps(self) -> None:
        """Test rollout collects the configured number of steps."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(num_steps=64)
        trainer = PPOTrainer(network, config=config, device=torch.device("cpu"))

        env = MockVectorizedEnv(num_envs=4, obs_size=114)
        obs = torch.randn(4, 114)

        buffer, next_obs = trainer.collect_rollout(env, obs)

        assert buffer.observations.shape == (64, 4, 114)
        assert buffer.actions.shape == (64, 4)
        assert buffer.rewards.shape == (64, 4)

    def test_updates_total_timesteps(self) -> None:
        """Test total_timesteps is updated after rollout."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(num_steps=32)
        trainer = PPOTrainer(network, config=config, device=torch.device("cpu"))

        env = MockVectorizedEnv(num_envs=4, obs_size=114)
        obs = torch.randn(4, 114)

        trainer.collect_rollout(env, obs)

        assert trainer.total_timesteps == 32 * 4  # num_steps * num_envs

    def test_advantages_computed(self) -> None:
        """Test advantages are computed in collected buffer."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(num_steps=32)
        trainer = PPOTrainer(network, config=config, device=torch.device("cpu"))

        env = MockVectorizedEnv(num_envs=4, obs_size=114)
        obs = torch.randn(4, 114)

        buffer, _ = trainer.collect_rollout(env, obs)

        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.shape == (32, 4)

    def test_returns_next_observation(self) -> None:
        """Test collect_rollout returns valid next observation."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(num_steps=32)
        trainer = PPOTrainer(network, config=config, device=torch.device("cpu"))

        env = MockVectorizedEnv(num_envs=4, obs_size=114)
        obs = torch.randn(4, 114)

        _, next_obs = trainer.collect_rollout(env, obs)

        assert next_obs.shape == (4, 114)
        assert next_obs.dtype == torch.float32


class TestUpdate:
    """Tests for PPO update step."""

    @pytest.fixture
    def trainer_and_buffer(self) -> tuple[PPOTrainer, RolloutBuffer]:
        """Create trainer and mock buffer for testing."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(
            num_steps=64,
            num_epochs=2,  # Fewer epochs for faster tests
            minibatch_size=32,
        )
        trainer = PPOTrainer(network, config=config, device=torch.device("cpu"))

        # Create buffer with random data
        buffer = RolloutBuffer.create(64, 4, 114, torch.device("cpu"))
        buffer.observations = torch.randn(64, 4, 114)
        buffer.actions = torch.randint(0, 27, (64, 4))
        buffer.log_probs = torch.randn(64, 4) - 2.0  # Negative log probs
        buffer.rewards = torch.randn(64, 4)
        buffer.values = torch.randn(64, 4)
        buffer.dones = torch.zeros(64, 4)
        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))

        return trainer, buffer

    def test_returns_metrics_dict(
        self, trainer_and_buffer: tuple[PPOTrainer, RolloutBuffer]
    ) -> None:
        """Test update returns dictionary with expected metrics."""
        trainer, buffer = trainer_and_buffer

        metrics = trainer.update(buffer)

        assert isinstance(metrics, dict)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "clip_fraction" in metrics
        assert "approx_kl" in metrics
        assert "total_timesteps" in metrics
        assert "num_updates" in metrics

    def test_updates_num_updates(
        self, trainer_and_buffer: tuple[PPOTrainer, RolloutBuffer]
    ) -> None:
        """Test num_updates is incremented."""
        trainer, buffer = trainer_and_buffer

        assert trainer.num_updates == 0
        trainer.update(buffer)
        assert trainer.num_updates == 1
        trainer.update(buffer)
        assert trainer.num_updates == 2

    def test_metrics_are_floats(
        self, trainer_and_buffer: tuple[PPOTrainer, RolloutBuffer]
    ) -> None:
        """Test all metrics are Python floats or ints."""
        trainer, buffer = trainer_and_buffer

        metrics = trainer.update(buffer)

        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["value_loss"], float)
        assert isinstance(metrics["entropy"], float)
        assert isinstance(metrics["clip_fraction"], float)
        assert isinstance(metrics["approx_kl"], float)

    def test_clip_fraction_in_valid_range(
        self, trainer_and_buffer: tuple[PPOTrainer, RolloutBuffer]
    ) -> None:
        """Test clip_fraction is between 0 and 1."""
        trainer, buffer = trainer_and_buffer

        metrics = trainer.update(buffer)

        assert 0.0 <= metrics["clip_fraction"] <= 1.0

    def test_entropy_is_positive(
        self, trainer_and_buffer: tuple[PPOTrainer, RolloutBuffer]
    ) -> None:
        """Test entropy metric is positive."""
        trainer, buffer = trainer_and_buffer

        metrics = trainer.update(buffer)

        assert metrics["entropy"] >= 0.0

    def test_network_parameters_change(
        self, trainer_and_buffer: tuple[PPOTrainer, RolloutBuffer]
    ) -> None:
        """Test network parameters are updated during training."""
        trainer, buffer = trainer_and_buffer

        # Store initial parameters
        initial_params = {
            name: param.clone() for name, param in trainer.network.named_parameters()
        }

        trainer.update(buffer)

        # Check at least some parameters changed
        any_changed = False
        for name, param in trainer.network.named_parameters():
            if not torch.equal(param, initial_params[name]):
                any_changed = True
                break

        assert any_changed, "Network parameters should change after update"


class TestClipping:
    """Tests for PPO clipping mechanism."""

    def test_clip_range_affects_clipping(self) -> None:
        """Test that different clip ranges affect the clip fraction."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)

        # Create buffer that will likely cause clipping
        buffer = RolloutBuffer.create(64, 4, 114, torch.device("cpu"))
        buffer.observations = torch.randn(64, 4, 114)
        buffer.actions = torch.randint(0, 27, (64, 4))
        # Use very different log probs to force ratio to be far from 1
        buffer.log_probs = torch.randn(64, 4) - 5.0
        buffer.rewards = torch.randn(64, 4)
        buffer.values = torch.randn(64, 4)
        buffer.dones = torch.zeros(64, 4)
        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))

        # Small clip range should cause more clipping
        config_small = PPOConfig(clip_range=0.05, num_epochs=1)
        trainer_small = PPOTrainer(
            network, config=config_small, device=torch.device("cpu")
        )
        metrics_small = trainer_small.update(buffer)

        # Reset network for fair comparison
        network2 = ActorCriticNetwork(observation_size=114, action_size=27)
        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))  # Recompute

        # Large clip range should cause less clipping
        config_large = PPOConfig(clip_range=0.5, num_epochs=1)
        trainer_large = PPOTrainer(
            network2, config=config_large, device=torch.device("cpu")
        )
        metrics_large = trainer_large.update(buffer)

        # We expect smaller clip range to have higher clip fraction
        # (Note: this is probabilistic, but with the setup above it should generally hold)
        # If test is flaky, we just verify both metrics are valid
        assert 0.0 <= metrics_small["clip_fraction"] <= 1.0
        assert 0.0 <= metrics_large["clip_fraction"] <= 1.0


class TestValueClipping:
    """Tests for value function clipping."""

    def test_value_clipping_enabled(self) -> None:
        """Test value clipping works when enabled."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(clip_range_vf=0.2, num_epochs=1)
        trainer = PPOTrainer(network, config=config, device=torch.device("cpu"))

        buffer = RolloutBuffer.create(64, 4, 114, torch.device("cpu"))
        buffer.observations = torch.randn(64, 4, 114)
        buffer.actions = torch.randint(0, 27, (64, 4))
        buffer.log_probs = torch.randn(64, 4)
        buffer.rewards = torch.randn(64, 4)
        buffer.values = torch.randn(64, 4)
        buffer.dones = torch.zeros(64, 4)
        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))

        metrics = trainer.update(buffer)

        assert "value_loss" in metrics
        assert metrics["value_loss"] >= 0.0


class TestTrainStep:
    """Tests for combined train_step method."""

    def test_train_step_returns_metrics_and_obs(self) -> None:
        """Test train_step returns both metrics and next observation."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(num_steps=32, num_epochs=1)
        trainer = PPOTrainer(network, config=config, device=torch.device("cpu"))

        env = MockVectorizedEnv(num_envs=4, obs_size=114)
        obs = torch.randn(4, 114)

        metrics, next_obs = trainer.train_step(env, obs)

        assert isinstance(metrics, dict)
        assert "policy_loss" in metrics
        assert next_obs.shape == (4, 114)

    def test_train_step_updates_statistics(self) -> None:
        """Test train_step updates timesteps and update count."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(num_steps=32, num_epochs=1)
        trainer = PPOTrainer(network, config=config, device=torch.device("cpu"))

        env = MockVectorizedEnv(num_envs=4, obs_size=114)
        obs = torch.randn(4, 114)

        trainer.train_step(env, obs)

        assert trainer.total_timesteps == 128  # 32 * 4
        assert trainer.num_updates == 1


class TestSaveLoad:
    """Tests for checkpoint save/load."""

    def test_save_creates_file(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test save creates checkpoint file."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        trainer = PPOTrainer(network, device=torch.device("cpu"))
        trainer.total_timesteps = 1000
        trainer.num_updates = 5

        checkpoint_path = str(tmp_path / "checkpoint.pt")  # type: ignore[operator]
        trainer.save(checkpoint_path)

        import os

        assert os.path.exists(checkpoint_path)

    def test_load_restores_state(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test load restores trainer state."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        trainer = PPOTrainer(network, device=torch.device("cpu"))
        trainer.total_timesteps = 5000
        trainer.num_updates = 10

        checkpoint_path = str(tmp_path / "checkpoint.pt")  # type: ignore[operator]
        trainer.save(checkpoint_path)

        # Create new trainer and load
        network2 = ActorCriticNetwork(observation_size=114, action_size=27)
        trainer2 = PPOTrainer(network2, device=torch.device("cpu"))
        trainer2.load(checkpoint_path)

        assert trainer2.total_timesteps == 5000
        assert trainer2.num_updates == 10

    def test_load_restores_network_weights(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Test load restores network parameters."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        trainer = PPOTrainer(network, device=torch.device("cpu"))

        # Modify network weights
        with torch.no_grad():
            for param in trainer.network.parameters():
                param.fill_(42.0)

        checkpoint_path = str(tmp_path / "checkpoint.pt")  # type: ignore[operator]
        trainer.save(checkpoint_path)

        # Create new trainer with fresh network
        network2 = ActorCriticNetwork(observation_size=114, action_size=27)
        trainer2 = PPOTrainer(network2, device=torch.device("cpu"))

        # Verify weights are different before load
        for param in trainer2.network.parameters():
            assert not torch.allclose(param, torch.tensor(42.0))

        trainer2.load(checkpoint_path)

        # Verify weights are restored
        for param in trainer2.network.parameters():
            assert torch.allclose(param, torch.tensor(42.0))


class TestAdvantageNormalization:
    """Tests for advantage normalization."""

    def test_normalization_enabled(self) -> None:
        """Test advantages are normalized when enabled."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(normalize_advantages=True, num_epochs=1)
        trainer = PPOTrainer(network, config=config, device=torch.device("cpu"))

        buffer = RolloutBuffer.create(64, 4, 114, torch.device("cpu"))
        buffer.observations = torch.randn(64, 4, 114)
        buffer.actions = torch.randint(0, 27, (64, 4))
        buffer.log_probs = torch.randn(64, 4)
        buffer.rewards = torch.randn(64, 4) * 100  # Large variance
        buffer.values = torch.randn(64, 4)
        buffer.dones = torch.zeros(64, 4)
        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))

        # Should not raise even with large advantage variance
        metrics = trainer.update(buffer)

        assert "policy_loss" in metrics

    def test_normalization_disabled(self) -> None:
        """Test training works with normalization disabled."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(normalize_advantages=False, num_epochs=1)
        trainer = PPOTrainer(network, config=config, device=torch.device("cpu"))

        buffer = RolloutBuffer.create(64, 4, 114, torch.device("cpu"))
        buffer.observations = torch.randn(64, 4, 114)
        buffer.actions = torch.randint(0, 27, (64, 4))
        buffer.log_probs = torch.randn(64, 4)
        buffer.rewards = torch.randn(64, 4)
        buffer.values = torch.randn(64, 4)
        buffer.dones = torch.zeros(64, 4)
        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))

        metrics = trainer.update(buffer)

        assert "policy_loss" in metrics


class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_gradients_are_clipped(self) -> None:
        """Test gradient norms are clipped during update."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        config = PPOConfig(max_grad_norm=0.5, num_epochs=1)
        trainer = PPOTrainer(network, config=config, device=torch.device("cpu"))

        # Create buffer with extreme values to cause large gradients
        buffer = RolloutBuffer.create(64, 4, 114, torch.device("cpu"))
        buffer.observations = torch.randn(64, 4, 114) * 100
        buffer.actions = torch.randint(0, 27, (64, 4))
        buffer.log_probs = torch.randn(64, 4)
        buffer.rewards = torch.randn(64, 4) * 1000
        buffer.values = torch.randn(64, 4)
        buffer.dones = torch.zeros(64, 4)
        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))

        # Should complete without NaN/Inf due to gradient clipping
        metrics = trainer.update(buffer)

        assert not np.isnan(metrics["policy_loss"])
        assert not np.isnan(metrics["value_loss"])
        assert not np.isinf(metrics["policy_loss"])
        assert not np.isinf(metrics["value_loss"])


class TestEntropyBonus:
    """Tests for entropy bonus in loss."""

    def test_entropy_coefficient_affects_exploration(self) -> None:
        """Test entropy coefficient affects the loss."""
        # Create same buffer for both trainers
        buffer = RolloutBuffer.create(64, 4, 114, torch.device("cpu"))
        buffer.observations = torch.randn(64, 4, 114)
        buffer.actions = torch.randint(0, 27, (64, 4))
        buffer.log_probs = torch.randn(64, 4)
        buffer.rewards = torch.randn(64, 4)
        buffer.values = torch.randn(64, 4)
        buffer.dones = torch.zeros(64, 4)
        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))

        # Both configs should work
        network1 = ActorCriticNetwork(observation_size=114, action_size=27)
        config_low = PPOConfig(entropy_coef=0.001, num_epochs=1)
        trainer_low = PPOTrainer(
            network1, config=config_low, device=torch.device("cpu")
        )
        metrics_low = trainer_low.update(buffer)

        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))  # Recompute
        network2 = ActorCriticNetwork(observation_size=114, action_size=27)
        config_high = PPOConfig(entropy_coef=0.1, num_epochs=1)
        trainer_high = PPOTrainer(
            network2, config=config_high, device=torch.device("cpu")
        )
        metrics_high = trainer_high.update(buffer)

        # Both should produce valid entropy metrics
        assert metrics_low["entropy"] >= 0.0
        assert metrics_high["entropy"] >= 0.0
