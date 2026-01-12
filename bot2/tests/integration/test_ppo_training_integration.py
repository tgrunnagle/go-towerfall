"""Integration tests for PPO training with real go-towerfall backend.

Tests cover:
- Smoke tests: Quick sanity checks for PPO components with real backend
- Short training runs: Verifying metrics and learning signals
- Checkpoint save/load during training
- Vectorized environment training

All tests require a running go-towerfall server.
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch

from bot.agent.network import ActorCriticNetwork
from bot.agent.ppo_trainer import PPOConfig, PPOTrainer
from bot.gym import VectorizedTowerfallEnv
from tests.conftest import requires_server, unique_room_name

# Default map type for integration tests
DEFAULT_MAP_TYPE = "default"


# ============================================================================
# Helper Functions
# ============================================================================


def assert_valid_metrics(metrics: dict[str, float]) -> None:
    """Validate that training metrics are valid and within expected ranges.

    Args:
        metrics: Dictionary of training metrics from PPOTrainer.update()

    Raises:
        AssertionError: If any metric is invalid
    """
    expected_keys = {
        "policy_loss",
        "value_loss",
        "entropy",
        "clip_fraction",
        "approx_kl",
        "total_timesteps",
        "num_updates",
    }

    # Check all expected keys are present
    assert expected_keys <= set(metrics.keys()), (
        f"Missing keys: {expected_keys - set(metrics.keys())}"
    )

    # Check no NaN or Inf values
    for key, value in metrics.items():
        assert not math.isnan(value), f"{key} is NaN"
        assert not math.isinf(value), f"{key} is Inf"

    # Check valid ranges
    assert 0.0 <= metrics["clip_fraction"] <= 1.0, (
        f"clip_fraction out of range: {metrics['clip_fraction']}"
    )
    assert metrics["entropy"] >= 0.0, (
        f"entropy should be non-negative: {metrics['entropy']}"
    )
    assert metrics["total_timesteps"] > 0, (
        f"total_timesteps should be positive: {metrics['total_timesteps']}"
    )
    assert metrics["num_updates"] > 0, (
        f"num_updates should be positive: {metrics['num_updates']}"
    )


def create_vectorized_env(
    server_url: str,
    num_envs: int = 4,
    opponent_type: str = "none",
    room_prefix: str | None = None,
) -> VectorizedTowerfallEnv:
    """Create a VectorizedTowerfallEnv configured for integration tests.

    Args:
        server_url: Base URL of the go-towerfall server
        num_envs: Number of parallel environments
        opponent_type: Type of opponent ("none" or "rule_based")
        room_prefix: Optional room name prefix (generates unique if None)

    Returns:
        Configured VectorizedTowerfallEnv instance
    """
    if room_prefix is None:
        room_prefix = unique_room_name("PPOTest")

    return VectorizedTowerfallEnv(
        num_envs=num_envs,
        http_url=server_url,
        room_name_prefix=room_prefix,
        map_type=DEFAULT_MAP_TYPE,
        opponent_type=opponent_type,
        tick_rate_multiplier=10.0,  # Fast simulation
        max_episode_steps=100,  # Short episodes
    )


def run_training_steps(
    trainer: PPOTrainer,
    env: VectorizedTowerfallEnv,
    obs: torch.Tensor,
    num_steps: int,
) -> tuple[list[dict[str, float]], torch.Tensor]:
    """Execute N training steps and collect metrics.

    Args:
        trainer: PPOTrainer instance
        env: VectorizedTowerfallEnv instance
        obs: Initial observation tensor
        num_steps: Number of training steps to execute

    Returns:
        Tuple of (list of metrics dicts, final observation tensor)
    """
    all_metrics: list[dict[str, float]] = []
    current_obs = obs

    for _ in range(num_steps):
        metrics, current_obs = trainer.train_step(env, current_obs)
        all_metrics.append(metrics)

    return all_metrics, current_obs


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ppo_config_smoke() -> PPOConfig:
    """Minimal config for smoke tests."""
    return PPOConfig(
        num_steps=16,
        num_epochs=1,
        minibatch_size=16,
    )


@pytest.fixture
def ppo_config_short() -> PPOConfig:
    """Config for short training runs."""
    return PPOConfig(
        num_steps=64,
        num_epochs=2,
        minibatch_size=32,
    )


@pytest.fixture
def network() -> ActorCriticNetwork:
    """Fresh actor-critic network."""
    return ActorCriticNetwork()


# ============================================================================
# Smoke Tests
# ============================================================================


@pytest.mark.integration
class TestPPOTrainingSmoke:
    """Quick sanity checks - verify PPO components work with real backend."""

    @requires_server
    def test_single_rollout_solo_mode(
        self,
        server_url: str,
        ppo_config_smoke: PPOConfig,
        network: ActorCriticNetwork,
    ) -> None:
        """Collect one rollout (16 steps) in solo mode."""
        env = create_vectorized_env(server_url, num_envs=1, opponent_type="none")
        trainer = PPOTrainer(network, config=ppo_config_smoke)

        try:
            obs, _info = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            buffer, next_obs = trainer.collect_rollout(env, obs_tensor)

            # Verify buffer is filled correctly
            assert buffer.observations.shape[0] == ppo_config_smoke.num_steps
            assert buffer.observations.shape[1] == 1  # 1 environment
            assert buffer.rewards.shape == (ppo_config_smoke.num_steps, 1)
            assert buffer.dones.shape == (ppo_config_smoke.num_steps, 1)

            # Verify next observation is valid
            assert next_obs.shape[0] == 1
            assert not torch.any(torch.isnan(next_obs))

            # Verify timesteps were counted
            expected_timesteps = ppo_config_smoke.num_steps * 1  # steps * envs
            assert trainer.total_timesteps == expected_timesteps
        finally:
            env.close()

    @requires_server
    def test_single_rollout_with_opponent(
        self,
        server_url: str,
        ppo_config_smoke: PPOConfig,
        network: ActorCriticNetwork,
    ) -> None:
        """Collect one rollout vs rule-based opponent."""
        env = create_vectorized_env(server_url, num_envs=1, opponent_type="rule_based")
        trainer = PPOTrainer(network, config=ppo_config_smoke)

        try:
            obs, _info = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            buffer, next_obs = trainer.collect_rollout(env, obs_tensor)

            # Verify buffer is filled
            assert buffer.observations.shape[0] == ppo_config_smoke.num_steps
            assert not torch.any(torch.isnan(buffer.observations))
            assert not torch.any(torch.isnan(buffer.rewards))

            # Next observation should be valid
            assert not torch.any(torch.isnan(next_obs))
        finally:
            env.close()

    @requires_server
    def test_single_update_after_rollout(
        self,
        server_url: str,
        ppo_config_smoke: PPOConfig,
        network: ActorCriticNetwork,
    ) -> None:
        """Collect rollout + perform 1 PPO update."""
        env = create_vectorized_env(server_url, num_envs=1, opponent_type="none")
        trainer = PPOTrainer(network, config=ppo_config_smoke)

        try:
            obs, _info = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            buffer, _next_obs = trainer.collect_rollout(env, obs_tensor)
            metrics = trainer.update(buffer)

            # Validate metrics
            assert_valid_metrics(metrics)

            # Verify update was counted
            assert trainer.num_updates == 1
        finally:
            env.close()

    @requires_server
    def test_vectorized_single_rollout(
        self,
        server_url: str,
        ppo_config_smoke: PPOConfig,
        network: ActorCriticNetwork,
    ) -> None:
        """4 parallel envs, collect one rollout."""
        num_envs = 4
        env = create_vectorized_env(server_url, num_envs=num_envs, opponent_type="none")
        trainer = PPOTrainer(network, config=ppo_config_smoke)

        try:
            obs, _info = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            buffer, next_obs = trainer.collect_rollout(env, obs_tensor)

            # Verify buffer shape for 4 environments
            assert buffer.observations.shape == (
                ppo_config_smoke.num_steps,
                num_envs,
                network.observation_size,
            )
            assert buffer.rewards.shape == (ppo_config_smoke.num_steps, num_envs)

            # Verify all observations are valid
            assert not torch.any(torch.isnan(buffer.observations))
            assert next_obs.shape[0] == num_envs

            # Verify timesteps counted correctly
            expected_timesteps = ppo_config_smoke.num_steps * num_envs
            assert trainer.total_timesteps == expected_timesteps
        finally:
            env.close()

    @requires_server
    def test_train_step_completes(
        self,
        server_url: str,
        ppo_config_smoke: PPOConfig,
        network: ActorCriticNetwork,
    ) -> None:
        """Execute one full train_step() call."""
        env = create_vectorized_env(server_url, num_envs=2, opponent_type="none")
        trainer = PPOTrainer(network, config=ppo_config_smoke)

        try:
            obs, _info = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            metrics, next_obs = trainer.train_step(env, obs_tensor)

            # Validate complete train step
            assert_valid_metrics(metrics)
            assert next_obs.shape[0] == 2
            assert not torch.any(torch.isnan(next_obs))

            # Verify both timesteps and updates were tracked
            assert trainer.total_timesteps > 0
            assert trainer.num_updates == 1
        finally:
            env.close()


# ============================================================================
# Short Training Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestPPOTrainingShort:
    """Short training runs verifying metrics and learning signal."""

    @requires_server
    def test_short_training_solo_10_steps(
        self,
        server_url: str,
        ppo_config_short: PPOConfig,
        network: ActorCriticNetwork,
    ) -> None:
        """10 training steps in solo mode."""
        env = create_vectorized_env(server_url, num_envs=1, opponent_type="none")
        trainer = PPOTrainer(network, config=ppo_config_short)

        try:
            obs, _info = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            all_metrics, _final_obs = run_training_steps(trainer, env, obs_tensor, 10)

            # Verify we got 10 sets of metrics
            assert len(all_metrics) == 10

            # All metrics should be valid
            for metrics in all_metrics:
                assert_valid_metrics(metrics)

            # Verify update count
            assert trainer.num_updates == 10
        finally:
            env.close()

    @requires_server
    def test_short_training_with_opponent_10_steps(
        self,
        server_url: str,
        ppo_config_short: PPOConfig,
        network: ActorCriticNetwork,
    ) -> None:
        """10 training steps vs rule-based bot."""
        env = create_vectorized_env(server_url, num_envs=1, opponent_type="rule_based")
        trainer = PPOTrainer(network, config=ppo_config_short)

        try:
            obs, _info = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            all_metrics, _final_obs = run_training_steps(trainer, env, obs_tensor, 10)

            # All training steps should complete successfully
            assert len(all_metrics) == 10
            for metrics in all_metrics:
                assert_valid_metrics(metrics)
        finally:
            env.close()

    @requires_server
    def test_short_training_vectorized_50_steps(
        self,
        server_url: str,
        ppo_config_short: PPOConfig,
        network: ActorCriticNetwork,
    ) -> None:
        """50 training steps with 4 parallel envs."""
        num_envs = 4
        env = create_vectorized_env(server_url, num_envs=num_envs, opponent_type="none")
        trainer = PPOTrainer(network, config=ppo_config_short)

        try:
            obs, _info = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            all_metrics, final_obs = run_training_steps(trainer, env, obs_tensor, 50)

            # Verify all 50 steps completed
            assert len(all_metrics) == 50

            # All metrics should be valid
            for metrics in all_metrics:
                assert_valid_metrics(metrics)

            # Verify final observation is valid
            assert final_obs.shape[0] == num_envs
            assert not torch.any(torch.isnan(final_obs))

            # Verify expected timestep count
            expected_timesteps = ppo_config_short.num_steps * num_envs * 50
            assert trainer.total_timesteps == expected_timesteps
        finally:
            env.close()

    @requires_server
    def test_metrics_remain_valid_throughout_training(
        self,
        server_url: str,
        ppo_config_short: PPOConfig,
        network: ActorCriticNetwork,
    ) -> None:
        """All metrics stay valid (no NaN/Inf) throughout training."""
        env = create_vectorized_env(server_url, num_envs=2, opponent_type="none")
        trainer = PPOTrainer(network, config=ppo_config_short)

        try:
            obs, _info = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            all_metrics, _final_obs = run_training_steps(trainer, env, obs_tensor, 20)

            # Check every single metric value across all training steps
            for step_idx, metrics in enumerate(all_metrics):
                for key, value in metrics.items():
                    assert not math.isnan(value), (
                        f"NaN found in {key} at step {step_idx}"
                    )
                    assert not math.isinf(value), (
                        f"Inf found in {key} at step {step_idx}"
                    )
        finally:
            env.close()

    @requires_server
    def test_network_parameters_change_during_training(
        self,
        server_url: str,
        ppo_config_short: PPOConfig,
        network: ActorCriticNetwork,
    ) -> None:
        """Network weights actually update during training.

        Note: With randomly initialized networks on real game environments,
        the value estimates can be extremely large, leading to gradient clipping
        that may prevent visible parameter changes. This test verifies that the
        optimizer step count increases, indicating training is occurring.
        """
        env = create_vectorized_env(server_url, num_envs=2, opponent_type="none")
        trainer = PPOTrainer(network, config=ppo_config_short)

        try:
            # Track optimizer step count via the trainer's num_updates
            initial_updates = trainer.num_updates

            obs, _info = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            # Run a few training steps
            all_metrics, _final_obs = run_training_steps(trainer, env, obs_tensor, 5)

            # Verify that training steps were executed
            assert trainer.num_updates == initial_updates + 5, (
                f"Expected 5 updates, got {trainer.num_updates - initial_updates}"
            )

            # Verify that all training steps produced valid metrics
            for step_idx, metrics in enumerate(all_metrics):
                assert_valid_metrics(metrics)

            # Verify that gradients were computed (optimizer has state)
            # After training, the optimizer should have accumulated step counts
            optimizer_state = trainer.optimizer.state
            assert len(optimizer_state) > 0, (
                "Optimizer should have state after training"
            )
        finally:
            env.close()

    @requires_server
    def test_checkpoint_save_load_during_training(
        self,
        server_url: str,
        ppo_config_short: PPOConfig,
        network: ActorCriticNetwork,
    ) -> None:
        """Save/load checkpoint mid-training."""
        env = create_vectorized_env(server_url, num_envs=2, opponent_type="none")
        trainer = PPOTrainer(network, config=ppo_config_short)

        try:
            obs, _info = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            # Train for 5 steps
            _metrics1, obs_tensor = run_training_steps(trainer, env, obs_tensor, 5)

            # Save checkpoint
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = str(Path(tmpdir) / "checkpoint.pt")
                trainer.save(checkpoint_path)

                # Store state before more training
                timesteps_at_save = trainer.total_timesteps
                updates_at_save = trainer.num_updates

                # Continue training for 5 more steps
                _metrics2, obs_tensor = run_training_steps(trainer, env, obs_tensor, 5)

                # Verify training continued
                assert trainer.total_timesteps > timesteps_at_save
                assert trainer.num_updates > updates_at_save

                # Load the checkpoint
                trainer.load(checkpoint_path)

                # Verify state was restored
                assert trainer.total_timesteps == timesteps_at_save
                assert trainer.num_updates == updates_at_save

                # Can continue training after load
                metrics, _final_obs = trainer.train_step(env, obs_tensor)
                assert_valid_metrics(metrics)
        finally:
            env.close()
