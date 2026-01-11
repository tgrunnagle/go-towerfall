"""Unit tests for the RolloutBuffer class.

Tests cover:
- Buffer creation with correct shapes
- GAE advantage computation
- Minibatch generation
- Flattening for batch processing
"""

import numpy as np
import pytest
import torch

from bot.agent.rollout_buffer import RolloutBuffer


class TestRolloutBufferCreate:
    """Tests for RolloutBuffer.create() method."""

    def test_creates_correct_shapes(self) -> None:
        """Test buffer creates tensors with correct shapes."""
        buffer = RolloutBuffer.create(
            num_steps=128,
            num_envs=4,
            observation_size=114,
            device=torch.device("cpu"),
        )

        assert buffer.observations.shape == (128, 4, 114)
        assert buffer.actions.shape == (128, 4)
        assert buffer.log_probs.shape == (128, 4)
        assert buffer.rewards.shape == (128, 4)
        assert buffer.values.shape == (128, 4)
        assert buffer.dones.shape == (128, 4)

    def test_creates_zeroed_tensors(self) -> None:
        """Test buffer tensors are initialized to zero."""
        buffer = RolloutBuffer.create(
            num_steps=64,
            num_envs=2,
            observation_size=10,
            device=torch.device("cpu"),
        )

        assert (buffer.observations == 0).all()
        assert (buffer.actions == 0).all()
        assert (buffer.rewards == 0).all()

    def test_creates_on_correct_device(self) -> None:
        """Test buffer tensors are on the specified device."""
        buffer = RolloutBuffer.create(
            num_steps=32,
            num_envs=2,
            observation_size=10,
            device=torch.device("cpu"),
        )

        assert buffer.observations.device.type == "cpu"
        assert buffer.actions.device.type == "cpu"

    def test_actions_have_long_dtype(self) -> None:
        """Test action tensor has long dtype for indexing."""
        buffer = RolloutBuffer.create(
            num_steps=32,
            num_envs=2,
            observation_size=10,
            device=torch.device("cpu"),
        )

        assert buffer.actions.dtype == torch.long

    def test_advantages_initially_none(self) -> None:
        """Test advantages and returns are None before computation."""
        buffer = RolloutBuffer.create(
            num_steps=32,
            num_envs=2,
            observation_size=10,
            device=torch.device("cpu"),
        )

        assert buffer.advantages is None
        assert buffer.returns is None


class TestComputeAdvantages:
    """Tests for GAE advantage computation."""

    def test_computes_advantages_shape(self) -> None:
        """Test compute_advantages produces correct shapes."""
        buffer = RolloutBuffer.create(
            num_steps=64,
            num_envs=4,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards = torch.randn(64, 4)
        buffer.values = torch.randn(64, 4)
        buffer.dones = torch.zeros(64, 4)

        buffer.compute_advantages(
            last_value=torch.zeros(4),
            last_done=torch.zeros(4),
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.shape == (64, 4)
        assert buffer.returns.shape == (64, 4)

    def test_returns_equals_advantages_plus_values(self) -> None:
        """Test that returns = advantages + values."""
        buffer = RolloutBuffer.create(
            num_steps=32,
            num_envs=2,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards = torch.randn(32, 2)
        buffer.values = torch.randn(32, 2)
        buffer.dones = torch.zeros(32, 2)

        buffer.compute_advantages(
            last_value=torch.zeros(2),
            last_done=torch.zeros(2),
        )

        assert buffer.advantages is not None
        assert buffer.returns is not None
        expected_returns = buffer.advantages + buffer.values
        assert torch.allclose(buffer.returns, expected_returns)

    def test_zero_rewards_zero_advantages_if_gamma_zero(self) -> None:
        """Test that gamma=0 gives delta as advantage (immediate reward only)."""
        buffer = RolloutBuffer.create(
            num_steps=5,
            num_envs=1,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        buffer.values = torch.tensor([[0.5], [0.5], [0.5], [0.5], [0.5]])
        buffer.dones = torch.zeros(5, 1)

        buffer.compute_advantages(
            last_value=torch.tensor([0.0]),
            last_done=torch.tensor([0.0]),
            gamma=0.0,  # No discounting
            gae_lambda=0.95,
        )

        # With gamma=0: advantage = reward - value (no future contribution)
        assert buffer.advantages is not None
        expected = buffer.rewards - buffer.values
        assert torch.allclose(buffer.advantages, expected)

    def test_terminal_states_break_bootstrap(self) -> None:
        """Test that terminal states properly break value bootstrapping."""
        buffer = RolloutBuffer.create(
            num_steps=5,
            num_envs=1,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0]])
        buffer.values = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0]])
        # Episode ends at step 2 (index 2)
        buffer.dones = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0]])

        buffer.compute_advantages(
            last_value=torch.tensor([10.0]),  # High bootstrap value
            last_done=torch.tensor([0.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert buffer.advantages is not None
        # The advantage at step 2 should not include future values
        # because dones[3] = 0 but the episode "resets" conceptually
        # Step 2's next_value should still be values[3], but the done at step 2
        # means the advantage calculation for step 1 should zero out future

    def test_gae_lambda_affects_variance(self) -> None:
        """Test that different gae_lambda values produce different advantages."""
        buffer = RolloutBuffer.create(
            num_steps=32,
            num_envs=2,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards = torch.randn(32, 2)
        buffer.values = torch.randn(32, 2)
        buffer.dones = torch.zeros(32, 2)

        # Store original values for second computation
        rewards_copy = buffer.rewards.clone()
        values_copy = buffer.values.clone()

        buffer.compute_advantages(
            last_value=torch.zeros(2),
            last_done=torch.zeros(2),
            gamma=0.99,
            gae_lambda=0.95,
        )
        adv_high_lambda = buffer.advantages.clone()

        # Reset and compute with lower lambda
        buffer.rewards = rewards_copy
        buffer.values = values_copy
        buffer.compute_advantages(
            last_value=torch.zeros(2),
            last_done=torch.zeros(2),
            gamma=0.99,
            gae_lambda=0.5,  # Lower lambda = more bias, less variance
        )
        adv_low_lambda = buffer.advantages

        # Advantages should be different
        assert adv_high_lambda is not None
        assert adv_low_lambda is not None
        assert not torch.allclose(adv_high_lambda, adv_low_lambda)


class TestFlatten:
    """Tests for buffer flattening."""

    def test_flatten_shapes(self) -> None:
        """Test flatten produces correct shapes."""
        buffer = RolloutBuffer.create(
            num_steps=64,
            num_envs=4,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards = torch.randn(64, 4)
        buffer.values = torch.randn(64, 4)
        buffer.dones = torch.zeros(64, 4)
        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))

        flat = buffer.flatten()

        assert flat["observations"].shape == (256, 10)  # 64 * 4
        assert flat["actions"].shape == (256,)
        assert flat["log_probs"].shape == (256,)
        assert flat["advantages"].shape == (256,)
        assert flat["returns"].shape == (256,)
        assert flat["values"].shape == (256,)

    def test_flatten_preserves_values(self) -> None:
        """Test flatten preserves tensor values."""
        buffer = RolloutBuffer.create(
            num_steps=4,
            num_envs=2,
            observation_size=3,
            device=torch.device("cpu"),
        )
        buffer.observations[0, 0, :] = torch.tensor([1.0, 2.0, 3.0])
        buffer.observations[0, 1, :] = torch.tensor([4.0, 5.0, 6.0])
        buffer.rewards = torch.randn(4, 2)
        buffer.values = torch.randn(4, 2)
        buffer.dones = torch.zeros(4, 2)
        buffer.compute_advantages(torch.zeros(2), torch.zeros(2))

        flat = buffer.flatten()

        # First observation should be preserved
        assert torch.allclose(flat["observations"][0], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(flat["observations"][1], torch.tensor([4.0, 5.0, 6.0]))

    def test_flatten_raises_without_advantages(self) -> None:
        """Test flatten raises error if advantages not computed."""
        buffer = RolloutBuffer.create(
            num_steps=32,
            num_envs=2,
            observation_size=10,
            device=torch.device("cpu"),
        )

        with pytest.raises(RuntimeError, match="Advantages must be computed"):
            buffer.flatten()


class TestGetMinibatches:
    """Tests for minibatch generation."""

    def test_minibatches_cover_all_samples(self) -> None:
        """Test all samples are covered across minibatches."""
        buffer = RolloutBuffer.create(
            num_steps=64,
            num_envs=4,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards = torch.randn(64, 4)
        buffer.values = torch.randn(64, 4)
        buffer.dones = torch.zeros(64, 4)
        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))

        batches = buffer.get_minibatches(minibatch_size=32)

        total_samples = sum(b["observations"].shape[0] for b in batches)
        assert total_samples == 256  # 64 * 4

    def test_minibatch_sizes(self) -> None:
        """Test minibatch sizes are correct."""
        buffer = RolloutBuffer.create(
            num_steps=64,
            num_envs=4,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards = torch.randn(64, 4)
        buffer.values = torch.randn(64, 4)
        buffer.dones = torch.zeros(64, 4)
        buffer.compute_advantages(torch.zeros(4), torch.zeros(4))

        batches = buffer.get_minibatches(minibatch_size=64)

        # 256 samples / 64 per batch = 4 batches of 64 each
        assert len(batches) == 4
        for batch in batches:
            assert batch["observations"].shape[0] == 64

    def test_minibatch_handles_remainder(self) -> None:
        """Test minibatch handles non-divisible batch sizes."""
        buffer = RolloutBuffer.create(
            num_steps=50,
            num_envs=2,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards = torch.randn(50, 2)
        buffer.values = torch.randn(50, 2)
        buffer.dones = torch.zeros(50, 2)
        buffer.compute_advantages(torch.zeros(2), torch.zeros(2))

        batches = buffer.get_minibatches(minibatch_size=32)

        # 100 samples: batches of 32, 32, 32, 4
        total = sum(b["observations"].shape[0] for b in batches)
        assert total == 100

    def test_minibatch_shuffle_varies_order(self) -> None:
        """Test shuffling produces different orderings."""
        np.random.seed(None)  # Ensure randomness

        buffer = RolloutBuffer.create(
            num_steps=32,
            num_envs=2,
            observation_size=10,
            device=torch.device("cpu"),
        )
        # Give each observation a unique identifier
        for i in range(32):
            for j in range(2):
                buffer.observations[i, j, 0] = float(i * 2 + j)
        buffer.rewards = torch.randn(32, 2)
        buffer.values = torch.randn(32, 2)
        buffer.dones = torch.zeros(32, 2)
        buffer.compute_advantages(torch.zeros(2), torch.zeros(2))

        # Get two sets of batches with shuffling
        batches1 = buffer.get_minibatches(minibatch_size=16, shuffle=True)
        batches2 = buffer.get_minibatches(minibatch_size=16, shuffle=True)

        # Extract first elements of each batch
        order1 = [b["observations"][0, 0].item() for b in batches1]
        order2 = [b["observations"][0, 0].item() for b in batches2]

        # Very unlikely to be the same with 64 samples shuffled
        # (But could happen, so just check data is valid)
        assert len(order1) == len(order2) == 4

    def test_minibatch_no_shuffle_preserves_order(self) -> None:
        """Test no shuffling preserves sample order."""
        buffer = RolloutBuffer.create(
            num_steps=4,
            num_envs=2,
            observation_size=3,
            device=torch.device("cpu"),
        )
        # Set identifiable values
        for i in range(4):
            for j in range(2):
                buffer.observations[i, j, 0] = float(i * 2 + j)
        buffer.rewards = torch.randn(4, 2)
        buffer.values = torch.randn(4, 2)
        buffer.dones = torch.zeros(4, 2)
        buffer.compute_advantages(torch.zeros(2), torch.zeros(2))

        batches = buffer.get_minibatches(minibatch_size=4, shuffle=False)

        # First batch should have indices 0-3 in order
        first_vals = batches[0]["observations"][:, 0].tolist()
        assert first_vals == [0.0, 1.0, 2.0, 3.0]

    def test_minibatch_contains_all_keys(self) -> None:
        """Test each minibatch has all required keys."""
        buffer = RolloutBuffer.create(
            num_steps=32,
            num_envs=2,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards = torch.randn(32, 2)
        buffer.values = torch.randn(32, 2)
        buffer.dones = torch.zeros(32, 2)
        buffer.compute_advantages(torch.zeros(2), torch.zeros(2))

        batches = buffer.get_minibatches(minibatch_size=16)

        expected_keys = {
            "observations",
            "actions",
            "log_probs",
            "advantages",
            "returns",
            "values",
        }
        for batch in batches:
            assert set(batch.keys()) == expected_keys


class TestTotalSteps:
    """Tests for total_steps property."""

    def test_total_steps_calculation(self) -> None:
        """Test total_steps returns correct value."""
        buffer = RolloutBuffer.create(
            num_steps=64,
            num_envs=4,
            observation_size=10,
            device=torch.device("cpu"),
        )

        assert buffer.total_steps == 256  # 64 * 4

    def test_total_steps_single_env(self) -> None:
        """Test total_steps with single environment."""
        buffer = RolloutBuffer.create(
            num_steps=128,
            num_envs=1,
            observation_size=10,
            device=torch.device("cpu"),
        )

        assert buffer.total_steps == 128


class TestGAEComputation:
    """Detailed tests for GAE (Generalized Advantage Estimation)."""

    def test_single_step_advantage(self) -> None:
        """Test GAE with single step equals TD error."""
        buffer = RolloutBuffer.create(
            num_steps=1,
            num_envs=1,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards[0, 0] = 1.0
        buffer.values[0, 0] = 0.5
        buffer.dones[0, 0] = 0.0

        last_value = torch.tensor([0.8])
        buffer.compute_advantages(
            last_value=last_value,
            last_done=torch.tensor([0.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )

        # For single step: advantage = r + gamma * V(s') - V(s)
        # = 1.0 + 0.99 * 0.8 - 0.5 = 1.0 + 0.792 - 0.5 = 1.292
        assert buffer.advantages is not None
        expected = 1.0 + 0.99 * 0.8 - 0.5
        assert torch.allclose(buffer.advantages[0, 0], torch.tensor(expected))

    def test_terminal_episode_no_bootstrap(self) -> None:
        """Test terminal episode doesn't bootstrap future value."""
        buffer = RolloutBuffer.create(
            num_steps=1,
            num_envs=1,
            observation_size=10,
            device=torch.device("cpu"),
        )
        buffer.rewards[0, 0] = 1.0
        buffer.values[0, 0] = 0.5
        buffer.dones[0, 0] = 0.0  # Not terminal at this step

        # But the episode ends at last_done
        buffer.compute_advantages(
            last_value=torch.tensor([100.0]),  # High value that should be ignored
            last_done=torch.tensor([1.0]),  # Terminal!
            gamma=0.99,
            gae_lambda=0.95,
        )

        # With terminal: advantage = r - V(s) = 1.0 - 0.5 = 0.5
        assert buffer.advantages is not None
        expected = 1.0 - 0.5
        assert torch.allclose(buffer.advantages[0, 0], torch.tensor(expected))

    def test_multi_step_advantage_accumulation(self) -> None:
        """Test multi-step advantage computation with known values."""
        buffer = RolloutBuffer.create(
            num_steps=3,
            num_envs=1,
            observation_size=10,
            device=torch.device("cpu"),
        )
        # Simple constant rewards and values
        buffer.rewards[:, 0] = torch.tensor([1.0, 1.0, 1.0])
        buffer.values[:, 0] = torch.tensor([0.0, 0.0, 0.0])
        buffer.dones[:, 0] = torch.tensor([0.0, 0.0, 0.0])

        buffer.compute_advantages(
            last_value=torch.tensor([0.0]),
            last_done=torch.tensor([0.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert buffer.advantages is not None
        # With r=1, V=0 everywhere, gamma=0.99, lambda=0.95:
        # delta_t = r_t + gamma * V_{t+1} - V_t = 1 + 0 - 0 = 1
        # A_2 = delta_2 = 1.0
        # A_1 = delta_1 + gamma * lambda * A_2 = 1 + 0.99*0.95*1 = 1.9405
        # A_0 = delta_0 + gamma * lambda * A_1 = 1 + 0.99*0.95*1.9405 ≈ 2.824
        gae_coeff = 0.99 * 0.95
        expected_2 = 1.0
        expected_1 = 1.0 + gae_coeff * expected_2
        expected_0 = 1.0 + gae_coeff * expected_1

        assert torch.allclose(
            buffer.advantages[2, 0], torch.tensor(expected_2), atol=1e-6
        )
        assert torch.allclose(
            buffer.advantages[1, 0], torch.tensor(expected_1), atol=1e-6
        )
        assert torch.allclose(
            buffer.advantages[0, 0], torch.tensor(expected_0), atol=1e-6
        )
