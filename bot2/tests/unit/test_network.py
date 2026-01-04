"""Unit tests for the PPO Actor-Critic network.

Tests cover:
- Network initialization with default and custom parameters
- Forward pass output shapes
- get_action_and_value() functionality
- get_value() functionality
- Deterministic action selection
- Weight initialization
- Device support (CPU/CUDA)
"""

import pytest
import torch

from bot.actions import ACTION_SPACE_SIZE
from bot.agent.network import ActorCriticNetwork
from bot.observation.observation_space import DEFAULT_CONFIG


class TestActorCriticNetworkInit:
    """Tests for network initialization."""

    def test_default_initialization(self) -> None:
        """Test network initializes with default parameters."""
        network = ActorCriticNetwork()

        assert network.observation_size == DEFAULT_CONFIG.total_size
        assert network.action_size == ACTION_SPACE_SIZE
        assert network.hidden_size == 256
        assert network.actor_hidden == 128
        assert network.critic_hidden == 128

    def test_custom_observation_size(self) -> None:
        """Test network with custom observation size."""
        network = ActorCriticNetwork(observation_size=114)

        assert network.observation_size == 114

    def test_custom_action_size(self) -> None:
        """Test network with custom action size."""
        network = ActorCriticNetwork(action_size=10)

        assert network.action_size == 10

    def test_custom_hidden_sizes(self) -> None:
        """Test network with custom hidden layer sizes."""
        network = ActorCriticNetwork(
            hidden_size=512,
            actor_hidden=256,
            critic_hidden=64,
        )

        assert network.hidden_size == 512
        assert network.actor_hidden == 256
        assert network.critic_hidden == 64

    def test_weight_initialization(self) -> None:
        """Test that weights are properly initialized."""
        network = ActorCriticNetwork()

        # Check that actor output layer has small weights (gain=0.01)
        actor_output_weight = network.actor[-1].weight
        assert actor_output_weight.abs().max() < 0.1

        # Check that biases are zero
        for module in network.modules():
            if isinstance(module, torch.nn.Linear):
                assert torch.allclose(module.bias, torch.zeros_like(module.bias))


class TestForwardPass:
    """Tests for forward pass functionality."""

    def test_forward_shapes_single(self) -> None:
        """Test forward pass with single observation."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(114)

        action_logits, value = network(obs)

        assert action_logits.shape == (27,)
        assert value.shape == ()  # scalar

    def test_forward_shapes_batch(self) -> None:
        """Test forward pass with batched observations."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(32, 114)

        action_logits, value = network(obs)

        assert action_logits.shape == (32, 27)
        assert value.shape == (32,)

    def test_forward_shapes_default_config(self) -> None:
        """Test forward pass with default configuration (includes map)."""
        network = ActorCriticNetwork()
        obs = torch.randn(16, DEFAULT_CONFIG.total_size)

        action_logits, value = network(obs)

        assert action_logits.shape == (16, ACTION_SPACE_SIZE)
        assert value.shape == (16,)

    def test_forward_output_types(self) -> None:
        """Test forward pass returns float tensors."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(8, 114)

        action_logits, value = network(obs)

        assert action_logits.dtype == torch.float32
        assert value.dtype == torch.float32


class TestGetActionAndValue:
    """Tests for get_action_and_value() method."""

    def test_output_shapes(self) -> None:
        """Test get_action_and_value output shapes."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(16, 114)

        action, log_prob, entropy, value = network.get_action_and_value(obs)

        assert action.shape == (16,)
        assert log_prob.shape == (16,)
        assert entropy.shape == (16,)
        assert value.shape == (16,)

    def test_actions_in_valid_range(self) -> None:
        """Test that sampled actions are within valid range."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(100, 114)

        action, _, _, _ = network.get_action_and_value(obs)

        assert (action >= 0).all()
        assert (action < 27).all()

    def test_log_probs_are_negative(self) -> None:
        """Test that log probabilities are negative (or zero for deterministic)."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(100, 114)

        _, log_prob, _, _ = network.get_action_and_value(obs)

        assert (log_prob <= 0).all()

    def test_entropy_is_non_negative(self) -> None:
        """Test that entropy values are non-negative."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(100, 114)

        _, _, entropy, _ = network.get_action_and_value(obs)

        assert (entropy >= 0).all()

    def test_provided_action_returns_correct_log_prob(self) -> None:
        """Test that providing an action returns log_prob for that specific action."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(8, 114)
        actions = torch.randint(0, 27, (8,))

        _, log_prob, _, _ = network.get_action_and_value(obs, action=actions)

        assert log_prob.shape == (8,)
        # Verify by computing manually
        action_logits, _ = network(obs)
        dist = torch.distributions.Categorical(logits=action_logits)
        expected_log_prob = dist.log_prob(actions)
        assert torch.allclose(log_prob, expected_log_prob)

    def test_deterministic_action_is_consistent(self) -> None:
        """Test that deterministic mode returns the same action."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(1, 114)

        action1, _, _, _ = network.get_action_and_value(obs, deterministic=True)
        action2, _, _, _ = network.get_action_and_value(obs, deterministic=True)

        assert action1 == action2

    def test_deterministic_action_is_argmax(self) -> None:
        """Test that deterministic action equals argmax of logits."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(8, 114)

        action, _, _, _ = network.get_action_and_value(obs, deterministic=True)

        action_logits, _ = network(obs)
        expected_action = action_logits.argmax(dim=-1)
        assert torch.equal(action, expected_action)

    def test_stochastic_sampling_varies(self) -> None:
        """Test that stochastic sampling produces variation."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        # Use uniform-ish logits to increase variation
        torch.manual_seed(42)
        network.actor[-1].weight.data.fill_(0.0)
        network.actor[-1].bias.data.fill_(0.0)

        obs = torch.randn(1, 114)
        actions = []
        for _ in range(100):
            action, _, _, _ = network.get_action_and_value(obs)
            actions.append(action.item())

        # With 27 actions and uniform distribution, we should see multiple unique actions
        unique_actions = len(set(actions))
        assert unique_actions > 1


class TestGetValue:
    """Tests for get_value() method."""

    def test_output_shape_batch(self) -> None:
        """Test get_value output shape with batch."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(16, 114)

        value = network.get_value(obs)

        assert value.shape == (16,)

    def test_output_shape_single(self) -> None:
        """Test get_value output shape with single observation."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(114)

        value = network.get_value(obs)

        assert value.shape == ()

    def test_value_matches_forward(self) -> None:
        """Test that get_value returns same value as forward pass."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(8, 114)

        value_from_get_value = network.get_value(obs)
        _, value_from_forward = network(obs)

        assert torch.allclose(value_from_get_value, value_from_forward)

    def test_value_matches_get_action_and_value(self) -> None:
        """Test that get_value returns same value as get_action_and_value."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(8, 114)

        value_from_get_value = network.get_value(obs)
        _, _, _, value_from_get_action = network.get_action_and_value(obs)

        assert torch.allclose(value_from_get_value, value_from_get_action)


class TestGetActionDistribution:
    """Tests for get_action_distribution() method."""

    def test_returns_categorical(self) -> None:
        """Test that get_action_distribution returns a Categorical distribution."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(8, 114)

        dist = network.get_action_distribution(obs)

        assert isinstance(dist, torch.distributions.Categorical)

    def test_distribution_shape(self) -> None:
        """Test distribution has correct batch shape."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(8, 114)

        dist = network.get_action_distribution(obs)

        assert dist.probs.shape == (8, 27)

    def test_probabilities_sum_to_one(self) -> None:
        """Test that action probabilities sum to 1."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(8, 114)

        dist = network.get_action_distribution(obs)

        prob_sums = dist.probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(8))


class TestDeviceSupport:
    """Tests for device (CPU/CUDA) support."""

    def test_cpu_execution(self) -> None:
        """Test network works on CPU."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        network = network.to("cpu")
        obs = torch.randn(8, 114, device="cpu")

        action, log_prob, entropy, value = network.get_action_and_value(obs)

        assert action.device.type == "cpu"
        assert log_prob.device.type == "cpu"
        assert entropy.device.type == "cpu"
        assert value.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self) -> None:
        """Test network works on CUDA."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        network = network.to("cuda")
        obs = torch.randn(8, 114, device="cuda")

        action, log_prob, entropy, value = network.get_action_and_value(obs)

        assert action.device.type == "cuda"
        assert log_prob.device.type == "cuda"
        assert entropy.device.type == "cuda"
        assert value.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_move_to_cuda_and_back(self) -> None:
        """Test network can be moved between devices."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)

        # Move to CUDA
        network = network.to("cuda")
        obs_cuda = torch.randn(4, 114, device="cuda")
        action_logits, value = network(obs_cuda)
        assert action_logits.device.type == "cuda"

        # Move back to CPU
        network = network.to("cpu")
        obs_cpu = torch.randn(4, 114, device="cpu")
        action_logits, value = network(obs_cpu)
        assert action_logits.device.type == "cpu"


class TestGradients:
    """Tests for gradient computation."""

    def test_gradients_flow_through_actor(self) -> None:
        """Test gradients flow through actor head."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(8, 114)

        action_logits, _ = network(obs)
        loss = action_logits.sum()
        loss.backward()

        # Check gradients exist in actor
        assert network.actor[-1].weight.grad is not None
        assert network.actor[-1].weight.grad.abs().sum() > 0

    def test_gradients_flow_through_critic(self) -> None:
        """Test gradients flow through critic head."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(8, 114)

        _, value = network(obs)
        loss = value.sum()
        loss.backward()

        # Check gradients exist in critic
        assert network.critic[-1].weight.grad is not None
        assert network.critic[-1].weight.grad.abs().sum() > 0

    def test_gradients_flow_through_shared_features(self) -> None:
        """Test gradients from both heads flow through shared features."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        obs = torch.randn(8, 114)

        action_logits, value = network(obs)
        loss = action_logits.sum() + value.sum()
        loss.backward()

        # Check gradients exist in shared features
        assert network.features[0].weight.grad is not None
        assert network.features[0].weight.grad.abs().sum() > 0


class TestIntegrationWithGameConstants:
    """Tests verifying integration with game-specific constants."""

    def test_default_observation_size_matches_config(self) -> None:
        """Test default observation size matches ObservationConfig."""
        network = ActorCriticNetwork()

        assert network.observation_size == DEFAULT_CONFIG.total_size
        # Default config includes map: 14 + 36 + 64 + 300 = 414
        assert network.observation_size == 414

    def test_default_action_size_matches_action_space(self) -> None:
        """Test default action size matches ACTION_SPACE_SIZE."""
        network = ActorCriticNetwork()

        assert network.action_size == ACTION_SPACE_SIZE
        assert network.action_size == 27

    def test_works_with_gym_observation_shape(self) -> None:
        """Test network works with observations shaped like gym environment."""
        import numpy as np

        network = ActorCriticNetwork()

        # Simulate gym observation (float32 numpy array)
        obs_np = np.random.randn(DEFAULT_CONFIG.total_size).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_np)

        action, log_prob, entropy, value = network.get_action_and_value(
            obs_tensor.unsqueeze(0)
        )

        assert action.shape == (1,)
        assert 0 <= action.item() < ACTION_SPACE_SIZE

    def test_works_with_vectorized_env_batch(self) -> None:
        """Test network works with batched observations from vectorized env."""
        import numpy as np

        network = ActorCriticNetwork()
        num_envs = 8

        # Simulate vectorized env observation (batch, obs_size)
        obs_np = np.random.randn(num_envs, DEFAULT_CONFIG.total_size).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_np)

        action, log_prob, entropy, value = network.get_action_and_value(obs_tensor)

        assert action.shape == (num_envs,)
        assert (action >= 0).all() and (action < ACTION_SPACE_SIZE).all()
