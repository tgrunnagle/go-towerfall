"""Integration tests for the model registry.

These tests verify the complete workflow of model registration, storage,
retrieval, and comparison across multiple generations.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from bot.agent.network import ActorCriticNetwork
from bot.training.registry import (
    ModelRegistry,
    TrainingMetrics,
)


@pytest.fixture
def temp_registry_path():
    """Create a temporary directory for registry storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "registry"


@pytest.fixture
def registry(temp_registry_path: Path) -> ModelRegistry:
    """Create a ModelRegistry with temporary storage."""
    return ModelRegistry(temp_registry_path)


class TestRegistryIntegration:
    """Integration tests for model registry workflows."""

    def test_register_retrieve_verify_weights_match(self, registry: ModelRegistry):
        """Integration test: register model -> retrieve model -> verify weights match."""
        # Create and train a network (simulate training by doing a forward pass)
        network = ActorCriticNetwork(observation_size=114, action_size=27)

        # Capture original weights for verification
        original_state = {k: v.clone() for k, v in network.state_dict().items()}

        # Run sample inference
        sample_obs = torch.randn(1, 114)
        original_logits, original_value = network(sample_obs)

        # Create metrics
        metrics = TrainingMetrics(
            total_episodes=500,
            total_timesteps=50000,
            average_reward=45.0,
            average_episode_length=450.0,
            win_rate=0.55,
            average_kills=2.0,
            average_deaths=1.8,
            kills_deaths_ratio=1.11,
        )

        hyperparams = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "clip_range": 0.2,
        }

        # Register model
        model_id = registry.register_model(
            model=network,
            generation=0,
            opponent_model_id=None,
            training_metrics=metrics,
            hyperparameters=hyperparams,
            training_duration_seconds=1800.0,
            notes="Integration test model",
        )

        # Retrieve model
        loaded_network, metadata = registry.get_model(model_id)

        # Verify weights match
        loaded_state = loaded_network.state_dict()
        for key in original_state:
            assert torch.allclose(original_state[key], loaded_state[key]), (
                f"Weight mismatch for {key}"
            )

        # Verify inference matches
        loaded_logits, loaded_value = loaded_network(sample_obs)
        assert torch.allclose(original_logits, loaded_logits)
        assert torch.allclose(original_value, loaded_value)

        # Verify metadata
        assert metadata.model_id == model_id
        assert metadata.generation == 0
        assert metadata.training_metrics.kills_deaths_ratio == 1.11
        assert metadata.notes == "Integration test model"

    def test_successive_model_registration(self, registry: ModelRegistry):
        """Integration test: register successive generations of models."""
        base_hyperparams = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
        }

        # Generation 0: initial model
        gen0_network = ActorCriticNetwork(observation_size=114, action_size=27)
        gen0_metrics = TrainingMetrics(
            total_episodes=1000,
            total_timesteps=100000,
            average_reward=30.0,
            average_episode_length=300.0,
            win_rate=0.4,
            average_kills=1.2,
            average_deaths=2.0,
            kills_deaths_ratio=0.6,
        )

        gen0_id = registry.register_model(
            model=gen0_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=gen0_metrics,
            hyperparameters=base_hyperparams,
            training_duration_seconds=3600.0,
        )

        assert gen0_id == "ppo_gen_000"
        assert registry.get_next_generation() == 1

        # Generation 1: trained against gen0
        gen1_network = ActorCriticNetwork(observation_size=114, action_size=27)
        gen1_metrics = TrainingMetrics(
            total_episodes=2000,
            total_timesteps=200000,
            average_reward=55.0,
            average_episode_length=450.0,
            win_rate=0.65,
            average_kills=2.5,
            average_deaths=1.5,
            kills_deaths_ratio=1.67,
        )

        gen1_id = registry.register_model(
            model=gen1_network,
            generation=1,
            opponent_model_id="ppo_gen_000",
            training_metrics=gen1_metrics,
            hyperparameters=base_hyperparams,
            training_duration_seconds=7200.0,
        )

        assert gen1_id == "ppo_gen_001"
        assert registry.get_next_generation() == 2

        # Generation 2: trained against gen1
        gen2_network = ActorCriticNetwork(observation_size=114, action_size=27)
        gen2_metrics = TrainingMetrics(
            total_episodes=3000,
            total_timesteps=300000,
            average_reward=75.0,
            average_episode_length=550.0,
            win_rate=0.8,
            average_kills=3.5,
            average_deaths=1.0,
            kills_deaths_ratio=3.5,
        )

        gen2_id = registry.register_model(
            model=gen2_network,
            generation=2,
            opponent_model_id="ppo_gen_001",
            training_metrics=gen2_metrics,
            hyperparameters=base_hyperparams,
            training_duration_seconds=10800.0,
        )

        assert gen2_id == "ppo_gen_002"

        # Verify model list
        models = registry.list_models()
        assert len(models) == 3
        assert [m.generation for m in models] == [0, 1, 2]

        # Verify best model
        best_result = registry.get_best_model()
        assert best_result is not None
        _, best_metadata = best_result
        assert best_metadata.model_id == "ppo_gen_002"
        assert best_metadata.training_metrics.kills_deaths_ratio == 3.5

        # Verify comparison
        assert registry.is_better_than("ppo_gen_001", "ppo_gen_000")
        assert registry.is_better_than("ppo_gen_002", "ppo_gen_001")
        assert registry.is_better_than("ppo_gen_002", "ppo_gen_000")
        assert not registry.is_better_than("ppo_gen_000", "ppo_gen_001")

        # Verify opponent tracking
        _, gen1_meta = registry.get_model("ppo_gen_001")
        assert gen1_meta.opponent_model_id == "ppo_gen_000"

        _, gen2_meta = registry.get_model("ppo_gen_002")
        assert gen2_meta.opponent_model_id == "ppo_gen_001"

    def test_registry_persistence(self, temp_registry_path: Path):
        """Integration test: verify registry persists across instances."""
        # Create first registry instance and register a model
        registry1 = ModelRegistry(temp_registry_path)

        network = ActorCriticNetwork(observation_size=114, action_size=27)
        metrics = TrainingMetrics(
            total_episodes=1000,
            total_timesteps=100000,
            average_reward=50.0,
            average_episode_length=500.0,
            win_rate=0.6,
            average_kills=2.5,
            average_deaths=1.5,
            kills_deaths_ratio=1.67,
        )

        model_id = registry1.register_model(
            model=network,
            generation=0,
            opponent_model_id=None,
            training_metrics=metrics,
            hyperparameters={"learning_rate": 3e-4},
            training_duration_seconds=3600.0,
        )

        # Create new registry instance pointing to same path
        registry2 = ModelRegistry(temp_registry_path)

        # Verify model is accessible from new instance
        models = registry2.list_models()
        assert len(models) == 1
        assert models[0].model_id == model_id

        # Verify can load the model
        loaded_network, metadata = registry2.get_model(model_id)
        assert isinstance(loaded_network, ActorCriticNetwork)
        assert metadata.training_metrics.kills_deaths_ratio == 1.67

    def test_registry_delete_cleanup(self, registry: ModelRegistry):
        """Integration test: verify delete properly cleans up files and index."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        metrics = TrainingMetrics(
            total_episodes=1000,
            total_timesteps=100000,
            average_reward=50.0,
            average_episode_length=500.0,
            win_rate=0.6,
            average_kills=2.5,
            average_deaths=1.5,
            kills_deaths_ratio=1.67,
        )

        # Register two models
        model_id_0 = registry.register_model(
            model=network,
            generation=0,
            opponent_model_id=None,
            training_metrics=metrics,
            hyperparameters={},
            training_duration_seconds=3600.0,
        )

        model_id_1 = registry.register_model(
            model=network,
            generation=1,
            opponent_model_id=model_id_0,
            training_metrics=metrics,
            hyperparameters={},
            training_duration_seconds=3600.0,
        )

        # Verify both models exist
        assert len(registry.list_models()) == 2

        # Delete first model
        assert registry.delete_model(model_id_0) is True

        # Verify only one model remains
        models = registry.list_models()
        assert len(models) == 1
        assert models[0].model_id == model_id_1

        # Verify files are cleaned up
        model_dir = registry.registry_path / "models" / model_id_0
        assert not model_dir.exists()

    def test_checkpoint_data_access(self, registry: ModelRegistry):
        """Integration test: verify checkpoint data access for training resumption."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)

        # Do some "training" to update optimizer state
        obs = torch.randn(32, 114)
        logits, value = network(obs)
        loss = logits.mean() + value.mean()
        loss.backward()
        optimizer.step()

        metrics = TrainingMetrics(
            total_episodes=1000,
            total_timesteps=100000,
            average_reward=50.0,
            average_episode_length=500.0,
            win_rate=0.6,
            average_kills=2.5,
            average_deaths=1.5,
            kills_deaths_ratio=1.67,
        )

        model_id = registry.register_model(
            model=network,
            generation=0,
            opponent_model_id=None,
            training_metrics=metrics,
            hyperparameters={"learning_rate": 3e-4},
            training_duration_seconds=3600.0,
            optimizer=optimizer,
            training_step=100,
        )

        # Get checkpoint data
        checkpoint = registry.get_checkpoint_data(model_id)

        assert checkpoint.training_step == 100
        assert checkpoint.total_timesteps == 100000
        assert "learning_rate" in checkpoint.hyperparameters

        # Optimizer state should be saved
        assert checkpoint.optimizer_state_dict is not None


class TestMultiGenerationTraining:
    """Tests simulating the successive training pipeline."""

    def test_full_successive_training_workflow(self, registry: ModelRegistry):
        """Simulate a complete successive training workflow.

        This test simulates:
        1. Training gen 0 against a base opponent
        2. Checking if gen 0 is "better" (K/D > threshold)
        3. Training gen 1 against gen 0
        4. Comparing generations and selecting the best
        """
        target_kd_ratio = 2.0  # Target K/D ratio to be considered "good"

        # Generation 0: Not good enough yet
        gen0_network = ActorCriticNetwork(observation_size=114, action_size=27)
        gen0_metrics = TrainingMetrics(
            total_episodes=1000,
            total_timesteps=100000,
            average_reward=35.0,
            average_episode_length=350.0,
            win_rate=0.45,
            average_kills=1.5,
            average_deaths=1.8,
            kills_deaths_ratio=0.83,  # Below threshold
        )

        gen0_id = registry.register_model(
            model=gen0_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=gen0_metrics,
            hyperparameters={"learning_rate": 3e-4},
            training_duration_seconds=3600.0,
        )

        # Check if gen0 meets threshold
        gen0_meta = registry.get_metadata(gen0_id)
        assert gen0_meta.training_metrics.kills_deaths_ratio < target_kd_ratio

        # Generation 1: Better but not there yet
        gen1_network = ActorCriticNetwork(observation_size=114, action_size=27)
        gen1_metrics = TrainingMetrics(
            total_episodes=2000,
            total_timesteps=200000,
            average_reward=55.0,
            average_episode_length=450.0,
            win_rate=0.6,
            average_kills=2.2,
            average_deaths=1.5,
            kills_deaths_ratio=1.47,  # Better but still below threshold
        )

        gen1_id = registry.register_model(
            model=gen1_network,
            generation=1,
            opponent_model_id=gen0_id,
            training_metrics=gen1_metrics,
            hyperparameters={"learning_rate": 3e-4},
            training_duration_seconds=7200.0,
        )

        # Verify gen1 is better than gen0
        assert registry.is_better_than(gen1_id, gen0_id)

        # Check if gen1 meets threshold
        gen1_meta = registry.get_metadata(gen1_id)
        assert gen1_meta.training_metrics.kills_deaths_ratio < target_kd_ratio

        # Generation 2: Meets the threshold!
        gen2_network = ActorCriticNetwork(observation_size=114, action_size=27)
        gen2_metrics = TrainingMetrics(
            total_episodes=3000,
            total_timesteps=300000,
            average_reward=80.0,
            average_episode_length=600.0,
            win_rate=0.75,
            average_kills=3.0,
            average_deaths=1.2,
            kills_deaths_ratio=2.5,  # Meets threshold!
        )

        gen2_id = registry.register_model(
            model=gen2_network,
            generation=2,
            opponent_model_id=gen1_id,
            training_metrics=gen2_metrics,
            hyperparameters={"learning_rate": 3e-4},
            training_duration_seconds=10800.0,
        )

        # Verify gen2 meets threshold
        gen2_meta = registry.get_metadata(gen2_id)
        assert gen2_meta.training_metrics.kills_deaths_ratio >= target_kd_ratio

        # Verify gen2 is the best model
        best_result = registry.get_best_model()
        assert best_result is not None
        _, best_meta = best_result
        assert best_meta.model_id == gen2_id

        # Verify lineage
        assert gen2_meta.opponent_model_id == gen1_id
        assert gen1_meta.opponent_model_id == gen0_id
        assert gen0_meta.opponent_model_id is None
