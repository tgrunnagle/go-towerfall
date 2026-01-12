"""Unit tests for the model registry."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from bot.agent.network import ActorCriticNetwork
from bot.training.registry import (
    ModelAlreadyExistsError,
    ModelMetadata,
    ModelNotFoundError,
    ModelRegistry,
    NetworkArchitecture,
    RegistryIndex,
    StorageBackend,
    TrainingMetrics,
)

# Fixtures


@pytest.fixture
def sample_metrics() -> TrainingMetrics:
    """Create sample training metrics for testing."""
    return TrainingMetrics(
        total_episodes=1000,
        total_timesteps=100000,
        average_reward=50.0,
        average_episode_length=500.0,
        win_rate=0.6,
        average_kills=2.5,
        average_deaths=1.5,
        kills_deaths_ratio=1.67,
    )


@pytest.fixture
def sample_hyperparams() -> dict:
    """Create sample hyperparameters for testing."""
    return {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "clip_range": 0.2,
        "n_epochs": 10,
    }


@pytest.fixture
def sample_network() -> ActorCriticNetwork:
    """Create a sample network for testing."""
    return ActorCriticNetwork(observation_size=114, action_size=27, hidden_size=256)


@pytest.fixture
def temp_registry_path():
    """Create a temporary directory for registry storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "registry"


@pytest.fixture
def registry(temp_registry_path: Path) -> ModelRegistry:
    """Create a ModelRegistry with temporary storage."""
    return ModelRegistry(temp_registry_path)


# TrainingMetrics Tests


class TestTrainingMetrics:
    """Tests for TrainingMetrics Pydantic model."""

    def test_create_valid_metrics(self, sample_metrics: TrainingMetrics):
        """Test creating valid training metrics."""
        assert sample_metrics.total_episodes == 1000
        assert sample_metrics.total_timesteps == 100000
        assert sample_metrics.average_reward == 50.0
        assert sample_metrics.kills_deaths_ratio == 1.67

    def test_metrics_validation_negative_episodes(self):
        """Test that negative episodes are rejected."""
        with pytest.raises(ValueError):
            TrainingMetrics(
                total_episodes=-1,
                total_timesteps=100000,
                average_reward=50.0,
                average_episode_length=500.0,
                win_rate=0.6,
                average_kills=2.5,
                average_deaths=1.5,
                kills_deaths_ratio=1.67,
            )

    def test_metrics_validation_win_rate_range(self):
        """Test that win_rate must be between 0 and 1."""
        with pytest.raises(ValueError):
            TrainingMetrics(
                total_episodes=100,
                total_timesteps=10000,
                average_reward=50.0,
                average_episode_length=500.0,
                win_rate=1.5,  # Invalid: > 1.0
                average_kills=2.5,
                average_deaths=1.5,
                kills_deaths_ratio=1.67,
            )

    def test_metrics_frozen(self, sample_metrics: TrainingMetrics):
        """Test that metrics are immutable."""
        with pytest.raises(Exception):  # Pydantic frozen model raises ValidationError
            sample_metrics.total_episodes = 999


class TestNetworkArchitecture:
    """Tests for NetworkArchitecture Pydantic model."""

    def test_create_valid_architecture(self):
        """Test creating valid network architecture."""
        arch = NetworkArchitecture(
            observation_size=114,
            action_size=27,
            hidden_size=256,
            actor_hidden=128,
            critic_hidden=128,
        )
        assert arch.observation_size == 114
        assert arch.action_size == 27

    def test_architecture_validation_positive_values(self):
        """Test that dimensions must be positive."""
        with pytest.raises(ValueError):
            NetworkArchitecture(
                observation_size=0,  # Invalid: must be > 0
                action_size=27,
            )


class TestModelMetadata:
    """Tests for ModelMetadata Pydantic model."""

    def test_create_full_metadata(self, sample_metrics: TrainingMetrics):
        """Test creating full model metadata."""
        arch = NetworkArchitecture(
            observation_size=114, action_size=27, hidden_size=256
        )
        metadata = ModelMetadata(
            model_id="ppo_gen_000",
            generation=0,
            created_at=datetime.now(timezone.utc),
            training_duration_seconds=3600.0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters={"learning_rate": 3e-4},
            architecture=arch,
            checkpoint_path="models/ppo_gen_000/model.pt",
            notes="First generation model",
        )
        assert metadata.model_id == "ppo_gen_000"
        assert metadata.generation == 0
        assert metadata.opponent_model_id is None
        assert metadata.notes == "First generation model"

    def test_metadata_serialization(self, sample_metrics: TrainingMetrics):
        """Test metadata serialization to dict."""
        arch = NetworkArchitecture(observation_size=114, action_size=27)
        metadata = ModelMetadata(
            model_id="ppo_gen_001",
            generation=1,
            created_at=datetime.now(timezone.utc),
            training_duration_seconds=7200.0,
            opponent_model_id="ppo_gen_000",
            training_metrics=sample_metrics,
            hyperparameters={},
            architecture=arch,
            checkpoint_path="models/ppo_gen_001/model.pt",
        )

        data = metadata.model_dump(mode="json")
        assert data["model_id"] == "ppo_gen_001"
        assert data["generation"] == 1
        assert isinstance(data["created_at"], str)  # ISO format string

        # Round-trip test
        restored = ModelMetadata.model_validate(data)
        assert restored.model_id == metadata.model_id
        assert (
            restored.training_metrics.kills_deaths_ratio
            == sample_metrics.kills_deaths_ratio
        )


# StorageBackend Tests


class TestStorageBackend:
    """Tests for StorageBackend class."""

    def test_initialize_creates_directories(self, temp_registry_path: Path):
        """Test that initialize creates required directories."""
        backend = StorageBackend(temp_registry_path)
        backend.initialize()

        assert temp_registry_path.exists()
        assert (temp_registry_path / "models").exists()
        assert (temp_registry_path / "index.json").exists()

    def test_model_exists_false_for_new_model(self, temp_registry_path: Path):
        """Test model_exists returns False for non-existent model."""
        backend = StorageBackend(temp_registry_path)
        backend.initialize()

        assert not backend.model_exists("ppo_gen_000")

    def test_get_next_generation_empty_registry(self, temp_registry_path: Path):
        """Test next generation is 0 for empty registry."""
        backend = StorageBackend(temp_registry_path)
        backend.initialize()

        assert backend.get_next_generation() == 0


class TestRegistryIndex:
    """Tests for RegistryIndex class."""

    def test_create_empty_index(self):
        """Test creating empty registry index."""
        index = RegistryIndex()
        assert len(index.models) == 0
        assert index.last_updated is not None

    def test_index_serialization_roundtrip(self, sample_metrics: TrainingMetrics):
        """Test index serialization and deserialization."""
        arch = NetworkArchitecture(observation_size=114, action_size=27)
        metadata = ModelMetadata(
            model_id="ppo_gen_000",
            generation=0,
            created_at=datetime.now(timezone.utc),
            training_duration_seconds=3600.0,
            training_metrics=sample_metrics,
            hyperparameters={},
            architecture=arch,
            checkpoint_path="models/ppo_gen_000/model.pt",
        )

        index = RegistryIndex(models={"ppo_gen_000": metadata})
        data = index.to_dict()

        restored = RegistryIndex.from_dict(data)
        assert "ppo_gen_000" in restored.models
        assert restored.models["ppo_gen_000"].generation == 0


# ModelRegistry Tests


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_registry_initialization(self, registry: ModelRegistry):
        """Test registry initializes correctly."""
        assert registry.registry_path.exists()

    def test_register_model(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_metrics: TrainingMetrics,
        sample_hyperparams: dict,
    ):
        """Test registering a new model."""
        model_id = registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )

        assert model_id == "ppo_gen_000"

        # Verify model files exist
        checkpoint_path = registry.registry_path / "models" / "ppo_gen_000" / "model.pt"
        metadata_path = (
            registry.registry_path / "models" / "ppo_gen_000" / "metadata.json"
        )
        assert checkpoint_path.exists()
        assert metadata_path.exists()

    def test_register_model_duplicate_raises_error(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_metrics: TrainingMetrics,
        sample_hyperparams: dict,
    ):
        """Test registering duplicate model raises error."""
        registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )

        with pytest.raises(ModelAlreadyExistsError):
            registry.register_model(
                model=sample_network,
                generation=0,  # Same generation
                opponent_model_id=None,
                training_metrics=sample_metrics,
                hyperparameters=sample_hyperparams,
                training_duration_seconds=3600.0,
            )

    def test_get_model(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_metrics: TrainingMetrics,
        sample_hyperparams: dict,
    ):
        """Test retrieving a registered model."""
        model_id = registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )

        network, metadata = registry.get_model(model_id)

        assert isinstance(network, ActorCriticNetwork)
        assert metadata.model_id == "ppo_gen_000"
        assert metadata.generation == 0
        assert metadata.training_metrics.total_episodes == sample_metrics.total_episodes

    def test_get_model_not_found(self, registry: ModelRegistry):
        """Test getting non-existent model raises error."""
        with pytest.raises(ModelNotFoundError):
            registry.get_model("nonexistent_model")

    def test_get_model_by_generation(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_metrics: TrainingMetrics,
        sample_hyperparams: dict,
    ):
        """Test retrieving model by generation number."""
        registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )

        result = registry.get_model_by_generation(0)
        assert result is not None
        network, metadata = result
        assert metadata.generation == 0

        # Non-existent generation returns None
        assert registry.get_model_by_generation(1) is None

    def test_get_latest_model(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_metrics: TrainingMetrics,
        sample_hyperparams: dict,
    ):
        """Test getting the latest model."""
        # Register two models
        registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )

        better_metrics = TrainingMetrics(
            total_episodes=2000,
            total_timesteps=200000,
            average_reward=75.0,
            average_episode_length=600.0,
            win_rate=0.8,
            average_kills=3.0,
            average_deaths=1.0,
            kills_deaths_ratio=3.0,
        )
        registry.register_model(
            model=sample_network,
            generation=1,
            opponent_model_id="ppo_gen_000",
            training_metrics=better_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=7200.0,
        )

        result = registry.get_latest_model()
        assert result is not None
        _, metadata = result
        assert metadata.generation == 1

    def test_get_latest_model_empty_registry(self, registry: ModelRegistry):
        """Test getting latest model from empty registry returns None."""
        assert registry.get_latest_model() is None

    def test_list_models(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_metrics: TrainingMetrics,
        sample_hyperparams: dict,
    ):
        """Test listing all models."""
        # Register two models
        registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )
        registry.register_model(
            model=sample_network,
            generation=1,
            opponent_model_id="ppo_gen_000",
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=7200.0,
        )

        models = registry.list_models()
        assert len(models) == 2
        assert models[0].generation == 0  # Sorted by generation
        assert models[1].generation == 1

    def test_get_best_model(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_hyperparams: dict,
    ):
        """Test getting the best model by K/D ratio."""
        weak_metrics = TrainingMetrics(
            total_episodes=1000,
            total_timesteps=100000,
            average_reward=30.0,
            average_episode_length=400.0,
            win_rate=0.4,
            average_kills=1.0,
            average_deaths=2.0,
            kills_deaths_ratio=0.5,
        )
        registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=weak_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )

        strong_metrics = TrainingMetrics(
            total_episodes=2000,
            total_timesteps=200000,
            average_reward=80.0,
            average_episode_length=600.0,
            win_rate=0.9,
            average_kills=4.0,
            average_deaths=1.0,
            kills_deaths_ratio=4.0,
        )
        registry.register_model(
            model=sample_network,
            generation=1,
            opponent_model_id="ppo_gen_000",
            training_metrics=strong_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=7200.0,
        )

        result = registry.get_best_model()
        assert result is not None
        _, metadata = result
        assert metadata.model_id == "ppo_gen_001"
        assert metadata.training_metrics.kills_deaths_ratio == 4.0

    def test_is_better_than(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_hyperparams: dict,
    ):
        """Test comparing two models."""
        weak_metrics = TrainingMetrics(
            total_episodes=1000,
            total_timesteps=100000,
            average_reward=30.0,
            average_episode_length=400.0,
            win_rate=0.4,
            average_kills=1.0,
            average_deaths=2.0,
            kills_deaths_ratio=0.5,
        )
        registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=weak_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )

        strong_metrics = TrainingMetrics(
            total_episodes=2000,
            total_timesteps=200000,
            average_reward=80.0,
            average_episode_length=600.0,
            win_rate=0.9,
            average_kills=4.0,
            average_deaths=1.0,
            kills_deaths_ratio=4.0,
        )
        registry.register_model(
            model=sample_network,
            generation=1,
            opponent_model_id="ppo_gen_000",
            training_metrics=strong_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=7200.0,
        )

        assert registry.is_better_than("ppo_gen_001", "ppo_gen_000") is True
        assert registry.is_better_than("ppo_gen_000", "ppo_gen_001") is False

    def test_is_better_than_model_not_found(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_metrics: TrainingMetrics,
        sample_hyperparams: dict,
    ):
        """Test is_better_than raises error for non-existent model."""
        registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )

        with pytest.raises(ModelNotFoundError):
            registry.is_better_than("ppo_gen_000", "nonexistent")

    def test_delete_model(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_metrics: TrainingMetrics,
        sample_hyperparams: dict,
    ):
        """Test deleting a model."""
        model_id = registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )

        assert registry.delete_model(model_id) is True

        # Verify model is gone
        with pytest.raises(ModelNotFoundError):
            registry.get_model(model_id)

        # Delete non-existent model returns False
        assert registry.delete_model("nonexistent") is False

    def test_get_next_generation(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_metrics: TrainingMetrics,
        sample_hyperparams: dict,
    ):
        """Test getting the next generation number."""
        assert registry.get_next_generation() == 0

        registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )

        assert registry.get_next_generation() == 1

        registry.register_model(
            model=sample_network,
            generation=1,
            opponent_model_id="ppo_gen_000",
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=7200.0,
        )

        assert registry.get_next_generation() == 2

    def test_get_metadata_without_loading_weights(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_metrics: TrainingMetrics,
        sample_hyperparams: dict,
    ):
        """Test getting metadata without loading full model."""
        registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
            notes="Test model",
        )

        metadata = registry.get_metadata("ppo_gen_000")
        assert metadata.model_id == "ppo_gen_000"
        assert metadata.notes == "Test model"

    def test_model_weights_preservation(
        self,
        registry: ModelRegistry,
        sample_network: ActorCriticNetwork,
        sample_metrics: TrainingMetrics,
        sample_hyperparams: dict,
    ):
        """Test that model weights are preserved after registration."""
        # Run input through original network
        obs = torch.randn(1, sample_network.observation_size)
        original_logits, original_value = sample_network(obs)

        model_id = registry.register_model(
            model=sample_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=sample_metrics,
            hyperparameters=sample_hyperparams,
            training_duration_seconds=3600.0,
        )

        # Load and compare
        loaded_network, _ = registry.get_model(model_id)
        loaded_logits, loaded_value = loaded_network(obs)

        assert torch.allclose(original_logits, loaded_logits)
        assert torch.allclose(original_value, loaded_value)


class TestModelIdGeneration:
    """Tests for model ID generation."""

    def test_model_id_format(self, registry: ModelRegistry):
        """Test model ID format."""
        assert registry._generate_model_id(0) == "ppo_gen_000"
        assert registry._generate_model_id(1) == "ppo_gen_001"
        assert registry._generate_model_id(10) == "ppo_gen_010"
        assert registry._generate_model_id(100) == "ppo_gen_100"
        assert registry._generate_model_id(999) == "ppo_gen_999"
