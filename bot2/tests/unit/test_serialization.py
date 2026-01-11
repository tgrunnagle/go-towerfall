"""Unit tests for model serialization functionality."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from bot.agent.network import ActorCriticNetwork
from bot.agent.serialization import (
    ModelCheckpoint,
    ModelMetadata,
    generate_model_filename,
    get_checkpoint_info,
    load_model,
    save_model,
)


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_create_minimal(self):
        """Test creating metadata with minimal required fields."""
        metadata = ModelMetadata(
            version="v1.0.0",
            created_at=datetime.now(timezone.utc),
            observation_size=114,
            action_size=27,
            hidden_size=256,
            actor_hidden=128,
            critic_hidden=128,
        )
        assert metadata.version == "v1.0.0"
        assert metadata.total_episodes == 0
        assert metadata.final_reward == 0.0
        assert metadata.opponent_version is None
        assert metadata.extra == {}

    def test_create_full(self):
        """Test creating metadata with all fields."""
        metadata = ModelMetadata(
            version="gen-005",
            created_at=datetime.now(timezone.utc),
            observation_size=100,
            action_size=20,
            hidden_size=256,
            actor_hidden=128,
            critic_hidden=128,
            total_episodes=500,
            final_reward=42.5,
            opponent_version="gen-004",
            extra={"custom_field": "value"},
        )
        assert metadata.version == "gen-005"
        assert metadata.total_episodes == 500
        assert metadata.final_reward == 42.5
        assert metadata.opponent_version == "gen-004"
        assert metadata.extra == {"custom_field": "value"}


class TestModelCheckpoint:
    """Tests for ModelCheckpoint dataclass."""

    def test_create_checkpoint(self):
        """Test creating a model checkpoint."""
        metadata = ModelMetadata(
            version="v1",
            created_at=datetime.now(timezone.utc),
            observation_size=114,
            action_size=27,
            hidden_size=256,
            actor_hidden=128,
            critic_hidden=128,
        )
        checkpoint = ModelCheckpoint(
            model_state_dict={"key": "value"},
            optimizer_state_dict=None,
            training_step=1000,
            total_timesteps=50000,
            metadata=metadata,
        )
        assert checkpoint.training_step == 1000
        assert checkpoint.total_timesteps == 50000
        assert checkpoint.optimizer_state_dict is None
        assert checkpoint.hyperparameters == {}


class TestSaveLoadRoundtrip:
    """Tests for save/load roundtrip functionality."""

    def test_save_load_preserves_weights(self):
        """Verify weights are preserved exactly after save/load."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)

        # Generate some random input to verify behavior
        obs = torch.randn(1, 114)
        original_logits, original_value = network(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1.0.0")

            loaded_network, checkpoint = load_model(path)
            loaded_logits, loaded_value = loaded_network(obs)

            assert torch.allclose(original_logits, loaded_logits)
            assert torch.allclose(original_value, loaded_value)
            assert checkpoint.metadata.version == "v1.0.0"

    def test_save_load_preserves_gradients_behavior(self):
        """Verify loaded network can still compute gradients."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            loaded_network, _ = load_model(path)
            loaded_network.train()  # Switch to training mode

            # Verify gradients can be computed
            obs = torch.randn(32, 114)
            logits, values = loaded_network(obs)
            loss = logits.mean() + values.mean()
            loss.backward()

            # Check that gradients exist
            for param in loaded_network.parameters():
                assert param.grad is not None


class TestMetadataPreservation:
    """Tests for metadata preservation in checkpoints."""

    def test_metadata_preservation(self):
        """Verify all metadata fields are saved and loaded correctly."""
        network = ActorCriticNetwork(observation_size=100, action_size=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(
                network,
                path,
                version="gen-005",
                training_step=1000,
                total_timesteps=50000,
                training_info={
                    "total_episodes": 500,
                    "final_reward": 42.5,
                    "opponent_version": "gen-004",
                },
            )

            _, checkpoint = load_model(path)

            assert checkpoint.metadata.version == "gen-005"
            assert checkpoint.metadata.observation_size == 100
            assert checkpoint.metadata.action_size == 20
            assert checkpoint.training_step == 1000
            assert checkpoint.total_timesteps == 50000
            assert checkpoint.metadata.total_episodes == 500
            assert checkpoint.metadata.final_reward == 42.5
            assert checkpoint.metadata.opponent_version == "gen-004"

    def test_architecture_preservation(self):
        """Verify architecture parameters are saved and loaded."""
        network = ActorCriticNetwork(
            observation_size=200,
            action_size=50,
            hidden_size=512,
            actor_hidden=256,
            critic_hidden=256,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            _, checkpoint = load_model(path)

            assert checkpoint.metadata.observation_size == 200
            assert checkpoint.metadata.action_size == 50
            assert checkpoint.metadata.hidden_size == 512
            assert checkpoint.metadata.actor_hidden == 256
            assert checkpoint.metadata.critic_hidden == 256

    def test_hyperparameters_preservation(self):
        """Verify hyperparameters are saved and loaded."""
        network = ActorCriticNetwork()
        hyperparams = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "clip_range": 0.2,
            "num_epochs": 10,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1", hyperparameters=hyperparams)

            _, checkpoint = load_model(path)

            assert checkpoint.hyperparameters == hyperparams

    def test_created_at_is_valid_datetime(self):
        """Verify created_at timestamp is properly parsed."""
        network = ActorCriticNetwork()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            _, checkpoint = load_model(path)

            assert isinstance(checkpoint.metadata.created_at, datetime)
            # Should be recent (within last minute)
            delta = datetime.now(timezone.utc) - checkpoint.metadata.created_at
            assert delta.total_seconds() < 60


class TestArchitectureValidation:
    """Tests for architecture validation during loading."""

    def test_architecture_mismatch_raises(self):
        """Loading into incompatible network should raise ValueError."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            # Try to load into network with different architecture
            wrong_network = ActorCriticNetwork(observation_size=50, action_size=10)

            with pytest.raises(ValueError, match="architecture mismatch"):
                load_model(path, network=wrong_network)

    def test_observation_size_mismatch(self):
        """Different observation size should raise ValueError."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            wrong_network = ActorCriticNetwork(observation_size=200, action_size=27)

            with pytest.raises(ValueError, match="architecture mismatch"):
                load_model(path, network=wrong_network)

    def test_action_size_mismatch(self):
        """Different action size should raise ValueError."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            wrong_network = ActorCriticNetwork(observation_size=114, action_size=10)

            with pytest.raises(ValueError, match="architecture mismatch"):
                load_model(path, network=wrong_network)

    def test_hidden_size_mismatch(self):
        """Different hidden size should raise ValueError."""
        network = ActorCriticNetwork(
            observation_size=114,
            action_size=27,
            hidden_size=256,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            wrong_network = ActorCriticNetwork(
                observation_size=114,
                action_size=27,
                hidden_size=512,
            )

            with pytest.raises(ValueError, match="hidden_size"):
                load_model(path, network=wrong_network)

    def test_actor_hidden_mismatch(self):
        """Different actor_hidden should raise ValueError."""
        network = ActorCriticNetwork(
            observation_size=114,
            action_size=27,
            actor_hidden=128,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            wrong_network = ActorCriticNetwork(
                observation_size=114,
                action_size=27,
                actor_hidden=256,
            )

            with pytest.raises(ValueError, match="actor_hidden"):
                load_model(path, network=wrong_network)

    def test_critic_hidden_mismatch(self):
        """Different critic_hidden should raise ValueError."""
        network = ActorCriticNetwork(
            observation_size=114,
            action_size=27,
            critic_hidden=128,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            wrong_network = ActorCriticNetwork(
                observation_size=114,
                action_size=27,
                critic_hidden=256,
            )

            with pytest.raises(ValueError, match="critic_hidden"):
                load_model(path, network=wrong_network)

    def test_multiple_mismatches_reported(self):
        """Multiple architecture mismatches should all be reported."""
        network = ActorCriticNetwork(
            observation_size=114,
            action_size=27,
            hidden_size=256,
            actor_hidden=128,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            wrong_network = ActorCriticNetwork(
                observation_size=114,
                action_size=27,
                hidden_size=512,
                actor_hidden=256,
            )

            with pytest.raises(ValueError) as exc_info:
                load_model(path, network=wrong_network)

            error_msg = str(exc_info.value)
            assert "hidden_size" in error_msg
            assert "actor_hidden" in error_msg

    def test_load_without_network_creates_correct_architecture(self):
        """Loading without providing network should create one with saved architecture."""
        network = ActorCriticNetwork(
            observation_size=114,
            action_size=27,
            hidden_size=256,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            # Load without providing network - should create new one with saved arch
            loaded_network, _ = load_model(path)

            assert loaded_network.hidden_size == 256


class TestErrorHandling:
    """Tests for error handling."""

    def test_file_not_found(self):
        """Loading non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.pt")

    def test_get_checkpoint_info_file_not_found(self):
        """get_checkpoint_info on non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_checkpoint_info("/nonexistent/path/model.pt")

    def test_invalid_checkpoint_missing_state_dict(self):
        """Checkpoint without model_state_dict should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.pt"
            torch.save({"metadata": {}}, path)

            with pytest.raises(ValueError, match="missing 'model_state_dict'"):
                load_model(path)

    def test_invalid_checkpoint_missing_metadata(self):
        """Checkpoint without metadata should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.pt"
            torch.save({"model_state_dict": {}}, path)

            with pytest.raises(ValueError, match="missing 'metadata'"):
                load_model(path)

    def test_invalid_checkpoint_missing_architecture(self):
        """Checkpoint without architecture should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.pt"
            torch.save(
                {
                    "model_state_dict": {},
                    "metadata": {
                        "version": "v1",
                        "created_at": "2024-01-01T00:00:00+00:00",
                    },
                },
                path,
            )

            with pytest.raises(ValueError, match="missing 'architecture'"):
                load_model(path)


class TestOptimizerState:
    """Tests for optimizer state preservation."""

    def test_optimizer_state_preservation(self):
        """Verify optimizer state can be saved and restored."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)

        # Perform a fake training step to modify optimizer state
        obs = torch.randn(32, 114)
        logits, values = network(obs)
        loss = logits.mean() + values.mean()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1", optimizer=optimizer)

            new_network = ActorCriticNetwork(observation_size=114, action_size=27)
            new_optimizer = torch.optim.Adam(new_network.parameters(), lr=3e-4)

            load_model(path, network=new_network, optimizer=new_optimizer)

            # Optimizer should have state populated
            assert len(new_optimizer.state) > 0

    def test_optimizer_none_when_not_saved(self):
        """Optimizer state should be None when not provided during save."""
        network = ActorCriticNetwork()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            _, checkpoint = load_model(path)

            assert checkpoint.optimizer_state_dict is None

    def test_optimizer_not_restored_when_not_provided(self):
        """Optimizer should not fail when not provided during load."""
        network = ActorCriticNetwork(observation_size=114, action_size=27)
        optimizer = torch.optim.Adam(network.parameters())

        obs = torch.randn(32, 114)
        logits, values = network(obs)
        loss = logits.mean() + values.mean()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1", optimizer=optimizer)

            # Load without optimizer - should not raise
            loaded_network, checkpoint = load_model(path)
            assert checkpoint.optimizer_state_dict is not None
            assert loaded_network is not None


class TestGetCheckpointInfo:
    """Tests for get_checkpoint_info function."""

    def test_get_checkpoint_info_returns_metadata(self):
        """Verify metadata can be read without loading full model."""
        network = ActorCriticNetwork()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="gen-003")

            info = get_checkpoint_info(path)

            assert info.version == "gen-003"
            assert info.observation_size == network.observation_size
            assert info.action_size == network.action_size

    def test_get_checkpoint_info_full_metadata(self):
        """Verify all metadata fields are returned."""
        network = ActorCriticNetwork(observation_size=100, action_size=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(
                network,
                path,
                version="gen-005",
                training_info={
                    "total_episodes": 1000,
                    "final_reward": 99.5,
                    "opponent_version": "gen-004",
                },
            )

            info = get_checkpoint_info(path)

            assert info.version == "gen-005"
            assert info.observation_size == 100
            assert info.action_size == 20
            assert info.total_episodes == 1000
            assert info.final_reward == 99.5
            assert info.opponent_version == "gen-004"

    def test_get_checkpoint_info_invalid_checkpoint(self):
        """get_checkpoint_info should raise on invalid checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.pt"
            torch.save({"invalid": "data"}, path)

            with pytest.raises(ValueError, match="missing 'metadata'"):
                get_checkpoint_info(path)


class TestDeviceTransfer:
    """Tests for device transfer functionality."""

    def test_load_onto_cpu(self):
        """Test loading model onto CPU explicitly."""
        network = ActorCriticNetwork()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            loaded, _ = load_model(path, device="cpu")
            assert next(loaded.parameters()).device.type == "cpu"

    def test_load_with_torch_device(self):
        """Test loading with torch.device object."""
        network = ActorCriticNetwork()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            device = torch.device("cpu")
            loaded, _ = load_model(path, device=device)
            assert next(loaded.parameters()).device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_onto_cuda(self):
        """Test loading model onto CUDA if available."""
        network = ActorCriticNetwork()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            loaded, _ = load_model(path, device="cuda")
            assert next(loaded.parameters()).device.type == "cuda"


class TestGenerateModelFilename:
    """Tests for generate_model_filename utility."""

    def test_basic_filename(self):
        """Test basic filename generation."""
        filename = generate_model_filename(prefix="ppo", version="gen-001")
        assert filename.startswith("ppo_gen-001_")
        assert filename.endswith(".pt")

    def test_filename_without_timestamp(self):
        """Test filename generation without timestamp."""
        filename = generate_model_filename(
            prefix="model", version="v2", include_timestamp=False
        )
        assert filename == "model_v2.pt"

    def test_custom_prefix(self):
        """Test custom prefix in filename."""
        filename = generate_model_filename(prefix="actor_critic", version="v1")
        assert filename.startswith("actor_critic_v1_")

    def test_timestamp_format(self):
        """Test that timestamp follows expected format."""
        filename = generate_model_filename(prefix="ppo", version="v1")
        # Format should be ppo_v1_YYYYMMDD-HHMMSS.pt
        parts = filename.replace(".pt", "").split("_")
        assert len(parts) == 3
        timestamp = parts[2]
        assert "-" in timestamp
        date_part, time_part = timestamp.split("-")
        assert len(date_part) == 8  # YYYYMMDD
        assert len(time_part) == 6  # HHMMSS


class TestDirectoryCreation:
    """Tests for automatic directory creation."""

    def test_creates_parent_directories(self):
        """save_model should create parent directories if they don't exist."""
        network = ActorCriticNetwork()

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "model.pt"
            assert not nested_path.parent.exists()

            save_model(network, nested_path, version="v1")

            assert nested_path.exists()
            assert nested_path.parent.exists()


class TestLoadedNetworkMode:
    """Tests for loaded network mode."""

    def test_loaded_network_in_eval_mode(self):
        """Loaded network should be in eval mode by default."""
        network = ActorCriticNetwork()
        network.train()  # Put in training mode

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            loaded, _ = load_model(path)

            assert not loaded.training  # Should be in eval mode


class TestDefaultValues:
    """Tests for default value handling."""

    def test_default_training_step(self):
        """training_step should default to 0."""
        network = ActorCriticNetwork()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            _, checkpoint = load_model(path)

            assert checkpoint.training_step == 0

    def test_default_total_timesteps(self):
        """total_timesteps should default to 0."""
        network = ActorCriticNetwork()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            _, checkpoint = load_model(path)

            assert checkpoint.total_timesteps == 0

    def test_default_training_info_fields(self):
        """Training info fields should have sensible defaults."""
        network = ActorCriticNetwork()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(network, path, version="v1")

            _, checkpoint = load_model(path)

            assert checkpoint.metadata.total_episodes == 0
            assert checkpoint.metadata.final_reward == 0.0
            assert checkpoint.metadata.opponent_version is None
