"""Model serialization utilities for PPO checkpoints.

This module provides functionality for saving and loading trained PPO models
with versioning support. It enables persisting trained models to disk, loading
them for inference or continued training, and tracking model versions as part
of the successive training pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from bot.agent.network import ActorCriticNetwork


@dataclass
class CheckpointMetadata:
    """Metadata for a saved model checkpoint.

    Attributes:
        version: Version string for this model (e.g., "v1.0.0", "gen-003")
        created_at: Timestamp when the checkpoint was created
        observation_size: Dimension of observation vector
        action_size: Number of discrete actions
        hidden_size: Size of shared feature extractor layers
        actor_hidden: Size of actor head hidden layer
        critic_hidden: Size of critic head hidden layer
        total_episodes: Number of episodes completed during training
        final_reward: Final mean reward achieved
        opponent_version: Version of opponent used during training
        extra: Additional custom metadata
    """

    version: str
    created_at: datetime
    observation_size: int
    action_size: int
    hidden_size: int
    actor_hidden: int
    critic_hidden: int
    total_episodes: int = 0
    final_reward: float = 0.0
    opponent_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCheckpoint:
    """Complete model checkpoint including weights and metadata.

    Attributes:
        model_state_dict: PyTorch state dict with network weights
        optimizer_state_dict: Optional optimizer state for training resumption
        training_step: Current training step/update number
        total_timesteps: Total environment steps collected
        metadata: Checkpoint metadata including version and architecture
        hyperparameters: Hyperparameters used during training
    """

    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None
    training_step: int
    total_timesteps: int
    metadata: CheckpointMetadata
    hyperparameters: dict[str, Any] = field(default_factory=dict)


def save_model(
    network: ActorCriticNetwork,
    path: str | Path,
    version: str,
    training_step: int = 0,
    total_timesteps: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    training_info: dict | None = None,
    hyperparameters: dict | None = None,
) -> Path:
    """Save a trained model checkpoint to disk.

    Args:
        network: The ActorCriticNetwork to save
        path: File path to save the checkpoint (.pt extension recommended)
        version: Version string for this model (e.g., "v1.0.0", "gen-003")
        training_step: Current training step/update number
        total_timesteps: Total environment steps collected
        optimizer: Optional optimizer to save state for training resumption
        training_info: Optional dict with training metrics (episodes, reward, opponent)
        hyperparameters: Optional dict of hyperparameters used during training

    Returns:
        Path to the saved checkpoint file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    training_info = training_info or {}

    checkpoint = {
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "training_step": training_step,
        "total_timesteps": total_timesteps,
        "metadata": {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "architecture": {
                "observation_size": network.observation_size,
                "action_size": network.action_size,
                "hidden_size": network.hidden_size,
                "actor_hidden": network.actor_hidden,
                "critic_hidden": network.critic_hidden,
            },
            "training_info": training_info,
        },
        "hyperparameters": hyperparameters or {},
    }

    torch.save(checkpoint, path)
    return path


def load_model(
    path: str | Path,
    device: torch.device | str = "cpu",
    network: ActorCriticNetwork | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[ActorCriticNetwork, ModelCheckpoint]:
    """Load a model checkpoint from disk.

    Args:
        path: Path to the checkpoint file
        device: Device to load the model onto ("cpu", "cuda", or torch.device)
        network: Optional existing network to load weights into.
                 If None, creates a new network with saved architecture.
        optimizer: Optional optimizer to restore state into

    Returns:
        Tuple of (network, checkpoint_data)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint format is invalid or incompatible
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Validate checkpoint format
    required_keys = ["model_state_dict", "metadata"]
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Invalid checkpoint: missing '{key}' key")

    # Validate metadata structure
    if "architecture" not in checkpoint["metadata"]:
        raise ValueError("Invalid checkpoint: missing 'architecture' in metadata")

    # Extract architecture from metadata
    arch = checkpoint["metadata"]["architecture"]

    # Create or validate network
    if network is None:
        network = ActorCriticNetwork(
            observation_size=arch["observation_size"],
            action_size=arch["action_size"],
            hidden_size=arch["hidden_size"],
            actor_hidden=arch["actor_hidden"],
            critic_hidden=arch["critic_hidden"],
        )
    else:
        # Validate network matches checkpoint architecture
        mismatches = []
        if network.observation_size != arch["observation_size"]:
            mismatches.append(
                f"observation_size: expected {arch['observation_size']}, "
                f"got {network.observation_size}"
            )
        if network.action_size != arch["action_size"]:
            mismatches.append(
                f"action_size: expected {arch['action_size']}, "
                f"got {network.action_size}"
            )
        if network.hidden_size != arch["hidden_size"]:
            mismatches.append(
                f"hidden_size: expected {arch['hidden_size']}, "
                f"got {network.hidden_size}"
            )
        if network.actor_hidden != arch["actor_hidden"]:
            mismatches.append(
                f"actor_hidden: expected {arch['actor_hidden']}, "
                f"got {network.actor_hidden}"
            )
        if network.critic_hidden != arch["critic_hidden"]:
            mismatches.append(
                f"critic_hidden: expected {arch['critic_hidden']}, "
                f"got {network.critic_hidden}"
            )
        if mismatches:
            raise ValueError(f"Network architecture mismatch: {'; '.join(mismatches)}")

    # Load weights
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(device)
    network.eval()  # Set to evaluation mode by default

    # Restore optimizer if provided
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Parse metadata into dataclass
    meta = checkpoint["metadata"]
    training_info = meta.get("training_info", {})

    metadata = CheckpointMetadata(
        version=meta["version"],
        created_at=datetime.fromisoformat(meta["created_at"]),
        observation_size=arch["observation_size"],
        action_size=arch["action_size"],
        hidden_size=arch["hidden_size"],
        actor_hidden=arch["actor_hidden"],
        critic_hidden=arch["critic_hidden"],
        total_episodes=training_info.get("total_episodes", 0),
        final_reward=training_info.get("final_reward", 0.0),
        opponent_version=training_info.get("opponent_version"),
    )

    checkpoint_data = ModelCheckpoint(
        model_state_dict=checkpoint["model_state_dict"],
        optimizer_state_dict=checkpoint.get("optimizer_state_dict"),
        training_step=checkpoint.get("training_step", 0),
        total_timesteps=checkpoint.get("total_timesteps", 0),
        metadata=metadata,
        hyperparameters=checkpoint.get("hyperparameters", {}),
    )

    return network, checkpoint_data


def get_checkpoint_info(path: str | Path) -> CheckpointMetadata:
    """Read checkpoint metadata from a checkpoint file.

    Useful for listing available models or checking compatibility.

    Note:
        This function loads the full checkpoint file to extract metadata.
        For large models, consider caching the result if calling repeatedly.

    Args:
        path: Path to the checkpoint file

    Returns:
        CheckpointMetadata with checkpoint information

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint (this still loads weights, but we only return metadata)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Validate checkpoint format
    if "metadata" not in checkpoint:
        raise ValueError("Invalid checkpoint: missing 'metadata' key")
    if "architecture" not in checkpoint["metadata"]:
        raise ValueError("Invalid checkpoint: missing 'architecture' in metadata")

    meta = checkpoint["metadata"]
    arch = meta["architecture"]
    training_info = meta.get("training_info", {})

    return CheckpointMetadata(
        version=meta["version"],
        created_at=datetime.fromisoformat(meta["created_at"]),
        observation_size=arch["observation_size"],
        action_size=arch["action_size"],
        hidden_size=arch["hidden_size"],
        actor_hidden=arch["actor_hidden"],
        critic_hidden=arch["critic_hidden"],
        total_episodes=training_info.get("total_episodes", 0),
        final_reward=training_info.get("final_reward", 0.0),
        opponent_version=training_info.get("opponent_version"),
    )


def generate_model_filename(
    prefix: str = "ppo",
    version: str = "v1",
    include_timestamp: bool = True,
) -> str:
    """Generate a consistent model filename.

    Args:
        prefix: Filename prefix (e.g., "ppo", "model")
        version: Version string (e.g., "v1", "gen-001")
        include_timestamp: Whether to include a UTC timestamp in the filename

    Returns:
        Filename string with .pt extension
    """
    if include_timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{prefix}_{version}_{timestamp}.pt"
    return f"{prefix}_{version}.pt"
