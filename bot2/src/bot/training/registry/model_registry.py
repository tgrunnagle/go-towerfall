"""Model Registry for storing, versioning, and retrieving trained PPO models.

This module provides the main ModelRegistry class for managing trained models
with associated metadata as part of the successive training pipeline.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from bot.agent.network import ActorCriticNetwork
from bot.agent.serialization import load_model, save_model
from bot.training.registry.model_metadata import (
    ModelMetadata,
    NetworkArchitecture,
    TrainingMetrics,
)
from bot.training.registry.storage_backend import StorageBackend

if TYPE_CHECKING:
    from bot.agent.serialization import ModelCheckpoint


class ModelNotFoundError(Exception):
    """Raised when a requested model is not found in the registry."""


class ModelAlreadyExistsError(Exception):
    """Raised when attempting to register a model with an existing ID."""


class ModelRegistry:
    """Registry for storing and retrieving trained PPO models.

    The model registry provides:
    - Persistent storage of trained models with versioning
    - Metadata tracking for performance comparison
    - Generation-based retrieval for successive training
    - Thread-safe operations with file locking

    Example usage:
        registry = ModelRegistry("/path/to/registry")

        # Register a new model
        model_id = registry.register_model(
            model=trained_network,
            generation=0,
            opponent_model_id=None,
            training_metrics=metrics,
            hyperparameters=config.model_dump(),
            training_duration_seconds=3600.0,
        )

        # Retrieve a model
        network, metadata = registry.get_model(model_id)

        # Get the best model for comparison
        best_network, best_meta = registry.get_best_model()

    Attributes:
        registry_path: Path to the registry storage directory
    """

    MODEL_ID_FORMAT = "ppo_gen_{generation:03d}"

    def __init__(self, registry_path: str | Path):
        """Initialize the model registry.

        Args:
            registry_path: Path to the registry storage directory.
                           Will be created if it doesn't exist.
        """
        self.registry_path = Path(registry_path)
        self._backend = StorageBackend(self.registry_path)
        self._backend.initialize()

    def _generate_model_id(self, generation: int) -> str:
        """Generate a model ID for a given generation.

        Args:
            generation: Generation number

        Returns:
            Model ID in format "ppo_gen_XXX"
        """
        return self.MODEL_ID_FORMAT.format(generation=generation)

    def register_model(
        self,
        model: ActorCriticNetwork,
        generation: int,
        opponent_model_id: str | None,
        training_metrics: TrainingMetrics,
        hyperparameters: dict,
        training_duration_seconds: float,
        optimizer: torch.optim.Optimizer | None = None,
        training_step: int = 0,
        notes: str | None = None,
    ) -> str:
        """Register a new model in the registry.

        Saves the model weights to disk and stores associated metadata
        in the registry index.

        Args:
            model: Trained ActorCriticNetwork to register
            generation: Generation number (0 for first model)
            opponent_model_id: ID of model trained against (None for gen 0)
            training_metrics: Performance metrics from training
            hyperparameters: PPO hyperparameters used during training
            training_duration_seconds: Total training time in seconds
            optimizer: Optional optimizer to save for training resumption
            training_step: Current training step/update number
            notes: Optional human-readable notes

        Returns:
            The assigned model_id

        Raises:
            ModelAlreadyExistsError: If model for this generation already exists
        """
        model_id = self._generate_model_id(generation)

        if self._backend.model_exists(model_id):
            raise ModelAlreadyExistsError(
                f"Model with ID '{model_id}' already exists in registry"
            )

        # Create checkpoint path relative to models directory
        checkpoint_rel_path = f"models/{model_id}/model.pt"
        checkpoint_path = self._backend.get_checkpoint_path(model_id)

        # Save the model checkpoint using existing serialization
        training_info = {
            "total_episodes": training_metrics.total_episodes,
            "final_reward": training_metrics.average_reward,
            "opponent_version": opponent_model_id,
        }

        save_model(
            network=model,
            path=checkpoint_path,
            version=model_id,
            training_step=training_step,
            total_timesteps=training_metrics.total_timesteps,
            optimizer=optimizer,
            training_info=training_info,
            hyperparameters=hyperparameters,
        )

        # Create and store metadata
        architecture = NetworkArchitecture(
            observation_size=model.observation_size,
            action_size=model.action_size,
            hidden_size=model.hidden_size,
            actor_hidden=model.actor_hidden,
            critic_hidden=model.critic_hidden,
        )

        metadata = ModelMetadata(
            model_id=model_id,
            generation=generation,
            created_at=datetime.now(timezone.utc),
            training_duration_seconds=training_duration_seconds,
            opponent_model_id=opponent_model_id,
            training_metrics=training_metrics,
            hyperparameters=hyperparameters,
            architecture=architecture,
            checkpoint_path=checkpoint_rel_path,
            notes=notes,
        )

        self._backend.save_metadata(metadata)

        return model_id

    def get_model(
        self,
        model_id: str,
        device: torch.device | str = "cpu",
    ) -> tuple[ActorCriticNetwork, ModelMetadata]:
        """Load a model and its metadata by ID.

        Args:
            model_id: Unique model identifier
            device: Device to load the model onto

        Returns:
            Tuple of (ActorCriticNetwork, ModelMetadata)

        Raises:
            ModelNotFoundError: If model is not found
        """
        metadata = self._backend.load_metadata(model_id)
        if metadata is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")

        checkpoint_path = self._backend.get_checkpoint_path(model_id)
        network, _ = load_model(checkpoint_path, device=device)

        return network, metadata

    def get_latest_model(
        self,
        device: torch.device | str = "cpu",
    ) -> tuple[ActorCriticNetwork, ModelMetadata] | None:
        """Get the most recently registered model.

        Args:
            device: Device to load the model onto

        Returns:
            Tuple of (ActorCriticNetwork, ModelMetadata) or None if empty
        """
        all_metadata = self._backend.list_all_metadata()
        if not all_metadata:
            return None

        # Sort by created_at and get the latest
        latest = max(all_metadata, key=lambda m: m.created_at)
        return self.get_model(latest.model_id, device=device)

    def get_model_by_generation(
        self,
        generation: int,
        device: torch.device | str = "cpu",
    ) -> tuple[ActorCriticNetwork, ModelMetadata] | None:
        """Get model for a specific generation.

        Args:
            generation: Generation number to retrieve
            device: Device to load the model onto

        Returns:
            Tuple of (ActorCriticNetwork, ModelMetadata) or None if not found
        """
        model_id = self._generate_model_id(generation)
        try:
            return self.get_model(model_id, device=device)
        except ModelNotFoundError:
            return None

    def list_models(self) -> list[ModelMetadata]:
        """List all registered models with their metadata.

        Returns:
            List of ModelMetadata for all registered models,
            sorted by generation
        """
        all_metadata = self._backend.list_all_metadata()
        return sorted(all_metadata, key=lambda m: m.generation)

    def get_best_model(
        self,
        device: torch.device | str = "cpu",
    ) -> tuple[ActorCriticNetwork, ModelMetadata] | None:
        """Get the model with the highest kills/deaths ratio.

        This is the primary comparison metric for the successive
        training pipeline.

        Args:
            device: Device to load the model onto

        Returns:
            Tuple of (ActorCriticNetwork, ModelMetadata) or None if empty
        """
        all_metadata = self._backend.list_all_metadata()
        if not all_metadata:
            return None

        best = max(
            all_metadata,
            key=lambda m: m.training_metrics.kills_deaths_ratio,
        )
        return self.get_model(best.model_id, device=device)

    def is_better_than(self, model_id_a: str, model_id_b: str) -> bool:
        """Compare two models based on kills/deaths ratio.

        Args:
            model_id_a: First model to compare
            model_id_b: Second model to compare

        Returns:
            True if model_a has higher K/D ratio than model_b

        Raises:
            ModelNotFoundError: If either model is not found
        """
        metadata_a = self._backend.load_metadata(model_id_a)
        metadata_b = self._backend.load_metadata(model_id_b)

        if metadata_a is None:
            raise ModelNotFoundError(f"Model '{model_id_a}' not found in registry")
        if metadata_b is None:
            raise ModelNotFoundError(f"Model '{model_id_b}' not found in registry")

        return (
            metadata_a.training_metrics.kills_deaths_ratio
            > metadata_b.training_metrics.kills_deaths_ratio
        )

    def delete_model(self, model_id: str) -> bool:
        """Remove a model from the registry.

        Deletes both the checkpoint files and metadata.

        Args:
            model_id: Unique model identifier

        Returns:
            True if model was deleted, False if not found
        """
        return self._backend.delete_model(model_id)

    def get_next_generation(self) -> int:
        """Get the next available generation number.

        Useful for determining the generation number for a new model
        in the successive training pipeline.

        Returns:
            Next generation number (0 if registry is empty)
        """
        return self._backend.get_next_generation()

    def get_checkpoint_data(
        self, model_id: str, device: torch.device | str = "cpu"
    ) -> "ModelCheckpoint":
        """Get the full checkpoint data for a model.

        Useful for accessing optimizer state for continued training.

        Args:
            model_id: Unique model identifier
            device: Device to load the checkpoint onto

        Returns:
            ModelCheckpoint with full checkpoint data

        Raises:
            ModelNotFoundError: If model is not found
        """
        metadata = self._backend.load_metadata(model_id)
        if metadata is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")

        checkpoint_path = self._backend.get_checkpoint_path(model_id)
        _, checkpoint = load_model(checkpoint_path, device=device)

        return checkpoint

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Get metadata for a model without loading weights.

        Useful for quick metadata queries without the overhead
        of loading the full model.

        Args:
            model_id: Unique model identifier

        Returns:
            ModelMetadata for the model

        Raises:
            ModelNotFoundError: If model is not found
        """
        metadata = self._backend.load_metadata(model_id)
        if metadata is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")
        return metadata
