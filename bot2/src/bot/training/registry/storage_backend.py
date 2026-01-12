"""File-based storage backend for the model registry.

This module provides the file system operations for storing and retrieving
model checkpoints and metadata with proper indexing for fast queries.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import filelock

from bot.training.registry.model_metadata import ModelMetadata


class RegistryIndex:
    """Index of all registered models for fast metadata queries.

    The index is stored as a JSON file and contains metadata for all
    registered models. It enables fast queries without loading model weights.

    Attributes:
        models: Dictionary mapping model_id to ModelMetadata
        last_updated: Timestamp of last index update
    """

    def __init__(
        self,
        models: dict[str, ModelMetadata] | None = None,
        last_updated: datetime | None = None,
    ):
        """Initialize the registry index.

        Args:
            models: Dictionary mapping model_id to metadata
            last_updated: Timestamp of last update
        """
        self.models = models or {}
        self.last_updated = last_updated or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Serialize index to dictionary for JSON storage."""
        return {
            "models": {
                model_id: metadata.model_dump(mode="json")
                for model_id, metadata in self.models.items()
            },
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RegistryIndex":
        """Deserialize index from dictionary."""
        models = {
            model_id: ModelMetadata.model_validate(meta)
            for model_id, meta in data.get("models", {}).items()
        }
        last_updated = datetime.fromisoformat(data["last_updated"])
        return cls(models=models, last_updated=last_updated)


class StorageBackend:
    """File-based storage backend for the model registry.

    This backend provides persistent storage for model checkpoints and
    metadata using the filesystem with proper locking for thread safety.

    Directory structure:
        registry_path/
        ├── index.json           # Model index with all metadata
        ├── index.json.lock      # Lock file for concurrent access
        └── models/
            ├── ppo_gen_000/
            │   ├── model.pt     # PyTorch model checkpoint
            │   └── metadata.json
            ├── ppo_gen_001/
            │   ├── model.pt
            │   └── metadata.json
            └── ...
    """

    INDEX_FILENAME = "index.json"
    MODELS_DIR = "models"
    MODEL_FILENAME = "model.pt"
    METADATA_FILENAME = "metadata.json"
    LOCK_TIMEOUT = 10.0  # seconds

    def __init__(self, registry_path: str | Path):
        """Initialize storage backend.

        Args:
            registry_path: Root directory for the registry storage
        """
        self.registry_path = Path(registry_path)
        self._index_path = self.registry_path / self.INDEX_FILENAME
        self._models_dir = self.registry_path / self.MODELS_DIR
        self._lock_path = self.registry_path / f"{self.INDEX_FILENAME}.lock"
        self._lock = filelock.FileLock(str(self._lock_path), timeout=self.LOCK_TIMEOUT)

    def initialize(self) -> None:
        """Initialize the storage directory structure.

        Creates the registry directory and initializes an empty index
        if they don't exist.
        """
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._models_dir.mkdir(exist_ok=True)

        if not self._index_path.exists():
            self._write_index(RegistryIndex())

    def _read_index(self) -> RegistryIndex:
        """Read the registry index from disk.

        Returns:
            RegistryIndex with all registered models

        Raises:
            FileNotFoundError: If index file doesn't exist
        """
        if not self._index_path.exists():
            return RegistryIndex()

        with open(self._index_path, encoding="utf-8") as f:
            data = json.load(f)
        return RegistryIndex.from_dict(data)

    def _write_index(self, index: RegistryIndex) -> None:
        """Write the registry index to disk.

        Args:
            index: RegistryIndex to write
        """
        index.last_updated = datetime.now(timezone.utc)
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(index.to_dict(), f, indent=2)

    def get_model_dir(self, model_id: str) -> Path:
        """Get the directory path for a model.

        Args:
            model_id: Unique model identifier

        Returns:
            Path to the model's directory
        """
        return self._models_dir / model_id

    def get_checkpoint_path(self, model_id: str) -> Path:
        """Get the checkpoint file path for a model.

        Args:
            model_id: Unique model identifier

        Returns:
            Path to the model's checkpoint file
        """
        return self.get_model_dir(model_id) / self.MODEL_FILENAME

    def get_metadata_path(self, model_id: str) -> Path:
        """Get the metadata file path for a model.

        Args:
            model_id: Unique model identifier

        Returns:
            Path to the model's metadata file
        """
        return self.get_model_dir(model_id) / self.METADATA_FILENAME

    def model_exists(self, model_id: str) -> bool:
        """Check if a model exists in the registry.

        Args:
            model_id: Unique model identifier

        Returns:
            True if model exists, False otherwise
        """
        return self.get_checkpoint_path(model_id).exists()

    def save_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata and update the index.

        This method is thread-safe and uses file locking to prevent
        concurrent writes.

        Args:
            metadata: ModelMetadata to save
        """
        model_dir = self.get_model_dir(metadata.model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save individual metadata file
        metadata_path = self.get_metadata_path(metadata.model_id)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2)

        # Update the index with locking
        with self._lock:
            index = self._read_index()
            index.models[metadata.model_id] = metadata
            self._write_index(index)

    def load_metadata(self, model_id: str) -> ModelMetadata | None:
        """Load metadata for a specific model.

        Args:
            model_id: Unique model identifier

        Returns:
            ModelMetadata if found, None otherwise
        """
        metadata_path = self.get_metadata_path(model_id)
        if not metadata_path.exists():
            return None

        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)
        return ModelMetadata.model_validate(data)

    def list_all_metadata(self) -> list[ModelMetadata]:
        """List metadata for all registered models.

        Uses the index for fast access without loading individual files.

        Returns:
            List of ModelMetadata for all registered models
        """
        with self._lock:
            index = self._read_index()
        return list(index.models.values())

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the registry.

        This method is thread-safe and removes both the model files
        and the index entry.

        Args:
            model_id: Unique model identifier

        Returns:
            True if model was deleted, False if not found
        """
        model_dir = self.get_model_dir(model_id)
        if not model_dir.exists():
            return False

        # Remove directory and contents
        shutil.rmtree(model_dir)

        # Update the index with locking
        with self._lock:
            index = self._read_index()
            if model_id in index.models:
                del index.models[model_id]
                self._write_index(index)

        return True

    def get_next_generation(self) -> int:
        """Get the next available generation number.

        Returns:
            Next generation number (0 if no models registered)
        """
        with self._lock:
            index = self._read_index()

        if not index.models:
            return 0

        max_gen = max(meta.generation for meta in index.models.values())
        return max_gen + 1
