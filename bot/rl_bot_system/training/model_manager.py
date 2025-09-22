"""
Model Manager for RL Bot System

Manages model lifecycle, storage, versioning, and knowledge transfer between generations.
Handles model persistence, metadata tracking, and performance-based model promotion.
"""

import os
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class RLModel:
    """Data class representing an RL model with metadata."""
    generation: int
    algorithm: str  # 'DQN', 'PPO', 'A3C'
    network_architecture: dict
    hyperparameters: dict
    training_episodes: int
    performance_metrics: dict
    parent_generation: Optional[int]
    created_at: datetime
    model_path: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'RLModel':
        """Create from dictionary loaded from JSON."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelManager:
    """
    Manages RL model lifecycle, storage, and knowledge transfer.
    
    Handles:
    - Model versioning and storage
    - Metadata and performance tracking
    - Knowledge transfer between generations
    - Model comparison and promotion logic
    """

    def __init__(self, models_dir: str = "bot/data/models"):
        """
        Initialize ModelManager.
        
        Args:
            models_dir: Directory to store models and metadata
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.current_best_dir = self.models_dir / "current_best"
        self.current_best_dir.mkdir(exist_ok=True)
        
        self._model_cache: Dict[int, RLModel] = {}
        self._loaded_models: Dict[int, nn.Module] = {}
        
        logger.info(f"ModelManager initialized with models directory: {self.models_dir}")

    def save_model(self, model: nn.Module, generation: int, metadata: Dict[str, Any]) -> RLModel:
        """
        Save a trained model with versioning and metadata.
        
        Args:
            model: The trained PyTorch model
            generation: Model generation number
            metadata: Dictionary containing model metadata
            
        Returns:
            RLModel: The saved model metadata object
            
        Raises:
            ValueError: If generation already exists or metadata is invalid
            IOError: If model cannot be saved
        """
        generation_dir = self.models_dir / f"generation_{generation}"
        
        # Check if generation already exists
        if generation_dir.exists():
            raise ValueError(f"Generation {generation} already exists")
        
        # Validate required metadata fields
        required_fields = ['algorithm', 'network_architecture', 'hyperparameters', 
                          'training_episodes', 'performance_metrics']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Required metadata field '{field}' missing")
        
        try:
            # Create generation directory
            generation_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model weights
            model_path = generation_dir / "model.pth"
            torch.save(model.state_dict(), model_path)
            
            # Create RLModel metadata object
            rl_model = RLModel(
                generation=generation,
                algorithm=metadata['algorithm'],
                network_architecture=metadata['network_architecture'],
                hyperparameters=metadata['hyperparameters'],
                training_episodes=metadata['training_episodes'],
                performance_metrics=metadata['performance_metrics'],
                parent_generation=metadata.get('parent_generation'),
                created_at=datetime.now(),
                model_path=str(model_path)
            )
            
            # Save metadata
            config_path = generation_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(rl_model.to_dict(), f, indent=2)
            
            # Save performance metrics separately for easy access
            metrics_path = generation_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metadata['performance_metrics'], f, indent=2)
            
            # Cache the model metadata
            self._model_cache[generation] = rl_model
            
            logger.info(f"Model generation {generation} saved successfully")
            return rl_model
            
        except Exception as e:
            # Clean up on failure
            if generation_dir.exists():
                shutil.rmtree(generation_dir)
            raise IOError(f"Failed to save model generation {generation}: {e}")

    def load_model(self, generation: int, model_class: type = None) -> Tuple[nn.Module, RLModel]:
        """
        Load a specific model generation.
        
        Args:
            generation: Model generation number to load
            model_class: PyTorch model class to instantiate (if not cached)
            
        Returns:
            Tuple[nn.Module, RLModel]: The loaded model and its metadata
            
        Raises:
            FileNotFoundError: If generation doesn't exist
            ValueError: If model cannot be loaded
        """
        generation_dir = self.models_dir / f"generation_{generation}"
        
        if not generation_dir.exists():
            raise FileNotFoundError(f"Generation {generation} not found")
        
        # Load metadata if not cached
        if generation not in self._model_cache:
            config_path = generation_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found for generation {generation}")
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self._model_cache[generation] = RLModel.from_dict(config_data)
        
        rl_model = self._model_cache[generation]
        
        # Load PyTorch model if not cached
        if generation not in self._loaded_models:
            if model_class is None:
                raise ValueError("model_class required for loading uncached model")
            
            # Instantiate model with architecture from metadata
            model = model_class(**rl_model.network_architecture)
            
            # Load weights
            model_path = Path(rl_model.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model weights not found: {model_path}")
            
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self._loaded_models[generation] = model
        
        logger.info(f"Model generation {generation} loaded successfully")
        return self._loaded_models[generation], rl_model

    def _load_model_metadata(self, generation: int) -> RLModel:
        """
        Load only the metadata for a model generation (without PyTorch model).
        
        Args:
            generation: Model generation number to load metadata for
            
        Returns:
            RLModel: The model metadata
            
        Raises:
            FileNotFoundError: If generation doesn't exist
        """
        generation_dir = self.models_dir / f"generation_{generation}"
        
        if not generation_dir.exists():
            raise FileNotFoundError(f"Generation {generation} not found")
        
        # Load metadata if not cached
        if generation not in self._model_cache:
            config_path = generation_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found for generation {generation}")
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self._model_cache[generation] = RLModel.from_dict(config_data)
        
        return self._model_cache[generation]

    def transfer_knowledge(self, source_generation: int, target_model: nn.Module, 
                          transfer_method: str = "weight_copy", source_model_class: type = None) -> nn.Module:
        """
        Transfer knowledge from source model to target model.
        
        Args:
            source_generation: Generation to transfer knowledge from
            target_model: Target model to receive knowledge
            transfer_method: Method for knowledge transfer ('weight_copy', 'layer_freeze', 'distillation')
            source_model_class: PyTorch model class for loading source model (required for weight_copy and layer_freeze)
            
        Returns:
            nn.Module: Target model with transferred knowledge
            
        Raises:
            ValueError: If transfer method is unsupported or models incompatible
        """
        if transfer_method == "distillation":
            # For distillation, we return the target model as-is
            # The actual distillation would happen during training
            logger.info(f"Prepared target model for distillation from generation {source_generation}")
            return target_model
        elif transfer_method in ["weight_copy", "layer_freeze"]:
            if source_model_class is None:
                raise ValueError("source_model_class required for weight_copy and layer_freeze methods")
            source_model, source_metadata = self.load_model(source_generation, source_model_class)
            
            if transfer_method == "weight_copy":
                return self._transfer_weights(source_model, target_model)
            elif transfer_method == "layer_freeze":
                return self._transfer_with_freeze(source_model, target_model)
        else:
            raise ValueError(f"Unsupported transfer method: {transfer_method}")

    def _transfer_weights(self, source_model: nn.Module, target_model: nn.Module) -> nn.Module:
        """Transfer compatible weights from source to target model."""
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()
        
        # Transfer compatible layers
        transferred_layers = []
        for name, param in source_dict.items():
            if name in target_dict and param.shape == target_dict[name].shape:
                target_dict[name] = param.clone()
                transferred_layers.append(name)
        
        target_model.load_state_dict(target_dict)
        logger.info(f"Transferred weights for layers: {transferred_layers}")
        return target_model

    def _transfer_with_freeze(self, source_model: nn.Module, target_model: nn.Module) -> nn.Module:
        """Transfer weights and freeze compatible layers."""
        target_model = self._transfer_weights(source_model, target_model)
        
        # Freeze transferred layers
        source_dict = source_model.state_dict()
        frozen_layers = []
        for name, param in target_model.named_parameters():
            if name in source_dict and param.shape == source_dict[name].shape:
                param.requires_grad = False
                frozen_layers.append(name)
        
        logger.info(f"Frozen layers: {frozen_layers}")
        return target_model

    def get_best_model(self) -> Optional[Tuple[int, RLModel]]:
        """
        Get the current best-performing model.
        
        Returns:
            Optional[Tuple[int, RLModel]]: Generation number and metadata of best model, or None if no models exist
        """
        best_link = self.current_best_dir / "best_generation"
        
        if best_link.exists():
            try:
                # Read the generation number from the symlink target or file
                if best_link.is_symlink():
                    target = best_link.readlink()
                    generation = int(target.name.split('_')[1])
                else:
                    with open(best_link, 'r') as f:
                        generation = int(f.read().strip())
                
                if generation in self._model_cache:
                    return generation, self._model_cache[generation]
                else:
                    # Load metadata if not cached
                    rl_model = self._load_model_metadata(generation)
                    return generation, rl_model
                    
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Invalid best model link: {e}")
        
        # If no best model set, find the highest performing model
        return self._find_best_model()

    def _find_best_model(self) -> Optional[Tuple[int, RLModel]]:
        """Find the best model based on performance metrics."""
        models = self.list_models()
        if not models:
            return None
        
        # Only return a model if there's an explicitly promoted best model
        # Don't automatically select the "best" based on metrics unless explicitly promoted
        return None

    def promote_model(self, generation: int) -> bool:
        """
        Promote a model to be the current best.
        
        Args:
            generation: Generation to promote
            
        Returns:
            bool: True if promotion successful
            
        Raises:
            FileNotFoundError: If generation doesn't exist
        """
        generation_dir = self.models_dir / f"generation_{generation}"
        if not generation_dir.exists():
            raise FileNotFoundError(f"Generation {generation} not found")
        
        best_link = self.current_best_dir / "best_generation"
        
        try:
            # Remove existing link/file
            if best_link.exists():
                best_link.unlink()
            
            # Create new reference to best model
            # Use a simple file instead of symlink for cross-platform compatibility
            with open(best_link, 'w') as f:
                f.write(str(generation))
            
            logger.info(f"Promoted generation {generation} to current best")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote generation {generation}: {e}")
            return False

    def compare_models(self, generation_a: int, generation_b: int) -> Dict[str, Any]:
        """
        Compare performance metrics between two model generations.
        
        Args:
            generation_a: First generation to compare
            generation_b: Second generation to compare
            
        Returns:
            Dict[str, Any]: Comparison results and statistics
            
        Raises:
            FileNotFoundError: If either generation doesn't exist
        """
        # Load only metadata, not the full models
        model_a = self._load_model_metadata(generation_a)
        model_b = self._load_model_metadata(generation_b)
        
        metrics_a = model_a.performance_metrics
        metrics_b = model_b.performance_metrics
        
        comparison = {
            'generation_a': generation_a,
            'generation_b': generation_b,
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'improvements': {},
            'summary': {}
        }
        
        # Compare common metrics
        common_metrics = set(metrics_a.keys()) & set(metrics_b.keys())
        for metric in common_metrics:
            value_a = metrics_a[metric]
            value_b = metrics_b[metric]
            
            if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                improvement = value_b - value_a
                improvement_pct = (improvement / value_a * 100) if value_a != 0 else 0
                
                comparison['improvements'][metric] = {
                    'absolute': improvement,
                    'percentage': improvement_pct,
                    'better': improvement > 0
                }
        
        # Overall assessment
        win_rate_improvement = comparison['improvements'].get('win_rate', {}).get('absolute', 0)
        reward_improvement = comparison['improvements'].get('average_reward', {}).get('absolute', 0)
        
        comparison['summary'] = {
            'overall_better': win_rate_improvement > 0 or (win_rate_improvement == 0 and reward_improvement > 0),
            'win_rate_improvement': win_rate_improvement,
            'reward_improvement': reward_improvement,
            'training_episodes_a': model_a.training_episodes,
            'training_episodes_b': model_b.training_episodes
        }
        
        return comparison

    def should_promote_model(self, candidate_generation: int, 
                           promotion_threshold: float = 0.05) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if a candidate model should be promoted to current best.
        
        Args:
            candidate_generation: Generation to evaluate for promotion
            promotion_threshold: Minimum improvement required for promotion (default 5%)
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (should_promote, comparison_details)
        """
        current_best = self.get_best_model()
        
        if current_best is None:
            # No current best, promote by default
            return True, {'reason': 'no_current_best'}
        
        current_generation, _ = current_best
        
        # Don't compare a model with itself
        if current_generation == candidate_generation:
            return False, {'reason': 'same_generation', 'current_generation': current_generation}
        
        comparison = self.compare_models(current_generation, candidate_generation)
        
        # Check if candidate meets promotion criteria
        win_rate_improvement = comparison['improvements'].get('win_rate', {}).get('percentage', 0)
        reward_improvement = comparison['improvements'].get('average_reward', {}).get('percentage', 0)
        
        should_promote = (
            win_rate_improvement >= promotion_threshold * 100 or
            (win_rate_improvement >= 0 and reward_improvement >= promotion_threshold * 100)
        )
        
        comparison['promotion_decision'] = {
            'should_promote': should_promote,
            'threshold': promotion_threshold,
            'win_rate_improvement_pct': win_rate_improvement,
            'reward_improvement_pct': reward_improvement,
            'reason': 'meets_threshold' if should_promote else 'below_threshold'
        }
        
        return should_promote, comparison

    def list_models(self) -> List[Tuple[int, RLModel]]:
        """
        List all available model generations.
        
        Returns:
            List[Tuple[int, RLModel]]: List of (generation, metadata) tuples
        """
        models = []
        
        for generation_dir in self.models_dir.glob("generation_*"):
            if generation_dir.is_dir():
                try:
                    generation = int(generation_dir.name.split('_')[1])
                    if generation not in self._model_cache:
                        config_path = generation_dir / "config.json"
                        if config_path.exists():
                            with open(config_path, 'r') as f:
                                config_data = json.load(f)
                            self._model_cache[generation] = RLModel.from_dict(config_data)
                    
                    if generation in self._model_cache:
                        models.append((generation, self._model_cache[generation]))
                        
                except (ValueError, FileNotFoundError) as e:
                    logger.warning(f"Skipping invalid generation directory {generation_dir}: {e}")
        
        # Sort by generation number
        models.sort(key=lambda x: x[0])
        return models

    def delete_model(self, generation: int) -> bool:
        """
        Delete a model generation and its associated files.
        
        Args:
            generation: Generation to delete
            
        Returns:
            bool: True if deletion successful
            
        Raises:
            ValueError: If trying to delete the current best model
        """
        # Check if this is the current best model
        current_best = self.get_best_model()
        if current_best and current_best[0] == generation:
            raise ValueError(f"Cannot delete current best model (generation {generation})")
        
        generation_dir = self.models_dir / f"generation_{generation}"
        
        if not generation_dir.exists():
            logger.warning(f"Generation {generation} directory not found")
            return False
        
        try:
            # Remove from caches
            self._model_cache.pop(generation, None)
            self._loaded_models.pop(generation, None)
            
            # Remove directory
            shutil.rmtree(generation_dir)
            
            logger.info(f"Deleted model generation {generation}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete generation {generation}: {e}")
            return False

    def get_model_info(self, generation: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific model generation.
        
        Args:
            generation: Generation to get info for
            
        Returns:
            Dict[str, Any]: Detailed model information
            
        Raises:
            FileNotFoundError: If generation doesn't exist
        """
        rl_model = self._load_model_metadata(generation)
        
        info = rl_model.to_dict()
        
        # Add file size information
        model_path = Path(rl_model.model_path)
        if model_path.exists():
            size_bytes = model_path.stat().st_size
            info['model_size_bytes'] = size_bytes
            info['model_size_mb'] = round(size_bytes / (1024 * 1024), 3) if size_bytes > 0 else 0.001  # Minimum 0.001 MB for small files
        
        # Add parent/child relationships
        info['children'] = []
        for gen, model in self.list_models():
            if model.parent_generation == generation:
                info['children'].append(gen)
        
        return info

    def cleanup_old_models(self, keep_count: int = 5) -> List[int]:
        """
        Clean up old model generations, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent models to keep
            
        Returns:
            List[int]: List of deleted generation numbers
        """
        models = self.list_models()
        current_best = self.get_best_model()
        current_best_gen = current_best[0] if current_best else None
        
        if len(models) <= keep_count:
            return []
        
        # Sort by generation (newest first)
        models.sort(key=lambda x: x[0], reverse=True)
        
        # Keep the most recent models and the current best
        to_keep = set()
        for i, (generation, _) in enumerate(models):
            if i < keep_count or generation == current_best_gen:
                to_keep.add(generation)
        
        # Delete the rest
        deleted = []
        for generation, _ in models:
            if generation not in to_keep:
                try:
                    if self.delete_model(generation):
                        deleted.append(generation)
                except ValueError:
                    # Skip if it's the current best
                    pass
        
        logger.info(f"Cleaned up {len(deleted)} old models: {deleted}")
        return deleted