"""
Tests for ModelManager class.

Tests model lifecycle management, versioning, knowledge transfer, and promotion logic.
"""

import os
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import torch
import torch.nn as nn

from rl_bot_system.training.model_manager import ModelManager, RLModel


class SimpleTestModel(nn.Module):
    """Simple neural network for testing."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TestRLModel:
    """Test RLModel data class."""
    
    def test_rl_model_creation(self):
        """Test RLModel creation and basic properties."""
        now = datetime.now()
        model = RLModel(
            generation=1,
            algorithm="DQN",
            network_architecture={"input_size": 10, "hidden_size": 20, "output_size": 5},
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
            training_episodes=1000,
            performance_metrics={"win_rate": 0.75, "average_reward": 100.5},
            parent_generation=None,
            created_at=now,
            model_path="/path/to/model.pth"
        )
        
        assert model.generation == 1
        assert model.algorithm == "DQN"
        assert model.network_architecture["input_size"] == 10
        assert model.hyperparameters["learning_rate"] == 0.001
        assert model.training_episodes == 1000
        assert model.performance_metrics["win_rate"] == 0.75
        assert model.parent_generation is None
        assert model.created_at == now
        assert model.model_path == "/path/to/model.pth"
    
    def test_rl_model_serialization(self):
        """Test RLModel to_dict and from_dict methods."""
        now = datetime.now()
        original = RLModel(
            generation=2,
            algorithm="PPO",
            network_architecture={"layers": [64, 32]},
            hyperparameters={"lr": 0.0003},
            training_episodes=500,
            performance_metrics={"reward": 85.2},
            parent_generation=1,
            created_at=now,
            model_path="/test/path.pth"
        )
        
        # Test serialization
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["generation"] == 2
        assert data["algorithm"] == "PPO"
        assert data["created_at"] == now.isoformat()
        
        # Test deserialization
        restored = RLModel.from_dict(data)
        assert restored.generation == original.generation
        assert restored.algorithm == original.algorithm
        assert restored.network_architecture == original.network_architecture
        assert restored.hyperparameters == original.hyperparameters
        assert restored.training_episodes == original.training_episodes
        assert restored.performance_metrics == original.performance_metrics
        assert restored.parent_generation == original.parent_generation
        assert restored.created_at == original.created_at
        assert restored.model_path == original.model_path


class TestModelManager:
    """Test ModelManager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model_manager(self, temp_dir):
        """Create ModelManager instance for testing."""
        return ModelManager(models_dir=temp_dir)
    
    @pytest.fixture
    def sample_model(self):
        """Create sample PyTorch model for testing."""
        return SimpleTestModel(input_size=10, hidden_size=20, output_size=5)
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return {
            "algorithm": "DQN",
            "network_architecture": {"input_size": 10, "hidden_size": 20, "output_size": 5},
            "hyperparameters": {"learning_rate": 0.001, "batch_size": 32, "gamma": 0.99},
            "training_episodes": 1000,
            "performance_metrics": {"win_rate": 0.75, "average_reward": 100.5, "episodes_won": 750},
            "parent_generation": None
        }
    
    def test_model_manager_initialization(self, temp_dir):
        """Test ModelManager initialization."""
        manager = ModelManager(models_dir=temp_dir)
        
        assert manager.models_dir == Path(temp_dir)
        assert manager.models_dir.exists()
        assert manager.current_best_dir.exists()
        assert isinstance(manager._model_cache, dict)
        assert isinstance(manager._loaded_models, dict)
    
    def test_save_model_success(self, model_manager, sample_model, sample_metadata):
        """Test successful model saving."""
        generation = 1
        
        # Save model
        rl_model = model_manager.save_model(sample_model, generation, sample_metadata)
        
        # Verify RLModel object
        assert isinstance(rl_model, RLModel)
        assert rl_model.generation == generation
        assert rl_model.algorithm == sample_metadata["algorithm"]
        assert rl_model.network_architecture == sample_metadata["network_architecture"]
        assert rl_model.hyperparameters == sample_metadata["hyperparameters"]
        assert rl_model.training_episodes == sample_metadata["training_episodes"]
        assert rl_model.performance_metrics == sample_metadata["performance_metrics"]
        assert rl_model.parent_generation == sample_metadata["parent_generation"]
        assert isinstance(rl_model.created_at, datetime)
        
        # Verify files were created
        generation_dir = model_manager.models_dir / f"generation_{generation}"
        assert generation_dir.exists()
        assert (generation_dir / "model.pth").exists()
        assert (generation_dir / "config.json").exists()
        assert (generation_dir / "metrics.json").exists()
        
        # Verify model is cached
        assert generation in model_manager._model_cache
        assert model_manager._model_cache[generation] == rl_model
    
    def test_save_model_duplicate_generation(self, model_manager, sample_model, sample_metadata):
        """Test saving model with duplicate generation raises error."""
        generation = 1
        
        # Save first model
        model_manager.save_model(sample_model, generation, sample_metadata)
        
        # Try to save duplicate generation
        with pytest.raises(ValueError, match="Generation 1 already exists"):
            model_manager.save_model(sample_model, generation, sample_metadata)
    
    def test_save_model_missing_metadata(self, model_manager, sample_model):
        """Test saving model with missing metadata raises error."""
        incomplete_metadata = {
            "algorithm": "DQN",
            # Missing required fields
        }
        
        with pytest.raises(ValueError, match="Required metadata field"):
            model_manager.save_model(sample_model, 1, incomplete_metadata)
    
    def test_load_model_success(self, model_manager, sample_model, sample_metadata):
        """Test successful model loading."""
        generation = 1
        
        # Save model first
        original_rl_model = model_manager.save_model(sample_model, generation, sample_metadata)
        
        # Clear caches to test loading from disk
        model_manager._model_cache.clear()
        model_manager._loaded_models.clear()
        
        # Load model
        loaded_model, loaded_rl_model = model_manager.load_model(generation, SimpleTestModel)
        
        # Verify loaded model
        assert isinstance(loaded_model, SimpleTestModel)
        assert isinstance(loaded_rl_model, RLModel)
        
        # Verify metadata matches
        assert loaded_rl_model.generation == original_rl_model.generation
        assert loaded_rl_model.algorithm == original_rl_model.algorithm
        assert loaded_rl_model.network_architecture == original_rl_model.network_architecture
        
        # Verify model weights are loaded correctly
        original_state = sample_model.state_dict()
        loaded_state = loaded_model.state_dict()
        
        for key in original_state:
            assert torch.allclose(original_state[key], loaded_state[key])
    
    def test_load_model_not_found(self, model_manager):
        """Test loading non-existent model raises error."""
        with pytest.raises(FileNotFoundError, match="Generation 999 not found"):
            model_manager.load_model(999, SimpleTestModel)
    
    def test_load_model_no_class_provided(self, model_manager, sample_model, sample_metadata):
        """Test loading model without model class raises error."""
        generation = 1
        
        # Save model first
        model_manager.save_model(sample_model, generation, sample_metadata)
        
        # Clear caches
        model_manager._model_cache.clear()
        model_manager._loaded_models.clear()
        
        # Try to load without model class
        with pytest.raises(ValueError, match="model_class required"):
            model_manager.load_model(generation)
    
    def test_transfer_knowledge_weight_copy(self, model_manager, sample_metadata):
        """Test knowledge transfer using weight copy method."""
        # Create source and target models
        source_model = SimpleTestModel(10, 20, 5)
        target_model = SimpleTestModel(10, 20, 5)
        
        # Initialize with different weights
        with torch.no_grad():
            for param in source_model.parameters():
                param.fill_(1.0)
            for param in target_model.parameters():
                param.fill_(0.0)
        
        # Save source model
        generation = 1
        model_manager.save_model(source_model, generation, sample_metadata)
        
        # Transfer knowledge
        result_model = model_manager.transfer_knowledge(generation, target_model, "weight_copy", SimpleTestModel)
        
        # Verify weights were copied
        source_state = source_model.state_dict()
        result_state = result_model.state_dict()
        
        for key in source_state:
            assert torch.allclose(source_state[key], result_state[key])
    
    def test_transfer_knowledge_layer_freeze(self, model_manager, sample_metadata):
        """Test knowledge transfer with layer freezing."""
        # Create models
        source_model = SimpleTestModel(10, 20, 5)
        target_model = SimpleTestModel(10, 20, 5)
        
        # Save source model
        generation = 1
        model_manager.save_model(source_model, generation, sample_metadata)
        
        # Transfer knowledge with freezing
        result_model = model_manager.transfer_knowledge(generation, target_model, "layer_freeze", SimpleTestModel)
        
        # Verify some layers are frozen
        frozen_count = sum(1 for param in result_model.parameters() if not param.requires_grad)
        assert frozen_count > 0
    
    def test_transfer_knowledge_distillation(self, model_manager, sample_metadata):
        """Test knowledge transfer preparation for distillation."""
        # Create models
        source_model = SimpleTestModel(10, 20, 5)
        target_model = SimpleTestModel(10, 20, 5)
        
        # Save source model
        generation = 1
        model_manager.save_model(source_model, generation, sample_metadata)
        
        # Prepare for distillation
        result_model = model_manager.transfer_knowledge(generation, target_model, "distillation")
        
        # For distillation, target model should be returned unchanged
        assert result_model is target_model
    
    def test_transfer_knowledge_invalid_method(self, model_manager, sample_metadata):
        """Test knowledge transfer with invalid method raises error."""
        source_model = SimpleTestModel(10, 20, 5)
        target_model = SimpleTestModel(10, 20, 5)
        
        # Save source model
        generation = 1
        model_manager.save_model(source_model, generation, sample_metadata)
        
        # Try invalid transfer method
        with pytest.raises(ValueError, match="Unsupported transfer method"):
            model_manager.transfer_knowledge(generation, target_model, "invalid_method", SimpleTestModel)
    
    def test_get_best_model_none_exists(self, model_manager):
        """Test get_best_model when no models exist."""
        result = model_manager.get_best_model()
        assert result is None
    
    def test_promote_and_get_best_model(self, model_manager, sample_model, sample_metadata):
        """Test model promotion and retrieval of best model."""
        generation = 1
        
        # Save model
        rl_model = model_manager.save_model(sample_model, generation, sample_metadata)
        
        # Promote model
        success = model_manager.promote_model(generation)
        assert success is True
        
        # Get best model
        best_generation, best_rl_model = model_manager.get_best_model()
        assert best_generation == generation
        assert best_rl_model.generation == rl_model.generation
    
    def test_promote_model_not_found(self, model_manager):
        """Test promoting non-existent model raises error."""
        with pytest.raises(FileNotFoundError, match="Generation 999 not found"):
            model_manager.promote_model(999)
    
    def test_compare_models(self, model_manager, sample_model, sample_metadata):
        """Test model comparison functionality."""
        # Create two models with different performance
        metadata_1 = sample_metadata.copy()
        metadata_1["performance_metrics"] = {"win_rate": 0.70, "average_reward": 90.0}
        
        metadata_2 = sample_metadata.copy()
        metadata_2["performance_metrics"] = {"win_rate": 0.80, "average_reward": 110.0}
        metadata_2["parent_generation"] = 1
        
        # Save both models
        model_manager.save_model(sample_model, 1, metadata_1)
        model_manager.save_model(sample_model, 2, metadata_2)
        
        # Compare models
        comparison = model_manager.compare_models(1, 2)
        
        # Verify comparison structure
        assert comparison["generation_a"] == 1
        assert comparison["generation_b"] == 2
        assert "metrics_a" in comparison
        assert "metrics_b" in comparison
        assert "improvements" in comparison
        assert "summary" in comparison
        
        # Verify improvements calculation
        assert comparison["improvements"]["win_rate"]["absolute"] == pytest.approx(0.10, rel=1e-6)
        assert comparison["improvements"]["win_rate"]["percentage"] == pytest.approx(14.29, rel=1e-2)
        assert comparison["improvements"]["win_rate"]["better"] is True
        
        assert comparison["improvements"]["average_reward"]["absolute"] == 20.0
        assert comparison["improvements"]["average_reward"]["better"] is True
        
        # Verify summary
        assert comparison["summary"]["overall_better"] is True
        assert comparison["summary"]["win_rate_improvement"] == pytest.approx(0.10, rel=1e-6)
        assert comparison["summary"]["reward_improvement"] == 20.0
    
    def test_should_promote_model_no_current_best(self, model_manager, sample_model, sample_metadata):
        """Test should_promote_model when no current best exists."""
        # Save a model
        model_manager.save_model(sample_model, 1, sample_metadata)
        
        # Check promotion decision
        should_promote, details = model_manager.should_promote_model(1)
        
        assert should_promote is True
        assert details["reason"] == "no_current_best"
    
    def test_should_promote_model_meets_threshold(self, model_manager, sample_model, sample_metadata):
        """Test should_promote_model when candidate meets threshold."""
        # Create models with different performance
        metadata_1 = sample_metadata.copy()
        metadata_1["performance_metrics"] = {"win_rate": 0.70, "average_reward": 90.0}
        
        metadata_2 = sample_metadata.copy()
        metadata_2["performance_metrics"] = {"win_rate": 0.80, "average_reward": 110.0}
        
        # Save and promote first model
        model_manager.save_model(sample_model, 1, metadata_1)
        model_manager.promote_model(1)
        
        # Save second model
        model_manager.save_model(sample_model, 2, metadata_2)
        
        # Check promotion decision (default threshold is 5%)
        should_promote, details = model_manager.should_promote_model(2)
        
        assert should_promote is True
        assert details["promotion_decision"]["should_promote"] is True
        assert details["promotion_decision"]["reason"] == "meets_threshold"
        assert details["promotion_decision"]["win_rate_improvement_pct"] > 5.0
    
    def test_should_promote_model_below_threshold(self, model_manager, sample_model, sample_metadata):
        """Test should_promote_model when candidate is below threshold."""
        # Create models with similar performance
        metadata_1 = sample_metadata.copy()
        metadata_1["performance_metrics"] = {"win_rate": 0.75, "average_reward": 100.0}
        
        metadata_2 = sample_metadata.copy()
        metadata_2["performance_metrics"] = {"win_rate": 0.76, "average_reward": 101.0}  # Only 1.33% improvement
        
        # Save and promote first model
        model_manager.save_model(sample_model, 1, metadata_1)
        model_manager.promote_model(1)
        
        # Save second model
        model_manager.save_model(sample_model, 2, metadata_2)
        
        # Check promotion decision with 5% threshold
        should_promote, details = model_manager.should_promote_model(2, promotion_threshold=0.05)
        
        assert should_promote is False
        assert details["promotion_decision"]["should_promote"] is False
        assert details["promotion_decision"]["reason"] == "below_threshold"
        assert details["promotion_decision"]["win_rate_improvement_pct"] < 5.0
    
    def test_list_models(self, model_manager, sample_model, sample_metadata):
        """Test listing all models."""
        # Initially no models
        models = model_manager.list_models()
        assert len(models) == 0
        
        # Save multiple models
        for generation in [1, 3, 2]:  # Save out of order
            metadata = sample_metadata.copy()
            metadata["performance_metrics"]["win_rate"] = 0.5 + generation * 0.1
            model_manager.save_model(sample_model, generation, metadata)
        
        # List models
        models = model_manager.list_models()
        
        # Verify all models are listed and sorted by generation
        assert len(models) == 3
        generations = [gen for gen, _ in models]
        assert generations == [1, 2, 3]  # Should be sorted
        
        # Verify metadata
        for generation, rl_model in models:
            assert rl_model.generation == generation
            assert rl_model.algorithm == sample_metadata["algorithm"]
    
    def test_delete_model_success(self, model_manager, sample_model, sample_metadata):
        """Test successful model deletion."""
        generation = 1
        
        # Save model
        model_manager.save_model(sample_model, generation, sample_metadata)
        
        # Verify model exists
        models = model_manager.list_models()
        assert len(models) == 1
        
        # Save another model and promote it as best to avoid deleting the current best
        generation_2 = 2
        metadata_2 = sample_metadata.copy()
        metadata_2["performance_metrics"]["win_rate"] = 0.85  # Better performance
        model_manager.save_model(sample_model, generation_2, metadata_2)
        model_manager.promote_model(generation_2)
        
        # Delete first model (not the current best)
        success = model_manager.delete_model(generation)
        assert success is True
        
        # Verify model is deleted
        models = model_manager.list_models()
        assert len(models) == 1  # Only generation 2 should remain
        assert models[0][0] == generation_2
        
        # Verify caches are cleared
        assert generation not in model_manager._model_cache
        assert generation not in model_manager._loaded_models
    
    def test_delete_model_current_best(self, model_manager, sample_model, sample_metadata):
        """Test deleting current best model raises error."""
        generation = 1
        
        # Save and promote model
        model_manager.save_model(sample_model, generation, sample_metadata)
        model_manager.promote_model(generation)
        
        # Try to delete current best
        with pytest.raises(ValueError, match="Cannot delete current best model"):
            model_manager.delete_model(generation)
    
    def test_delete_model_not_found(self, model_manager):
        """Test deleting non-existent model."""
        success = model_manager.delete_model(999)
        assert success is False
    
    def test_get_model_info(self, model_manager, sample_model, sample_metadata):
        """Test getting detailed model information."""
        generation = 1
        
        # Save model
        rl_model = model_manager.save_model(sample_model, generation, sample_metadata)
        
        # Get model info
        info = model_manager.get_model_info(generation)
        
        # Verify info structure
        assert info["generation"] == generation
        assert info["algorithm"] == sample_metadata["algorithm"]
        assert info["network_architecture"] == sample_metadata["network_architecture"]
        assert info["hyperparameters"] == sample_metadata["hyperparameters"]
        assert info["training_episodes"] == sample_metadata["training_episodes"]
        assert info["performance_metrics"] == sample_metadata["performance_metrics"]
        assert "model_size_bytes" in info
        assert "model_size_mb" in info
        assert "children" in info
        
        # Verify file size information
        assert info["model_size_bytes"] > 0
        assert info["model_size_mb"] > 0
    
    def test_get_model_info_with_children(self, model_manager, sample_model, sample_metadata):
        """Test getting model info with parent-child relationships."""
        # Save parent model
        parent_metadata = sample_metadata.copy()
        model_manager.save_model(sample_model, 1, parent_metadata)
        
        # Save child model
        child_metadata = sample_metadata.copy()
        child_metadata["parent_generation"] = 1
        model_manager.save_model(sample_model, 2, child_metadata)
        
        # Get parent info
        parent_info = model_manager.get_model_info(1)
        assert parent_info["children"] == [2]
        
        # Get child info
        child_info = model_manager.get_model_info(2)
        assert child_info["parent_generation"] == 1
        assert child_info["children"] == []
    
    def test_cleanup_old_models(self, model_manager, sample_model, sample_metadata):
        """Test cleanup of old models."""
        # Save multiple models
        for generation in range(1, 8):  # Save 7 models
            metadata = sample_metadata.copy()
            model_manager.save_model(sample_model, generation, metadata)
        
        # Promote model 5 as current best
        model_manager.promote_model(5)
        
        # Cleanup keeping only 3 models
        deleted = model_manager.cleanup_old_models(keep_count=3)
        
        # Verify cleanup results
        remaining_models = model_manager.list_models()
        remaining_generations = [gen for gen, _ in remaining_models]
        
        # Should keep: 7, 6, 5 (most recent 3) and 5 is also current best
        # Should delete: 1, 2, 3, 4
        expected_remaining = {5, 6, 7}  # 5 is kept as current best, 6,7 as most recent
        assert set(remaining_generations) == expected_remaining
        
        # Verify deleted list
        expected_deleted = [1, 2, 3, 4]
        assert sorted(deleted) == expected_deleted
    
    def test_cleanup_old_models_insufficient_models(self, model_manager, sample_model, sample_metadata):
        """Test cleanup when there are fewer models than keep_count."""
        # Save only 2 models
        for generation in [1, 2]:
            metadata = sample_metadata.copy()
            model_manager.save_model(sample_model, generation, metadata)
        
        # Try to cleanup keeping 5 models
        deleted = model_manager.cleanup_old_models(keep_count=5)
        
        # No models should be deleted
        assert len(deleted) == 0
        
        # All models should remain
        remaining_models = model_manager.list_models()
        assert len(remaining_models) == 2