"""
Unit tests for the cohort-based training system.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from bot.rl_bot_system.training.cohort_training import (
    CohortTrainingSystem,
    CohortConfig,
    OpponentConfig,
    OpponentSelectionStrategy,
    DifficultyProgression,
    EpisodeOpponentSetup,
    CohortMetrics
)
from bot.rl_bot_system.training.model_manager import ModelManager, RLModel
from bot.rl_bot_system.rules_based.rules_based_bot import DifficultyLevel


class TestCohortTrainingSystem:
    """Test cases for CohortTrainingSystem."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock ModelManager."""
        manager = Mock(spec=ModelManager)
        
        # Mock available models
        mock_models = [
            (1, RLModel(
                generation=1,
                algorithm="DQN",
                network_architecture={},
                hyperparameters={},
                training_episodes=1000,
                performance_metrics={"win_rate": 0.6, "average_reward": 120.0},
                parent_generation=None,
                created_at=datetime.now(),
                model_path="/path/to/gen1.pth"
            )),
            (2, RLModel(
                generation=2,
                algorithm="PPO",
                network_architecture={},
                hyperparameters={},
                training_episodes=1500,
                performance_metrics={"win_rate": 0.75, "average_reward": 150.0},
                parent_generation=1,
                created_at=datetime.now(),
                model_path="/path/to/gen2.pth"
            ))
        ]
        
        manager.list_models.return_value = mock_models
        return manager

    @pytest.fixture
    def cohort_config(self):
        """Create a test cohort configuration."""
        return CohortConfig(
            cohort_size=4,
            max_enemy_count=2,
            min_enemy_count=1,
            selection_strategy=OpponentSelectionStrategy.WEIGHTED_PERFORMANCE,
            difficulty_progression=DifficultyProgression.LINEAR,
            include_rules_based=True,
            rules_based_ratio=0.5,
            include_previous_generations=True
        )

    @pytest.fixture
    def cohort_system(self, mock_model_manager, cohort_config):
        """Create a CohortTrainingSystem instance."""
        return CohortTrainingSystem(mock_model_manager, cohort_config)

    @pytest.mark.asyncio
    async def test_initialize_cohort(self, cohort_system):
        """Test cohort initialization."""
        await cohort_system.initialize_cohort(current_generation=3)
        
        assert cohort_system.current_generation == 3
        assert len(cohort_system.available_opponents) > 0
        assert len(cohort_system.active_cohort) <= cohort_system.config.cohort_size
        
        # Check that both rules-based and RL opponents are included
        opponent_types = {opp.opponent_type for opp in cohort_system.active_cohort}
        assert "rules_based" in opponent_types
        assert "rl_model" in opponent_types

    @pytest.mark.asyncio
    async def test_discover_available_opponents(self, cohort_system):
        """Test opponent discovery."""
        # Set current generation to 3 so that generations 1 and 2 are available
        cohort_system.current_generation = 3
        await cohort_system._discover_available_opponents()
        
        # Should have rules-based opponents for each difficulty level
        rules_based_count = sum(
            1 for opp in cohort_system.available_opponents.values()
            if opp.opponent_type == "rules_based"
        )
        assert rules_based_count == len(DifficultyLevel)
        
        # Should have RL model opponents from previous generations
        rl_model_count = sum(
            1 for opp in cohort_system.available_opponents.values()
            if opp.opponent_type == "rl_model"
        )
        assert rl_model_count == 2  # Based on mock data

    @pytest.mark.asyncio
    async def test_select_episode_opponents_random(self, cohort_system):
        """Test random opponent selection."""
        cohort_system.config.selection_strategy = OpponentSelectionStrategy.RANDOM
        await cohort_system.initialize_cohort(current_generation=3)
        
        setup = await cohort_system.select_episode_opponents("test_episode", 0.5)
        
        assert isinstance(setup, EpisodeOpponentSetup)
        assert setup.episode_id == "test_episode"
        assert cohort_system.config.min_enemy_count <= setup.enemy_count <= cohort_system.config.max_enemy_count
        assert len(setup.opponents) == setup.enemy_count
        assert 0.0 <= setup.expected_challenge <= 1.0

    @pytest.mark.asyncio
    async def test_select_episode_opponents_weighted(self, cohort_system):
        """Test weighted performance opponent selection."""
        cohort_system.config.selection_strategy = OpponentSelectionStrategy.WEIGHTED_PERFORMANCE
        await cohort_system.initialize_cohort(current_generation=3)
        
        setup = await cohort_system.select_episode_opponents("test_episode", 0.3)
        
        assert len(setup.opponents) == setup.enemy_count
        # Higher-weighted opponents should be more likely to be selected
        # (This is probabilistic, so we just check the structure)
        for opponent in setup.opponents:
            assert opponent in cohort_system.active_cohort

    @pytest.mark.asyncio
    async def test_select_episode_opponents_round_robin(self, cohort_system):
        """Test round-robin opponent selection."""
        cohort_system.config.selection_strategy = OpponentSelectionStrategy.ROUND_ROBIN
        await cohort_system.initialize_cohort(current_generation=3)
        
        # Select opponents for multiple episodes
        setups = []
        for i in range(5):
            setup = await cohort_system.select_episode_opponents(f"episode_{i}", 0.2)
            setups.append(setup)
        
        # Check that round-robin index advances
        assert cohort_system.round_robin_index > 0
        
        # All opponents should eventually be selected
        all_selected_opponents = set()
        for setup in setups:
            for opponent in setup.opponents:
                all_selected_opponents.add(opponent.opponent_id)
        
        # Should have good coverage of available opponents
        assert len(all_selected_opponents) >= min(len(cohort_system.active_cohort), 3)

    @pytest.mark.asyncio
    async def test_difficulty_progression_linear(self, cohort_system):
        """Test linear difficulty progression."""
        cohort_system.config.difficulty_progression = DifficultyProgression.LINEAR
        cohort_system.config.progression_rate = 0.5  # Reduced rate for more predictable testing
        await cohort_system.initialize_cohort(current_generation=3)
        
        # Test progression at different training stages
        initial_difficulty = cohort_system.current_difficulty
        
        await cohort_system.select_episode_opponents("episode_1", 0.1)
        early_difficulty = cohort_system.current_difficulty
        
        await cohort_system.select_episode_opponents("episode_2", 0.5)
        mid_difficulty = cohort_system.current_difficulty
        
        await cohort_system.select_episode_opponents("episode_3", 1.0)
        late_difficulty = cohort_system.current_difficulty
        
        # Difficulty should increase with training progress
        assert early_difficulty >= initial_difficulty
        assert mid_difficulty >= early_difficulty
        assert late_difficulty >= mid_difficulty

    @pytest.mark.asyncio
    async def test_difficulty_progression_adaptive(self, cohort_system):
        """Test adaptive difficulty progression."""
        cohort_system.config.difficulty_progression = DifficultyProgression.ADAPTIVE
        await cohort_system.initialize_cohort(current_generation=3)
        
        initial_difficulty = cohort_system.current_difficulty
        
        # Simulate high win rate (should increase difficulty)
        cohort_system.recent_performance = [1.0] * 20  # 100% win rate
        cohort_system._update_adaptive_difficulty()
        high_performance_difficulty = cohort_system.current_difficulty
        
        # Simulate low win rate (should decrease difficulty)
        cohort_system.recent_performance = [0.0] * 20  # 0% win rate
        cohort_system._update_adaptive_difficulty()
        low_performance_difficulty = cohort_system.current_difficulty
        
        assert high_performance_difficulty >= initial_difficulty
        assert low_performance_difficulty <= high_performance_difficulty

    @pytest.mark.asyncio
    async def test_update_episode_results(self, cohort_system):
        """Test updating episode results and metrics."""
        await cohort_system.initialize_cohort(current_generation=3)
        
        setup = await cohort_system.select_episode_opponents("test_episode", 0.5)
        
        # Simulate episode results
        results = {
            "won": True,
            "total_reward": 150.0,
            "episode_length": 100
        }
        
        initial_episodes = cohort_system.metrics.total_episodes
        
        await cohort_system.update_episode_results("test_episode", setup, results)
        
        # Check metrics were updated
        assert cohort_system.metrics.total_episodes == initial_episodes + 1
        assert len(cohort_system.recent_performance) > 0
        assert cohort_system.recent_performance[-1] == 1.0  # Won = 1.0
        
        # Check opponent-specific metrics
        for opponent in setup.opponents:
            assert opponent.opponent_id in cohort_system.metrics.win_rates_by_opponent
            assert opponent.opponent_id in cohort_system.metrics.opponent_usage

    def test_determine_enemy_count(self, cohort_system):
        """Test enemy count determination."""
        cohort_system.config.min_enemy_count = 1
        cohort_system.config.max_enemy_count = 3
        
        # Early training should use fewer enemies
        early_count = cohort_system._determine_enemy_count(0.0)
        assert early_count >= cohort_system.config.min_enemy_count
        
        # Late training should use more enemies
        late_count = cohort_system._determine_enemy_count(1.0)
        assert late_count <= cohort_system.config.max_enemy_count
        assert late_count >= early_count

    def test_estimate_opponent_difficulty(self, cohort_system):
        """Test opponent difficulty estimation."""
        # Test rules-based opponent
        rules_opponent = OpponentConfig(
            opponent_id="test_rules",
            opponent_type="rules_based",
            difficulty_level=DifficultyLevel.ADVANCED
        )
        
        rules_difficulty = cohort_system._estimate_opponent_difficulty(rules_opponent)
        assert 0.0 <= rules_difficulty <= 1.0
        assert rules_difficulty > 0.5  # Advanced should be > 0.5
        
        # Test RL model opponent
        rl_opponent = OpponentConfig(
            opponent_id="test_rl",
            opponent_type="rl_model",
            generation=5,
            performance_metrics={"win_rate": 0.8}
        )
        
        rl_difficulty = cohort_system._estimate_opponent_difficulty(rl_opponent)
        assert 0.0 <= rl_difficulty <= 1.0

    def test_calculate_opponent_diversity(self, cohort_system):
        """Test opponent diversity calculation."""
        opponent1 = OpponentConfig(
            opponent_id="rules_beginner",
            opponent_type="rules_based",
            difficulty_level=DifficultyLevel.BEGINNER
        )
        
        opponent2 = OpponentConfig(
            opponent_id="rl_gen_2",
            opponent_type="rl_model",
            generation=2,
            performance_metrics={"win_rate": 0.7}
        )
        
        # Diversity between different types should be high
        diversity = cohort_system._calculate_opponent_diversity(opponent1, [opponent2])
        assert diversity > 0.0
        
        # Diversity with empty list should be 1.0
        empty_diversity = cohort_system._calculate_opponent_diversity(opponent1, [])
        assert empty_diversity == 1.0

    def test_select_diverse_subset(self, cohort_system):
        """Test diverse subset selection."""
        opponents = [
            OpponentConfig(
                opponent_id=f"opponent_{i}",
                opponent_type="rules_based" if i % 2 == 0 else "rl_model",
                difficulty_level=DifficultyLevel.BEGINNER if i < 2 else DifficultyLevel.EXPERT,
                generation=i if i % 2 == 1 else None,
                selection_weight=0.5 + i * 0.1
            )
            for i in range(6)
        ]
        
        # Select diverse subset
        selected = cohort_system._select_diverse_subset(opponents, 3)
        
        assert len(selected) == 3
        assert len(set(opp.opponent_id for opp in selected)) == 3  # No duplicates
        
        # Should include the strongest opponent
        strongest = max(opponents, key=lambda x: x.selection_weight)
        assert strongest in selected

    def test_get_cohort_info(self, cohort_system):
        """Test cohort information retrieval."""
        cohort_system.current_generation = 3
        cohort_system.training_episode_count = 50
        cohort_system.current_difficulty = 0.6
        
        # Add some mock opponents
        cohort_system.active_cohort = [
            OpponentConfig(
                opponent_id="test_opponent",
                opponent_type="rules_based",
                difficulty_level=DifficultyLevel.INTERMEDIATE,
                usage_count=10,
                selection_weight=0.8
            )
        ]
        
        info = cohort_system.get_cohort_info()
        
        assert info["currentGeneration"] == 3
        assert info["trainingEpisodes"] == 50
        assert info["currentDifficulty"] == 0.6
        assert len(info["opponents"]) == 1
        assert info["opponents"][0]["id"] == "test_opponent"
        assert "config" in info
        assert "metrics" in info

    @pytest.mark.asyncio
    async def test_curriculum_learning_selection(self, cohort_system):
        """Test curriculum learning opponent selection."""
        cohort_system.config.selection_strategy = OpponentSelectionStrategy.CURRICULUM_LEARNING
        cohort_system.config.curriculum_stages = 3
        await cohort_system.initialize_cohort(current_generation=3)
        
        # Test early stage (should select easier opponents)
        early_setup = await cohort_system.select_episode_opponents("early", 0.1)
        
        # Test late stage (should select harder opponents)
        late_setup = await cohort_system.select_episode_opponents("late", 0.9)
        
        # Both should have valid setups
        assert len(early_setup.opponents) > 0
        assert len(late_setup.opponents) > 0
        
        # Expected challenge should generally increase
        # (This is probabilistic due to random selection within stages)
        assert early_setup.expected_challenge >= 0.0
        assert late_setup.expected_challenge >= 0.0

    def test_calculate_expected_challenge(self, cohort_system):
        """Test expected challenge calculation."""
        opponents = [
            OpponentConfig(
                opponent_id="easy",
                opponent_type="rules_based",
                difficulty_level=DifficultyLevel.BEGINNER,
                performance_metrics={"win_rate": 0.3}
            ),
            OpponentConfig(
                opponent_id="hard",
                opponent_type="rules_based",
                difficulty_level=DifficultyLevel.EXPERT,
                performance_metrics={"win_rate": 0.8}
            )
        ]
        
        challenge = cohort_system._calculate_expected_challenge(opponents)
        
        assert 0.0 <= challenge <= 1.0
        
        # Multiple opponents should increase challenge
        single_challenge = cohort_system._calculate_expected_challenge([opponents[0]])
        assert challenge >= single_challenge

    @pytest.mark.asyncio
    async def test_load_opponent_instances(self, cohort_system):
        """Test loading opponent instances."""
        # Set up active cohort
        cohort_system.active_cohort = [
            OpponentConfig(
                opponent_id="rules_test",
                opponent_type="rules_based",
                difficulty_level=DifficultyLevel.INTERMEDIATE
            ),
            OpponentConfig(
                opponent_id="rl_test",
                opponent_type="rl_model",
                generation=2,
                model_path="/path/to/model.pth",
                performance_metrics={"win_rate": 0.7}
            )
        ]
        
        await cohort_system._load_opponent_instances()
        
        # Check that instances were loaded
        assert "rules_test" in cohort_system.opponent_pool
        assert "rl_test" in cohort_system.opponent_pool
        
        # Rules-based should be a bot instance
        rules_instance = cohort_system.opponent_pool["rules_test"]
        assert hasattr(rules_instance, 'difficulty')
        
        # RL model should be model info dict
        rl_instance = cohort_system.opponent_pool["rl_test"]
        assert isinstance(rl_instance, dict)
        assert "generation" in rl_instance


if __name__ == "__main__":
    pytest.main([__file__])