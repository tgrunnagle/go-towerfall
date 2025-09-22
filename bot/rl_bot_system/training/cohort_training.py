"""
Cohort-based training system for successive RL bot generations.

This module implements opponent selection from previous bot generations,
configurable cohort sizes and selection strategies, variable enemy counts,
and difficulty progression for multi-agent training scenarios.
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
import numpy as np

from bot.rl_bot_system.training.model_manager import ModelManager, RLModel
from bot.rl_bot_system.rules_based.rules_based_bot import RulesBasedBot, DifficultyLevel


class OpponentSelectionStrategy(Enum):
    """Strategies for selecting opponents from the cohort."""
    RANDOM = "random"  # Random selection from cohort
    WEIGHTED_PERFORMANCE = "weighted_performance"  # Weight by performance metrics
    ROUND_ROBIN = "round_robin"  # Cycle through all opponents
    DIFFICULTY_PROGRESSION = "difficulty_progression"  # Start easy, get harder
    DIVERSE_SAMPLING = "diverse_sampling"  # Maximize opponent diversity
    CURRICULUM_LEARNING = "curriculum_learning"  # Structured learning progression


class DifficultyProgression(Enum):
    """Difficulty progression modes for training."""
    STATIC = "static"  # Fixed difficulty throughout training
    LINEAR = "linear"  # Linear increase in difficulty
    EXPONENTIAL = "exponential"  # Exponential difficulty curve
    ADAPTIVE = "adaptive"  # Adapt based on performance
    STAGED = "staged"  # Discrete difficulty stages


@dataclass
class OpponentConfig:
    """Configuration for an opponent in training."""
    opponent_id: str
    opponent_type: str  # 'rules_based', 'rl_model', 'hybrid'
    generation: Optional[int] = None  # For RL models
    difficulty_level: Optional[DifficultyLevel] = None  # For rules-based bots
    model_path: Optional[str] = None  # Path to model weights
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    selection_weight: float = 1.0
    last_used: Optional[datetime] = None
    usage_count: int = 0


@dataclass
class CohortConfig:
    """Configuration for cohort-based training."""
    cohort_size: int = 5  # Number of different opponents to include
    max_enemy_count: int = 3  # Maximum enemies per episode
    min_enemy_count: int = 1  # Minimum enemies per episode
    selection_strategy: OpponentSelectionStrategy = OpponentSelectionStrategy.WEIGHTED_PERFORMANCE
    difficulty_progression: DifficultyProgression = DifficultyProgression.ADAPTIVE
    
    # Strategy-specific parameters
    performance_weight_decay: float = 0.9  # How much to weight recent performance
    diversity_threshold: float = 0.3  # Minimum diversity for diverse sampling
    curriculum_stages: int = 5  # Number of curriculum stages
    
    # Progression parameters
    progression_rate: float = 0.1  # Rate of difficulty increase
    adaptation_window: int = 100  # Episodes to consider for adaptation
    performance_threshold: float = 0.7  # Win rate threshold for progression
    
    # Cohort composition
    include_rules_based: bool = True  # Include rules-based opponents
    rules_based_ratio: float = 0.3  # Ratio of rules-based to RL opponents
    include_previous_generations: bool = True  # Include previous RL generations
    max_generation_gap: int = 5  # Maximum gap between current and opponent generations


@dataclass
class EpisodeOpponentSetup:
    """Setup configuration for opponents in a training episode."""
    episode_id: str
    enemy_count: int
    opponents: List[OpponentConfig]
    difficulty_level: float  # 0.0 to 1.0
    selection_rationale: str
    expected_challenge: float  # 0.0 to 1.0


@dataclass
class CohortMetrics:
    """Metrics tracking for cohort-based training."""
    total_episodes: int = 0
    opponent_usage: Dict[str, int] = field(default_factory=dict)
    win_rates_by_opponent: Dict[str, float] = field(default_factory=dict)
    average_episode_difficulty: float = 0.0
    diversity_score: float = 0.0
    progression_stage: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class CohortTrainingSystem:
    """
    Manages cohort-based training with opponent selection and difficulty progression.
    
    Provides functionality for:
    - Selecting opponents from previous bot generations
    - Configurable cohort size and selection strategies
    - Variable enemy counts per training episode
    - Difficulty progression and multi-agent scenarios
    """

    def __init__(
        self,
        model_manager: ModelManager,
        config: Optional[CohortConfig] = None
    ):
        self.model_manager = model_manager
        self.config = config or CohortConfig()
        
        # Cohort management
        self.available_opponents: Dict[str, OpponentConfig] = {}
        self.active_cohort: List[OpponentConfig] = []
        self.opponent_pool: Dict[str, Any] = {}  # Loaded opponent instances
        
        # Training state
        self.current_generation: int = 0
        self.training_episode_count: int = 0
        self.metrics = CohortMetrics()
        
        # Selection strategy state
        self.round_robin_index: int = 0
        self.curriculum_stage: int = 0
        self.recent_performance: List[float] = []
        
        # Difficulty progression state
        self.current_difficulty: float = 0.1  # Start easy
        self.progression_history: List[Tuple[int, float]] = []
        
        self._logger = logging.getLogger(__name__)

    async def initialize_cohort(self, current_generation: int) -> None:
        """
        Initialize the cohort for training a specific generation.
        
        Args:
            current_generation: The generation being trained
        """
        self.current_generation = current_generation
        self._logger.info(f"Initializing cohort for generation {current_generation}")
        
        # Discover available opponents
        await self._discover_available_opponents()
        
        # Build the active cohort
        await self._build_active_cohort()
        
        # Load opponent instances
        await self._load_opponent_instances()
        
        # Reset training state
        self.training_episode_count = 0
        self.metrics = CohortMetrics()
        
        self._logger.info(
            f"Cohort initialized with {len(self.active_cohort)} opponents: "
            f"{[opp.opponent_id for opp in self.active_cohort]}"
        )

    async def select_episode_opponents(
        self,
        episode_id: str,
        training_progress: float = 0.0
    ) -> EpisodeOpponentSetup:
        """
        Select opponents for a training episode.
        
        Args:
            episode_id: Unique identifier for the episode
            training_progress: Training progress (0.0 to 1.0)
            
        Returns:
            EpisodeOpponentSetup with selected opponents and configuration
        """
        # Determine enemy count based on progression
        enemy_count = self._determine_enemy_count(training_progress)
        
        # Update difficulty based on progression mode
        self._update_difficulty_progression(training_progress)
        
        # Select opponents based on strategy
        selected_opponents = await self._select_opponents_by_strategy(
            enemy_count, training_progress
        )
        
        # Calculate expected challenge level
        expected_challenge = self._calculate_expected_challenge(selected_opponents)
        
        # Create episode setup
        setup = EpisodeOpponentSetup(
            episode_id=episode_id,
            enemy_count=enemy_count,
            opponents=selected_opponents,
            difficulty_level=self.current_difficulty,
            selection_rationale=self._get_selection_rationale(),
            expected_challenge=expected_challenge
        )
        
        # Update usage tracking
        self._update_opponent_usage(selected_opponents)
        
        self.training_episode_count += 1
        
        self._logger.debug(
            f"Episode {episode_id}: {enemy_count} opponents selected "
            f"(difficulty: {self.current_difficulty:.2f}, challenge: {expected_challenge:.2f})"
        )
        
        return setup

    async def update_episode_results(
        self,
        episode_id: str,
        setup: EpisodeOpponentSetup,
        results: Dict[str, Any]
    ) -> None:
        """
        Update cohort metrics based on episode results.
        
        Args:
            episode_id: Episode identifier
            setup: The episode setup that was used
            results: Episode results including win/loss and performance metrics
        """
        won = results.get('won', False)
        episode_reward = results.get('total_reward', 0.0)
        
        # Update win rates by opponent
        for opponent in setup.opponents:
            opponent_id = opponent.opponent_id
            
            if opponent_id not in self.metrics.win_rates_by_opponent:
                self.metrics.win_rates_by_opponent[opponent_id] = 0.0
            
            # Update running average win rate
            current_wins = self.metrics.opponent_usage.get(opponent_id, 0)
            if current_wins > 0:
                current_rate = self.metrics.win_rates_by_opponent[opponent_id]
                new_rate = (current_rate * (current_wins - 1) + (1.0 if won else 0.0)) / current_wins
                self.metrics.win_rates_by_opponent[opponent_id] = new_rate
            else:
                self.metrics.win_rates_by_opponent[opponent_id] = 1.0 if won else 0.0
        
        # Update recent performance for adaptive progression
        self.recent_performance.append(1.0 if won else 0.0)
        if len(self.recent_performance) > self.config.adaptation_window:
            self.recent_performance.pop(0)
        
        # Update global metrics
        self.metrics.total_episodes += 1
        self.metrics.average_episode_difficulty = (
            (self.metrics.average_episode_difficulty * (self.metrics.total_episodes - 1) + 
             setup.difficulty_level) / self.metrics.total_episodes
        )
        
        # Calculate diversity score
        self.metrics.diversity_score = self._calculate_cohort_diversity()
        
        self.metrics.last_updated = datetime.now()
        
        self._logger.debug(
            f"Episode {episode_id} results updated: won={won}, "
            f"reward={episode_reward:.2f}, difficulty={setup.difficulty_level:.2f}"
        )

    def get_cohort_info(self) -> Dict[str, Any]:
        """Get information about the current cohort."""
        return {
            "currentGeneration": self.current_generation,
            "cohortSize": len(self.active_cohort),
            "availableOpponents": len(self.available_opponents),
            "trainingEpisodes": self.training_episode_count,
            "currentDifficulty": self.current_difficulty,
            "progressionStage": self.curriculum_stage,
            "config": {
                "cohortSize": self.config.cohort_size,
                "maxEnemyCount": self.config.max_enemy_count,
                "minEnemyCount": self.config.min_enemy_count,
                "selectionStrategy": self.config.selection_strategy.value,
                "difficultyProgression": self.config.difficulty_progression.value
            },
            "opponents": [
                {
                    "id": opp.opponent_id,
                    "type": opp.opponent_type,
                    "generation": opp.generation,
                    "difficulty": opp.difficulty_level.value if opp.difficulty_level else None,
                    "usageCount": opp.usage_count,
                    "selectionWeight": opp.selection_weight
                }
                for opp in self.active_cohort
            ],
            "metrics": {
                "totalEpisodes": self.metrics.total_episodes,
                "averageDifficulty": self.metrics.average_episode_difficulty,
                "diversityScore": self.metrics.diversity_score,
                "winRatesByOpponent": self.metrics.win_rates_by_opponent,
                "opponentUsage": self.metrics.opponent_usage
            }
        }

    async def _discover_available_opponents(self) -> None:
        """Discover all available opponents for the cohort."""
        self.available_opponents.clear()
        
        # Add rules-based opponents if enabled
        if self.config.include_rules_based:
            for difficulty in DifficultyLevel:
                opponent_id = f"rules_based_{difficulty.value}"
                self.available_opponents[opponent_id] = OpponentConfig(
                    opponent_id=opponent_id,
                    opponent_type="rules_based",
                    difficulty_level=difficulty,
                    performance_metrics=self._estimate_rules_based_performance(difficulty),
                    selection_weight=self._calculate_rules_based_weight(difficulty)
                )
        
        # Add previous RL model generations if enabled
        if self.config.include_previous_generations:
            available_models = self.model_manager.list_models()
            
            for generation, model_metadata in available_models:
                # Skip future generations and respect generation gap limit
                if (generation >= self.current_generation or 
                    self.current_generation - generation > self.config.max_generation_gap):
                    continue
                
                opponent_id = f"rl_gen_{generation}"
                self.available_opponents[opponent_id] = OpponentConfig(
                    opponent_id=opponent_id,
                    opponent_type="rl_model",
                    generation=generation,
                    model_path=model_metadata.model_path,
                    performance_metrics=model_metadata.performance_metrics,
                    selection_weight=self._calculate_model_weight(model_metadata)
                )
        
        self._logger.info(f"Discovered {len(self.available_opponents)} available opponents")

    async def _build_active_cohort(self) -> None:
        """Build the active cohort from available opponents."""
        self.active_cohort.clear()
        
        # Calculate target composition
        target_rules_based = int(self.config.cohort_size * self.config.rules_based_ratio)
        target_rl_models = self.config.cohort_size - target_rules_based
        
        # Select rules-based opponents
        rules_based_opponents = [
            opp for opp in self.available_opponents.values() 
            if opp.opponent_type == "rules_based"
        ]
        
        if rules_based_opponents:
            # Sort by difficulty and select diverse set
            rules_based_opponents.sort(key=lambda x: x.difficulty_level.value)
            selected_rules = self._select_diverse_subset(
                rules_based_opponents, target_rules_based
            )
            self.active_cohort.extend(selected_rules)
        
        # Select RL model opponents
        rl_opponents = [
            opp for opp in self.available_opponents.values() 
            if opp.opponent_type == "rl_model"
        ]
        
        if rl_opponents:
            # Sort by generation (newer first) and performance
            rl_opponents.sort(
                key=lambda x: (x.generation or 0, x.performance_metrics.get('win_rate', 0.0)),
                reverse=True
            )
            selected_rl = self._select_diverse_subset(rl_opponents, target_rl_models)
            self.active_cohort.extend(selected_rl)
        
        # Fill remaining slots if needed
        remaining_slots = self.config.cohort_size - len(self.active_cohort)
        if remaining_slots > 0:
            remaining_opponents = [
                opp for opp in self.available_opponents.values()
                if opp not in self.active_cohort
            ]
            
            if remaining_opponents:
                # Sort by selection weight
                remaining_opponents.sort(key=lambda x: x.selection_weight, reverse=True)
                self.active_cohort.extend(remaining_opponents[:remaining_slots])

    def _select_diverse_subset(
        self, 
        opponents: List[OpponentConfig], 
        target_count: int
    ) -> List[OpponentConfig]:
        """Select a diverse subset of opponents."""
        if len(opponents) <= target_count:
            return opponents
        
        if target_count == 0:
            return []
        
        # Use weighted selection based on diversity and performance
        selected = []
        remaining = opponents.copy()
        
        # Always include the strongest opponent
        if remaining:
            strongest = max(remaining, key=lambda x: x.selection_weight)
            selected.append(strongest)
            remaining.remove(strongest)
        
        # Select remaining opponents to maximize diversity
        while len(selected) < target_count and remaining:
            best_candidate = None
            best_diversity_score = -1
            
            for candidate in remaining:
                diversity_score = self._calculate_opponent_diversity(candidate, selected)
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            elif remaining:
                # If no best candidate found, just take the first remaining
                selected.append(remaining[0])
                remaining.remove(remaining[0])
        
        return selected

    async def _load_opponent_instances(self) -> None:
        """Load instances of all opponents in the active cohort."""
        self.opponent_pool.clear()
        
        for opponent in self.active_cohort:
            try:
                if opponent.opponent_type == "rules_based":
                    # Create rules-based bot instance
                    bot_instance = RulesBasedBot(opponent.difficulty_level)
                    self.opponent_pool[opponent.opponent_id] = bot_instance
                    
                elif opponent.opponent_type == "rl_model":
                    # Load RL model (placeholder - would need actual model loading)
                    # This would integrate with the actual RL model loading system
                    model_info = {
                        'generation': opponent.generation,
                        'model_path': opponent.model_path,
                        'performance_metrics': opponent.performance_metrics
                    }
                    self.opponent_pool[opponent.opponent_id] = model_info
                    
                self._logger.debug(f"Loaded opponent instance: {opponent.opponent_id}")
                
            except Exception as e:
                self._logger.error(f"Failed to load opponent {opponent.opponent_id}: {e}")

    def _determine_enemy_count(self, training_progress: float) -> int:
        """Determine the number of enemies for this episode based on progression."""
        if self.config.min_enemy_count == self.config.max_enemy_count:
            return self.config.min_enemy_count
        
        # Progressive increase in enemy count
        progress_factor = min(1.0, training_progress + self.current_difficulty * 0.5)
        
        enemy_range = self.config.max_enemy_count - self.config.min_enemy_count
        additional_enemies = int(progress_factor * enemy_range)
        
        return self.config.min_enemy_count + additional_enemies

    def _update_difficulty_progression(self, training_progress: float) -> None:
        """Update the current difficulty based on progression mode."""
        if self.config.difficulty_progression == DifficultyProgression.STATIC:
            return
        
        elif self.config.difficulty_progression == DifficultyProgression.LINEAR:
            self.current_difficulty = min(1.0, self.current_difficulty + training_progress * self.config.progression_rate)
        
        elif self.config.difficulty_progression == DifficultyProgression.EXPONENTIAL:
            self.current_difficulty = min(1.0, training_progress ** (1.0 / self.config.progression_rate))
        
        elif self.config.difficulty_progression == DifficultyProgression.ADAPTIVE:
            self._update_adaptive_difficulty()
        
        elif self.config.difficulty_progression == DifficultyProgression.STAGED:
            self._update_staged_difficulty(training_progress)

    def _update_adaptive_difficulty(self) -> None:
        """Update difficulty based on recent performance."""
        if len(self.recent_performance) < 10:  # Need minimum data
            return
        
        recent_win_rate = sum(self.recent_performance[-50:]) / len(self.recent_performance[-50:])
        
        # Increase difficulty if performing well, decrease if struggling
        if recent_win_rate > self.config.performance_threshold:
            self.current_difficulty = min(1.0, self.current_difficulty + 0.05)
        elif recent_win_rate < self.config.performance_threshold - 0.2:
            self.current_difficulty = max(0.1, self.current_difficulty - 0.03)

    def _update_staged_difficulty(self, training_progress: float) -> None:
        """Update difficulty using discrete stages."""
        stage_size = 1.0 / self.config.curriculum_stages
        new_stage = min(self.config.curriculum_stages - 1, int(training_progress / stage_size))
        
        if new_stage > self.curriculum_stage:
            self.curriculum_stage = new_stage
            self.current_difficulty = (new_stage + 1) / self.config.curriculum_stages
            self._logger.info(f"Advanced to curriculum stage {new_stage} (difficulty: {self.current_difficulty:.2f})")

    async def _select_opponents_by_strategy(
        self,
        enemy_count: int,
        training_progress: float
    ) -> List[OpponentConfig]:
        """Select opponents based on the configured strategy."""
        if enemy_count > len(self.active_cohort):
            # If we need more opponents than available, use all and repeat some
            selected = self.active_cohort.copy()
            remaining_count = enemy_count - len(selected)
            
            # Add strongest opponents to fill remaining slots
            strongest_opponents = sorted(
                self.active_cohort, 
                key=lambda x: x.selection_weight, 
                reverse=True
            )
            selected.extend(strongest_opponents[:remaining_count])
            
            return selected
        
        strategy = self.config.selection_strategy
        
        if strategy == OpponentSelectionStrategy.RANDOM:
            return random.sample(self.active_cohort, enemy_count)
        
        elif strategy == OpponentSelectionStrategy.WEIGHTED_PERFORMANCE:
            return self._weighted_performance_selection(enemy_count)
        
        elif strategy == OpponentSelectionStrategy.ROUND_ROBIN:
            return self._round_robin_selection(enemy_count)
        
        elif strategy == OpponentSelectionStrategy.DIFFICULTY_PROGRESSION:
            return self._difficulty_progression_selection(enemy_count, training_progress)
        
        elif strategy == OpponentSelectionStrategy.DIVERSE_SAMPLING:
            return self._diverse_sampling_selection(enemy_count)
        
        elif strategy == OpponentSelectionStrategy.CURRICULUM_LEARNING:
            return self._curriculum_learning_selection(enemy_count, training_progress)
        
        else:
            # Fallback to random
            return random.sample(self.active_cohort, enemy_count)

    def _weighted_performance_selection(self, enemy_count: int) -> List[OpponentConfig]:
        """Select opponents weighted by their performance metrics."""
        weights = [opp.selection_weight for opp in self.active_cohort]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]
        else:
            probabilities = [1.0 / len(weights)] * len(weights)
        
        # Sample without replacement
        selected_indices = np.random.choice(
            len(self.active_cohort),
            size=min(enemy_count, len(self.active_cohort)),
            replace=False,
            p=probabilities
        )
        
        return [self.active_cohort[i] for i in selected_indices]

    def _round_robin_selection(self, enemy_count: int) -> List[OpponentConfig]:
        """Select opponents using round-robin strategy."""
        selected = []
        
        for i in range(enemy_count):
            index = (self.round_robin_index + i) % len(self.active_cohort)
            selected.append(self.active_cohort[index])
        
        self.round_robin_index = (self.round_robin_index + enemy_count) % len(self.active_cohort)
        return selected

    def _difficulty_progression_selection(
        self, 
        enemy_count: int, 
        training_progress: float
    ) -> List[OpponentConfig]:
        """Select opponents based on difficulty progression."""
        # Sort opponents by estimated difficulty
        sorted_opponents = sorted(
            self.active_cohort,
            key=lambda x: self._estimate_opponent_difficulty(x)
        )
        
        # Select opponents based on current difficulty level
        target_difficulty = self.current_difficulty
        
        selected = []
        for opponent in sorted_opponents:
            opp_difficulty = self._estimate_opponent_difficulty(opponent)
            
            # Select opponents within difficulty range
            if abs(opp_difficulty - target_difficulty) <= 0.3:
                selected.append(opponent)
                
                if len(selected) >= enemy_count:
                    break
        
        # Fill remaining slots if needed
        while len(selected) < enemy_count and len(selected) < len(self.active_cohort):
            remaining = [opp for opp in self.active_cohort if opp not in selected]
            if remaining:
                # Add closest difficulty opponent
                closest = min(
                    remaining,
                    key=lambda x: abs(self._estimate_opponent_difficulty(x) - target_difficulty)
                )
                selected.append(closest)
        
        return selected

    def _diverse_sampling_selection(self, enemy_count: int) -> List[OpponentConfig]:
        """Select opponents to maximize diversity."""
        if enemy_count >= len(self.active_cohort):
            return self.active_cohort.copy()
        
        selected = []
        remaining = self.active_cohort.copy()
        
        # Start with a random opponent
        first_opponent = random.choice(remaining)
        selected.append(first_opponent)
        remaining.remove(first_opponent)
        
        # Select remaining opponents to maximize diversity
        while len(selected) < enemy_count and remaining:
            best_candidate = max(
                remaining,
                key=lambda x: self._calculate_opponent_diversity(x, selected)
            )
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        
        return selected

    def _curriculum_learning_selection(
        self, 
        enemy_count: int, 
        training_progress: float
    ) -> List[OpponentConfig]:
        """Select opponents based on curriculum learning principles."""
        # Determine curriculum stage
        stage_progress = training_progress * self.config.curriculum_stages
        current_stage = int(stage_progress)
        stage_blend = stage_progress - current_stage
        
        # Select opponents appropriate for current stage
        stage_opponents = self._get_curriculum_stage_opponents(current_stage)
        
        # Blend with next stage if appropriate
        if stage_blend > 0.5 and current_stage < self.config.curriculum_stages - 1:
            next_stage_opponents = self._get_curriculum_stage_opponents(current_stage + 1)
            
            # Mix current and next stage opponents
            current_count = max(1, int(enemy_count * (1.0 - stage_blend)))
            next_count = enemy_count - current_count
            
            selected = random.sample(stage_opponents, min(current_count, len(stage_opponents)))
            if next_stage_opponents and next_count > 0:
                selected.extend(random.sample(next_stage_opponents, min(next_count, len(next_stage_opponents))))
        else:
            selected = random.sample(stage_opponents, min(enemy_count, len(stage_opponents)))
        
        # Fill remaining slots if needed
        while len(selected) < enemy_count:
            remaining = [opp for opp in self.active_cohort if opp not in selected]
            if not remaining:
                break
            selected.append(random.choice(remaining))
        
        return selected

    def _get_curriculum_stage_opponents(self, stage: int) -> List[OpponentConfig]:
        """Get opponents appropriate for a specific curriculum stage."""
        stage_difficulty = stage / max(1, self.config.curriculum_stages - 1)
        
        appropriate_opponents = []
        for opponent in self.active_cohort:
            opp_difficulty = self._estimate_opponent_difficulty(opponent)
            
            # Include opponents within stage difficulty range
            if abs(opp_difficulty - stage_difficulty) <= 0.4:
                appropriate_opponents.append(opponent)
        
        # Ensure we have at least some opponents
        if not appropriate_opponents:
            appropriate_opponents = self.active_cohort.copy()
        
        return appropriate_opponents

    def _calculate_expected_challenge(self, opponents: List[OpponentConfig]) -> float:
        """Calculate the expected challenge level of the selected opponents."""
        if not opponents:
            return 0.0
        
        total_challenge = 0.0
        for opponent in opponents:
            # Base challenge from opponent difficulty
            base_challenge = self._estimate_opponent_difficulty(opponent)
            
            # Adjust for performance metrics
            win_rate = opponent.performance_metrics.get('win_rate', 0.5)
            performance_modifier = win_rate * 0.5 + 0.5  # Scale 0.5-1.0
            
            total_challenge += base_challenge * performance_modifier
        
        # Account for multiple opponents (more opponents = higher challenge)
        multi_opponent_modifier = 1.0 + (len(opponents) - 1) * 0.3
        
        return min(1.0, (total_challenge / len(opponents)) * multi_opponent_modifier)

    def _estimate_opponent_difficulty(self, opponent: OpponentConfig) -> float:
        """Estimate the difficulty level of an opponent (0.0 to 1.0)."""
        if opponent.opponent_type == "rules_based":
            difficulty_map = {
                DifficultyLevel.BEGINNER: 0.2,
                DifficultyLevel.INTERMEDIATE: 0.5,
                DifficultyLevel.ADVANCED: 0.7,
                DifficultyLevel.EXPERT: 0.9
            }
            return difficulty_map.get(opponent.difficulty_level, 0.5)
        
        elif opponent.opponent_type == "rl_model":
            # Estimate based on generation and performance
            generation_factor = min(1.0, (opponent.generation or 0) / 10.0)  # Assume max 10 generations
            performance_factor = opponent.performance_metrics.get('win_rate', 0.5)
            
            return (generation_factor * 0.6 + performance_factor * 0.4)
        
        return 0.5  # Default difficulty

    def _calculate_opponent_diversity(
        self, 
        candidate: OpponentConfig, 
        selected: List[OpponentConfig]
    ) -> float:
        """Calculate how diverse a candidate is compared to already selected opponents."""
        if not selected:
            return 1.0
        
        diversity_score = 0.0
        
        for selected_opponent in selected:
            # Type diversity (base score)
            type_diversity = 0.5 if candidate.opponent_type != selected_opponent.opponent_type else 0.1
            diversity_score += type_diversity
            
            # Difficulty diversity
            candidate_diff = self._estimate_opponent_difficulty(candidate)
            selected_diff = self._estimate_opponent_difficulty(selected_opponent)
            difficulty_distance = abs(candidate_diff - selected_diff)
            diversity_score += difficulty_distance
            
            # Generation diversity (for RL models)
            if (candidate.opponent_type == "rl_model" and 
                selected_opponent.opponent_type == "rl_model"):
                gen_distance = abs((candidate.generation or 0) - (selected_opponent.generation or 0))
                diversity_score += min(0.5, gen_distance / 5.0)  # Normalize by max expected gap
            
            # ID diversity (ensure we don't select the same opponent)
            if candidate.opponent_id != selected_opponent.opponent_id:
                diversity_score += 0.1
        
        return diversity_score / len(selected)

    def _calculate_cohort_diversity(self) -> float:
        """Calculate the overall diversity score of the current cohort."""
        if len(self.active_cohort) <= 1:
            return 0.0
        
        total_diversity = 0.0
        comparisons = 0
        
        for i, opponent1 in enumerate(self.active_cohort):
            for j, opponent2 in enumerate(self.active_cohort[i+1:], i+1):
                diversity = self._calculate_opponent_diversity(opponent1, [opponent2])
                total_diversity += diversity
                comparisons += 1
        
        return total_diversity / max(1, comparisons)

    def _update_opponent_usage(self, selected_opponents: List[OpponentConfig]) -> None:
        """Update usage tracking for selected opponents."""
        for opponent in selected_opponents:
            opponent.usage_count += 1
            opponent.last_used = datetime.now()
            
            # Update metrics
            if opponent.opponent_id not in self.metrics.opponent_usage:
                self.metrics.opponent_usage[opponent.opponent_id] = 0
            self.metrics.opponent_usage[opponent.opponent_id] += 1

    def _get_selection_rationale(self) -> str:
        """Get a human-readable rationale for the current selection strategy."""
        strategy = self.config.selection_strategy
        
        rationales = {
            OpponentSelectionStrategy.RANDOM: "Random selection for unbiased sampling",
            OpponentSelectionStrategy.WEIGHTED_PERFORMANCE: "Performance-weighted selection for challenging opponents",
            OpponentSelectionStrategy.ROUND_ROBIN: "Round-robin selection for balanced exposure",
            OpponentSelectionStrategy.DIFFICULTY_PROGRESSION: f"Difficulty-based selection (level: {self.current_difficulty:.2f})",
            OpponentSelectionStrategy.DIVERSE_SAMPLING: "Diverse sampling for varied training experiences",
            OpponentSelectionStrategy.CURRICULUM_LEARNING: f"Curriculum learning (stage: {self.curriculum_stage})"
        }
        
        return rationales.get(strategy, "Unknown selection strategy")

    def _estimate_rules_based_performance(self, difficulty: DifficultyLevel) -> Dict[str, float]:
        """Estimate performance metrics for rules-based bots."""
        performance_estimates = {
            DifficultyLevel.BEGINNER: {'win_rate': 0.2, 'average_reward': 50.0},
            DifficultyLevel.INTERMEDIATE: {'win_rate': 0.4, 'average_reward': 100.0},
            DifficultyLevel.ADVANCED: {'win_rate': 0.6, 'average_reward': 150.0},
            DifficultyLevel.EXPERT: {'win_rate': 0.8, 'average_reward': 200.0}
        }
        return performance_estimates.get(difficulty, {'win_rate': 0.5, 'average_reward': 100.0})

    def _calculate_rules_based_weight(self, difficulty: DifficultyLevel) -> float:
        """Calculate selection weight for rules-based bots."""
        weight_map = {
            DifficultyLevel.BEGINNER: 0.6,
            DifficultyLevel.INTERMEDIATE: 0.8,
            DifficultyLevel.ADVANCED: 1.0,
            DifficultyLevel.EXPERT: 1.2
        }
        return weight_map.get(difficulty, 1.0)

    def _calculate_model_weight(self, model_metadata: RLModel) -> float:
        """Calculate selection weight for RL models."""
        # Base weight on performance and recency
        performance_weight = model_metadata.performance_metrics.get('win_rate', 0.5)
        recency_weight = min(1.0, model_metadata.generation / 10.0)  # Newer models weighted higher
        
        return (performance_weight * 0.7 + recency_weight * 0.3)