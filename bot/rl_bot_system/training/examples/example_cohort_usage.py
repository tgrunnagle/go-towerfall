"""
Example usage of the cohort-based training system.

This example demonstrates how to set up and use the CohortTrainingSystem
for training successive RL bot generations with different opponent selection
strategies and difficulty progression modes.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from bot.rl_bot_system.training.cohort_training import (
    CohortTrainingSystem,
    CohortConfig,
    OpponentSelectionStrategy,
    DifficultyProgression
)
from bot.rl_bot_system.training.model_manager import ModelManager
from bot.rl_bot_system.training.training_session import TrainingSession, TrainingConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_cohort_training():
    """
    Example of basic cohort-based training setup.
    """
    logger.info("=== Basic Cohort Training Example ===")
    
    # Initialize model manager
    model_manager = ModelManager("bot/data/models")
    
    # Configure cohort training
    cohort_config = CohortConfig(
        cohort_size=5,
        max_enemy_count=2,
        min_enemy_count=1,
        selection_strategy=OpponentSelectionStrategy.WEIGHTED_PERFORMANCE,
        difficulty_progression=DifficultyProgression.LINEAR,
        include_rules_based=True,
        rules_based_ratio=0.4,
        include_previous_generations=True,
        max_generation_gap=3
    )
    
    # Create cohort training system
    cohort_system = CohortTrainingSystem(model_manager, cohort_config)
    
    # Initialize for training generation 3
    current_generation = 3
    await cohort_system.initialize_cohort(current_generation)
    
    # Display cohort information
    cohort_info = cohort_system.get_cohort_info()
    logger.info(f"Initialized cohort for generation {current_generation}")
    logger.info(f"Active cohort size: {cohort_info['cohortSize']}")
    logger.info(f"Available opponents: {cohort_info['availableOpponents']}")
    
    # Display opponent details
    logger.info("Active opponents:")
    for opponent in cohort_info['opponents']:
        logger.info(f"  - {opponent['id']}: {opponent['type']} "
                   f"(gen: {opponent['generation']}, difficulty: {opponent['difficulty']})")
    
    # Simulate training episodes
    logger.info("\nSimulating training episodes...")
    
    for episode in range(10):
        training_progress = episode / 10.0
        
        # Select opponents for this episode
        setup = await cohort_system.select_episode_opponents(
            f"episode_{episode:03d}",
            training_progress
        )
        
        logger.info(f"Episode {episode}: {setup.enemy_count} opponents, "
                   f"difficulty: {setup.difficulty_level:.2f}, "
                   f"challenge: {setup.expected_challenge:.2f}")
        
        # Simulate episode execution and results
        episode_results = simulate_episode_execution(setup)
        
        # Update cohort metrics
        await cohort_system.update_episode_results(
            setup.episode_id,
            setup,
            episode_results
        )
    
    # Display final metrics
    final_info = cohort_system.get_cohort_info()
    logger.info(f"\nFinal metrics after {final_info['trainingEpisodes']} episodes:")
    logger.info(f"Average difficulty: {final_info['metrics']['averageDifficulty']:.2f}")
    logger.info(f"Diversity score: {final_info['metrics']['diversityScore']:.2f}")
    
    logger.info("Opponent usage:")
    for opponent_id, usage_count in final_info['metrics']['opponentUsage'].items():
        win_rate = final_info['metrics']['winRatesByOpponent'].get(opponent_id, 0.0)
        logger.info(f"  - {opponent_id}: {usage_count} uses, {win_rate:.1%} win rate")


async def example_curriculum_learning():
    """
    Example of curriculum learning with staged difficulty progression.
    """
    logger.info("\n=== Curriculum Learning Example ===")
    
    model_manager = ModelManager("bot/data/models")
    
    # Configure curriculum learning
    cohort_config = CohortConfig(
        cohort_size=6,
        max_enemy_count=3,
        min_enemy_count=1,
        selection_strategy=OpponentSelectionStrategy.CURRICULUM_LEARNING,
        difficulty_progression=DifficultyProgression.STAGED,
        curriculum_stages=4,
        performance_threshold=0.7,
        include_rules_based=True,
        rules_based_ratio=0.5
    )
    
    cohort_system = CohortTrainingSystem(model_manager, cohort_config)
    await cohort_system.initialize_cohort(current_generation=4)
    
    logger.info("Curriculum learning with 4 stages")
    
    # Simulate curriculum progression
    episodes_per_stage = 25
    total_episodes = cohort_config.curriculum_stages * episodes_per_stage
    
    for episode in range(total_episodes):
        training_progress = episode / total_episodes
        stage = int(training_progress * cohort_config.curriculum_stages)
        
        setup = await cohort_system.select_episode_opponents(
            f"curriculum_episode_{episode:03d}",
            training_progress
        )
        
        if episode % episodes_per_stage == 0:
            logger.info(f"\n--- Stage {stage} (Progress: {training_progress:.1%}) ---")
        
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Stage {stage}, "
                       f"{setup.enemy_count} enemies, "
                       f"difficulty: {setup.difficulty_level:.2f}")
        
        # Simulate episode with curriculum-appropriate results
        episode_results = simulate_curriculum_episode(setup, stage, cohort_config.curriculum_stages)
        
        await cohort_system.update_episode_results(
            setup.episode_id,
            setup,
            episode_results
        )
    
    # Show curriculum progression results
    final_info = cohort_system.get_cohort_info()
    logger.info(f"\nCurriculum completed:")
    logger.info(f"Final stage: {final_info['progressionStage']}")
    logger.info(f"Final difficulty: {final_info['currentDifficulty']:.2f}")


async def example_adaptive_difficulty():
    """
    Example of adaptive difficulty progression based on performance.
    """
    logger.info("\n=== Adaptive Difficulty Example ===")
    
    model_manager = ModelManager("bot/data/models")
    
    # Configure adaptive difficulty
    cohort_config = CohortConfig(
        cohort_size=4,
        max_enemy_count=2,
        min_enemy_count=1,
        selection_strategy=OpponentSelectionStrategy.DIVERSE_SAMPLING,
        difficulty_progression=DifficultyProgression.ADAPTIVE,
        adaptation_window=20,
        performance_threshold=0.6,
        progression_rate=0.05
    )
    
    cohort_system = CohortTrainingSystem(model_manager, cohort_config)
    await cohort_system.initialize_cohort(current_generation=2)
    
    logger.info("Adaptive difficulty based on performance")
    
    # Simulate varying performance to trigger adaptation
    performance_phases = [
        ("struggling", 0.3, 30),  # Low win rate - should decrease difficulty
        ("improving", 0.7, 30),   # High win rate - should increase difficulty
        ("stable", 0.5, 20)       # Moderate win rate - should stabilize
    ]
    
    episode_count = 0
    
    for phase_name, target_win_rate, phase_episodes in performance_phases:
        logger.info(f"\n--- {phase_name.title()} Phase (target win rate: {target_win_rate:.1%}) ---")
        
        phase_start_difficulty = cohort_system.current_difficulty
        
        for i in range(phase_episodes):
            setup = await cohort_system.select_episode_opponents(
                f"adaptive_episode_{episode_count:03d}",
                episode_count / 80.0  # Total episodes across all phases
            )
            
            # Simulate results based on target performance
            episode_results = simulate_performance_based_episode(setup, target_win_rate)
            
            await cohort_system.update_episode_results(
                setup.episode_id,
                setup,
                episode_results
            )
            
            if i % 10 == 0:
                recent_win_rate = sum(cohort_system.recent_performance[-10:]) / min(10, len(cohort_system.recent_performance))
                logger.info(f"Episode {episode_count}: difficulty: {setup.difficulty_level:.2f}, "
                           f"recent win rate: {recent_win_rate:.1%}")
            
            episode_count += 1
        
        phase_end_difficulty = cohort_system.current_difficulty
        difficulty_change = phase_end_difficulty - phase_start_difficulty
        
        logger.info(f"Phase completed: difficulty changed by {difficulty_change:+.2f} "
                   f"({phase_start_difficulty:.2f} → {phase_end_difficulty:.2f})")


async def example_multi_agent_scenarios():
    """
    Example of multi-agent training scenarios with variable enemy counts.
    """
    logger.info("\n=== Multi-Agent Scenarios Example ===")
    
    model_manager = ModelManager("bot/data/models")
    
    # Configure for multi-agent scenarios
    cohort_config = CohortConfig(
        cohort_size=8,
        max_enemy_count=4,  # Up to 4 enemies
        min_enemy_count=1,
        selection_strategy=OpponentSelectionStrategy.ROUND_ROBIN,
        difficulty_progression=DifficultyProgression.LINEAR,
        progression_rate=0.8
    )
    
    cohort_system = CohortTrainingSystem(model_manager, cohort_config)
    await cohort_system.initialize_cohort(current_generation=5)
    
    logger.info("Multi-agent training scenarios")
    
    # Track enemy count distribution
    enemy_count_distribution = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for episode in range(50):
        training_progress = episode / 50.0
        
        setup = await cohort_system.select_episode_opponents(
            f"multiagent_episode_{episode:03d}",
            training_progress
        )
        
        enemy_count_distribution[setup.enemy_count] += 1
        
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: {setup.enemy_count} enemies, "
                       f"progress: {training_progress:.1%}, "
                       f"challenge: {setup.expected_challenge:.2f}")
            
            # Show opponent composition
            opponent_types = {}
            for opponent in setup.opponents:
                opp_type = opponent.opponent_type
                opponent_types[opp_type] = opponent_types.get(opp_type, 0) + 1
            
            logger.info(f"  Opponent composition: {dict(opponent_types)}")
        
        # Simulate multi-agent episode
        episode_results = simulate_multiagent_episode(setup)
        
        await cohort_system.update_episode_results(
            setup.episode_id,
            setup,
            episode_results
        )
    
    # Show enemy count distribution
    logger.info("\nEnemy count distribution:")
    for count, frequency in enemy_count_distribution.items():
        percentage = frequency / 50 * 100
        logger.info(f"  {count} enemies: {frequency} episodes ({percentage:.1f}%)")


def simulate_episode_execution(setup) -> Dict[str, Any]:
    """
    Simulate episode execution and return results.
    This is a placeholder for actual RL training integration.
    """
    import random
    
    # Simulate results based on expected challenge
    base_win_probability = max(0.1, 0.8 - setup.expected_challenge)
    won = random.random() < base_win_probability
    
    # Simulate reward based on performance
    base_reward = 100.0
    challenge_bonus = setup.expected_challenge * 50.0
    performance_modifier = 1.5 if won else 0.7
    
    total_reward = (base_reward + challenge_bonus) * performance_modifier
    
    return {
        "won": won,
        "total_reward": total_reward,
        "episode_length": random.randint(50, 200),
        "challenge_level": setup.expected_challenge
    }


def simulate_curriculum_episode(setup, stage: int, total_stages: int) -> Dict[str, Any]:
    """Simulate episode results appropriate for curriculum stage."""
    import random
    
    # Win rate should improve as we progress through curriculum
    stage_progress = stage / max(1, total_stages - 1)
    base_win_rate = 0.3 + stage_progress * 0.5  # 30% to 80% win rate progression
    
    # Adjust for episode difficulty
    difficulty_modifier = 1.0 - setup.difficulty_level * 0.3
    win_probability = base_win_rate * difficulty_modifier
    
    won = random.random() < win_probability
    
    return {
        "won": won,
        "total_reward": random.uniform(80, 180) * (1.5 if won else 0.8),
        "episode_length": random.randint(60, 150),
        "curriculum_stage": stage
    }


def simulate_performance_based_episode(setup, target_win_rate: float) -> Dict[str, Any]:
    """Simulate episode results targeting a specific win rate."""
    import random
    
    # Add some noise to the target win rate
    actual_win_rate = target_win_rate + random.uniform(-0.1, 0.1)
    actual_win_rate = max(0.0, min(1.0, actual_win_rate))
    
    won = random.random() < actual_win_rate
    
    return {
        "won": won,
        "total_reward": random.uniform(70, 160) * (1.3 if won else 0.9),
        "episode_length": random.randint(40, 180),
        "target_performance": target_win_rate
    }


def simulate_multiagent_episode(setup) -> Dict[str, Any]:
    """Simulate multi-agent episode results."""
    import random
    
    # More enemies = higher challenge but potentially higher rewards
    enemy_count_modifier = 1.0 + (setup.enemy_count - 1) * 0.2
    base_win_rate = 0.6 / enemy_count_modifier  # Harder with more enemies
    
    won = random.random() < base_win_rate
    
    # Reward scales with enemy count and difficulty
    base_reward = 80.0 * enemy_count_modifier
    difficulty_bonus = setup.difficulty_level * 40.0
    
    total_reward = (base_reward + difficulty_bonus) * (1.4 if won else 0.8)
    
    return {
        "won": won,
        "total_reward": total_reward,
        "episode_length": random.randint(80, 250),
        "enemy_count": setup.enemy_count,
        "multiagent_scenario": True
    }


async def main():
    """Run all cohort training examples."""
    try:
        await example_basic_cohort_training()
        await example_curriculum_learning()
        await example_adaptive_difficulty()
        await example_multi_agent_scenarios()
        
        logger.info("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())