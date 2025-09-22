"""
Example usage of the RL training engine.

This script demonstrates how to use the training engine to train successive
generations of RL models with different algorithms.
"""

import asyncio
import logging
import numpy as np
from pathlib import Path

from bot.rl_bot_system.training.training_engine import TrainingEngine, TrainingConfig
from bot.rl_bot_system.training.model_manager import ModelManager
from bot.rl_bot_system.training.cohort_training import CohortTrainingSystem, CohortConfig
from bot.rl_bot_system.training.hyperparameter_tuning import HyperparameterTuner
from bot.rl_bot_system.environment import GameEnvironment, TrainingMode
from bot.game_client import GameClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockGameEnvironment:
    """Mock game environment for demonstration."""
    
    def __init__(self):
        self.observation_space = type('obj', (object,), {'shape': (20,)})()
        self.action_space = type('obj', (object,), {'n': 9, 'sample': lambda: np.random.randint(0, 9)})()
        self.training_mode = TrainingMode.TRAINING
        self.step_count = 0
        
    def reset(self):
        self.step_count = 0
        state = np.random.random(20)
        info = {}
        return state, info
    
    def step(self, action):
        self.step_count += 1
        next_state = np.random.random(20)
        
        # Simple reward function for demonstration
        reward = np.random.uniform(-1, 1)
        if self.step_count > 50:  # Episode ends after 50 steps
            reward += 5.0 if np.random.random() > 0.5 else -5.0  # Win/loss bonus
        
        terminated = self.step_count >= 50
        truncated = False
        info = {}
        
        return next_state, reward, terminated, truncated, info
    
    def set_training_mode(self, mode):
        self.training_mode = mode


def demonstrate_basic_training():
    """Demonstrate basic training with a single algorithm."""
    logger.info("=== Basic Training Demonstration ===")
    
    # Set up components
    model_manager = ModelManager(models_dir="bot/data/demo_models")
    training_engine = TrainingEngine(model_manager, device='cpu')
    mock_env = MockGameEnvironment()
    
    # Configure training
    config = TrainingConfig(
        algorithm='DQN',
        max_episodes=50,
        max_steps_per_episode=50,
        evaluation_frequency=20,
        evaluation_episodes=5,
        use_behavior_cloning=False,  # Skip for demo
        hyperparameters={
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'epsilon_decay': 0.99,
            'batch_size': 32
        }
    )
    
    # Train first generation
    logger.info("Training Generation 1 (DQN)")
    model_gen1 = training_engine.train_next_generation(
        generation=1,
        config=config,
        environment=mock_env
    )
    
    logger.info(f"Generation 1 completed. Performance: {model_gen1.performance_metrics}")
    
    return model_gen1, training_engine, mock_env


def demonstrate_successive_training():
    """Demonstrate successive training with knowledge transfer."""
    logger.info("=== Successive Training Demonstration ===")
    
    # Get first generation from basic training
    model_gen1, training_engine, mock_env = demonstrate_basic_training()
    
    # Train second generation with knowledge transfer
    config_gen2 = TrainingConfig(
        algorithm='DQN',
        max_episodes=30,
        max_steps_per_episode=50,
        evaluation_frequency=15,
        evaluation_episodes=3,
        use_behavior_cloning=False,
        hyperparameters={
            'learning_rate': 5e-4,  # Lower learning rate for fine-tuning
            'gamma': 0.99,
            'epsilon': 0.5,  # Start with lower exploration
            'epsilon_decay': 0.995,
            'batch_size': 32
        }
    )
    
    logger.info("Training Generation 2 (with knowledge transfer)")
    model_gen2 = training_engine.train_next_generation(
        generation=2,
        config=config_gen2,
        environment=mock_env,
        previous_model=model_gen1
    )
    
    logger.info(f"Generation 2 completed. Performance: {model_gen2.performance_metrics}")
    
    # Compare performance
    gen1_final_eval = model_gen1.performance_metrics['final_evaluation']
    gen2_final_eval = model_gen2.performance_metrics['final_evaluation']
    
    logger.info(f"Generation 1 win rate: {gen1_final_eval['win_rate']:.2%}")
    logger.info(f"Generation 2 win rate: {gen2_final_eval['win_rate']:.2%}")
    
    improvement = gen2_final_eval['win_rate'] - gen1_final_eval['win_rate']
    logger.info(f"Improvement: {improvement:+.2%}")
    
    return model_gen2


def demonstrate_algorithm_comparison():
    """Demonstrate training with different algorithms."""
    logger.info("=== Algorithm Comparison Demonstration ===")
    
    model_manager = ModelManager(models_dir="bot/data/algorithm_comparison")
    mock_env = MockGameEnvironment()
    
    algorithms = ['DQN', 'PPO', 'A2C']
    results = {}
    
    for algorithm in algorithms:
        logger.info(f"Training with {algorithm}")
        
        training_engine = TrainingEngine(model_manager, device='cpu')
        
        config = TrainingConfig(
            algorithm=algorithm,
            max_episodes=30,
            max_steps_per_episode=50,
            evaluation_frequency=15,
            evaluation_episodes=5,
            use_behavior_cloning=False
        )
        
        model = training_engine.train_next_generation(
            generation=1,
            config=config,
            environment=mock_env
        )
        
        final_eval = model.performance_metrics['final_evaluation']
        results[algorithm] = {
            'win_rate': final_eval['win_rate'],
            'avg_reward': final_eval['average_reward'],
            'training_time': model.performance_metrics['training_time']
        }
        
        logger.info(f"{algorithm} - Win Rate: {final_eval['win_rate']:.2%}, "
                   f"Avg Reward: {final_eval['average_reward']:.2f}")
    
    # Print comparison
    logger.info("\n=== Algorithm Comparison Results ===")
    for algorithm, metrics in results.items():
        logger.info(f"{algorithm:>5}: Win Rate: {metrics['win_rate']:>6.2%}, "
                   f"Avg Reward: {metrics['avg_reward']:>6.2f}, "
                   f"Time: {metrics['training_time']:>6.1f}s")
    
    return results


def demonstrate_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning."""
    logger.info("=== Hyperparameter Tuning Demonstration ===")
    
    mock_env = MockGameEnvironment()
    
    def objective_function(hyperparameters):
        """Objective function for hyperparameter tuning."""
        model_manager = ModelManager(models_dir="bot/data/hp_tuning")
        training_engine = TrainingEngine(model_manager, device='cpu')
        
        config = TrainingConfig(
            algorithm='DQN',
            max_episodes=20,  # Shorter for tuning
            max_steps_per_episode=30,
            evaluation_episodes=3,
            use_behavior_cloning=False,
            hyperparameters=hyperparameters
        )
        
        try:
            model = training_engine.train_next_generation(
                generation=1,
                config=config,
                environment=mock_env
            )
            
            # Return win rate as the objective
            return model.performance_metrics['final_evaluation']['win_rate']
            
        except Exception as e:
            logger.error(f"Training failed with hyperparameters {hyperparameters}: {e}")
            return 0.0  # Return poor score for failed runs
    
    # Set up hyperparameter tuner
    tuner = HyperparameterTuner(
        algorithm='DQN',
        tuning_strategy='random',
        n_trials=5  # Small number for demo
    )
    
    # Run tuning
    best_config = tuner.tune(objective_function, maximize=True)
    
    logger.info(f"Best hyperparameters found:")
    for param, value in best_config.parameters.items():
        logger.info(f"  {param}: {value}")
    logger.info(f"Best score: {best_config.score:.2%}")
    
    return best_config


def demonstrate_cohort_training():
    """Demonstrate cohort-based training."""
    logger.info("=== Cohort Training Demonstration ===")
    
    # This is a simplified demonstration since we don't have
    # the full cohort system implemented in this example
    
    model_manager = ModelManager(models_dir="bot/data/cohort_training")
    
    # Create a mock cohort system
    cohort_system = type('CohortSystem', (), {
        'select_opponents': lambda self, generation, config: [],
        'update_performance': lambda self, generation, metrics: None
    })()
    
    training_engine = TrainingEngine(
        model_manager=model_manager,
        cohort_system=cohort_system,
        device='cpu'
    )
    
    mock_env = MockGameEnvironment()
    
    # Train multiple generations
    models = []
    for generation in range(1, 4):
        logger.info(f"Training Generation {generation}")
        
        config = TrainingConfig(
            algorithm='PPO',
            max_episodes=25,
            max_steps_per_episode=40,
            evaluation_episodes=3,
            use_behavior_cloning=False
        )
        
        previous_model = models[-1] if models else None
        
        model = training_engine.train_next_generation(
            generation=generation,
            config=config,
            environment=mock_env,
            previous_model=previous_model
        )
        
        models.append(model)
        
        final_eval = model.performance_metrics['final_evaluation']
        logger.info(f"Generation {generation} - Win Rate: {final_eval['win_rate']:.2%}")
    
    # Show progression
    logger.info("\n=== Generation Progression ===")
    for i, model in enumerate(models, 1):
        final_eval = model.performance_metrics['final_evaluation']
        logger.info(f"Generation {i}: {final_eval['win_rate']:.2%} win rate")
    
    return models


def main():
    """Run all demonstrations."""
    logger.info("Starting RL Training Engine Demonstrations")
    
    try:
        # Basic training
        demonstrate_basic_training()
        
        # Successive training
        demonstrate_successive_training()
        
        # Algorithm comparison
        demonstrate_algorithm_comparison()
        
        # Hyperparameter tuning
        demonstrate_hyperparameter_tuning()
        
        # Cohort training
        demonstrate_cohort_training()
        
        logger.info("All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()