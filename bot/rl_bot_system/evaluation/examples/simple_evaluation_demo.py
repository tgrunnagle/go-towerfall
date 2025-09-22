"""
Simple demonstration of EvaluationManager functionality.

This example shows the core evaluation features without complex imports.
Run this from the bot/ directory with: python -m rl_bot_system.evaluation.examples.simple_evaluation_demo
"""

import logging
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_evaluation_system():
    """Create a mock evaluation system for demonstration."""
    # Import here to avoid path issues
    from bot.rl_bot_system.evaluation.evaluator import EvaluationManager, EvaluationResult
    from bot.rl_bot_system.training.model_manager import RLModel
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    
    # Create mock model manager
    mock_model_manager = Mock()
    
    # Mock model metadata
    model_metadata = RLModel(
        generation=1,
        algorithm='DQN',
        network_architecture={'input_size': 100, 'hidden_sizes': [256, 128]},
        hyperparameters={'learning_rate': 0.001, 'gamma': 0.99},
        training_episodes=5000,
        performance_metrics={'win_rate': 0.75, 'average_reward': 85.0},
        parent_generation=0,
        created_at=datetime.now(),
        model_path='mock_path'
    )
    
    mock_model_manager._load_model_metadata.return_value = model_metadata
    mock_model_manager.load_model.return_value = (Mock(), model_metadata)
    
    # Create evaluation manager
    evaluator = EvaluationManager(
        model_manager=mock_model_manager,
        results_dir=f"{temp_dir}/evaluations"
    )
    
    return evaluator, temp_dir


def main():
    """Demonstrate EvaluationManager functionality."""
    logger.info("=== EvaluationManager Demo ===")
    
    # Create mock system
    evaluator, temp_dir = create_mock_evaluation_system()
    
    try:
        # Demo 1: Basic evaluation
        logger.info("\n1. Running basic model evaluation...")
        
        evaluation_result = evaluator.run_evaluation(
            model_generation=1,
            opponent_generations=[0],
            episodes_per_opponent=20
        )
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  - Model Generation: {evaluation_result.model_generation}")
        logger.info(f"  - Total Games: {evaluation_result.total_games}")
        logger.info(f"  - Win Rate: {evaluation_result.win_rate:.3f}")
        logger.info(f"  - Average Reward: {evaluation_result.average_reward:.2f}")
        
        # Demo 2: Performance metrics
        logger.info("\n2. Performance metrics analysis...")
        
        metrics = evaluation_result.performance_metrics
        logger.info(f"Performance Metrics:")
        logger.info(f"  - Strategic Diversity: {metrics.get('strategic_diversity', 0):.3f}")
        logger.info(f"  - Performance Consistency: {metrics.get('performance_consistency', 0):.3f}")
        logger.info(f"  - Episode Efficiency: {metrics.get('episode_efficiency_mean', 0):.3f}")
        
        # Demo 3: Generate reports
        logger.info("\n3. Generating evaluation reports...")
        
        json_report = evaluator.generate_report(evaluation_result, output_format='json')
        markdown_report = evaluator.generate_report(evaluation_result, output_format='markdown')
        
        logger.info(f"Reports generated:")
        logger.info(f"  - JSON: {json_report}")
        logger.info(f"  - Markdown: {markdown_report}")
        
        # Demo 4: Statistical comparison
        logger.info("\n4. Statistical comparison between generations...")
        
        comparison = evaluator.compare_generations(
            generation_a=1,
            generation_b=0,
            comparison_episodes=50
        )
        
        summary = comparison['summary']
        logger.info(f"Comparison Results:")
        logger.info(f"  - Better Generation: {summary['better_generation']}")
        logger.info(f"  - Win Rate Difference: {summary['win_rate_difference']:.3f}")
        logger.info(f"  - Statistically Significant: {summary['statistically_significant']}")
        
        # Demo 5: Tournament evaluation
        logger.info("\n5. Tournament evaluation...")
        
        tournament = evaluator.tournament_evaluation(
            generations=[0, 1, 2],
            episodes_per_matchup=15
        )
        
        logger.info(f"Tournament Results:")
        logger.info(f"  - Winner: Generation {tournament['winner']}")
        logger.info(f"  - Rankings:")
        
        for i, ranking in enumerate(tournament['rankings']):
            logger.info(f"    {i+1}. Generation {ranking['generation']} - "
                       f"Win Rate: {ranking['win_rate']:.3f}")
        
        # Demo 6: List evaluations
        logger.info("\n6. Listing saved evaluations...")
        
        evaluations = evaluator.list_evaluations()
        logger.info(f"Found {len(evaluations)} evaluations:")
        
        for eval_id, eval_result in evaluations:
            logger.info(f"  - {eval_id}: Gen {eval_result.model_generation}, "
                       f"Win Rate: {eval_result.win_rate:.3f}")
        
        logger.info("\n=== Demo Complete ===")
        logger.info("The EvaluationManager provides:")
        logger.info("✓ Systematic model testing against multiple opponents")
        logger.info("✓ Statistical comparison between model generations")
        logger.info("✓ Comprehensive performance metrics calculation")
        logger.info("✓ Report generation in multiple formats")
        logger.info("✓ Tournament-style evaluation")
        logger.info("✓ Evaluation result storage and retrieval")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()