"""
Example usage of the EvaluationManager for model evaluation and comparison.

This example demonstrates:
1. Setting up the evaluation system
2. Running evaluations against multiple opponents
3. Comparing model generations statistically
4. Generating evaluation reports
5. Running tournament-style evaluations
"""

import logging
from pathlib import Path

from bot.rl_bot_system.evaluation import EvaluationManager
from bot.rl_bot_system.training.model_manager import ModelManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate EvaluationManager usage."""
    
    # Initialize model manager and evaluation manager
    model_manager = ModelManager("data/examples/models")
    evaluator = EvaluationManager(
        model_manager=model_manager,
        results_dir="data/examples/evaluations"
    )
    
    logger.info("=== RL Bot Evaluation System Demo ===")
    
    # Example 1: Basic model evaluation
    logger.info("\n1. Running basic model evaluation...")
    
    try:
        # Evaluate generation 2 against generations 0 and 1
        evaluation_result = evaluator.run_evaluation(
            model_generation=2,
            opponent_generations=[0, 1],
            episodes_per_opponent=50
        )
        
        logger.info(f"Evaluation completed:")
        logger.info(f"  - Model Generation: {evaluation_result.model_generation}")
        logger.info(f"  - Total Games: {evaluation_result.total_games}")
        logger.info(f"  - Win Rate: {evaluation_result.win_rate:.3f}")
        logger.info(f"  - Average Reward: {evaluation_result.average_reward:.2f}")
        
        # Generate reports in different formats
        logger.info("\n2. Generating evaluation reports...")
        
        json_report = evaluator.generate_report(evaluation_result, output_format='json')
        markdown_report = evaluator.generate_report(evaluation_result, output_format='markdown')
        html_report = evaluator.generate_report(evaluation_result, output_format='html')
        
        logger.info(f"Reports generated:")
        logger.info(f"  - JSON: {json_report}")
        logger.info(f"  - Markdown: {markdown_report}")
        logger.info(f"  - HTML: {html_report}")
        
    except FileNotFoundError as e:
        logger.warning(f"Model not found for basic evaluation: {e}")
        logger.info("This is expected if no models have been trained yet.")
    
    # Example 2: Statistical comparison between generations
    logger.info("\n3. Running statistical comparison...")
    
    try:
        comparison_result = evaluator.compare_generations(
            generation_a=2,
            generation_b=1,
            comparison_episodes=100,
            significance_level=0.05
        )
        
        logger.info(f"Comparison results:")
        logger.info(f"  - Generation A: {comparison_result['generation_a']}")
        logger.info(f"  - Generation B: {comparison_result['generation_b']}")
        
        summary = comparison_result['summary']
        logger.info(f"  - Better Generation: {summary['better_generation']}")
        logger.info(f"  - Win Rate Difference: {summary['win_rate_difference']:.3f}")
        logger.info(f"  - Statistically Significant: {summary['statistically_significant']}")
        logger.info(f"  - Practically Significant: {summary['practical_significance']}")
        
        # Show statistical test results
        stats = comparison_result['statistical_tests']
        logger.info(f"  - T-test p-value: {stats['reward_t_test']['p_value']:.4f}")
        logger.info(f"  - Effect size (Cohen's d): {stats['effect_size_cohens_d']:.3f}")
        
    except FileNotFoundError as e:
        logger.warning(f"Models not found for comparison: {e}")
        logger.info("This is expected if fewer than 2 models have been trained.")
    
    # Example 3: Tournament evaluation
    logger.info("\n4. Running tournament evaluation...")
    
    try:
        tournament_result = evaluator.tournament_evaluation(
            generations=[0, 1, 2, 3],
            episodes_per_matchup=25
        )
        
        logger.info(f"Tournament results:")
        logger.info(f"  - Tournament ID: {tournament_result['tournament_id']}")
        logger.info(f"  - Participants: {tournament_result['generations']}")
        logger.info(f"  - Winner: Generation {tournament_result['winner']}")
        
        # Show rankings
        logger.info("  - Final Rankings:")
        for i, ranking in enumerate(tournament_result['rankings']):
            logger.info(f"    {i+1}. Generation {ranking['generation']} - "
                       f"Win Rate: {ranking['win_rate']:.3f}, "
                       f"Score: {ranking['score']:.3f}")
        
    except FileNotFoundError as e:
        logger.warning(f"Models not found for tournament: {e}")
        logger.info("This is expected if fewer than 4 models have been trained.")
    
    # Example 4: List and load previous evaluations
    logger.info("\n5. Listing previous evaluations...")
    
    evaluations = evaluator.list_evaluations()
    logger.info(f"Found {len(evaluations)} previous evaluations:")
    
    for eval_id, eval_result in evaluations[:3]:  # Show first 3
        logger.info(f"  - {eval_id}: Gen {eval_result.model_generation}, "
                   f"Win Rate: {eval_result.win_rate:.3f}, "
                   f"Date: {eval_result.evaluation_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Example 5: Advanced evaluation configuration
    logger.info("\n6. Advanced evaluation with custom configuration...")
    
    try:
        # Custom evaluation configuration
        evaluation_config = {
            'reward_scaling': 1.0,
            'episode_timeout': 1000,
            'random_seed': 42
        }
        
        advanced_evaluation = evaluator.run_evaluation(
            model_generation=1,
            opponent_generations=[0],
            episodes_per_opponent=30,
            evaluation_config=evaluation_config
        )
        
        logger.info(f"Advanced evaluation completed:")
        logger.info(f"  - Performance metrics available: {list(advanced_evaluation.performance_metrics.keys())}")
        
        # Show detailed performance metrics
        metrics = advanced_evaluation.performance_metrics
        if 'strategic_diversity' in metrics:
            logger.info(f"  - Strategic Diversity: {metrics['strategic_diversity']:.3f}")
        if 'performance_consistency' in metrics:
            logger.info(f"  - Performance Consistency: {metrics['performance_consistency']:.3f}")
        if 'episode_efficiency_mean' in metrics:
            logger.info(f"  - Episode Efficiency: {metrics['episode_efficiency_mean']:.3f}")
        
    except FileNotFoundError as e:
        logger.warning(f"Models not found for advanced evaluation: {e}")
    
    logger.info("\n=== Evaluation Demo Complete ===")
    logger.info("To use the evaluation system with real models:")
    logger.info("1. Train some models using the TrainingEngine")
    logger.info("2. Run evaluations to compare their performance")
    logger.info("3. Use statistical comparisons to validate improvements")
    logger.info("4. Generate reports for analysis and documentation")


if __name__ == "__main__":
    main()