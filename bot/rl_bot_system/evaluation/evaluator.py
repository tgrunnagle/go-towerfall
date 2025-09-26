"""
Model Evaluation System for RL Bot System

Provides systematic model testing, statistical comparison between generations,
performance metrics calculation, and evaluation report generation with visualizations.
"""

import os
import json
import logging
import asyncio
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    VISUALIZATION_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    SCIPY_AVAILABLE = False

from rl_bot_system.training.model_manager import ModelManager, RLModel
from game_client import GameClient

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Data class representing evaluation results for a model."""
    model_generation: int
    opponent_generations: List[int]
    total_games: int
    wins: int
    losses: int
    draws: int
    average_reward: float
    win_rate: float
    performance_metrics: Dict[str, Any]
    evaluation_date: datetime
    evaluation_id: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['evaluation_date'] = self.evaluation_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'EvaluationResult':
        """Create from dictionary loaded from JSON."""
        data = data.copy()
        data['evaluation_date'] = datetime.fromisoformat(data['evaluation_date'])
        return cls(**data)


@dataclass
class GameEpisode:
    """Data class representing a single game episode."""
    episode_id: str
    model_generation: int
    opponent_generation: int
    states: List[dict]
    actions: List[int]
    rewards: List[float]
    total_reward: float
    episode_length: int
    game_result: str  # 'win', 'loss', 'draw'
    episode_metrics: Dict[str, Any]


class EvaluationManager:
    """
    Manages systematic model testing and performance evaluation.
    
    Provides:
    - Statistical comparison between model generations
    - Performance metrics calculation (win rate, rewards, strategic diversity)
    - Evaluation report generation with visualizations
    - Tournament-style evaluation between multiple generations
    """

    def __init__(
        self,
        model_manager: ModelManager,
        results_dir: str = "bot/data/evaluations",
        game_client_factory=None
    ):
        """
        Initialize EvaluationManager.
        
        Args:
            model_manager: ModelManager instance for loading models
            results_dir: Directory to store evaluation results
            game_client_factory: Factory function to create GameClient instances
        """
        self.model_manager = model_manager
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Default game client factory if none provided
        self.game_client_factory = game_client_factory or (lambda: GameClient())
        
        # Cache for evaluation results
        self._results_cache: Dict[str, EvaluationResult] = {}
        
        logger.info(f"EvaluationManager initialized with results directory: {self.results_dir}")

    def run_evaluation(
        self,
        model_generation: int,
        opponent_generations: List[int],
        episodes_per_opponent: int = 100,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Run systematic evaluation of a model against specified opponents.
        
        Args:
            model_generation: Generation of model to evaluate
            opponent_generations: List of opponent generations to test against
            episodes_per_opponent: Number of episodes to run against each opponent
            evaluation_config: Additional configuration for evaluation
            
        Returns:
            EvaluationResult: Comprehensive evaluation results
            
        Raises:
            FileNotFoundError: If model generation doesn't exist
            ValueError: If evaluation configuration is invalid
        """
        evaluation_id = f"eval_{model_generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting evaluation {evaluation_id} for generation {model_generation}")
        
        # Validate model exists
        try:
            model, model_metadata = self.model_manager.load_model(model_generation)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model generation {model_generation} not found")
        
        # Initialize evaluation tracking
        total_games = 0
        total_wins = 0
        total_losses = 0
        total_draws = 0
        total_rewards = []
        episode_data = []
        
        # Evaluate against each opponent
        for opponent_gen in opponent_generations:
            logger.info(f"Evaluating against generation {opponent_gen}")
            
            # Run episodes against this opponent
            opponent_results = self._run_episodes_against_opponent(
                model_generation,
                opponent_gen,
                episodes_per_opponent,
                evaluation_config
            )
            
            # Aggregate results
            total_games += len(opponent_results)
            for episode in opponent_results:
                if episode.game_result == 'win':
                    total_wins += 1
                elif episode.game_result == 'loss':
                    total_losses += 1
                else:
                    total_draws += 1
                
                total_rewards.append(episode.total_reward)
                episode_data.append(episode)
        
        # Calculate performance metrics
        win_rate = total_wins / total_games if total_games > 0 else 0.0
        average_reward = statistics.mean(total_rewards) if total_rewards else 0.0
        
        # Calculate additional performance metrics
        performance_metrics = self._calculate_performance_metrics(episode_data)
        
        # Create evaluation result
        evaluation_result = EvaluationResult(
            model_generation=model_generation,
            opponent_generations=opponent_generations,
            total_games=total_games,
            wins=total_wins,
            losses=total_losses,
            draws=total_draws,
            average_reward=average_reward,
            win_rate=win_rate,
            performance_metrics=performance_metrics,
            evaluation_date=datetime.now(),
            evaluation_id=evaluation_id
        )
        
        # Save evaluation results
        self._save_evaluation_result(evaluation_result, episode_data)
        
        # Cache result
        self._results_cache[evaluation_id] = evaluation_result
        
        logger.info(f"Evaluation {evaluation_id} completed: {total_wins}/{total_games} wins ({win_rate:.3f})")
        
        return evaluation_result

    def _run_episodes_against_opponent(
        self,
        model_generation: int,
        opponent_generation: int,
        num_episodes: int,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> List[GameEpisode]:
        """
        Run multiple episodes between two model generations.
        
        Args:
            model_generation: Generation of model being evaluated
            opponent_generation: Generation of opponent model
            num_episodes: Number of episodes to run
            evaluation_config: Additional configuration
            
        Returns:
            List[GameEpisode]: Results from all episodes
        """
        episodes = []
        
        # For now, simulate episodes since we don't have full game integration
        # In a real implementation, this would use the GameClient to run actual games
        for i in range(num_episodes):
            episode = self._simulate_episode(
                model_generation,
                opponent_generation,
                f"ep_{model_generation}_vs_{opponent_generation}_{i}",
                evaluation_config
            )
            episodes.append(episode)
        
        return episodes

    def _simulate_episode(
        self,
        model_generation: int,
        opponent_generation: int,
        episode_id: str,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> GameEpisode:
        """
        Simulate a single game episode between two models.
        
        This is a placeholder implementation. In a real system, this would:
        1. Load both models
        2. Create a game environment
        3. Run the models against each other
        4. Record states, actions, and rewards
        
        Args:
            model_generation: Generation of model being evaluated
            opponent_generation: Generation of opponent model
            episode_id: Unique identifier for this episode
            evaluation_config: Additional configuration
            
        Returns:
            GameEpisode: Simulated episode results
        """
        # Simulate episode based on model performance characteristics
        # This is a simplified simulation for demonstration
        
        # Get model metadata to inform simulation
        try:
            model_metadata = self.model_manager._load_model_metadata(model_generation)
            opponent_metadata = self.model_manager._load_model_metadata(opponent_generation)
        except FileNotFoundError:
            # Use default values if metadata not available
            model_metadata = None
            opponent_metadata = None
        
        # Simulate based on generation difference and training episodes
        if model_metadata and opponent_metadata:
            model_strength = model_metadata.training_episodes + model_generation * 1000
            opponent_strength = opponent_metadata.training_episodes + opponent_generation * 1000
            
            # Higher generation and more training episodes = higher win probability
            win_probability = 0.5 + 0.3 * (model_strength - opponent_strength) / max(model_strength, opponent_strength, 1)
            win_probability = max(0.1, min(0.9, win_probability))  # Clamp between 0.1 and 0.9
        else:
            # Default simulation
            win_probability = 0.5 + 0.1 * (model_generation - opponent_generation)
            win_probability = max(0.1, min(0.9, win_probability))
        
        # Simulate episode outcome
        random_outcome = np.random.random()
        if random_outcome < win_probability:
            game_result = 'win'
            base_reward = 100
        elif random_outcome < win_probability + 0.05:  # Small chance of draw
            game_result = 'draw'
            base_reward = 0
        else:
            game_result = 'loss'
            base_reward = -50
        
        # Add some variance to reward
        reward_variance = np.random.normal(0, 20)
        total_reward = base_reward + reward_variance
        
        # Simulate episode length (50-200 steps)
        episode_length = np.random.randint(50, 201)
        
        # Generate simulated states, actions, and rewards
        states = [{'step': i, 'simulated': True} for i in range(episode_length)]
        actions = [np.random.randint(0, 9) for _ in range(episode_length)]  # 9 possible actions
        rewards = [total_reward / episode_length for _ in range(episode_length)]
        
        # Calculate episode metrics
        episode_metrics = {
            'action_diversity': len(set(actions)) / len(actions) if actions else 0,
            'average_step_reward': total_reward / episode_length if episode_length > 0 else 0,
            'win_probability_used': win_probability,
            'model_strength': model_metadata.training_episodes if model_metadata else 0,
            'opponent_strength': opponent_metadata.training_episodes if opponent_metadata else 0
        }
        
        return GameEpisode(
            episode_id=episode_id,
            model_generation=model_generation,
            opponent_generation=opponent_generation,
            states=states,
            actions=actions,
            rewards=rewards,
            total_reward=total_reward,
            episode_length=episode_length,
            game_result=game_result,
            episode_metrics=episode_metrics
        )

    def _calculate_performance_metrics(self, episodes: List[GameEpisode]) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from episode data.
        
        Args:
            episodes: List of game episodes
            
        Returns:
            Dict containing various performance metrics
        """
        if not episodes:
            return {}
        
        # Basic statistics
        rewards = [ep.total_reward for ep in episodes]
        episode_lengths = [ep.episode_length for ep in episodes]
        
        # Win rate by opponent
        win_rates_by_opponent = {}
        for episode in episodes:
            opp_gen = episode.opponent_generation
            if opp_gen not in win_rates_by_opponent:
                win_rates_by_opponent[opp_gen] = {'wins': 0, 'total': 0}
            
            win_rates_by_opponent[opp_gen]['total'] += 1
            if episode.game_result == 'win':
                win_rates_by_opponent[opp_gen]['wins'] += 1
        
        # Calculate win rates
        for opp_gen in win_rates_by_opponent:
            data = win_rates_by_opponent[opp_gen]
            data['win_rate'] = data['wins'] / data['total'] if data['total'] > 0 else 0
        
        # Strategic diversity (action entropy)
        all_actions = []
        for episode in episodes:
            all_actions.extend(episode.actions)
        
        action_diversity = 0
        if all_actions:
            action_counts = {}
            for action in all_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            total_actions = len(all_actions)
            action_entropy = 0
            for count in action_counts.values():
                prob = count / total_actions
                if prob > 0:
                    action_entropy -= prob * np.log2(prob)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(action_counts)) if len(action_counts) > 1 else 1
            action_diversity = action_entropy / max_entropy if max_entropy > 0 else 0
        
        # Performance consistency
        reward_std = statistics.stdev(rewards) if len(rewards) > 1 else 0
        reward_consistency = 1 / (1 + reward_std / 100) if reward_std > 0 else 1
        
        # Episode efficiency (reward per step)
        episode_efficiencies = []
        for episode in episodes:
            if episode.episode_length > 0:
                efficiency = episode.total_reward / episode.episode_length
                episode_efficiencies.append(efficiency)
        
        return {
            'reward_mean': statistics.mean(rewards),
            'reward_std': reward_std,
            'reward_min': min(rewards),
            'reward_max': max(rewards),
            'reward_median': statistics.median(rewards),
            'episode_length_mean': statistics.mean(episode_lengths),
            'episode_length_std': statistics.stdev(episode_lengths) if len(episode_lengths) > 1 else 0,
            'win_rates_by_opponent': win_rates_by_opponent,
            'strategic_diversity': action_diversity,
            'performance_consistency': reward_consistency,
            'episode_efficiency_mean': statistics.mean(episode_efficiencies) if episode_efficiencies else 0,
            'episode_efficiency_std': statistics.stdev(episode_efficiencies) if len(episode_efficiencies) > 1 else 0,
            'total_episodes': len(episodes),
            'unique_opponents': len(set(ep.opponent_generation for ep in episodes))
        }

    def compare_generations(
        self,
        generation_a: int,
        generation_b: int,
        comparison_episodes: int = 200,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison between two model generations.
        
        Args:
            generation_a: First generation to compare
            generation_b: Second generation to compare
            comparison_episodes: Number of episodes for head-to-head comparison
            significance_level: Statistical significance level for tests
            
        Returns:
            Dict containing detailed comparison results and statistical tests
            
        Raises:
            FileNotFoundError: If either generation doesn't exist
        """
        logger.info(f"Comparing generations {generation_a} vs {generation_b}")
        
        # Validate both generations exist
        for gen in [generation_a, generation_b]:
            try:
                self.model_manager._load_model_metadata(gen)
            except FileNotFoundError:
                raise FileNotFoundError(f"Generation {gen} not found")
        
        # Run head-to-head episodes
        episodes_a_vs_b = self._run_episodes_against_opponent(
            generation_a, generation_b, comparison_episodes // 2
        )
        episodes_b_vs_a = self._run_episodes_against_opponent(
            generation_b, generation_a, comparison_episodes // 2
        )
        
        # Extract performance data
        rewards_a = [ep.total_reward for ep in episodes_a_vs_b]
        rewards_b = [ep.total_reward for ep in episodes_b_vs_a]
        
        wins_a = sum(1 for ep in episodes_a_vs_b if ep.game_result == 'win')
        wins_b = sum(1 for ep in episodes_b_vs_a if ep.game_result == 'win')
        
        # Basic statistics
        total_games_a = len(episodes_a_vs_b)
        total_games_b = len(episodes_b_vs_a)
        losses_a = total_games_a - wins_a
        losses_b = total_games_b - wins_b
        
        win_rate_a = wins_a / total_games_a if total_games_a > 0 else 0
        win_rate_b = wins_b / total_games_b if total_games_b > 0 else 0
        
        # Statistical tests (if scipy is available)
        if SCIPY_AVAILABLE and stats is not None:
            # T-test for reward differences
            t_stat, t_p_value = stats.ttest_ind(rewards_a, rewards_b)
            
            # Mann-Whitney U test (non-parametric alternative)
            u_stat, u_p_value = stats.mannwhitneyu(rewards_a, rewards_b, alternative='two-sided')
            
            # Chi-square test for win rate differences
            chi2_stat, chi2_p_value = stats.chi2_contingency([[wins_a, losses_a], [wins_b, losses_b]])[:2]
            
            # Wilson score interval for win rates
            def wilson_score_interval(successes, trials, confidence=0.95):
                if trials == 0:
                    return 0, 0
                z = stats.norm.ppf((1 + confidence) / 2)
                p = successes / trials
                denominator = 1 + z**2 / trials
                centre = (p + z**2 / (2 * trials)) / denominator
                margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
                return centre - margin, centre + margin
            
            ci_a = wilson_score_interval(wins_a, total_games_a)
            ci_b = wilson_score_interval(wins_b, total_games_b)
        else:
            # Fallback to basic statistics without scipy
            t_stat, t_p_value = 0.0, 1.0
            u_stat, u_p_value = 0.0, 1.0
            chi2_stat, chi2_p_value = 0.0, 1.0
            ci_a = (win_rate_a, win_rate_a)
            ci_b = (win_rate_b, win_rate_b)
        
        # Effect size (Cohen's d for rewards)
        pooled_std = np.sqrt(((len(rewards_a) - 1) * np.var(rewards_a, ddof=1) + 
                             (len(rewards_b) - 1) * np.var(rewards_b, ddof=1)) / 
                            (len(rewards_a) + len(rewards_b) - 2))
        cohens_d = (np.mean(rewards_a) - np.mean(rewards_b)) / pooled_std if pooled_std > 0 else 0
        
        comparison_result = {
            'generation_a': generation_a,
            'generation_b': generation_b,
            'episodes_compared': comparison_episodes,
            
            # Basic performance metrics
            'performance_a': {
                'win_rate': win_rate_a,
                'wins': wins_a,
                'total_games': total_games_a,
                'average_reward': np.mean(rewards_a),
                'reward_std': np.std(rewards_a),
                'win_rate_ci': ci_a
            },
            'performance_b': {
                'win_rate': win_rate_b,
                'wins': wins_b,
                'total_games': total_games_b,
                'average_reward': np.mean(rewards_b),
                'reward_std': np.std(rewards_b),
                'win_rate_ci': ci_b
            },
            
            # Statistical tests
            'statistical_tests': {
                'reward_t_test': {
                    'statistic': t_stat,
                    'p_value': t_p_value,
                    'significant': t_p_value < significance_level
                },
                'reward_mannwhitney': {
                    'statistic': u_stat,
                    'p_value': u_p_value,
                    'significant': u_p_value < significance_level
                },
                'win_rate_chi2': {
                    'statistic': chi2_stat,
                    'p_value': chi2_p_value,
                    'significant': chi2_p_value < significance_level
                },
                'effect_size_cohens_d': cohens_d
            },
            
            # Summary
            'summary': {
                'better_generation': generation_a if win_rate_a > win_rate_b else generation_b,
                'win_rate_difference': abs(win_rate_a - win_rate_b),
                'reward_difference': np.mean(rewards_a) - np.mean(rewards_b),
                'statistically_significant': (t_p_value < significance_level or 
                                            chi2_p_value < significance_level),
                'practical_significance': abs(cohens_d) > 0.2,  # Small effect size threshold
                'confidence_level': 1 - significance_level
            }
        }
        
        logger.info(f"Comparison complete: Gen {generation_a} vs Gen {generation_b} - "
                   f"Better: Gen {comparison_result['summary']['better_generation']}")
        
        return comparison_result

    def generate_report(
        self,
        evaluation_result: EvaluationResult,
        include_visualizations: bool = True,
        output_format: str = 'html'
    ) -> str:
        """
        Generate comprehensive evaluation report with visualizations.
        
        Args:
            evaluation_result: Evaluation results to generate report for
            include_visualizations: Whether to include charts and graphs
            output_format: Output format ('html', 'markdown', 'json')
            
        Returns:
            str: Path to generated report file
            
        Raises:
            ValueError: If output format is unsupported
        """
        if output_format not in ['html', 'markdown', 'json']:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        report_dir = self.results_dir / "reports"
        report_dir.mkdir(exist_ok=True)
        
        timestamp = evaluation_result.evaluation_date.strftime('%Y%m%d_%H%M%S')
        report_filename = f"evaluation_report_gen{evaluation_result.model_generation}_{timestamp}.{output_format}"
        report_path = report_dir / report_filename
        
        if output_format == 'json':
            # JSON report
            with open(report_path, 'w') as f:
                json.dump(evaluation_result.to_dict(), f, indent=2, default=str)
        
        elif output_format == 'markdown':
            # Markdown report
            report_content = self._generate_markdown_report(evaluation_result, include_visualizations)
            with open(report_path, 'w') as f:
                f.write(report_content)
        
        elif output_format == 'html':
            # HTML report
            report_content = self._generate_html_report(evaluation_result, include_visualizations)
            with open(report_path, 'w') as f:
                f.write(report_content)
        
        logger.info(f"Evaluation report generated: {report_path}")
        return str(report_path)

    def _generate_markdown_report(self, result: EvaluationResult, include_viz: bool) -> str:
        """Generate markdown format evaluation report."""
        content = f"""# Evaluation Report - Generation {result.model_generation}

**Evaluation ID:** {result.evaluation_id}  
**Date:** {result.evaluation_date.strftime('%Y-%m-%d %H:%M:%S')}  
**Opponents:** {', '.join(map(str, result.opponent_generations))}

## Summary

- **Total Games:** {result.total_games}
- **Win Rate:** {result.win_rate:.3f} ({result.wins}/{result.total_games})
- **Average Reward:** {result.average_reward:.2f}
- **Wins:** {result.wins}
- **Losses:** {result.losses}
- **Draws:** {result.draws}

## Performance Metrics

"""
        
        # Add performance metrics
        for metric, value in result.performance_metrics.items():
            if isinstance(value, dict):
                content += f"### {metric.replace('_', ' ').title()}\n\n"
                for sub_metric, sub_value in value.items():
                    content += f"- **{sub_metric}:** {sub_value}\n"
                content += "\n"
            else:
                content += f"- **{metric.replace('_', ' ').title()}:** {value}\n"
        
        content += "\n## Detailed Analysis\n\n"
        
        # Win rates by opponent
        if 'win_rates_by_opponent' in result.performance_metrics:
            content += "### Performance vs Each Opponent\n\n"
            content += "| Opponent Generation | Win Rate | Wins/Total |\n"
            content += "|-------------------|----------|------------|\n"
            
            for opp_gen, data in result.performance_metrics['win_rates_by_opponent'].items():
                win_rate = data['win_rate']
                wins = data['wins']
                total = data['total']
                content += f"| {opp_gen} | {win_rate:.3f} | {wins}/{total} |\n"
        
        return content

    def _generate_html_report(self, result: EvaluationResult, include_viz: bool) -> str:
        """Generate HTML format evaluation report."""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report - Generation {result.model_generation}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .metric-value {{ font-weight: bold; color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart-container {{ margin: 20px 0; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Evaluation Report - Generation {result.model_generation}</h1>
        <p><strong>Evaluation ID:</strong> {result.evaluation_id}</p>
        <p><strong>Date:</strong> {result.evaluation_date.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Opponents:</strong> {', '.join(map(str, result.opponent_generations))}</p>
    </div>

    <h2>Summary</h2>
    <div class="metric">Total Games: <span class="metric-value">{result.total_games}</span></div>
    <div class="metric">Win Rate: <span class="metric-value">{result.win_rate:.3f}</span> ({result.wins}/{result.total_games})</div>
    <div class="metric">Average Reward: <span class="metric-value">{result.average_reward:.2f}</span></div>
    <div class="metric">Wins: <span class="metric-value">{result.wins}</span></div>
    <div class="metric">Losses: <span class="metric-value">{result.losses}</span></div>
    <div class="metric">Draws: <span class="metric-value">{result.draws}</span></div>

    <h2>Performance Metrics</h2>
"""
        
        # Add performance metrics
        for metric, value in result.performance_metrics.items():
            if isinstance(value, dict) and metric == 'win_rates_by_opponent':
                html_content += """
    <h3>Performance vs Each Opponent</h3>
    <table>
        <tr><th>Opponent Generation</th><th>Win Rate</th><th>Wins/Total</th></tr>
"""
                for opp_gen, data in value.items():
                    win_rate = data['win_rate']
                    wins = data['wins']
                    total = data['total']
                    html_content += f"        <tr><td>{opp_gen}</td><td>{win_rate:.3f}</td><td>{wins}/{total}</td></tr>\n"
                html_content += "    </table>\n"
            elif not isinstance(value, dict):
                html_content += f"    <div class=\"metric\">{metric.replace('_', ' ').title()}: <span class=\"metric-value\">{value}</span></div>\n"
        
        html_content += """
</body>
</html>"""
        
        return html_content

    def _save_evaluation_result(self, result: EvaluationResult, episodes: List[GameEpisode]):
        """Save evaluation result and episode data to disk."""
        # Save main result
        result_file = self.results_dir / f"{result.evaluation_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Save episode data (optional, for detailed analysis)
        episodes_file = self.results_dir / f"{result.evaluation_id}_episodes.json"
        episodes_data = [asdict(ep) for ep in episodes]
        with open(episodes_file, 'w') as f:
            json.dump(episodes_data, f, indent=2, default=str)

    def load_evaluation_result(self, evaluation_id: str) -> EvaluationResult:
        """
        Load a previously saved evaluation result.
        
        Args:
            evaluation_id: ID of evaluation to load
            
        Returns:
            EvaluationResult: Loaded evaluation result
            
        Raises:
            FileNotFoundError: If evaluation result not found
        """
        if evaluation_id in self._results_cache:
            return self._results_cache[evaluation_id]
        
        result_file = self.results_dir / f"{evaluation_id}.json"
        if not result_file.exists():
            raise FileNotFoundError(f"Evaluation result {evaluation_id} not found")
        
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        result = EvaluationResult.from_dict(result_data)
        self._results_cache[evaluation_id] = result
        
        return result

    def list_evaluations(self) -> List[Tuple[str, EvaluationResult]]:
        """
        List all available evaluation results.
        
        Returns:
            List of (evaluation_id, EvaluationResult) tuples
        """
        evaluations = []
        
        for result_file in self.results_dir.glob("*.json"):
            if not result_file.name.endswith("_episodes.json"):
                evaluation_id = result_file.stem
                try:
                    result = self.load_evaluation_result(evaluation_id)
                    evaluations.append((evaluation_id, result))
                except Exception as e:
                    logger.warning(f"Failed to load evaluation {evaluation_id}: {e}")
        
        # Sort by evaluation date
        evaluations.sort(key=lambda x: x[1].evaluation_date, reverse=True)
        
        return evaluations

    def tournament_evaluation(
        self,
        generations: List[int],
        episodes_per_matchup: int = 50
    ) -> Dict[str, Any]:
        """
        Run tournament-style evaluation between multiple generations.
        
        Args:
            generations: List of generations to include in tournament
            episodes_per_matchup: Number of episodes per head-to-head matchup
            
        Returns:
            Dict containing tournament results and rankings
        """
        logger.info(f"Starting tournament evaluation with generations: {generations}")
        
        tournament_id = f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize tournament tracking
        matchup_results = {}
        generation_stats = {gen: {'wins': 0, 'losses': 0, 'draws': 0, 'total_reward': 0} 
                           for gen in generations}
        
        # Run all pairwise matchups
        for i, gen_a in enumerate(generations):
            for j, gen_b in enumerate(generations):
                if i != j:  # Don't play against self
                    matchup_key = f"{gen_a}_vs_{gen_b}"
                    
                    # Run episodes
                    episodes = self._run_episodes_against_opponent(
                        gen_a, gen_b, episodes_per_matchup
                    )
                    
                    # Aggregate results
                    wins = sum(1 for ep in episodes if ep.game_result == 'win')
                    losses = sum(1 for ep in episodes if ep.game_result == 'loss')
                    draws = sum(1 for ep in episodes if ep.game_result == 'draw')
                    total_reward = sum(ep.total_reward for ep in episodes)
                    
                    matchup_results[matchup_key] = {
                        'wins': wins,
                        'losses': losses,
                        'draws': draws,
                        'total_reward': total_reward,
                        'win_rate': wins / len(episodes) if episodes else 0,
                        'episodes': len(episodes)
                    }
                    
                    # Update generation stats
                    generation_stats[gen_a]['wins'] += wins
                    generation_stats[gen_a]['losses'] += losses
                    generation_stats[gen_a]['draws'] += draws
                    generation_stats[gen_a]['total_reward'] += total_reward
        
        # Calculate final rankings
        rankings = []
        for gen in generations:
            stats = generation_stats[gen]
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            win_rate = stats['wins'] / total_games if total_games > 0 else 0
            avg_reward = stats['total_reward'] / total_games if total_games > 0 else 0
            
            rankings.append({
                'generation': gen,
                'win_rate': win_rate,
                'average_reward': avg_reward,
                'total_wins': stats['wins'],
                'total_games': total_games,
                'score': win_rate * 0.7 + (avg_reward / 100) * 0.3  # Combined score
            })
        
        # Sort by combined score
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        tournament_result = {
            'tournament_id': tournament_id,
            'generations': generations,
            'episodes_per_matchup': episodes_per_matchup,
            'matchup_results': matchup_results,
            'generation_stats': generation_stats,
            'rankings': rankings,
            'winner': rankings[0]['generation'] if rankings else None,
            'evaluation_date': datetime.now()
        }
        
        # Save tournament results
        tournament_file = self.results_dir / f"{tournament_id}.json"
        with open(tournament_file, 'w') as f:
            json.dump(tournament_result, f, indent=2, default=str)
        
        logger.info(f"Tournament complete. Winner: Generation {tournament_result['winner']}")
        
        return tournament_result