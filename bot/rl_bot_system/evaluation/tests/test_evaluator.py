"""
Tests for EvaluationManager and related evaluation components.
"""

import unittest
import tempfile
import shutil
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from rl_bot_system.evaluation.evaluator import (
    EvaluationManager, EvaluationResult, GameEpisode
)
from rl_bot_system.training.model_manager import ModelManager, RLModel


class TestEvaluationResult(unittest.TestCase):
    """Test EvaluationResult data class."""
    
    def test_evaluation_result_creation(self):
        """Test creating an EvaluationResult."""
        result = EvaluationResult(
            model_generation=1,
            opponent_generations=[0, 2],
            total_games=100,
            wins=75,
            losses=20,
            draws=5,
            average_reward=85.5,
            win_rate=0.75,
            performance_metrics={'test_metric': 1.0},
            evaluation_date=datetime.now(),
            evaluation_id='test_eval_1'
        )
        
        self.assertEqual(result.model_generation, 1)
        self.assertEqual(result.opponent_generations, [0, 2])
        self.assertEqual(result.total_games, 100)
        self.assertEqual(result.wins, 75)
        self.assertEqual(result.win_rate, 0.75)
    
    def test_evaluation_result_serialization(self):
        """Test EvaluationResult to_dict and from_dict."""
        original = EvaluationResult(
            model_generation=1,
            opponent_generations=[0],
            total_games=50,
            wins=30,
            losses=20,
            draws=0,
            average_reward=75.0,
            win_rate=0.6,
            performance_metrics={'metric1': 1.0},
            evaluation_date=datetime(2024, 1, 1, 12, 0, 0),
            evaluation_id='test_eval'
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = EvaluationResult.from_dict(data)
        
        self.assertEqual(original.model_generation, restored.model_generation)
        self.assertEqual(original.evaluation_date, restored.evaluation_date)
        self.assertEqual(original.performance_metrics, restored.performance_metrics)


class TestGameEpisode(unittest.TestCase):
    """Test GameEpisode data class."""
    
    def test_game_episode_creation(self):
        """Test creating a GameEpisode."""
        episode = GameEpisode(
            episode_id='ep_1',
            model_generation=1,
            opponent_generation=0,
            states=[{'step': 0}, {'step': 1}],
            actions=[1, 2],
            rewards=[10.0, 15.0],
            total_reward=25.0,
            episode_length=2,
            game_result='win',
            episode_metrics={'diversity': 0.8}
        )
        
        self.assertEqual(episode.episode_id, 'ep_1')
        self.assertEqual(episode.model_generation, 1)
        self.assertEqual(episode.total_reward, 25.0)
        self.assertEqual(episode.game_result, 'win')


class TestEvaluationManager(unittest.TestCase):
    """Test EvaluationManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.results_dir = Path(self.temp_dir) / "evaluations"
        
        # Create mock model manager
        self.model_manager = Mock(spec=ModelManager)
        
        # Create evaluation manager
        self.evaluator = EvaluationManager(
            model_manager=self.model_manager,
            results_dir=str(self.results_dir)
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_evaluation_manager_initialization(self):
        """Test EvaluationManager initialization."""
        self.assertEqual(self.evaluator.model_manager, self.model_manager)
        self.assertTrue(self.evaluator.results_dir.exists())
        self.assertEqual(str(self.evaluator.results_dir), str(self.results_dir))
    
    def test_calculate_performance_metrics_empty(self):
        """Test performance metrics calculation with empty episodes."""
        metrics = self.evaluator._calculate_performance_metrics([])
        self.assertEqual(metrics, {})
    
    def test_calculate_performance_metrics_basic(self):
        """Test performance metrics calculation with basic episodes."""
        episodes = [
            GameEpisode(
                episode_id='ep_1',
                model_generation=1,
                opponent_generation=0,
                states=[],
                actions=[1, 2, 1, 3],
                rewards=[10.0, 15.0],
                total_reward=25.0,
                episode_length=4,
                game_result='win',
                episode_metrics={}
            ),
            GameEpisode(
                episode_id='ep_2',
                model_generation=1,
                opponent_generation=0,
                states=[],
                actions=[2, 2, 2, 2],
                rewards=[5.0, 10.0],
                total_reward=15.0,
                episode_length=4,
                game_result='loss',
                episode_metrics={}
            )
        ]
        
        metrics = self.evaluator._calculate_performance_metrics(episodes)
        
        # Check basic metrics
        self.assertEqual(metrics['total_episodes'], 2)
        self.assertEqual(metrics['reward_mean'], 20.0)  # (25 + 15) / 2
        self.assertEqual(metrics['episode_length_mean'], 4.0)
        
        # Check win rates by opponent
        self.assertIn('win_rates_by_opponent', metrics)
        self.assertEqual(metrics['win_rates_by_opponent'][0]['total'], 2)
        self.assertEqual(metrics['win_rates_by_opponent'][0]['wins'], 1)
        self.assertEqual(metrics['win_rates_by_opponent'][0]['win_rate'], 0.5)
        
        # Check strategic diversity (action entropy)
        self.assertIn('strategic_diversity', metrics)
        self.assertGreater(metrics['strategic_diversity'], 0)
    
    def test_simulate_episode(self):
        """Test episode simulation."""
        # Mock model metadata
        model_metadata = RLModel(
            generation=1,
            algorithm='DQN',
            network_architecture={},
            hyperparameters={},
            training_episodes=1000,
            performance_metrics={},
            parent_generation=0,
            created_at=datetime.now(),
            model_path='test_path'
        )
        
        self.model_manager._load_model_metadata.return_value = model_metadata
        
        episode = self.evaluator._simulate_episode(1, 0, 'test_ep', None)
        
        self.assertEqual(episode.model_generation, 1)
        self.assertEqual(episode.opponent_generation, 0)
        self.assertEqual(episode.episode_id, 'test_ep')
        self.assertIn(episode.game_result, ['win', 'loss', 'draw'])
        self.assertGreater(episode.episode_length, 0)
        self.assertEqual(len(episode.states), episode.episode_length)
        self.assertEqual(len(episode.actions), episode.episode_length)
        self.assertEqual(len(episode.rewards), episode.episode_length)
    
    def test_run_evaluation_basic(self):
        """Test basic evaluation run."""
        # Mock model loading
        mock_model = Mock()
        model_metadata = RLModel(
            generation=1,
            algorithm='DQN',
            network_architecture={},
            hyperparameters={},
            training_episodes=1000,
            performance_metrics={},
            parent_generation=0,
            created_at=datetime.now(),
            model_path='test_path'
        )
        
        self.model_manager.load_model.return_value = (mock_model, model_metadata)
        self.model_manager._load_model_metadata.return_value = model_metadata
        
        # Run evaluation
        result = self.evaluator.run_evaluation(
            model_generation=1,
            opponent_generations=[0],
            episodes_per_opponent=10
        )
        
        # Verify result
        self.assertEqual(result.model_generation, 1)
        self.assertEqual(result.opponent_generations, [0])
        self.assertEqual(result.total_games, 10)
        self.assertGreaterEqual(result.wins + result.losses + result.draws, result.total_games)
        self.assertIsInstance(result.win_rate, float)
        self.assertIsInstance(result.average_reward, float)
        self.assertIsInstance(result.performance_metrics, dict)
    
    def test_run_evaluation_model_not_found(self):
        """Test evaluation with non-existent model."""
        self.model_manager.load_model.side_effect = FileNotFoundError("Model not found")
        
        with self.assertRaises(FileNotFoundError):
            self.evaluator.run_evaluation(1, [0], 10)
    
    def test_compare_generations(self):
        """Test generation comparison."""
        # Mock model metadata
        model_metadata = RLModel(
            generation=1,
            algorithm='DQN',
            network_architecture={},
            hyperparameters={},
            training_episodes=1000,
            performance_metrics={},
            parent_generation=0,
            created_at=datetime.now(),
            model_path='test_path'
        )
        
        self.model_manager._load_model_metadata.return_value = model_metadata
        
        # Run comparison
        comparison = self.evaluator.compare_generations(1, 0, comparison_episodes=20)
        
        # Verify comparison structure
        self.assertEqual(comparison['generation_a'], 1)
        self.assertEqual(comparison['generation_b'], 0)
        self.assertEqual(comparison['episodes_compared'], 20)
        
        self.assertIn('performance_a', comparison)
        self.assertIn('performance_b', comparison)
        self.assertIn('statistical_tests', comparison)
        self.assertIn('summary', comparison)
        
        # Check statistical tests
        stats = comparison['statistical_tests']
        self.assertIn('reward_t_test', stats)
        self.assertIn('reward_mannwhitney', stats)
        self.assertIn('win_rate_chi2', stats)
        self.assertIn('effect_size_cohens_d', stats)
        
        # Check summary
        summary = comparison['summary']
        self.assertIn('better_generation', summary)
        self.assertIn('statistically_significant', summary)
        self.assertIn('practical_significance', summary)
    
    def test_compare_generations_model_not_found(self):
        """Test comparison with non-existent model."""
        self.model_manager._load_model_metadata.side_effect = FileNotFoundError("Model not found")
        
        with self.assertRaises(FileNotFoundError):
            self.evaluator.compare_generations(1, 0)
    
    def test_generate_report_json(self):
        """Test JSON report generation."""
        result = EvaluationResult(
            model_generation=1,
            opponent_generations=[0],
            total_games=50,
            wins=30,
            losses=20,
            draws=0,
            average_reward=75.0,
            win_rate=0.6,
            performance_metrics={'test_metric': 1.0},
            evaluation_date=datetime.now(),
            evaluation_id='test_eval'
        )
        
        report_path = self.evaluator.generate_report(result, output_format='json')
        
        # Verify report file exists
        self.assertTrue(Path(report_path).exists())
        
        # Verify report content
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        self.assertEqual(report_data['model_generation'], 1)
        self.assertEqual(report_data['win_rate'], 0.6)
    
    def test_generate_report_markdown(self):
        """Test Markdown report generation."""
        result = EvaluationResult(
            model_generation=1,
            opponent_generations=[0],
            total_games=50,
            wins=30,
            losses=20,
            draws=0,
            average_reward=75.0,
            win_rate=0.6,
            performance_metrics={
                'reward_mean': 75.0,
                'win_rates_by_opponent': {
                    0: {'win_rate': 0.6, 'wins': 30, 'total': 50}
                }
            },
            evaluation_date=datetime.now(),
            evaluation_id='test_eval'
        )
        
        report_path = self.evaluator.generate_report(result, output_format='markdown')
        
        # Verify report file exists
        self.assertTrue(Path(report_path).exists())
        
        # Verify report content
        with open(report_path, 'r') as f:
            content = f.read()
        
        self.assertIn('# Evaluation Report - Generation 1', content)
        self.assertIn('**Win Rate:** 0.600', content)
        self.assertIn('**Total Games:** 50', content)
    
    def test_generate_report_html(self):
        """Test HTML report generation."""
        result = EvaluationResult(
            model_generation=1,
            opponent_generations=[0],
            total_games=50,
            wins=30,
            losses=20,
            draws=0,
            average_reward=75.0,
            win_rate=0.6,
            performance_metrics={'test_metric': 1.0},
            evaluation_date=datetime.now(),
            evaluation_id='test_eval'
        )
        
        report_path = self.evaluator.generate_report(result, output_format='html')
        
        # Verify report file exists
        self.assertTrue(Path(report_path).exists())
        
        # Verify report content
        with open(report_path, 'r') as f:
            content = f.read()
        
        self.assertIn('<title>Evaluation Report - Generation 1</title>', content)
        self.assertIn('Win Rate:', content)
        self.assertIn('0.600', content)
    
    def test_generate_report_invalid_format(self):
        """Test report generation with invalid format."""
        result = EvaluationResult(
            model_generation=1,
            opponent_generations=[0],
            total_games=50,
            wins=30,
            losses=20,
            draws=0,
            average_reward=75.0,
            win_rate=0.6,
            performance_metrics={},
            evaluation_date=datetime.now(),
            evaluation_id='test_eval'
        )
        
        with self.assertRaises(ValueError):
            self.evaluator.generate_report(result, output_format='invalid')
    
    def test_save_and_load_evaluation_result(self):
        """Test saving and loading evaluation results."""
        result = EvaluationResult(
            model_generation=1,
            opponent_generations=[0],
            total_games=50,
            wins=30,
            losses=20,
            draws=0,
            average_reward=75.0,
            win_rate=0.6,
            performance_metrics={'test_metric': 1.0},
            evaluation_date=datetime.now(),
            evaluation_id='test_eval_save_load'
        )
        
        # Save result
        self.evaluator._save_evaluation_result(result, [])
        
        # Load result
        loaded_result = self.evaluator.load_evaluation_result('test_eval_save_load')
        
        # Verify loaded result
        self.assertEqual(loaded_result.model_generation, result.model_generation)
        self.assertEqual(loaded_result.win_rate, result.win_rate)
        self.assertEqual(loaded_result.evaluation_id, result.evaluation_id)
    
    def test_load_evaluation_result_not_found(self):
        """Test loading non-existent evaluation result."""
        with self.assertRaises(FileNotFoundError):
            self.evaluator.load_evaluation_result('nonexistent_eval')
    
    def test_list_evaluations(self):
        """Test listing evaluation results."""
        # Create and save test results
        result1 = EvaluationResult(
            model_generation=1,
            opponent_generations=[0],
            total_games=50,
            wins=30,
            losses=20,
            draws=0,
            average_reward=75.0,
            win_rate=0.6,
            performance_metrics={},
            evaluation_date=datetime(2024, 1, 1),
            evaluation_id='eval_1'
        )
        
        result2 = EvaluationResult(
            model_generation=2,
            opponent_generations=[0, 1],
            total_games=100,
            wins=70,
            losses=30,
            draws=0,
            average_reward=85.0,
            win_rate=0.7,
            performance_metrics={},
            evaluation_date=datetime(2024, 1, 2),
            evaluation_id='eval_2'
        )
        
        self.evaluator._save_evaluation_result(result1, [])
        self.evaluator._save_evaluation_result(result2, [])
        
        # List evaluations
        evaluations = self.evaluator.list_evaluations()
        
        # Verify results (should be sorted by date, newest first)
        self.assertEqual(len(evaluations), 2)
        self.assertEqual(evaluations[0][0], 'eval_2')  # Newer evaluation first
        self.assertEqual(evaluations[1][0], 'eval_1')
        self.assertEqual(evaluations[0][1].model_generation, 2)
        self.assertEqual(evaluations[1][1].model_generation, 1)
    
    def test_tournament_evaluation(self):
        """Test tournament-style evaluation."""
        # Mock model metadata
        model_metadata = RLModel(
            generation=1,
            algorithm='DQN',
            network_architecture={},
            hyperparameters={},
            training_episodes=1000,
            performance_metrics={},
            parent_generation=0,
            created_at=datetime.now(),
            model_path='test_path'
        )
        
        self.model_manager._load_model_metadata.return_value = model_metadata
        
        # Run tournament
        tournament_result = self.evaluator.tournament_evaluation(
            generations=[0, 1, 2],
            episodes_per_matchup=10
        )
        
        # Verify tournament structure
        self.assertIn('tournament_id', tournament_result)
        self.assertEqual(tournament_result['generations'], [0, 1, 2])
        self.assertEqual(tournament_result['episodes_per_matchup'], 10)
        
        self.assertIn('matchup_results', tournament_result)
        self.assertIn('generation_stats', tournament_result)
        self.assertIn('rankings', tournament_result)
        self.assertIn('winner', tournament_result)
        
        # Check that all matchups were run
        matchups = tournament_result['matchup_results']
        expected_matchups = ['0_vs_1', '0_vs_2', '1_vs_0', '1_vs_2', '2_vs_0', '2_vs_1']
        for matchup in expected_matchups:
            self.assertIn(matchup, matchups)
        
        # Check rankings structure
        rankings = tournament_result['rankings']
        self.assertEqual(len(rankings), 3)
        for ranking in rankings:
            self.assertIn('generation', ranking)
            self.assertIn('win_rate', ranking)
            self.assertIn('average_reward', ranking)
            self.assertIn('score', ranking)
        
        # Verify winner is determined
        self.assertIsNotNone(tournament_result['winner'])
        self.assertIn(tournament_result['winner'], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()