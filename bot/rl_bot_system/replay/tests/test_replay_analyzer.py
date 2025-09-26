"""
Unit tests for replay analyzer functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from rl_bot_system.replay.replay_analyzer import (
    ReplayAnalyzer, AnalysisConfig, BehaviorPattern
)
from rl_bot_system.evaluation.evaluator import GameEpisode


class TestReplayAnalyzer(unittest.TestCase):
    """Test ReplayAnalyzer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = AnalysisConfig(
            min_pattern_frequency=2,
            min_confidence_threshold=0.5,
            sequence_length=3
        )
        self.analyzer = ReplayAnalyzer(self.config)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def create_test_episode(
        self, 
        episode_id: str, 
        actions: list = None, 
        rewards: list = None,
        game_result: str = "win",
        model_generation: int = 1
    ) -> GameEpisode:
        """Create a test episode with specified parameters."""
        if actions is None:
            actions = [0, 1, 2, 1, 0]
        if rewards is None:
            rewards = [1.0, 2.0, 3.0, 2.0, 1.0]
            
        states = [{"step": i, "x": i * 10} for i in range(len(actions))]
        
        return GameEpisode(
            episode_id=episode_id,
            model_generation=model_generation,
            opponent_generation=0,
            states=states,
            actions=actions,
            rewards=rewards,
            total_reward=sum(rewards),
            episode_length=len(actions),
            game_result=game_result,
            episode_metrics={}
        )
        
    def test_analyze_empty_episodes(self):
        """Test analyzing empty episode list."""
        result = self.analyzer.analyze_episodes([])
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "No episodes provided for analysis")
        
    def test_analyze_basic_episodes(self):
        """Test basic episode analysis."""
        episodes = [
            self.create_test_episode("ep1", [0, 1, 2], [1.0, 2.0, 3.0], "win"),
            self.create_test_episode("ep2", [0, 1, 2], [2.0, 3.0, 4.0], "win"),
            self.create_test_episode("ep3", [1, 2, 0], [1.0, 1.0, 2.0], "loss")
        ]
        
        result = self.analyzer.analyze_episodes(episodes)
        
        self.assertIn("episode_count", result)
        self.assertEqual(result["episode_count"], 3)
        
        self.assertIn("model_generations", result)
        self.assertIn("performance_stats", result)
        self.assertIn("patterns", result)
        self.assertIn("behavioral_insights", result)
        
        # Check performance stats
        perf_stats = result["performance_stats"]
        self.assertEqual(perf_stats["total_episodes"], 3)
        self.assertAlmostEqual(perf_stats["win_rate"], 2/3, places=2)
        
    def test_action_sequence_detection(self):
        """Test detection of action sequence patterns."""
        # Create episodes with repeated action sequences
        episodes = [
            self.create_test_episode("ep1", [0, 1, 2, 0, 1]),
            self.create_test_episode("ep2", [0, 1, 2, 1, 0]),
            self.create_test_episode("ep3", [0, 1, 2, 2, 1])
        ]
        
        result = self.analyzer.analyze_episodes(episodes)
        
        # Should detect the common sequence [0, 1, 2]
        action_patterns = [p for p in result["patterns"] if p["pattern_type"] == "action_sequence"]
        
        # Check if we found any action sequence patterns
        self.assertGreater(len(action_patterns), 0)
        
        # Find the [0, 1, 2] pattern
        target_pattern = None
        for pattern in action_patterns:
            if pattern["metadata"]["sequence"] == (0, 1, 2):
                target_pattern = pattern
                break
                
        self.assertIsNotNone(target_pattern)
        self.assertEqual(target_pattern["frequency"], 3)  # Appears in all 3 episodes
        
    def test_state_preference_detection(self):
        """Test detection of state preference patterns."""
        # Create episodes with consistent state features
        episodes = []
        for i in range(5):
            states = [{"x": 10 + j, "y": 20 + j} for j in range(3)]
            episode = GameEpisode(
                episode_id=f"ep{i}",
                model_generation=1,
                opponent_generation=0,
                states=states,
                actions=[0, 1, 2],
                rewards=[1.0, 2.0, 3.0],
                total_reward=6.0,
                episode_length=3,
                game_result="win",
                episode_metrics={}
            )
            episodes.append(episode)
            
        result = self.analyzer.analyze_episodes(episodes)
        
        # Should detect state preferences
        state_patterns = [p for p in result["patterns"] if p["pattern_type"] == "state_preference"]
        
        # Should find patterns for x and y coordinates
        self.assertGreater(len(state_patterns), 0)
        
    def test_reward_pattern_analysis(self):
        """Test reward pattern analysis."""
        episodes = [
            self.create_test_episode("ep1", [0, 1, 2], [0.1, 5.0, 0.1]),  # Big reward in middle
            self.create_test_episode("ep2", [0, 1, 2], [0.1, 4.0, 0.1]),  # Big reward in middle
            self.create_test_episode("ep3", [0, 1, 2], [0.1, 6.0, 0.1])   # Big reward in middle
        ]
        
        result = self.analyzer.analyze_episodes(episodes)
        
        reward_patterns = [p for p in result["patterns"] if p["pattern_type"] == "reward_pattern"]
        
        # Should detect timing pattern for significant rewards
        self.assertGreater(len(reward_patterns), 0)
        
    def test_strategic_behavior_detection(self):
        """Test strategic behavior pattern detection."""
        # Create episodes with different lengths for wins vs losses
        episodes = [
            self.create_test_episode("ep1", [0, 1], [1.0, 2.0], "win", 1),      # Short win
            self.create_test_episode("ep2", [0, 1], [1.0, 2.0], "win", 1),      # Short win
            self.create_test_episode("ep3", [0, 1, 2, 0, 1], [1.0, 1.0, 1.0, 1.0, 1.0], "loss", 1),  # Long loss
            self.create_test_episode("ep4", [0, 1, 2, 0, 1], [1.0, 1.0, 1.0, 1.0, 1.0], "loss", 1)   # Long loss
        ]
        
        result = self.analyzer.analyze_episodes(episodes)
        
        strategic_patterns = [p for p in result["patterns"] if p["pattern_type"] == "strategy"]
        
        # Should detect quick wins strategy
        self.assertGreater(len(strategic_patterns), 0)
        
        quick_wins_pattern = None
        for pattern in strategic_patterns:
            if "quick_wins" in pattern["metadata"].get("strategy_type", ""):
                quick_wins_pattern = pattern
                break
                
        self.assertIsNotNone(quick_wins_pattern)
        
    def test_compare_generations(self):
        """Test comparing patterns across generations."""
        episodes_gen1 = [
            self.create_test_episode("ep1", [0, 1, 2], [1.0, 2.0, 3.0], "win", 1),
            self.create_test_episode("ep2", [0, 1, 2], [2.0, 3.0, 4.0], "win", 1)
        ]
        
        episodes_gen2 = [
            self.create_test_episode("ep3", [1, 2, 0], [3.0, 4.0, 5.0], "win", 2),
            self.create_test_episode("ep4", [1, 2, 0], [4.0, 5.0, 6.0], "win", 2)
        ]
        
        episodes_by_generation = {
            1: episodes_gen1,
            2: episodes_gen2
        }
        
        result = self.analyzer.compare_generations(episodes_by_generation)
        
        self.assertIn("generations_analyzed", result)
        self.assertIn("generation_comparisons", result)
        self.assertIn("evolution_trends", result)
        self.assertIn("performance_progression", result)
        
        self.assertEqual(result["generations_analyzed"], [1, 2])
        
    def test_behavioral_insights_generation(self):
        """Test generation of behavioral insights."""
        episodes = [
            self.create_test_episode("ep1", [0, 1, 2], [1.0, 2.0, 3.0], "win"),
            self.create_test_episode("ep2", [0, 1, 2], [2.0, 3.0, 4.0], "win"),
            self.create_test_episode("ep3", [0, 1, 2], [1.5, 2.5, 3.5], "win")
        ]
        
        result = self.analyzer.analyze_episodes(episodes)
        
        insights = result["behavioral_insights"]
        
        self.assertIn("dominant_strategies", insights)
        self.assertIn("consistency_metrics", insights)
        self.assertIn("improvement_areas", insights)
        
        # Check consistency metrics
        consistency = insights["consistency_metrics"]
        self.assertIn("reward_consistency", consistency)
        self.assertIn("performance_trend", consistency)
        
    def test_export_analysis_json(self):
        """Test exporting analysis results as JSON."""
        episodes = [
            self.create_test_episode("ep1", [0, 1, 2], [1.0, 2.0, 3.0], "win")
        ]
        
        result = self.analyzer.analyze_episodes(episodes)
        
        output_path = Path(self.temp_dir) / "analysis_export"
        exported_path = self.analyzer.export_analysis(result, str(output_path), "json")
        
        self.assertTrue(Path(exported_path).exists())
        self.assertTrue(exported_path.endswith('.json'))
        
        # Verify file content
        import json
        with open(exported_path, 'r') as f:
            exported_data = json.load(f)
            
        self.assertIn("episode_count", exported_data)
        self.assertEqual(exported_data["episode_count"], 1)
        
    def test_export_analysis_csv(self):
        """Test exporting analysis results as CSV."""
        episodes = [
            self.create_test_episode("ep1", [0, 1, 2], [1.0, 2.0, 3.0], "win")
        ]
        
        result = self.analyzer.analyze_episodes(episodes)
        
        output_path = Path(self.temp_dir) / "analysis_export"
        exported_path = self.analyzer.export_analysis(result, str(output_path), "csv")
        
        self.assertTrue(Path(exported_path).exists())
        self.assertTrue(exported_path.endswith('.csv'))
        
    def test_export_analysis_html(self):
        """Test exporting analysis results as HTML."""
        episodes = [
            self.create_test_episode("ep1", [0, 1, 2], [1.0, 2.0, 3.0], "win")
        ]
        
        result = self.analyzer.analyze_episodes(episodes)
        
        output_path = Path(self.temp_dir) / "analysis_export"
        exported_path = self.analyzer.export_analysis(result, str(output_path), "html")
        
        self.assertTrue(Path(exported_path).exists())
        self.assertTrue(exported_path.endswith('.html'))
        
        # Verify HTML content
        with open(exported_path, 'r') as f:
            html_content = f.read()
            
        self.assertIn("<html>", html_content)
        self.assertIn("Replay Analysis Report", html_content)
        
    def test_unsupported_export_format(self):
        """Test handling of unsupported export formats."""
        episodes = [
            self.create_test_episode("ep1", [0, 1, 2], [1.0, 2.0, 3.0], "win")
        ]
        
        result = self.analyzer.analyze_episodes(episodes)
        
        output_path = Path(self.temp_dir) / "analysis_export"
        
        with self.assertRaises(ValueError):
            self.analyzer.export_analysis(result, str(output_path), "unsupported")
            
    def test_configuration_options(self):
        """Test different configuration options."""
        # Test with different thresholds
        strict_config = AnalysisConfig(
            min_pattern_frequency=5,
            min_confidence_threshold=0.9,
            sequence_length=4
        )
        
        strict_analyzer = ReplayAnalyzer(strict_config)
        
        episodes = [
            self.create_test_episode("ep1", [0, 1, 2], [1.0, 2.0, 3.0], "win")
        ]
        
        result = strict_analyzer.analyze_episodes(episodes)
        
        # With strict thresholds, should find fewer patterns
        self.assertIn("patterns", result)
        
    def test_pattern_confidence_calculation(self):
        """Test that pattern confidence is calculated correctly."""
        # Create episodes where a pattern appears in 2 out of 3 episodes
        episodes = [
            self.create_test_episode("ep1", [0, 1, 2, 0, 1]),
            self.create_test_episode("ep2", [0, 1, 2, 1, 0]),
            self.create_test_episode("ep3", [1, 2, 0, 1, 2])  # Different pattern
        ]
        
        result = self.analyzer.analyze_episodes(episodes)
        
        action_patterns = [p for p in result["patterns"] if p["pattern_type"] == "action_sequence"]
        
        # Find pattern that appears in 2/3 episodes
        for pattern in action_patterns:
            if pattern["frequency"] == 2:
                expected_confidence = 2 / 3  # 2 episodes out of 3
                self.assertAlmostEqual(pattern["confidence"], expected_confidence, places=2)


if __name__ == '__main__':
    unittest.main()