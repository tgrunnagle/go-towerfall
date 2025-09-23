"""
Unit tests for replay manager functionality.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path

from bot.rl_bot_system.replay.replay_manager import ReplayManager
from bot.rl_bot_system.replay.episode_recorder import RecordingConfig
from bot.rl_bot_system.replay.experience_buffer import BufferConfig
from bot.rl_bot_system.replay.replay_analyzer import AnalysisConfig
from bot.rl_bot_system.evaluation.evaluator import GameEpisode


class TestReplayManager(unittest.TestCase):
    """Test ReplayManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configurations
        self.recording_config = RecordingConfig(
            storage_path=self.temp_dir,
            max_episodes_per_file=2,
            auto_cleanup=False
        )
        
        self.buffer_config = BufferConfig(
            max_size=50,
            min_size_for_sampling=3
        )
        
        self.analysis_config = AnalysisConfig(
            min_pattern_frequency=2,
            min_confidence_threshold=0.5
        )
        
        self.manager = ReplayManager(
            storage_path=self.temp_dir,
            recording_config=self.recording_config,
            buffer_config=self.buffer_config,
            analysis_config=self.analysis_config
        )
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def create_test_episode(self, episode_id: str = "test_ep", length: int = 3) -> GameEpisode:
        """Create a test episode for testing."""
        return GameEpisode(
            episode_id=episode_id,
            model_generation=1,
            opponent_generation=0,
            states=[{"step": i, "x": i * 10} for i in range(length)],
            actions=list(range(length)),
            rewards=[float(i) for i in range(length)],
            total_reward=sum(range(length)),
            episode_length=length,
            game_result="win",
            episode_metrics={}
        )
        
    def test_start_session(self):
        """Test starting a replay session."""
        session_metadata = {"experiment": "test", "version": "1.0"}
        
        session_id = self.manager.start_session("test_session", session_metadata)
        
        self.assertEqual(session_id, "test_session")
        self.assertEqual(self.manager.current_session_id, "test_session")
        
        # Check session metadata file
        session_path = Path(self.temp_dir) / "test_session"
        metadata_file = session_path / "replay_session_metadata.json"
        
        self.assertTrue(metadata_file.exists())
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        self.assertEqual(metadata["session_id"], "test_session")
        self.assertEqual(metadata["user_metadata"], session_metadata)
        self.assertIn("components", metadata)
        
    def test_record_episode(self):
        """Test recording an episode."""
        self.manager.start_session("test_session")
        
        states = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        actions = [0, 1]
        rewards = [1.0, 2.0]
        
        episode_id = self.manager.record_episode(
            states=states,
            actions=actions,
            rewards=rewards,
            model_generation=1,
            opponent_generation=0,
            game_result="win",
            add_to_buffer=True
        )
        
        self.assertIsInstance(episode_id, str)
        
        # Check that episode was added to buffer
        self.assertEqual(len(self.manager.experience_buffer), 1)  # 2 states -> 1 transition
        
    def test_record_game_episode(self):
        """Test recording an existing GameEpisode object."""
        self.manager.start_session("test_session")
        
        episode = self.create_test_episode("test_ep", 4)
        
        episode_id = self.manager.record_game_episode(episode, add_to_buffer=True)
        
        self.assertEqual(episode_id, "test_ep")
        
        # Check that episode was added to buffer
        self.assertEqual(len(self.manager.experience_buffer), 3)  # 4 states -> 3 transitions
        
    def test_get_training_batch(self):
        """Test getting training batches."""
        self.manager.start_session("test_session")
        
        # Add enough episodes to meet minimum sampling requirement
        for i in range(3):
            episode = self.create_test_episode(f"ep{i}", 3)
            self.manager.record_game_episode(episode)
        
        # Should have 6 transitions total (3 episodes * 2 transitions each)
        self.assertEqual(len(self.manager.experience_buffer), 6)
        
        # Get training batch
        batch = self.manager.get_training_batch(4)
        
        self.assertEqual(len(batch), 4)
        
    def test_get_episode_batch(self):
        """Test getting episode batches."""
        self.manager.start_session("test_session")
        
        # Add episodes
        for i in range(3):
            episode = self.create_test_episode(f"ep{i}", 3)
            self.manager.record_game_episode(episode)
        
        # Get episode batch
        episode_batch = self.manager.get_episode_batch(2)
        
        self.assertEqual(len(episode_batch), 2)
        
        # Each item should be a list of transitions
        for episode_transitions in episode_batch:
            self.assertIsInstance(episode_transitions, list)
            self.assertGreater(len(episode_transitions), 0)
            
    def test_load_episodes_from_session(self):
        """Test loading episodes from a session."""
        session_id = self.manager.start_session("load_test_session")
        
        # Record some episodes
        episodes = []
        for i in range(3):
            episode = self.create_test_episode(f"ep{i}", 3)
            self.manager.record_game_episode(episode, add_to_buffer=False)
            episodes.append(episode)
            
            # Small delay to ensure different timestamps for file creation
            import time
            time.sleep(0.01)
        
        # End session to flush episodes to disk
        self.manager.end_session()
        
        # Load episodes back
        loaded_episodes = self.manager.load_episodes_from_session(session_id)
        
        self.assertEqual(len(loaded_episodes), 3)
        
        # Check episode IDs match
        loaded_ids = {ep.episode_id for ep in loaded_episodes}
        original_ids = {ep.episode_id for ep in episodes}
        self.assertEqual(loaded_ids, original_ids)
        
    def test_load_nonexistent_session(self):
        """Test loading from a nonexistent session."""
        with self.assertRaises(ValueError):
            self.manager.load_episodes_from_session("nonexistent_session")
            
    def test_analyze_session(self):
        """Test analyzing a session."""
        session_id = self.manager.start_session("analyze_test_session")
        
        # Record episodes with patterns
        for i in range(3):
            episode = self.create_test_episode(f"ep{i}", 4)
            self.manager.record_game_episode(episode, add_to_buffer=False)
        
        self.manager.end_session()
        
        # Analyze session
        analysis_result = self.manager.analyze_session(session_id)
        
        self.assertIn("episode_count", analysis_result)
        self.assertEqual(analysis_result["episode_count"], 3)
        self.assertIn("patterns", analysis_result)
        
    def test_analyze_episodes(self):
        """Test analyzing a list of episodes."""
        episodes = [
            self.create_test_episode("ep1", 3),
            self.create_test_episode("ep2", 3),
            self.create_test_episode("ep3", 3)
        ]
        
        analysis_result = self.manager.analyze_episodes(episodes)
        
        self.assertIn("episode_count", analysis_result)
        self.assertEqual(analysis_result["episode_count"], 3)
        
    def test_compare_generations(self):
        """Test comparing generations."""
        episodes_gen1 = [
            self.create_test_episode("ep1", 3),
            self.create_test_episode("ep2", 3)
        ]
        
        episodes_gen2 = [
            self.create_test_episode("ep3", 4),
            self.create_test_episode("ep4", 4)
        ]
        
        # Set different generations
        for ep in episodes_gen2:
            ep.model_generation = 2
        
        episodes_by_generation = {
            1: episodes_gen1,
            2: episodes_gen2
        }
        
        comparison_result = self.manager.compare_generations(episodes_by_generation)
        
        self.assertIn("generations_analyzed", comparison_result)
        self.assertEqual(comparison_result["generations_analyzed"], [1, 2])
        
    def test_export_episodes_json(self):
        """Test exporting episodes as JSON."""
        episodes = [
            self.create_test_episode("ep1", 3),
            self.create_test_episode("ep2", 3)
        ]
        
        output_path = Path(self.temp_dir) / "export_test"
        exported_path = self.manager.export_episodes(episodes, str(output_path), "json")
        
        self.assertTrue(Path(exported_path).exists())
        self.assertTrue(exported_path.endswith('.json'))
        
        # Verify content
        with open(exported_path, 'r') as f:
            exported_data = json.load(f)
            
        self.assertEqual(len(exported_data), 2)
        self.assertEqual(exported_data[0]["episode_id"], "ep1")
        
    def test_export_episodes_csv(self):
        """Test exporting episodes as CSV."""
        episodes = [
            self.create_test_episode("ep1", 3),
            self.create_test_episode("ep2", 3)
        ]
        
        output_path = Path(self.temp_dir) / "export_test"
        exported_path = self.manager.export_episodes(episodes, str(output_path), "csv")
        
        self.assertTrue(Path(exported_path).exists())
        self.assertTrue(exported_path.endswith('.csv'))
        
    def test_export_episodes_pickle(self):
        """Test exporting episodes as pickle."""
        episodes = [
            self.create_test_episode("ep1", 3),
            self.create_test_episode("ep2", 3)
        ]
        
        output_path = Path(self.temp_dir) / "export_test"
        exported_path = self.manager.export_episodes(episodes, str(output_path), "pickle")
        
        self.assertTrue(Path(exported_path).exists())
        self.assertTrue(exported_path.endswith('.pkl'))
        
        # Verify content
        import pickle
        with open(exported_path, 'rb') as f:
            exported_episodes = pickle.load(f)
            
        self.assertEqual(len(exported_episodes), 2)
        self.assertEqual(exported_episodes[0].episode_id, "ep1")
        
    def test_export_unsupported_format(self):
        """Test exporting with unsupported format."""
        episodes = [self.create_test_episode("ep1", 3)]
        
        output_path = Path(self.temp_dir) / "export_test"
        
        with self.assertRaises(ValueError):
            self.manager.export_episodes(episodes, str(output_path), "unsupported")
            
    def test_export_analysis(self):
        """Test exporting analysis results."""
        episodes = [self.create_test_episode("ep1", 3)]
        analysis_result = self.manager.analyze_episodes(episodes)
        
        output_path = Path(self.temp_dir) / "analysis_export"
        exported_path = self.manager.export_analysis(analysis_result, str(output_path), "json")
        
        self.assertTrue(Path(exported_path).exists())
        
    def test_end_session(self):
        """Test ending a session."""
        self.manager.start_session("end_test_session")
        
        # Record some episodes
        for i in range(2):
            episode = self.create_test_episode(f"ep{i}", 3)
            self.manager.record_game_episode(episode)
        
        session_summary = self.manager.end_session()
        
        self.assertIn("total_episodes", session_summary)
        self.assertEqual(session_summary["total_episodes"], 2)
        self.assertIn("experience_buffer_stats", session_summary)
        
        # Check session state was reset
        self.assertIsNone(self.manager.current_session_id)
        
    def test_get_available_sessions(self):
        """Test getting available sessions."""
        # Create a few sessions
        session_ids = []
        for i in range(3):
            session_id = self.manager.start_session(f"session_{i}")
            session_ids.append(session_id)
            self.manager.end_session()
        
        available_sessions = self.manager.get_available_sessions()
        
        self.assertEqual(len(available_sessions), 3)
        
        # Check session IDs are present
        available_ids = {session["session_id"] for session in available_sessions}
        expected_ids = set(session_ids)
        self.assertEqual(available_ids, expected_ids)
        
    def test_clear_experience_buffer(self):
        """Test clearing the experience buffer."""
        self.manager.start_session("clear_test_session")
        
        # Add some episodes
        episode = self.create_test_episode("ep1", 4)
        self.manager.record_game_episode(episode)
        
        self.assertGreater(len(self.manager.experience_buffer), 0)
        
        self.manager.clear_experience_buffer()
        
        self.assertEqual(len(self.manager.experience_buffer), 0)
        
    def test_get_system_stats(self):
        """Test getting system statistics."""
        stats = self.manager.get_system_stats()
        
        self.assertIn("current_session", stats)
        self.assertIn("recorder_stats", stats)
        self.assertIn("buffer_stats", stats)
        self.assertIn("available_sessions", stats)
        self.assertIn("storage_path", stats)
        
        # Test with active session
        self.manager.start_session("stats_test_session")
        episode = self.create_test_episode("ep1", 3)
        self.manager.record_game_episode(episode)
        
        stats = self.manager.get_system_stats()
        self.assertEqual(stats["current_session"], "stats_test_session")
        
    def test_record_without_buffer(self):
        """Test recording episodes without adding to buffer."""
        self.manager.start_session("no_buffer_test")
        
        episode = self.create_test_episode("ep1", 4)
        
        # Record without adding to buffer
        episode_id = self.manager.record_game_episode(episode, add_to_buffer=False)
        
        self.assertEqual(episode_id, "ep1")
        
        # Buffer should be empty
        self.assertEqual(len(self.manager.experience_buffer), 0)
        
    def test_session_metadata_persistence(self):
        """Test that session metadata is properly persisted."""
        metadata = {"experiment": "test", "model": "DQN", "version": "1.0"}
        
        session_id = self.manager.start_session("metadata_test", metadata)
        self.manager.end_session()
        
        # Check metadata file
        session_path = Path(self.temp_dir) / session_id
        metadata_file = session_path / "replay_session_metadata.json"
        
        with open(metadata_file, 'r') as f:
            saved_metadata = json.load(f)
            
        self.assertEqual(saved_metadata["user_metadata"], metadata)
        self.assertIn("session_summary", saved_metadata)


if __name__ == '__main__':
    unittest.main()