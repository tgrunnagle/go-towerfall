"""
Unit tests for episode recorder functionality.
"""

import unittest
import tempfile
import shutil
import json
import pickle
from pathlib import Path
from unittest.mock import patch

from rl_bot_system.replay.episode_recorder import EpisodeRecorder, RecordingConfig
from rl_bot_system.evaluation.evaluator import GameEpisode


class TestEpisodeRecorder(unittest.TestCase):
    """Test EpisodeRecorder functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = RecordingConfig(
            storage_path=self.temp_dir,
            max_episodes_per_file=2,  # Small for testing
            auto_cleanup=False  # Disable for testing
        )
        self.recorder = EpisodeRecorder(self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_start_recording_session(self):
        """Test starting a recording session."""
        session_id = self.recorder.start_recording_session("test_session")
        
        self.assertEqual(session_id, "test_session")
        self.assertEqual(self.recorder.current_session_id, "test_session")
        
        # Check session directory was created
        session_path = Path(self.temp_dir) / "test_session"
        self.assertTrue(session_path.exists())
        
        # Check metadata file was created
        metadata_file = session_path / "session_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata["session_id"], "test_session")
        self.assertIn("start_time", metadata)
        
    def test_record_episode(self):
        """Test recording a single episode."""
        self.recorder.start_recording_session("test_session")
        
        states = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        actions = [0, 1]
        rewards = [1.0, 2.0]
        
        episode_id = self.recorder.record_episode(
            states=states,
            actions=actions,
            rewards=rewards,
            model_generation=1,
            opponent_generation=0,
            game_result="win"
        )
        
        self.assertIsInstance(episode_id, str)
        self.assertEqual(len(self.recorder.current_episode_buffer), 1)
        self.assertEqual(self.recorder.total_episodes_recorded, 1)
        
        # Check episode data
        episode = self.recorder.current_episode_buffer[0]
        self.assertEqual(episode.states, states)
        self.assertEqual(episode.actions, actions)
        self.assertEqual(episode.rewards, rewards)
        self.assertEqual(episode.total_reward, 3.0)
        self.assertEqual(episode.game_result, "win")
        
    def test_record_game_episode(self):
        """Test recording an existing GameEpisode object."""
        self.recorder.start_recording_session("test_session")
        
        episode = GameEpisode(
            episode_id="test_episode",
            model_generation=1,
            opponent_generation=0,
            states=[{"x": 1}],
            actions=[0],
            rewards=[1.0],
            total_reward=1.0,
            episode_length=1,
            game_result="win",
            episode_metrics={}
        )
        
        episode_id = self.recorder.record_game_episode(episode)
        
        self.assertEqual(episode_id, "test_episode")
        self.assertEqual(len(self.recorder.current_episode_buffer), 1)
        
    def test_buffer_flush(self):
        """Test automatic buffer flushing when max episodes reached."""
        self.recorder.start_recording_session("test_session")
        
        # Record episodes to trigger flush
        for i in range(3):  # max_episodes_per_file is 2
            self.recorder.record_episode(
                states=[{"step": i}],
                actions=[i],
                rewards=[float(i)],
                model_generation=1
            )
        
        # Buffer should have been flushed and contain only the last episode
        self.assertEqual(len(self.recorder.current_episode_buffer), 1)
        
        # Check that episode file was created
        session_path = Path(self.temp_dir) / "test_session"
        episode_files = list(session_path.glob("episodes_*.pkl"))
        self.assertEqual(len(episode_files), 1)
        
        # Load and verify episodes
        with open(episode_files[0], 'rb') as f:
            saved_episodes = pickle.load(f)
        
        self.assertEqual(len(saved_episodes), 2)
        
    def test_end_recording_session(self):
        """Test ending a recording session."""
        self.recorder.start_recording_session("test_session")
        
        # Record some episodes
        for i in range(2):
            self.recorder.record_episode(
                states=[{"step": i}],
                actions=[i],
                rewards=[float(i)],
                model_generation=1
            )
        
        summary = self.recorder.end_recording_session()
        
        self.assertIn("total_episodes", summary)
        self.assertEqual(summary["total_episodes"], 2)
        self.assertIn("duration_seconds", summary)
        
        # Check session state was reset
        self.assertIsNone(self.recorder.current_session_id)
        self.assertEqual(len(self.recorder.current_episode_buffer), 0)
        self.assertEqual(self.recorder.total_episodes_recorded, 0)
        
    def test_recording_config_options(self):
        """Test different recording configuration options."""
        # Test with states/actions/rewards disabled
        config = RecordingConfig(
            storage_path=self.temp_dir,
            record_states=False,
            record_actions=False,
            record_rewards=False
        )
        recorder = EpisodeRecorder(config)
        recorder.start_recording_session("config_test")
        
        recorder.record_episode(
            states=[{"x": 1}],
            actions=[0],
            rewards=[1.0],
            model_generation=1
        )
        
        episode = recorder.current_episode_buffer[0]
        self.assertEqual(episode.states, [])
        self.assertEqual(episode.actions, [])
        self.assertEqual(episode.rewards, [])
        
    def test_get_recording_stats(self):
        """Test getting recording statistics."""
        stats = self.recorder.get_recording_stats()
        
        self.assertIn("current_session_id", stats)
        self.assertIn("episodes_in_buffer", stats)
        self.assertIn("total_episodes_recorded", stats)
        self.assertIn("storage_path", stats)
        self.assertIn("config", stats)
        
        # Test with active session
        self.recorder.start_recording_session("stats_test")
        self.recorder.record_episode(
            states=[{"x": 1}],
            actions=[0],
            rewards=[1.0],
            model_generation=1
        )
        
        stats = self.recorder.get_recording_stats()
        self.assertEqual(stats["current_session_id"], "stats_test")
        self.assertEqual(stats["episodes_in_buffer"], 1)
        self.assertEqual(stats["total_episodes_recorded"], 1)


if __name__ == '__main__':
    unittest.main()