"""
Unit tests for experience buffer functionality.
"""

import unittest
import random
from unittest.mock import patch

from bot.rl_bot_system.replay.experience_buffer import (
    ExperienceBuffer, BufferConfig, ExperienceTransition
)
from bot.rl_bot_system.evaluation.evaluator import GameEpisode


class TestExperienceBuffer(unittest.TestCase):
    """Test ExperienceBuffer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = BufferConfig(
            max_size=100,
            min_size_for_sampling=5,
            prioritized_replay=False
        )
        self.buffer = ExperienceBuffer(self.config)
        
    def create_test_episode(self, episode_id: str = "test_ep", length: int = 5) -> GameEpisode:
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
        
    def test_add_episode(self):
        """Test adding an episode to the buffer."""
        episode = self.create_test_episode("ep1", 3)
        
        self.buffer.add_episode(episode)
        
        # Should have 2 transitions (3 states -> 2 transitions)
        self.assertEqual(len(self.buffer), 2)
        self.assertEqual(self.buffer.episodes_processed, 1)
        self.assertEqual(self.buffer.total_experiences_added, 2)
        
    def test_add_multiple_episodes(self):
        """Test adding multiple episodes."""
        episodes = [
            self.create_test_episode("ep1", 3),
            self.create_test_episode("ep2", 4),
            self.create_test_episode("ep3", 2)
        ]
        
        self.buffer.add_episodes(episodes)
        
        # Total transitions: (3-1) + (4-1) + (2-1) = 2 + 3 + 1 = 6
        self.assertEqual(len(self.buffer), 6)
        self.assertEqual(self.buffer.episodes_processed, 3)
        
    def test_sample_batch_insufficient_data(self):
        """Test sampling when buffer doesn't have enough data."""
        episode = self.create_test_episode("ep1", 3)
        self.buffer.add_episode(episode)
        
        # Buffer has 2 experiences, but min_size_for_sampling is 5
        with self.assertRaises(ValueError):
            self.buffer.sample_batch(3)
            
    def test_sample_batch_sufficient_data(self):
        """Test sampling when buffer has enough data."""
        # Add enough episodes to meet minimum sampling requirement
        for i in range(3):
            episode = self.create_test_episode(f"ep{i}", 4)
            self.buffer.add_episode(episode)
        
        # Should have 9 transitions total (3 episodes * 3 transitions each)
        self.assertEqual(len(self.buffer), 9)
        
        # Sample a batch
        batch = self.buffer.sample_batch(5)
        
        self.assertEqual(len(batch), 5)
        self.assertIsInstance(batch[0], ExperienceTransition)
        
    def test_sample_episode_batch(self):
        """Test sampling complete episodes."""
        episodes = [
            self.create_test_episode("ep1", 3),
            self.create_test_episode("ep2", 4),
            self.create_test_episode("ep3", 2)
        ]
        
        self.buffer.add_episodes(episodes)
        
        episode_batch = self.buffer.sample_episode_batch(2)
        
        self.assertEqual(len(episode_batch), 2)
        
        # Each episode should be a list of transitions
        for episode_transitions in episode_batch:
            self.assertIsInstance(episode_transitions, list)
            self.assertTrue(len(episode_transitions) > 0)
            self.assertIsInstance(episode_transitions[0], ExperienceTransition)
            
            # Check transitions are ordered by step_index
            for i in range(1, len(episode_transitions)):
                self.assertGreaterEqual(
                    episode_transitions[i].step_index,
                    episode_transitions[i-1].step_index
                )
                
    def test_get_recent_experiences(self):
        """Test getting recent experiences."""
        episodes = [
            self.create_test_episode("ep1", 3),
            self.create_test_episode("ep2", 3)
        ]
        
        self.buffer.add_episodes(episodes)
        
        recent = self.buffer.get_recent_experiences(3)
        
        self.assertEqual(len(recent), 3)
        
        # Should be the most recently added experiences
        for transition in recent:
            self.assertIsInstance(transition, ExperienceTransition)
            
    def test_experience_transition_extraction(self):
        """Test that transitions are correctly extracted from episodes."""
        episode = self.create_test_episode("ep1", 4)
        
        self.buffer.add_episode(episode)
        
        # Should have 3 transitions (4 states -> 3 transitions)
        transitions = list(self.buffer.buffer)
        self.assertEqual(len(transitions), 3)
        
        # Check first transition
        first_transition = transitions[0]
        self.assertEqual(first_transition.state, {"step": 0, "x": 0})
        self.assertEqual(first_transition.action, 0)
        self.assertEqual(first_transition.reward, 0.0)
        self.assertEqual(first_transition.next_state, {"step": 1, "x": 10})
        self.assertFalse(first_transition.done)
        self.assertEqual(first_transition.episode_id, "ep1")
        self.assertEqual(first_transition.step_index, 0)
        
        # Check last transition (should be marked as done)
        last_transition = transitions[-1]
        self.assertTrue(last_transition.done)
        
    def test_buffer_max_size_limit(self):
        """Test that buffer respects maximum size limit."""
        # Create buffer with small max size
        small_config = BufferConfig(max_size=5, min_size_for_sampling=1)
        small_buffer = ExperienceBuffer(small_config)
        
        # Add more episodes than buffer can hold
        for i in range(10):
            episode = self.create_test_episode(f"ep{i}", 3)
            small_buffer.add_episode(episode)
        
        # Buffer should not exceed max size
        self.assertLessEqual(len(small_buffer), 5)
        
    def test_clear_buffer(self):
        """Test clearing the buffer."""
        episode = self.create_test_episode("ep1", 3)
        self.buffer.add_episode(episode)
        
        self.assertGreater(len(self.buffer), 0)
        
        self.buffer.clear()
        
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.total_experiences_added, 0)
        self.assertEqual(self.buffer.episodes_processed, 0)
        
    def test_get_statistics(self):
        """Test getting buffer statistics."""
        stats = self.buffer.get_statistics()
        
        self.assertIn("total_experiences", stats)
        self.assertIn("unique_episodes", stats)
        self.assertIn("buffer_utilization", stats)
        self.assertIn("ready_for_sampling", stats)
        self.assertIn("config", stats)
        
        # Add some data and check stats update
        episodes = [
            self.create_test_episode("ep1", 3),
            self.create_test_episode("ep2", 4)
        ]
        self.buffer.add_episodes(episodes)
        
        stats = self.buffer.get_statistics()
        self.assertEqual(stats["total_experiences"], 5)  # (3-1) + (4-1)
        self.assertEqual(stats["unique_episodes"], 2)
        self.assertTrue(stats["ready_for_sampling"])  # 5 >= min_size_for_sampling (5)
        
    def test_prioritized_replay_config(self):
        """Test prioritized replay configuration."""
        prioritized_config = BufferConfig(
            max_size=100,
            min_size_for_sampling=5,
            prioritized_replay=True,
            alpha=0.6,
            beta=0.4
        )
        
        prioritized_buffer = ExperienceBuffer(prioritized_config)
        
        # Add episode
        episode = self.create_test_episode("ep1", 6)
        prioritized_buffer.add_episode(episode)
        
        # Should have priorities initialized
        self.assertIsNotNone(prioritized_buffer.priorities)
        self.assertEqual(len(prioritized_buffer.priorities), len(prioritized_buffer.buffer))
        
    def test_empty_episode_handling(self):
        """Test handling of episodes with missing data."""
        # Episode with no states
        empty_episode = GameEpisode(
            episode_id="empty",
            model_generation=1,
            opponent_generation=0,
            states=[],
            actions=[],
            rewards=[],
            total_reward=0.0,
            episode_length=0,
            game_result="unknown",
            episode_metrics={}
        )
        
        initial_size = len(self.buffer)
        self.buffer.add_episode(empty_episode)
        
        # Buffer size should not change
        self.assertEqual(len(self.buffer), initial_size)
        
    def test_iterator_interface(self):
        """Test that buffer can be iterated over."""
        episode = self.create_test_episode("ep1", 4)
        self.buffer.add_episode(episode)
        
        transitions = list(self.buffer)
        
        self.assertEqual(len(transitions), 3)
        for transition in transitions:
            self.assertIsInstance(transition, ExperienceTransition)


class TestPrioritizedExperienceBuffer(unittest.TestCase):
    """Test prioritized replay functionality."""
    
    def setUp(self):
        """Set up test environment with prioritized replay."""
        self.config = BufferConfig(
            max_size=100,
            min_size_for_sampling=5,
            prioritized_replay=True,
            alpha=0.6,
            beta=0.4
        )
        self.buffer = ExperienceBuffer(self.config)
        
    def create_test_episode(self, episode_id: str = "test_ep", length: int = 5) -> GameEpisode:
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
        
    def test_prioritized_sampling(self):
        """Test prioritized sampling functionality."""
        # Add enough episodes for sampling
        for i in range(3):
            episode = self.create_test_episode(f"ep{i}", 4)
            self.buffer.add_episode(episode)
        
        # Sample batch using prioritized replay
        batch = self.buffer.sample_batch(5)
        
        self.assertEqual(len(batch), 5)
        for transition in batch:
            self.assertIsInstance(transition, ExperienceTransition)
            
    def test_update_priorities(self):
        """Test updating priorities for experiences."""
        episode = self.create_test_episode("ep1", 6)
        self.buffer.add_episode(episode)
        
        # Update priorities for first few experiences
        indices = [0, 1, 2]
        new_priorities = [0.8, 0.6, 0.9]
        
        self.buffer.update_priorities(indices, new_priorities)
        
        # Verify priorities were updated (with epsilon added)
        for i, expected_priority in zip(indices, new_priorities):
            actual_priority = self.buffer.priorities[i]
            self.assertAlmostEqual(
                actual_priority, 
                expected_priority + self.config.epsilon,
                places=6
            )


if __name__ == '__main__':
    unittest.main()