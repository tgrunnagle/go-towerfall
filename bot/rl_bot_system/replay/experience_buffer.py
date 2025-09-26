"""
Experience buffer for storing and retrieving training batches from recorded episodes.
"""

import random
import numpy as np
from collections import deque
from typing import List, Dict, Any, Tuple, Optional, Iterator
from dataclasses import dataclass

from rl_bot_system.evaluation.evaluator import GameEpisode


@dataclass
class ExperienceTransition:
    """Single experience transition for RL training."""
    state: dict
    action: int
    reward: float
    next_state: dict
    done: bool
    episode_id: str
    step_index: int


@dataclass
class BufferConfig:
    """Configuration for experience buffer."""
    max_size: int = 100000
    min_size_for_sampling: int = 1000
    prioritized_replay: bool = False
    alpha: float = 0.6  # Prioritization exponent
    beta: float = 0.4   # Importance sampling exponent
    epsilon: float = 1e-6  # Small constant to prevent zero probabilities


class ExperienceBuffer:
    """
    Experience replay buffer for storing and sampling training experiences.
    
    Features:
    - Efficient storage of state-action-reward transitions
    - Random and prioritized sampling for training
    - Episode-based experience extraction
    - Memory management with configurable limits
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        """
        Initialize the experience buffer.
        
        Args:
            config: Buffer configuration options
        """
        self.config = config or BufferConfig()
        self.buffer = deque(maxlen=self.config.max_size)
        self.priorities = deque(maxlen=self.config.max_size) if self.config.prioritized_replay else None
        
        # Statistics
        self.total_experiences_added = 0
        self.episodes_processed = 0
        
    def add_episode(self, episode: GameEpisode):
        """
        Add all transitions from a game episode to the buffer.
        
        Args:
            episode: GameEpisode containing states, actions, and rewards
        """
        if not episode.states or not episode.actions or not episode.rewards:
            return
            
        # Extract transitions from episode
        transitions = self._extract_transitions(episode)
        
        for transition in transitions:
            self.buffer.append(transition)
            
            if self.config.prioritized_replay:
                # Initialize with maximum priority for new experiences
                max_priority = max(self.priorities) if self.priorities else 1.0
                self.priorities.append(max_priority)
                
        self.total_experiences_added += len(transitions)
        self.episodes_processed += 1
        
    def add_episodes(self, episodes: List[GameEpisode]):
        """
        Add multiple episodes to the buffer.
        
        Args:
            episodes: List of GameEpisode objects
        """
        for episode in episodes:
            self.add_episode(episode)
    
    def sample_batch(self, batch_size: int) -> List[ExperienceTransition]:
        """
        Sample a batch of experiences for training.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List[ExperienceTransition]: Sampled experiences
            
        Raises:
            ValueError: If buffer doesn't have enough experiences
        """
        if len(self.buffer) < self.config.min_size_for_sampling:
            raise ValueError(
                f"Buffer has {len(self.buffer)} experiences, "
                f"need at least {self.config.min_size_for_sampling}"
            )
            
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
        if self.config.prioritized_replay:
            return self._sample_prioritized(batch_size)
        else:
            return self._sample_uniform(batch_size)
    
    def sample_episode_batch(self, num_episodes: int) -> List[List[ExperienceTransition]]:
        """
        Sample complete episodes for training.
        
        Args:
            num_episodes: Number of complete episodes to sample
            
        Returns:
            List[List[ExperienceTransition]]: List of episodes, each containing transitions
        """
        # Group transitions by episode
        episodes_dict = {}
        for transition in self.buffer:
            episode_id = transition.episode_id
            if episode_id not in episodes_dict:
                episodes_dict[episode_id] = []
            episodes_dict[episode_id].append(transition)
            
        # Sort transitions within each episode by step index
        for episode_id in episodes_dict:
            episodes_dict[episode_id].sort(key=lambda t: t.step_index)
            
        # Sample episodes
        available_episodes = list(episodes_dict.keys())
        if num_episodes > len(available_episodes):
            num_episodes = len(available_episodes)
            
        sampled_episode_ids = random.sample(available_episodes, num_episodes)
        return [episodes_dict[episode_id] for episode_id in sampled_episode_ids]
    
    def get_recent_experiences(self, num_experiences: int) -> List[ExperienceTransition]:
        """
        Get the most recently added experiences.
        
        Args:
            num_experiences: Number of recent experiences to retrieve
            
        Returns:
            List[ExperienceTransition]: Most recent experiences
        """
        if num_experiences > len(self.buffer):
            num_experiences = len(self.buffer)
            
        return list(self.buffer)[-num_experiences:]
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update priorities for prioritized replay.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        if not self.config.prioritized_replay or self.priorities is None:
            return
            
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority + self.config.epsilon
    
    def clear(self):
        """Clear all experiences from the buffer."""
        self.buffer.clear()
        if self.priorities:
            self.priorities.clear()
        self.total_experiences_added = 0
        self.episodes_processed = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dict[str, Any]: Buffer statistics
        """
        episode_ids = set(t.episode_id for t in self.buffer)
        
        return {
            "total_experiences": len(self.buffer),
            "unique_episodes": len(episode_ids),
            "total_experiences_added": self.total_experiences_added,
            "episodes_processed": self.episodes_processed,
            "buffer_utilization": len(self.buffer) / self.config.max_size,
            "ready_for_sampling": len(self.buffer) >= self.config.min_size_for_sampling,
            "config": {
                "max_size": self.config.max_size,
                "min_size_for_sampling": self.config.min_size_for_sampling,
                "prioritized_replay": self.config.prioritized_replay
            }
        }
    
    def _extract_transitions(self, episode: GameEpisode) -> List[ExperienceTransition]:
        """
        Extract individual transitions from a game episode.
        
        Args:
            episode: GameEpisode to extract transitions from
            
        Returns:
            List[ExperienceTransition]: Individual transitions
        """
        transitions = []
        
        for i in range(len(episode.states) - 1):
            transition = ExperienceTransition(
                state=episode.states[i],
                action=episode.actions[i] if i < len(episode.actions) else 0,
                reward=episode.rewards[i] if i < len(episode.rewards) else 0.0,
                next_state=episode.states[i + 1],
                done=False,
                episode_id=episode.episode_id,
                step_index=i
            )
            transitions.append(transition)
            
        # Mark the last transition as done
        if transitions:
            transitions[-1].done = True
            
        return transitions
    
    def _sample_uniform(self, batch_size: int) -> List[ExperienceTransition]:
        """Sample experiences uniformly at random."""
        return random.sample(list(self.buffer), batch_size)
    
    def _sample_prioritized(self, batch_size: int) -> List[ExperienceTransition]:
        """Sample experiences using prioritized replay."""
        if self.priorities is None:
            return self._sample_uniform(batch_size)
            
        # Convert priorities to probabilities
        priorities_array = np.array(self.priorities)
        probabilities = priorities_array ** self.config.alpha
        probabilities = probabilities / probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(
            len(self.buffer), 
            size=batch_size, 
            p=probabilities,
            replace=False
        )
        
        # Return sampled experiences
        return [self.buffer[idx] for idx in indices]
    
    def __len__(self) -> int:
        """Return the number of experiences in the buffer."""
        return len(self.buffer)
    
    def __iter__(self) -> Iterator[ExperienceTransition]:
        """Iterate over all experiences in the buffer."""
        return iter(self.buffer)