"""
Episode recording system for capturing game episodes during training and evaluation.
"""

import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

from bot.rl_bot_system.evaluation.evaluator import GameEpisode


@dataclass
class RecordingConfig:
    """Configuration for episode recording."""
    storage_path: str = "bot/data/replays"
    max_episodes_per_file: int = 100
    compression: bool = True
    record_states: bool = True
    record_actions: bool = True
    record_rewards: bool = True
    record_metadata: bool = True
    auto_cleanup: bool = True
    max_storage_mb: int = 1000


class EpisodeRecorder:
    """
    Records game episodes during training and evaluation for later analysis.
    
    Features:
    - Real-time episode recording during gameplay
    - Configurable storage formats (JSON, pickle)
    - Automatic file management and cleanup
    - Metadata tracking for episodes
    """
    
    def __init__(self, config: Optional[RecordingConfig] = None):
        """
        Initialize the episode recorder.
        
        Args:
            config: Recording configuration options
        """
        self.config = config or RecordingConfig()
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Current recording session
        self.current_session_id = None
        self.current_episode_buffer = []
        
        # Statistics
        self.total_episodes_recorded = 0
        self.session_start_time = None
        
    def start_recording_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new recording session.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            str: Session ID for this recording session
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
        self.current_session_id = session_id
        self.current_episode_buffer = []
        self.session_start_time = datetime.now()
        
        # Create session directory
        session_path = self.storage_path / session_id
        session_path.mkdir(exist_ok=True)
        
        # Save session metadata
        session_metadata = {
            "session_id": session_id,
            "start_time": self.session_start_time.isoformat(),
            "config": asdict(self.config)
        }
        
        with open(session_path / "session_metadata.json", 'w') as f:
            json.dump(session_metadata, f, indent=2)
            
        return session_id
    
    def record_episode(
        self,
        states: List[dict],
        actions: List[int], 
        rewards: List[float],
        model_generation: int,
        opponent_generation: int = -1,
        game_result: str = "unknown",
        episode_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a single game episode.
        
        Args:
            states: List of game states during the episode
            actions: List of actions taken during the episode
            rewards: List of rewards received during the episode
            model_generation: Generation of the model that played
            opponent_generation: Generation of the opponent model
            game_result: Result of the game ('win', 'loss', 'draw')
            episode_metadata: Additional metadata for the episode
            
        Returns:
            str: Episode ID for the recorded episode
        """
        if self.current_session_id is None:
            self.start_recording_session()
            
        episode_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create GameEpisode object
        episode = GameEpisode(
            episode_id=episode_id,
            model_generation=model_generation,
            opponent_generation=opponent_generation,
            states=states if self.config.record_states else [],
            actions=actions if self.config.record_actions else [],
            rewards=rewards if self.config.record_rewards else [],
            total_reward=sum(rewards) if rewards else 0.0,
            episode_length=len(states) if states else len(actions),
            game_result=game_result,
            episode_metrics=episode_metadata or {}
        )
        
        # Add to current buffer
        self.current_episode_buffer.append(episode)
        self.total_episodes_recorded += 1
        
        # Check if we need to flush buffer to disk
        if len(self.current_episode_buffer) >= self.config.max_episodes_per_file:
            self._flush_episode_buffer()
            
        return episode_id
    
    def record_game_episode(self, episode: GameEpisode) -> str:
        """
        Record an existing GameEpisode object.
        
        Args:
            episode: GameEpisode object to record
            
        Returns:
            str: Episode ID
        """
        if self.current_session_id is None:
            self.start_recording_session()
            
        self.current_episode_buffer.append(episode)
        self.total_episodes_recorded += 1
        
        if len(self.current_episode_buffer) >= self.config.max_episodes_per_file:
            self._flush_episode_buffer()
            
        return episode.episode_id
    
    def end_recording_session(self) -> Dict[str, Any]:
        """
        End the current recording session and flush remaining episodes.
        
        Returns:
            Dict[str, Any]: Session summary statistics
        """
        if self.current_session_id is None:
            return {}
            
        # Flush any remaining episodes
        if self.current_episode_buffer:
            self._flush_episode_buffer()
            
        # Update session metadata with final statistics
        session_path = self.storage_path / self.current_session_id
        session_metadata_path = session_path / "session_metadata.json"
        
        if session_metadata_path.exists():
            with open(session_metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
            
        end_time = datetime.now()
        duration = (end_time - self.session_start_time).total_seconds() if self.session_start_time else 0
        
        metadata.update({
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_episodes": self.total_episodes_recorded,
            "episodes_per_second": self.total_episodes_recorded / duration if duration > 0 else 0
        })
        
        with open(session_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Reset session state
        session_summary = metadata.copy()
        self.current_session_id = None
        self.current_episode_buffer = []
        self.total_episodes_recorded = 0
        self.session_start_time = None
        
        return session_summary
    
    def _flush_episode_buffer(self):
        """Flush the current episode buffer to disk."""
        if not self.current_episode_buffer or self.current_session_id is None:
            return
            
        session_path = self.storage_path / self.current_session_id
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # Include microseconds
        
        # Save as pickle for efficient storage and loading
        pickle_path = session_path / f"episodes_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.current_episode_buffer, f)
            
        # Also save as JSON for human readability (optional)
        if not self.config.compression:
            json_path = session_path / f"episodes_{timestamp}.json"
            episodes_dict = [asdict(episode) for episode in self.current_episode_buffer]
            with open(json_path, 'w') as f:
                json.dump(episodes_dict, f, indent=2)
        
        # Clear buffer
        self.current_episode_buffer = []
        
        # Perform cleanup if needed
        if self.config.auto_cleanup:
            self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """Clean up old episode files if storage limit is exceeded."""
        try:
            total_size_mb = sum(
                f.stat().st_size for f in self.storage_path.rglob('*') 
                if f.is_file()
            ) / (1024 * 1024)
            
            if total_size_mb > self.config.max_storage_mb:
                # Remove oldest session directories
                session_dirs = [d for d in self.storage_path.iterdir() if d.is_dir()]
                session_dirs.sort(key=lambda x: x.stat().st_mtime)
                
                while total_size_mb > self.config.max_storage_mb * 0.8 and session_dirs:
                    oldest_dir = session_dirs.pop(0)
                    if oldest_dir.name != self.current_session_id:  # Don't delete current session
                        import shutil
                        shutil.rmtree(oldest_dir)
                        
                        # Recalculate size
                        total_size_mb = sum(
                            f.stat().st_size for f in self.storage_path.rglob('*') 
                            if f.is_file()
                        ) / (1024 * 1024)
                        
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """
        Get current recording statistics.
        
        Returns:
            Dict[str, Any]: Recording statistics
        """
        return {
            "current_session_id": self.current_session_id,
            "episodes_in_buffer": len(self.current_episode_buffer),
            "total_episodes_recorded": self.total_episodes_recorded,
            "session_start_time": self.session_start_time.isoformat() if self.session_start_time else None,
            "storage_path": str(self.storage_path),
            "config": asdict(self.config)
        }