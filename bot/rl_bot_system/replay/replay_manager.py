"""
Main replay manager that coordinates episode recording, storage, analysis, and experience replay.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from rl_bot_system.evaluation.evaluator import GameEpisode
from .episode_recorder import EpisodeRecorder, RecordingConfig
from .experience_buffer import ExperienceBuffer, BufferConfig
from .replay_analyzer import ReplayAnalyzer, AnalysisConfig


class ReplayManager:
    """
    Central manager for all replay system functionality.
    
    Coordinates:
    - Episode recording during training and evaluation
    - Experience buffer management for training batch retrieval
    - Replay analysis for behavior pattern detection
    - Episode export for external analysis
    """
    
    def __init__(
        self,
        storage_path: str = "data/replays",
        recording_config: Optional[RecordingConfig] = None,
        buffer_config: Optional[BufferConfig] = None,
        analysis_config: Optional[AnalysisConfig] = None
    ):
        """
        Initialize the replay manager.
        
        Args:
            storage_path: Base path for storing replay data
            recording_config: Configuration for episode recording
            buffer_config: Configuration for experience buffer
            analysis_config: Configuration for replay analysis
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        if recording_config is None:
            recording_config = RecordingConfig(storage_path=storage_path)
        self.recorder = EpisodeRecorder(recording_config)
        
        self.experience_buffer = ExperienceBuffer(buffer_config)
        self.analyzer = ReplayAnalyzer(analysis_config)
        
        # Session management
        self.current_session_id = None
        self.session_metadata = {}
        
    def start_session(
        self, 
        session_name: Optional[str] = None,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new replay session for recording episodes.
        
        Args:
            session_name: Optional name for the session
            session_metadata: Additional metadata for the session
            
        Returns:
            str: Session ID
        """
        self.current_session_id = self.recorder.start_recording_session(session_name)
        self.session_metadata = session_metadata or {}
        
        # Save session metadata
        session_path = self.storage_path / self.current_session_id
        metadata_file = session_path / "replay_session_metadata.json"
        
        full_metadata = {
            "session_id": self.current_session_id,
            "start_time": datetime.now().isoformat(),
            "user_metadata": self.session_metadata,
            "components": {
                "recorder_config": self.recorder.config.__dict__,
                "buffer_config": self.experience_buffer.config.__dict__,
                "analyzer_config": self.analyzer.config.__dict__
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(full_metadata, f, indent=2)
            
        return self.current_session_id
    
    def record_episode(
        self,
        states: List[dict],
        actions: List[int],
        rewards: List[float],
        model_generation: int,
        opponent_generation: int = -1,
        game_result: str = "unknown",
        episode_metadata: Optional[Dict[str, Any]] = None,
        add_to_buffer: bool = True
    ) -> str:
        """
        Record a game episode and optionally add to experience buffer.
        
        Args:
            states: List of game states during the episode
            actions: List of actions taken during the episode
            rewards: List of rewards received during the episode
            model_generation: Generation of the model that played
            opponent_generation: Generation of the opponent model
            game_result: Result of the game ('win', 'loss', 'draw')
            episode_metadata: Additional metadata for the episode
            add_to_buffer: Whether to add episode to experience buffer
            
        Returns:
            str: Episode ID
        """
        # Record episode
        episode_id = self.recorder.record_episode(
            states=states,
            actions=actions,
            rewards=rewards,
            model_generation=model_generation,
            opponent_generation=opponent_generation,
            game_result=game_result,
            episode_metadata=episode_metadata
        )
        
        # Add to experience buffer if requested
        if add_to_buffer and states and actions and rewards:
            episode = GameEpisode(
                episode_id=episode_id,
                model_generation=model_generation,
                opponent_generation=opponent_generation,
                states=states,
                actions=actions,
                rewards=rewards,
                total_reward=sum(rewards),
                episode_length=len(states),
                game_result=game_result,
                episode_metrics=episode_metadata or {}
            )
            self.experience_buffer.add_episode(episode)
        
        return episode_id
    
    def record_game_episode(self, episode: GameEpisode, add_to_buffer: bool = True) -> str:
        """
        Record an existing GameEpisode object.
        
        Args:
            episode: GameEpisode object to record
            add_to_buffer: Whether to add episode to experience buffer
            
        Returns:
            str: Episode ID
        """
        episode_id = self.recorder.record_game_episode(episode)
        
        if add_to_buffer:
            self.experience_buffer.add_episode(episode)
            
        return episode_id
    
    def get_training_batch(self, batch_size: int) -> List:
        """
        Get a batch of experiences for training.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List: Sampled experiences for training
        """
        return self.experience_buffer.sample_batch(batch_size)
    
    def get_episode_batch(self, num_episodes: int) -> List[List]:
        """
        Get a batch of complete episodes for training.
        
        Args:
            num_episodes: Number of complete episodes to sample
            
        Returns:
            List[List]: List of episodes, each containing transitions
        """
        return self.experience_buffer.sample_episode_batch(num_episodes)
    
    def load_episodes_from_session(self, session_id: str) -> List[GameEpisode]:
        """
        Load all episodes from a specific recording session.
        
        Args:
            session_id: ID of the session to load episodes from
            
        Returns:
            List[GameEpisode]: Episodes from the session
        """
        session_path = self.storage_path / session_id
        if not session_path.exists():
            raise ValueError(f"Session {session_id} not found")
            
        episodes = []
        
        # Load all episode files from the session
        for episode_file in session_path.glob("episodes_*.pkl"):
            try:
                with open(episode_file, 'rb') as f:
                    session_episodes = pickle.load(f)
                    episodes.extend(session_episodes)
            except Exception as e:
                print(f"Warning: Could not load {episode_file}: {e}")
                
        # Also try JSON files if pickle files are not available
        if not episodes:
            for episode_file in session_path.glob("episodes_*.json"):
                try:
                    with open(episode_file, 'r') as f:
                        episodes_data = json.load(f)
                        for ep_data in episodes_data:
                            episode = GameEpisode(**ep_data)
                            episodes.append(episode)
                except Exception as e:
                    print(f"Warning: Could not load {episode_file}: {e}")
        
        return episodes
    
    def analyze_session(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze all episodes from a specific session.
        
        Args:
            session_id: ID of the session to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        episodes = self.load_episodes_from_session(session_id)
        return self.analyzer.analyze_episodes(episodes)
    
    def analyze_episodes(self, episodes: List[GameEpisode]) -> Dict[str, Any]:
        """
        Analyze a list of episodes for behavior patterns.
        
        Args:
            episodes: List of GameEpisode objects to analyze
            
        Returns:
            Dict[str, Any]: Analysis results including patterns and statistics
        """
        return self.analyzer.analyze_episodes(episodes)
    
    def compare_generations(
        self, 
        episodes_by_generation: Dict[int, List[GameEpisode]]
    ) -> Dict[str, Any]:
        """
        Compare behavior patterns across different model generations.
        
        Args:
            episodes_by_generation: Episodes grouped by model generation
            
        Returns:
            Dict[str, Any]: Comparative analysis results
        """
        return self.analyzer.compare_generations(episodes_by_generation)
    
    def export_episodes(
        self, 
        episodes: List[GameEpisode], 
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        Export episodes to external format for analysis.
        
        Args:
            episodes: Episodes to export
            output_path: Path to save the exported data
            format: Export format ('json', 'csv', 'pickle')
            
        Returns:
            str: Path to the exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            json_path = output_path.with_suffix('.json')
            episodes_data = []
            
            for episode in episodes:
                episode_dict = {
                    "episode_id": episode.episode_id,
                    "model_generation": episode.model_generation,
                    "opponent_generation": episode.opponent_generation,
                    "states": episode.states,
                    "actions": episode.actions,
                    "rewards": episode.rewards,
                    "total_reward": episode.total_reward,
                    "episode_length": episode.episode_length,
                    "game_result": episode.game_result,
                    "episode_metrics": episode.episode_metrics
                }
                episodes_data.append(episode_dict)
            
            with open(json_path, 'w') as f:
                json.dump(episodes_data, f, indent=2)
            
            return str(json_path)
            
        elif format.lower() == "csv":
            import csv
            csv_path = output_path.with_suffix('.csv')
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'episode_id', 'model_generation', 'opponent_generation',
                    'total_reward', 'episode_length', 'game_result'
                ])
                
                # Write episode data
                for episode in episodes:
                    writer.writerow([
                        episode.episode_id,
                        episode.model_generation,
                        episode.opponent_generation,
                        episode.total_reward,
                        episode.episode_length,
                        episode.game_result
                    ])
            
            return str(csv_path)
            
        elif format.lower() == "pickle":
            pickle_path = output_path.with_suffix('.pkl')
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(episodes, f)
            
            return str(pickle_path)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def export_analysis(
        self, 
        analysis_results: Dict[str, Any], 
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        Export analysis results to file.
        
        Args:
            analysis_results: Results from analyze_episodes or compare_generations
            output_path: Path to save the analysis
            format: Export format ('json', 'csv', 'html')
            
        Returns:
            str: Path to the exported file
        """
        return self.analyzer.export_analysis(analysis_results, output_path, format)
    
    def end_session(self) -> Dict[str, Any]:
        """
        End the current recording session.
        
        Returns:
            Dict[str, Any]: Session summary statistics
        """
        if self.current_session_id is None:
            return {}
            
        session_summary = self.recorder.end_recording_session()
        
        # Add buffer statistics to summary
        session_summary["experience_buffer_stats"] = self.experience_buffer.get_statistics()
        
        # Update session metadata file
        session_path = self.storage_path / self.current_session_id
        metadata_file = session_path / "replay_session_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
            
        metadata.update({
            "end_time": datetime.now().isoformat(),
            "session_summary": session_summary
        })
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.current_session_id = None
        self.session_metadata = {}
        
        return session_summary
    
    def get_available_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of available recording sessions.
        
        Returns:
            List[Dict[str, Any]]: List of session information
        """
        sessions = []
        
        for session_dir in self.storage_path.iterdir():
            if session_dir.is_dir():
                metadata_file = session_dir / "replay_session_metadata.json"
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        sessions.append(metadata)
                    except Exception as e:
                        print(f"Warning: Could not load metadata for {session_dir.name}: {e}")
                else:
                    # Create basic metadata for sessions without metadata files
                    sessions.append({
                        "session_id": session_dir.name,
                        "start_time": "unknown",
                        "user_metadata": {}
                    })
        
        return sorted(sessions, key=lambda x: x.get("start_time", ""), reverse=True)
    
    def clear_experience_buffer(self):
        """Clear the experience buffer."""
        self.experience_buffer.clear()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        return {
            "current_session": self.current_session_id,
            "recorder_stats": self.recorder.get_recording_stats(),
            "buffer_stats": self.experience_buffer.get_statistics(),
            "available_sessions": len(self.get_available_sessions()),
            "storage_path": str(self.storage_path)
        }