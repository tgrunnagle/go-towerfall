"""
Episode replay system for spectator interface.

This module provides functionality to replay recorded episodes with
pause/rewind controls and side-by-side model comparison.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

from rl_bot_system.evaluation.evaluator import GameEpisode
from rl_bot_system.replay.replay_manager import ReplayManager
from rl_bot_system.spectator.spectator_manager import SpectatorManager, SpectatorMode

logger = logging.getLogger(__name__)


class ReplayState(Enum):
    """Replay playback states."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    REWINDING = "rewinding"
    FAST_FORWARD = "fast_forward"


@dataclass
class ReplayControls:
    """Replay control settings."""
    playback_speed: float = 1.0  # 1.0 = normal speed, 0.5 = half speed, 2.0 = double speed
    auto_loop: bool = False
    show_frame_info: bool = True
    show_decision_overlay: bool = True
    comparison_mode: bool = False


@dataclass
class ReplayFrame:
    """Single frame of episode replay."""
    frame_index: int
    timestamp: float
    game_state: Dict[str, Any]
    action_taken: Optional[int]
    reward_received: float
    model_decision: Optional[Dict[str, Any]]
    frame_metadata: Dict[str, Any]


class EpisodeReplayManager:
    """
    Manages episode replay functionality for spectator interface.
    
    Provides controls for playing back recorded episodes with pause/rewind,
    side-by-side model comparison, and detailed frame-by-frame analysis.
    """
    
    def __init__(
        self,
        replay_manager: ReplayManager,
        spectator_manager: SpectatorManager
    ):
        self.replay_manager = replay_manager
        self.spectator_manager = spectator_manager
        
        # Active replay sessions
        self._active_replays: Dict[str, 'ReplaySession'] = {}
        
        # Replay event callbacks
        self._frame_callbacks: Dict[str, List[Callable[[str, ReplayFrame], Awaitable[None]]]] = {}
        self._state_callbacks: Dict[str, List[Callable[[str, ReplayState], Awaitable[None]]]] = {}
    
    async def start_episode_replay(
        self,
        session_id: str,
        episode_id: str,
        controls: Optional[ReplayControls] = None
    ) -> str:
        """
        Start replaying a specific episode.
        
        Args:
            session_id: Spectator session ID
            episode_id: ID of the episode to replay
            controls: Replay control settings
            
        Returns:
            Replay session ID
        """
        if controls is None:
            controls = ReplayControls()
        
        # Load episode data
        episode = await self._load_episode(episode_id)
        if not episode:
            raise ValueError(f"Episode {episode_id} not found")
        
        # Create replay session
        replay_session = ReplaySession(
            session_id=session_id,
            episode=episode,
            controls=controls,
            replay_manager=self
        )
        
        replay_id = f"replay_{session_id}_{episode_id}"
        self._active_replays[replay_id] = replay_session
        
        # Initialize callbacks for this replay
        self._frame_callbacks[replay_id] = []
        self._state_callbacks[replay_id] = []
        
        logger.info(f"Started episode replay {replay_id} for session {session_id}")
        return replay_id
    
    async def start_comparison_replay(
        self,
        session_id: str,
        episode_ids: List[str],
        controls: Optional[ReplayControls] = None
    ) -> str:
        """
        Start side-by-side comparison replay of multiple episodes.
        
        Args:
            session_id: Spectator session ID
            episode_ids: List of episode IDs to compare (max 4)
            controls: Replay control settings
            
        Returns:
            Comparison replay session ID
        """
        if len(episode_ids) > 4:
            raise ValueError("Maximum 4 episodes can be compared simultaneously")
        
        if controls is None:
            controls = ReplayControls(comparison_mode=True)
        else:
            controls.comparison_mode = True
        
        # Load all episodes
        episodes = []
        for episode_id in episode_ids:
            episode = await self._load_episode(episode_id)
            if not episode:
                raise ValueError(f"Episode {episode_id} not found")
            episodes.append(episode)
        
        # Create comparison replay session
        comparison_session = ComparisonReplaySession(
            session_id=session_id,
            episodes=episodes,
            controls=controls,
            replay_manager=self
        )
        
        replay_id = f"comparison_{session_id}_{'_'.join(episode_ids[:2])}"
        self._active_replays[replay_id] = comparison_session
        
        # Initialize callbacks
        self._frame_callbacks[replay_id] = []
        self._state_callbacks[replay_id] = []
        
        logger.info(f"Started comparison replay {replay_id} with {len(episodes)} episodes")
        return replay_id
    
    async def control_replay(
        self,
        replay_id: str,
        command: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send control command to replay session.
        
        Args:
            replay_id: Replay session ID
            command: Control command (play, pause, stop, seek, speed, etc.)
            parameters: Command parameters
            
        Returns:
            True if command was executed successfully
        """
        if replay_id not in self._active_replays:
            return False
        
        replay_session = self._active_replays[replay_id]
        return await replay_session.handle_command(command, parameters or {})
    
    async def get_replay_status(self, replay_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a replay session.
        
        Args:
            replay_id: Replay session ID
            
        Returns:
            Replay status information or None if not found
        """
        if replay_id not in self._active_replays:
            return None
        
        replay_session = self._active_replays[replay_id]
        return replay_session.get_status()
    
    async def stop_replay(self, replay_id: str) -> bool:
        """
        Stop and clean up a replay session.
        
        Args:
            replay_id: Replay session ID
            
        Returns:
            True if replay was stopped successfully
        """
        if replay_id not in self._active_replays:
            return False
        
        replay_session = self._active_replays[replay_id]
        await replay_session.stop()
        
        # Clean up
        del self._active_replays[replay_id]
        if replay_id in self._frame_callbacks:
            del self._frame_callbacks[replay_id]
        if replay_id in self._state_callbacks:
            del self._state_callbacks[replay_id]
        
        logger.info(f"Stopped replay session {replay_id}")
        return True
    
    def register_frame_callback(
        self,
        replay_id: str,
        callback: Callable[[str, ReplayFrame], Awaitable[None]]
    ) -> None:
        """
        Register callback for replay frame updates.
        
        Args:
            replay_id: Replay session ID
            callback: Async function to call with frame updates
        """
        if replay_id not in self._frame_callbacks:
            self._frame_callbacks[replay_id] = []
        
        self._frame_callbacks[replay_id].append(callback)
    
    def register_state_callback(
        self,
        replay_id: str,
        callback: Callable[[str, ReplayState], Awaitable[None]]
    ) -> None:
        """
        Register callback for replay state changes.
        
        Args:
            replay_id: Replay session ID
            callback: Async function to call with state changes
        """
        if replay_id not in self._state_callbacks:
            self._state_callbacks[replay_id] = []
        
        self._state_callbacks[replay_id].append(callback)
    
    async def _notify_frame_callbacks(self, replay_id: str, frame: ReplayFrame) -> None:
        """Notify all frame callbacks for a replay session."""
        if replay_id in self._frame_callbacks:
            for callback in self._frame_callbacks[replay_id]:
                try:
                    await callback(replay_id, frame)
                except Exception as e:
                    logger.error(f"Error in frame callback: {e}")
    
    async def _notify_state_callbacks(self, replay_id: str, state: ReplayState) -> None:
        """Notify all state callbacks for a replay session."""
        if replay_id in self._state_callbacks:
            for callback in self._state_callbacks[replay_id]:
                try:
                    await callback(replay_id, state)
                except Exception as e:
                    logger.error(f"Error in state callback: {e}")
    
    async def _load_episode(self, episode_id: str) -> Optional[GameEpisode]:
        """Load episode data from storage."""
        try:
            # Try to find episode in available sessions
            sessions = self.replay_manager.get_available_sessions()
            
            for session_info in sessions:
                session_id = session_info["session_id"]
                episodes = self.replay_manager.load_episodes_from_session(session_id)
                
                for episode in episodes:
                    if episode.episode_id == episode_id:
                        return episode
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading episode {episode_id}: {e}")
            return None
    
    async def cleanup(self) -> None:
        """Clean up all active replay sessions."""
        replay_ids = list(self._active_replays.keys())
        for replay_id in replay_ids:
            await self.stop_replay(replay_id)
        
        logger.info("EpisodeReplayManager cleanup completed")


class ReplaySession:
    """
    Single episode replay session.
    
    Manages playback of a single episode with full control capabilities.
    """
    
    def __init__(
        self,
        session_id: str,
        episode: GameEpisode,
        controls: ReplayControls,
        replay_manager: EpisodeReplayManager
    ):
        self.session_id = session_id
        self.episode = episode
        self.controls = controls
        self.replay_manager = replay_manager
        
        # Playback state
        self.current_frame = 0
        self.state = ReplayState.STOPPED
        self.playback_task: Optional[asyncio.Task] = None
        
        # Convert episode to frames
        self.frames = self._create_frames()
        
        # Timing
        self.frame_duration = 1.0 / 60.0  # 60 FPS base
        self.last_frame_time = 0.0
    
    def _create_frames(self) -> List[ReplayFrame]:
        """Convert episode data to replay frames."""
        frames = []
        
        for i in range(len(self.episode.states)):
            frame = ReplayFrame(
                frame_index=i,
                timestamp=i * self.frame_duration,
                game_state=self.episode.states[i] if i < len(self.episode.states) else {},
                action_taken=self.episode.actions[i] if i < len(self.episode.actions) else None,
                reward_received=self.episode.rewards[i] if i < len(self.episode.rewards) else 0.0,
                model_decision=None,  # Would be populated if available
                frame_metadata={
                    "episode_id": self.episode.episode_id,
                    "model_generation": self.episode.model_generation,
                    "opponent_generation": getattr(self.episode, 'opponent_generation', -1)
                }
            )
            frames.append(frame)
        
        return frames
    
    async def handle_command(self, command: str, parameters: Dict[str, Any]) -> bool:
        """Handle replay control command."""
        try:
            if command == "play":
                await self._play()
            elif command == "pause":
                await self._pause()
            elif command == "stop":
                await self._stop()
            elif command == "seek":
                frame_index = parameters.get("frame", 0)
                await self._seek(frame_index)
            elif command == "speed":
                speed = parameters.get("speed", 1.0)
                await self._set_speed(speed)
            elif command == "step":
                direction = parameters.get("direction", 1)  # 1 for forward, -1 for backward
                await self._step(direction)
            else:
                logger.warning(f"Unknown replay command: {command}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling replay command {command}: {e}")
            return False
    
    async def _play(self) -> None:
        """Start or resume playback."""
        if self.state == ReplayState.PLAYING:
            return
        
        self.state = ReplayState.PLAYING
        await self.replay_manager._notify_state_callbacks(
            f"replay_{self.session_id}_{self.episode.episode_id}",
            self.state
        )
        
        # Start playback task
        if self.playback_task:
            self.playback_task.cancel()
        
        self.playback_task = asyncio.create_task(self._playback_loop())
    
    async def _pause(self) -> None:
        """Pause playback."""
        if self.state != ReplayState.PLAYING:
            return
        
        self.state = ReplayState.PAUSED
        await self.replay_manager._notify_state_callbacks(
            f"replay_{self.session_id}_{self.episode.episode_id}",
            self.state
        )
        
        if self.playback_task:
            self.playback_task.cancel()
            self.playback_task = None
    
    async def _stop(self) -> None:
        """Stop playback and reset to beginning."""
        self.state = ReplayState.STOPPED
        self.current_frame = 0
        
        await self.replay_manager._notify_state_callbacks(
            f"replay_{self.session_id}_{self.episode.episode_id}",
            self.state
        )
        
        if self.playback_task:
            self.playback_task.cancel()
            self.playback_task = None
    
    async def _seek(self, frame_index: int) -> None:
        """Seek to specific frame."""
        frame_index = max(0, min(frame_index, len(self.frames) - 1))
        self.current_frame = frame_index
        
        # Send current frame
        if self.frames:
            await self.replay_manager._notify_frame_callbacks(
                f"replay_{self.session_id}_{self.episode.episode_id}",
                self.frames[self.current_frame]
            )
    
    async def _set_speed(self, speed: float) -> None:
        """Set playback speed."""
        self.controls.playback_speed = max(0.1, min(speed, 10.0))
    
    async def _step(self, direction: int) -> None:
        """Step one frame forward or backward."""
        new_frame = self.current_frame + direction
        new_frame = max(0, min(new_frame, len(self.frames) - 1))
        
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            
            if self.frames:
                await self.replay_manager._notify_frame_callbacks(
                    f"replay_{self.session_id}_{self.episode.episode_id}",
                    self.frames[self.current_frame]
                )
    
    async def _playback_loop(self) -> None:
        """Main playback loop."""
        try:
            while self.state == ReplayState.PLAYING and self.current_frame < len(self.frames):
                # Send current frame
                if self.frames:
                    await self.replay_manager._notify_frame_callbacks(
                        f"replay_{self.session_id}_{self.episode.episode_id}",
                        self.frames[self.current_frame]
                    )
                
                # Advance frame
                self.current_frame += 1
                
                # Check for end of episode
                if self.current_frame >= len(self.frames):
                    if self.controls.auto_loop:
                        self.current_frame = 0
                    else:
                        await self._stop()
                        break
                
                # Wait for next frame based on speed
                frame_delay = self.frame_duration / self.controls.playback_speed
                await asyncio.sleep(frame_delay)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in playback loop: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current replay status."""
        return {
            "episode_id": self.episode.episode_id,
            "state": self.state.value,
            "current_frame": self.current_frame,
            "total_frames": len(self.frames),
            "playback_speed": self.controls.playback_speed,
            "progress": self.current_frame / len(self.frames) if self.frames else 0.0,
            "episode_info": {
                "model_generation": self.episode.model_generation,
                "opponent_generation": getattr(self.episode, 'opponent_generation', -1),
                "total_reward": self.episode.total_reward,
                "game_result": self.episode.game_result,
                "episode_length": self.episode.episode_length
            }
        }
    
    async def stop(self) -> None:
        """Stop and clean up the replay session."""
        await self._stop()


class ComparisonReplaySession:
    """
    Side-by-side comparison replay session.
    
    Manages synchronized playback of multiple episodes for comparison.
    """
    
    def __init__(
        self,
        session_id: str,
        episodes: List[GameEpisode],
        controls: ReplayControls,
        replay_manager: EpisodeReplayManager
    ):
        self.session_id = session_id
        self.episodes = episodes
        self.controls = controls
        self.replay_manager = replay_manager
        
        # Create individual replay sessions
        self.replay_sessions = []
        for i, episode in enumerate(episodes):
            session = ReplaySession(
                session_id=f"{session_id}_comparison_{i}",
                episode=episode,
                controls=controls,
                replay_manager=replay_manager
            )
            self.replay_sessions.append(session)
        
        # Synchronized state
        self.state = ReplayState.STOPPED
        self.sync_frame = 0
        self.max_frames = max(len(session.frames) for session in self.replay_sessions)
    
    async def handle_command(self, command: str, parameters: Dict[str, Any]) -> bool:
        """Handle synchronized replay control command."""
        try:
            # Apply command to all replay sessions
            results = []
            for session in self.replay_sessions:
                result = await session.handle_command(command, parameters)
                results.append(result)
            
            # Update synchronized state
            if command in ["play", "pause", "stop"]:
                if command == "play":
                    self.state = ReplayState.PLAYING
                elif command == "pause":
                    self.state = ReplayState.PAUSED
                elif command == "stop":
                    self.state = ReplayState.STOPPED
                
                # Notify state change
                replay_id = f"comparison_{self.session_id}"
                await self.replay_manager._notify_state_callbacks(replay_id, self.state)
            
            return all(results)
            
        except Exception as e:
            logger.error(f"Error handling comparison replay command {command}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current comparison replay status."""
        episode_statuses = []
        for session in self.replay_sessions:
            episode_statuses.append(session.get_status())
        
        return {
            "comparison_mode": True,
            "state": self.state.value,
            "episode_count": len(self.episodes),
            "episodes": episode_statuses,
            "synchronized_frame": self.sync_frame,
            "max_frames": self.max_frames,
            "playback_speed": self.controls.playback_speed
        }
    
    async def stop(self) -> None:
        """Stop and clean up all comparison replay sessions."""
        for session in self.replay_sessions:
            await session.stop()
        
        self.state = ReplayState.STOPPED