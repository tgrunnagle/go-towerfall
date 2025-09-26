"""
FastAPI endpoints for episode replay functionality.

This module provides REST API endpoints for managing episode replay sessions,
including single episode replay and side-by-side model comparison.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from rl_bot_system.spectator.episode_replay import (
    EpisodeReplayManager,
    ReplayControls,
    ReplayState
)
from rl_bot_system.spectator.spectator_manager import SpectatorManager
from rl_bot_system.replay.replay_manager import ReplayManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/replay", tags=["replay"])

# Global instances (would be injected in production)
replay_manager: Optional[ReplayManager] = None
spectator_manager: Optional[SpectatorManager] = None
episode_replay_manager: Optional[EpisodeReplayManager] = None


# Request/Response Models
class ReplayControlsModel(BaseModel):
    """Replay control settings."""
    playback_speed: float = Field(default=1.0, ge=0.1, le=10.0)
    auto_loop: bool = False
    show_frame_info: bool = True
    show_decision_overlay: bool = True
    comparison_mode: bool = False


class StartReplayRequest(BaseModel):
    """Request to start episode replay."""
    session_id: str
    episode_id: str
    controls: Optional[ReplayControlsModel] = None


class StartComparisonRequest(BaseModel):
    """Request to start comparison replay."""
    session_id: str
    episode_ids: List[str] = Field(..., min_items=2, max_items=4)
    controls: Optional[ReplayControlsModel] = None


class ReplayCommandRequest(BaseModel):
    """Request to send replay control command."""
    command: str
    parameters: Optional[Dict[str, Any]] = None


class ReplayStatusResponse(BaseModel):
    """Replay status response."""
    replay_id: str
    status: Dict[str, Any]
    timestamp: datetime


class EpisodeListResponse(BaseModel):
    """Response with list of available episodes."""
    episodes: List[Dict[str, Any]]
    total_count: int
    session_info: Dict[str, Any]


# Dependency injection
def get_replay_manager() -> ReplayManager:
    """Get replay manager instance."""
    if replay_manager is None:
        raise HTTPException(status_code=500, detail="Replay manager not initialized")
    return replay_manager


def get_spectator_manager() -> SpectatorManager:
    """Get spectator manager instance."""
    if spectator_manager is None:
        raise HTTPException(status_code=500, detail="Spectator manager not initialized")
    return spectator_manager


def get_episode_replay_manager() -> EpisodeReplayManager:
    """Get episode replay manager instance."""
    if episode_replay_manager is None:
        raise HTTPException(status_code=500, detail="Episode replay manager not initialized")
    return episode_replay_manager


# API Endpoints
@router.get("/sessions", response_model=List[Dict[str, Any]])
async def list_replay_sessions(
    rm: ReplayManager = Depends(get_replay_manager)
) -> List[Dict[str, Any]]:
    """
    List all available replay sessions.
    
    Returns:
        List of replay session information
    """
    try:
        sessions = rm.get_available_sessions()
        return sessions
    except Exception as e:
        logger.error(f"Error listing replay sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/episodes", response_model=EpisodeListResponse)
async def list_session_episodes(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    rm: ReplayManager = Depends(get_replay_manager)
) -> EpisodeListResponse:
    """
    List episodes from a specific replay session.
    
    Args:
        session_id: ID of the replay session
        limit: Maximum number of episodes to return
        offset: Number of episodes to skip
        
    Returns:
        List of episodes with metadata
    """
    try:
        # Load episodes from session
        episodes = rm.load_episodes_from_session(session_id)
        
        # Apply pagination
        total_count = len(episodes)
        paginated_episodes = episodes[offset:offset + limit]
        
        # Convert to response format
        episode_data = []
        for episode in paginated_episodes:
            episode_info = {
                "episode_id": episode.episode_id,
                "model_generation": episode.model_generation,
                "opponent_generation": getattr(episode, 'opponent_generation', -1),
                "total_reward": episode.total_reward,
                "episode_length": episode.episode_length,
                "game_result": episode.game_result,
                "episode_metrics": getattr(episode, 'episode_metrics', {})
            }
            episode_data.append(episode_info)
        
        # Get session info
        sessions = rm.get_available_sessions()
        session_info = next((s for s in sessions if s["session_id"] == session_id), {})
        
        return EpisodeListResponse(
            episodes=episode_data,
            total_count=total_count,
            session_info=session_info
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing episodes for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=Dict[str, str])
async def start_episode_replay(
    request: StartReplayRequest,
    erm: EpisodeReplayManager = Depends(get_episode_replay_manager)
) -> Dict[str, str]:
    """
    Start replaying a specific episode.
    
    Args:
        request: Replay start request
        
    Returns:
        Replay session information
    """
    try:
        # Convert controls if provided
        controls = None
        if request.controls:
            controls = ReplayControls(
                playback_speed=request.controls.playback_speed,
                auto_loop=request.controls.auto_loop,
                show_frame_info=request.controls.show_frame_info,
                show_decision_overlay=request.controls.show_decision_overlay,
                comparison_mode=request.controls.comparison_mode
            )
        
        # Start replay
        replay_id = await erm.start_episode_replay(
            session_id=request.session_id,
            episode_id=request.episode_id,
            controls=controls
        )
        
        return {
            "replay_id": replay_id,
            "status": "started",
            "message": f"Episode replay started for episode {request.episode_id}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting episode replay: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-comparison", response_model=Dict[str, str])
async def start_comparison_replay(
    request: StartComparisonRequest,
    erm: EpisodeReplayManager = Depends(get_episode_replay_manager)
) -> Dict[str, str]:
    """
    Start side-by-side comparison replay of multiple episodes.
    
    Args:
        request: Comparison replay start request
        
    Returns:
        Comparison replay session information
    """
    try:
        # Convert controls if provided
        controls = None
        if request.controls:
            controls = ReplayControls(
                playback_speed=request.controls.playback_speed,
                auto_loop=request.controls.auto_loop,
                show_frame_info=request.controls.show_frame_info,
                show_decision_overlay=request.controls.show_decision_overlay,
                comparison_mode=True  # Always true for comparison
            )
        
        # Start comparison replay
        replay_id = await erm.start_comparison_replay(
            session_id=request.session_id,
            episode_ids=request.episode_ids,
            controls=controls
        )
        
        return {
            "replay_id": replay_id,
            "status": "started",
            "message": f"Comparison replay started for {len(request.episode_ids)} episodes"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting comparison replay: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/control/{replay_id}", response_model=Dict[str, Any])
async def control_replay(
    replay_id: str,
    request: ReplayCommandRequest,
    erm: EpisodeReplayManager = Depends(get_episode_replay_manager)
) -> Dict[str, Any]:
    """
    Send control command to replay session.
    
    Args:
        replay_id: Replay session ID
        request: Control command request
        
    Returns:
        Command execution result
    """
    try:
        success = await erm.control_replay(
            replay_id=replay_id,
            command=request.command,
            parameters=request.parameters
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Replay session not found")
        
        # Get updated status
        status = await erm.get_replay_status(replay_id)
        
        return {
            "success": True,
            "command": request.command,
            "replay_status": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error controlling replay {replay_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{replay_id}", response_model=ReplayStatusResponse)
async def get_replay_status(
    replay_id: str,
    erm: EpisodeReplayManager = Depends(get_episode_replay_manager)
) -> ReplayStatusResponse:
    """
    Get current status of a replay session.
    
    Args:
        replay_id: Replay session ID
        
    Returns:
        Current replay status
    """
    try:
        status = await erm.get_replay_status(replay_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Replay session not found")
        
        return ReplayStatusResponse(
            replay_id=replay_id,
            status=status,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting replay status for {replay_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/stop/{replay_id}", response_model=Dict[str, str])
async def stop_replay(
    replay_id: str,
    erm: EpisodeReplayManager = Depends(get_episode_replay_manager)
) -> Dict[str, str]:
    """
    Stop and clean up a replay session.
    
    Args:
        replay_id: Replay session ID
        
    Returns:
        Stop confirmation
    """
    try:
        success = await erm.stop_replay(replay_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Replay session not found")
        
        return {
            "replay_id": replay_id,
            "status": "stopped",
            "message": "Replay session stopped successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping replay {replay_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes/{episode_id}/analysis", response_model=Dict[str, Any])
async def get_episode_analysis(
    episode_id: str,
    rm: ReplayManager = Depends(get_replay_manager)
) -> Dict[str, Any]:
    """
    Get detailed analysis of a specific episode.
    
    Args:
        episode_id: ID of the episode to analyze
        
    Returns:
        Episode analysis results
    """
    try:
        # Find and load the episode
        sessions = rm.get_available_sessions()
        episode = None
        
        for session_info in sessions:
            session_id = session_info["session_id"]
            episodes = rm.load_episodes_from_session(session_id)
            
            for ep in episodes:
                if ep.episode_id == episode_id:
                    episode = ep
                    break
            
            if episode:
                break
        
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")
        
        # Analyze the episode
        analysis = rm.analyze_episodes([episode])
        
        return {
            "episode_id": episode_id,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing episode {episode_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-episodes", response_model=Dict[str, Any])
async def compare_episodes(
    episode_ids: List[str],
    rm: ReplayManager = Depends(get_replay_manager)
) -> Dict[str, Any]:
    """
    Compare multiple episodes and their behavior patterns.
    
    Args:
        episode_ids: List of episode IDs to compare
        
    Returns:
        Comparative analysis results
    """
    try:
        if len(episode_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 episodes required for comparison")
        
        if len(episode_ids) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 episodes can be compared")
        
        # Load all episodes
        sessions = rm.get_available_sessions()
        episodes = []
        episodes_by_generation = {}
        
        for episode_id in episode_ids:
            episode = None
            
            for session_info in sessions:
                session_id = session_info["session_id"]
                session_episodes = rm.load_episodes_from_session(session_id)
                
                for ep in session_episodes:
                    if ep.episode_id == episode_id:
                        episode = ep
                        break
                
                if episode:
                    break
            
            if not episode:
                raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
            
            episodes.append(episode)
            
            # Group by generation for comparison
            gen = episode.model_generation
            if gen not in episodes_by_generation:
                episodes_by_generation[gen] = []
            episodes_by_generation[gen].append(episode)
        
        # Perform comparison analysis
        if len(episodes_by_generation) > 1:
            # Multi-generation comparison
            comparison = rm.compare_generations(episodes_by_generation)
        else:
            # Single generation analysis
            comparison = rm.analyze_episodes(episodes)
        
        return {
            "episode_ids": episode_ids,
            "comparison_type": "multi_generation" if len(episodes_by_generation) > 1 else "single_generation",
            "analysis": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing episodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Initialization function
def initialize_replay_api(
    rm: ReplayManager,
    sm: SpectatorManager,
    erm: EpisodeReplayManager
) -> None:
    """
    Initialize the replay API with manager instances.
    
    Args:
        rm: ReplayManager instance
        sm: SpectatorManager instance
        erm: EpisodeReplayManager instance
    """
    global replay_manager, spectator_manager, episode_replay_manager
    
    replay_manager = rm
    spectator_manager = sm
    episode_replay_manager = erm
    
    logger.info("Replay API initialized successfully")