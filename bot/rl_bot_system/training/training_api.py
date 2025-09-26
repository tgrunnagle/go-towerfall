"""
Training API for managing training sessions and room requests.

This module provides HTTP API endpoints for creating training rooms,
managing training sessions, and monitoring training progress.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from rl_bot_system.training.session_manager import SessionManager, ResourceLimits
from rl_bot_system.training.training_session import TrainingConfig, TrainingMode
from rl_bot_system.training.batch_episode_manager import BatchEpisodeManager


class TrainingAPI:
    """
    HTTP API for training session management.
    
    Provides endpoints for:
    - Creating and managing training sessions
    - Requesting training rooms with speed multipliers
    - Monitoring training progress and metrics
    - Managing batch episode execution
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        batch_manager: Optional[BatchEpisodeManager] = None,
        game_server_url: str = "http://localhost:4000",
        ws_url: str = "ws://localhost:4000/ws"
    ):
        self.session_manager = session_manager or SessionManager(
            game_server_url=game_server_url,
            ws_url=ws_url
        )
        self.batch_manager = batch_manager or BatchEpisodeManager(
            game_server_url=game_server_url,
            ws_url=ws_url
        )
        
        self._logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """Start the training API and underlying managers."""
        await self.session_manager.start()
        await self.batch_manager.start()
        self._logger.info("Training API started")

    async def stop(self) -> None:
        """Stop the training API and clean up resources."""
        await self.session_manager.stop()
        await self.batch_manager.stop()
        self._logger.info("Training API stopped")

    # Session Management Endpoints

    async def create_training_session(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new training session.
        
        Request format:
        {
            "config": {
                "speedMultiplier": 10.0,
                "headlessMode": true,
                "maxEpisodes": 1000,
                "parallelEpisodes": 4,
                "trainingMode": "training",
                "spectatorEnabled": false
            },
            "priority": 1,
            "sessionId": "optional_custom_id"
        }
        """
        try:
            # Parse configuration
            config_data = request_data.get("config", {})
            config = TrainingConfig(
                speed_multiplier=config_data.get("speedMultiplier", 1.0),
                headless_mode=config_data.get("headlessMode", False),
                max_episodes=config_data.get("maxEpisodes", 1000),
                episode_timeout=config_data.get("episodeTimeout", 300),
                parallel_episodes=config_data.get("parallelEpisodes", 1),
                training_mode=TrainingMode(config_data.get("trainingMode", "realtime")),
                room_password=config_data.get("roomPassword"),
                spectator_enabled=config_data.get("spectatorEnabled", False),
                auto_cleanup=config_data.get("autoCleanup", True)
            )
            
            # Create session
            session_id = await self.session_manager.create_session(
                config=config,
                priority=request_data.get("priority", 0),
                session_id=request_data.get("sessionId")
            )
            
            return {
                "success": True,
                "sessionId": session_id,
                "message": f"Training session {session_id} created and queued"
            }
            
        except Exception as e:
            self._logger.error(f"Failed to create training session: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to create training session"
            }

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a specific training session."""
        try:
            session = await self.session_manager.get_session(session_id)
            if not session:
                return {
                    "success": False,
                    "error": "Session not found",
                    "message": f"Training session {session_id} not found"
                }
            
            session_info = await session.get_session_info()
            return {
                "success": True,
                "session": session_info
            }
            
        except Exception as e:
            self._logger.error(f"Failed to get session info: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve session information"
            }

    async def list_all_sessions(self) -> Dict[str, Any]:
        """List all training sessions and their status."""
        try:
            sessions_info = await self.session_manager.get_all_sessions_info()
            global_metrics = await self.session_manager.get_global_metrics()
            
            return {
                "success": True,
                "sessions": sessions_info,
                "globalMetrics": global_metrics
            }
            
        except Exception as e:
            self._logger.error(f"Failed to list sessions: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve sessions list"
            }

    async def pause_session(self, session_id: str) -> Dict[str, Any]:
        """Pause a training session."""
        try:
            success = await self.session_manager.pause_session(session_id)
            if success:
                return {
                    "success": True,
                    "message": f"Session {session_id} paused"
                }
            else:
                return {
                    "success": False,
                    "error": "Session not found",
                    "message": f"Session {session_id} not found or cannot be paused"
                }
                
        except Exception as e:
            self._logger.error(f"Failed to pause session: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to pause session"
            }

    async def resume_session(self, session_id: str) -> Dict[str, Any]:
        """Resume a paused training session."""
        try:
            success = await self.session_manager.resume_session(session_id)
            if success:
                return {
                    "success": True,
                    "message": f"Session {session_id} resumed"
                }
            else:
                return {
                    "success": False,
                    "error": "Session not found",
                    "message": f"Session {session_id} not found or cannot be resumed"
                }
                
        except Exception as e:
            self._logger.error(f"Failed to resume session: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to resume session"
            }

    async def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Stop a training session."""
        try:
            success = await self.session_manager.stop_session(session_id)
            if success:
                return {
                    "success": True,
                    "message": f"Session {session_id} stopped"
                }
            else:
                return {
                    "success": False,
                    "error": "Session not found",
                    "message": f"Session {session_id} not found"
                }
                
        except Exception as e:
            self._logger.error(f"Failed to stop session: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to stop session"
            }

    # Training Room Management Endpoints

    async def request_training_room(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request a training room with specified configuration.
        
        Request format:
        {
            "speedMultiplier": 10.0,
            "headlessMode": true,
            "maxPlayers": 8,
            "password": "optional_password",
            "spectatorEnabled": true
        }
        """
        try:
            # This would typically interface with the game server
            # For now, we'll return a mock response
            room_config = {
                "roomType": "training",
                "speedMultiplier": request_data.get("speedMultiplier", 1.0),
                "headlessMode": request_data.get("headlessMode", False),
                "maxPlayers": request_data.get("maxPlayers", 8),
                "password": request_data.get("password"),
                "spectatorEnabled": request_data.get("spectatorEnabled", False)
            }
            
            # Generate mock room data
            room_code = f"TR{datetime.now().strftime('%H%M%S')}"
            room_id = f"room_{room_code.lower()}"
            
            return {
                "success": True,
                "roomCode": room_code,
                "roomId": room_id,
                "config": room_config,
                "message": f"Training room {room_code} created"
            }
            
        except Exception as e:
            self._logger.error(f"Failed to create training room: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to create training room"
            }

    async def get_training_room_info(self, room_code: str) -> Dict[str, Any]:
        """Get information about a training room."""
        try:
            # This would query the game server for room information
            # For now, return mock data
            return {
                "success": True,
                "roomCode": room_code,
                "status": "active",
                "playerCount": 2,
                "maxPlayers": 8,
                "speedMultiplier": 10.0,
                "headlessMode": True,
                "spectatorEnabled": True
            }
            
        except Exception as e:
            self._logger.error(f"Failed to get room info: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve room information"
            }

    # Batch Episode Management Endpoints

    async def submit_episode_batch(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a batch of episodes for execution.
        
        Request format:
        {
            "batchId": "batch_001",
            "episodeIds": ["ep_001", "ep_002", "ep_003"],
            "roomCode": "TR123456",
            "roomPassword": "optional_password",
            "maxParallel": 4,
            "timeoutSeconds": 300
        }
        """
        try:
            await self.batch_manager.submit_batch(
                batch_id=request_data["batchId"],
                episode_ids=request_data["episodeIds"],
                room_code=request_data["roomCode"],
                room_password=request_data.get("roomPassword"),
                max_parallel=request_data.get("maxParallel"),
                timeout_seconds=request_data.get("timeoutSeconds")
            )
            
            return {
                "success": True,
                "batchId": request_data["batchId"],
                "episodeCount": len(request_data["episodeIds"]),
                "message": f"Batch {request_data['batchId']} submitted for execution"
            }
            
        except Exception as e:
            self._logger.error(f"Failed to submit episode batch: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to submit episode batch"
            }

    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get status of an episode batch."""
        try:
            status = await self.batch_manager.get_batch_status(batch_id)
            if status:
                return {
                    "success": True,
                    "batch": status
                }
            else:
                return {
                    "success": False,
                    "error": "Batch not found",
                    "message": f"Batch {batch_id} not found"
                }
                
        except Exception as e:
            self._logger.error(f"Failed to get batch status: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve batch status"
            }

    async def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """Cancel an episode batch."""
        try:
            success = await self.batch_manager.cancel_batch(batch_id)
            if success:
                return {
                    "success": True,
                    "message": f"Batch {batch_id} cancelled"
                }
            else:
                return {
                    "success": False,
                    "error": "Batch not found",
                    "message": f"Batch {batch_id} not found"
                }
                
        except Exception as e:
            self._logger.error(f"Failed to cancel batch: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to cancel batch"
            }

    # System Status and Monitoring Endpoints

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and resource utilization."""
        try:
            sessions_info = await self.session_manager.get_all_sessions_info()
            global_metrics = await self.session_manager.get_global_metrics()
            
            return {
                "success": True,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "sessions": {
                    "active": len(sessions_info["activeSessions"]),
                    "completed": len(sessions_info["completedSessions"]),
                    "queued": sessions_info["queuedSessions"]
                },
                "resources": sessions_info["resourceStatus"],
                "metrics": global_metrics
            }
            
        except Exception as e:
            self._logger.error(f"Failed to get system status: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve system status"
            }

    async def get_training_metrics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get training metrics for a specific session or all sessions."""
        try:
            if session_id:
                session = await self.session_manager.get_session(session_id)
                if not session:
                    return {
                        "success": False,
                        "error": "Session not found",
                        "message": f"Session {session_id} not found"
                    }
                
                session_info = await session.get_session_info()
                return {
                    "success": True,
                    "sessionId": session_id,
                    "metrics": session_info["metrics"]
                }
            else:
                global_metrics = await self.session_manager.get_global_metrics()
                return {
                    "success": True,
                    "globalMetrics": global_metrics
                }
                
        except Exception as e:
            self._logger.error(f"Failed to get training metrics: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve training metrics"
            }

    # Configuration Management Endpoints

    async def get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        default_config = TrainingConfig()
        return {
            "success": True,
            "config": {
                "speedMultiplier": default_config.speed_multiplier,
                "headlessMode": default_config.headless_mode,
                "maxEpisodes": default_config.max_episodes,
                "episodeTimeout": default_config.episode_timeout,
                "parallelEpisodes": default_config.parallel_episodes,
                "trainingMode": default_config.training_mode.value,
                "spectatorEnabled": default_config.spectator_enabled,
                "autoCleanup": default_config.auto_cleanup
            }
        }

    async def validate_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a training configuration."""
        try:
            # Validate configuration parameters
            errors = []
            
            speed_multiplier = config_data.get("speedMultiplier", 1.0)
            if not isinstance(speed_multiplier, (int, float)) or speed_multiplier <= 0 or speed_multiplier > 100:
                errors.append("speedMultiplier must be a number between 0.1 and 100")
            
            max_episodes = config_data.get("maxEpisodes", 1000)
            if not isinstance(max_episodes, int) or max_episodes <= 0:
                errors.append("maxEpisodes must be a positive integer")
            
            parallel_episodes = config_data.get("parallelEpisodes", 1)
            if not isinstance(parallel_episodes, int) or parallel_episodes <= 0 or parallel_episodes > 16:
                errors.append("parallelEpisodes must be an integer between 1 and 16")
            
            training_mode = config_data.get("trainingMode", "realtime")
            valid_modes = [mode.value for mode in TrainingMode]
            if training_mode not in valid_modes:
                errors.append(f"trainingMode must be one of: {valid_modes}")
            
            if errors:
                return {
                    "success": True,  # Changed to True - validation completed successfully
                    "valid": False,
                    "errors": errors
                }
            else:
                return {
                    "success": True,
                    "valid": True,
                    "message": "Configuration is valid"
                }
                
        except Exception as e:
            self._logger.error(f"Failed to validate config: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to validate configuration"
            }