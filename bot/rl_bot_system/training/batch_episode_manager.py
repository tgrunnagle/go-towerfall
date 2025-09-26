"""
Batch episode manager for parallel training episode execution.

This module provides the BatchEpisodeManager class for efficiently managing
multiple parallel training episodes with load balancing and resource optimization.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from game_client import GameClient
from rl_bot_system.training.training_session import EpisodeResult


class EpisodeStatus(Enum):
    """Status of individual episodes."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EpisodeBatch:
    """A batch of episodes to be executed in parallel."""
    batch_id: str
    episode_ids: List[str]
    room_code: str
    room_password: Optional[str]
    max_parallel: int
    timeout_seconds: int
    created_at: datetime
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class EpisodeTask:
    """Individual episode task information."""
    episode_id: str
    batch_id: str
    status: EpisodeStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[EpisodeResult] = None
    error: Optional[str] = None
    retry_count: int = 0


class BatchEpisodeManager:
    """
    Manages parallel execution of training episodes in batches.
    
    Provides functionality for:
    - Batch episode queuing and execution
    - Load balancing across parallel workers
    - Episode retry logic and error handling
    - Resource optimization and cleanup
    """

    def __init__(
        self,
        max_parallel_episodes: int = 8,
        max_retries: int = 3,
        episode_timeout: int = 300,
        game_server_url: str = "http://localhost:4000",
        ws_url: str = "ws://localhost:4000/ws"
    ):
        self.max_parallel_episodes = max_parallel_episodes
        self.max_retries = max_retries
        self.episode_timeout = episode_timeout
        self.game_server_url = game_server_url
        self.ws_url = ws_url
        
        # Episode management
        self.active_batches: Dict[str, EpisodeBatch] = {}
        self.episode_tasks: Dict[str, EpisodeTask] = {}
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        
        # Worker management
        self.active_workers: Dict[str, asyncio.Task] = {}
        self.worker_semaphore = asyncio.Semaphore(max_parallel_episodes)
        
        # Event handlers
        self.episode_complete_handlers: List[Callable[[EpisodeResult], Awaitable[None]]] = []
        self.batch_complete_handlers: List[Callable[[str, List[EpisodeResult]], Awaitable[None]]] = []
        self.episode_failed_handlers: List[Callable[[str, str], Awaitable[None]]] = []
        
        # Background tasks
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._worker_monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self._logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """Start the batch episode manager."""
        self._logger.info("Starting batch episode manager")
        
        # Start background tasks
        self._batch_processor_task = asyncio.create_task(self._process_batch_queue())
        self._worker_monitor_task = asyncio.create_task(self._monitor_workers())
        
        self._logger.info(f"Batch episode manager started with {self.max_parallel_episodes} max parallel episodes")

    async def stop(self) -> None:
        """Stop the batch episode manager and clean up."""
        self._logger.info("Stopping batch episode manager")
        self._shutdown_event.set()
        
        # Cancel all active workers
        for worker_id, worker_task in self.active_workers.items():
            if not worker_task.done():
                worker_task.cancel()
                self._logger.debug(f"Cancelled worker {worker_id}")
        
        # Wait for workers to complete
        if self.active_workers:
            await asyncio.gather(*self.active_workers.values(), return_exceptions=True)
        
        # Cancel background tasks
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
        if self._worker_monitor_task:
            self._worker_monitor_task.cancel()
        
        # Wait for background tasks
        tasks = [t for t in [self._batch_processor_task, self._worker_monitor_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self._logger.info("Batch episode manager stopped")

    async def submit_batch(
        self,
        batch_id: str,
        episode_ids: List[str],
        room_code: str,
        room_password: Optional[str] = None,
        max_parallel: Optional[int] = None,
        timeout_seconds: Optional[int] = None
    ) -> None:
        """
        Submit a batch of episodes for execution.
        
        Args:
            batch_id: Unique identifier for the batch
            episode_ids: List of episode IDs to execute
            room_code: Game room code for episodes
            room_password: Optional room password
            max_parallel: Maximum parallel episodes for this batch
            timeout_seconds: Timeout for individual episodes
        """
        batch = EpisodeBatch(
            batch_id=batch_id,
            episode_ids=episode_ids,
            room_code=room_code,
            room_password=room_password,
            max_parallel=max_parallel or self.max_parallel_episodes,
            timeout_seconds=timeout_seconds or self.episode_timeout,
            created_at=datetime.now()
        )
        
        # Create episode tasks
        for episode_id in episode_ids:
            self.episode_tasks[episode_id] = EpisodeTask(
                episode_id=episode_id,
                batch_id=batch_id,
                status=EpisodeStatus.QUEUED
            )
        
        # Add batch to queue
        await self.batch_queue.put(batch)
        self.active_batches[batch_id] = batch
        
        self._logger.info(f"Submitted batch {batch_id} with {len(episode_ids)} episodes")

    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a batch."""
        batch = self.active_batches.get(batch_id)
        if not batch:
            return None
        
        episode_statuses = {}
        for episode_id in batch.episode_ids:
            task = self.episode_tasks.get(episode_id)
            if task:
                episode_statuses[episode_id] = {
                    "status": task.status.value,
                    "startTime": task.start_time.isoformat() if task.start_time else None,
                    "endTime": task.end_time.isoformat() if task.end_time else None,
                    "retryCount": task.retry_count,
                    "error": task.error
                }
        
        # Calculate batch statistics
        completed = sum(1 for task in self.episode_tasks.values() 
                       if task.batch_id == batch_id and task.status == EpisodeStatus.COMPLETED)
        failed = sum(1 for task in self.episode_tasks.values() 
                    if task.batch_id == batch_id and task.status == EpisodeStatus.FAILED)
        running = sum(1 for task in self.episode_tasks.values() 
                     if task.batch_id == batch_id and task.status == EpisodeStatus.RUNNING)
        
        return {
            "batchId": batch_id,
            "totalEpisodes": len(batch.episode_ids),
            "completed": completed,
            "failed": failed,
            "running": running,
            "queued": len(batch.episode_ids) - completed - failed - running,
            "createdAt": batch.created_at.isoformat(),
            "episodes": episode_statuses
        }

    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch and all its episodes."""
        batch = self.active_batches.get(batch_id)
        if not batch:
            return False
        
        # Cancel all episodes in the batch
        cancelled_count = 0
        for episode_id in batch.episode_ids:
            task = self.episode_tasks.get(episode_id)
            if task and task.status in [EpisodeStatus.QUEUED, EpisodeStatus.RUNNING]:
                task.status = EpisodeStatus.CANCELLED
                cancelled_count += 1
        
        # Remove from active batches
        del self.active_batches[batch_id]
        
        self._logger.info(f"Cancelled batch {batch_id}, cancelled {cancelled_count} episodes")
        return True

    def register_episode_complete_handler(
        self,
        handler: Callable[[EpisodeResult], Awaitable[None]]
    ) -> None:
        """Register a handler for episode completion events."""
        self.episode_complete_handlers.append(handler)

    def register_batch_complete_handler(
        self,
        handler: Callable[[str, List[EpisodeResult]], Awaitable[None]]
    ) -> None:
        """Register a handler for batch completion events."""
        self.batch_complete_handlers.append(handler)

    def register_episode_failed_handler(
        self,
        handler: Callable[[str, str], Awaitable[None]]
    ) -> None:
        """Register a handler for episode failure events."""
        self.episode_failed_handlers.append(handler)

    async def _process_batch_queue(self) -> None:
        """Process batches from the queue."""
        while not self._shutdown_event.is_set():
            try:
                # Get next batch (with timeout to allow shutdown)
                try:
                    batch = await asyncio.wait_for(
                        self.batch_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the batch
                await self._process_batch(batch)
                
            except Exception as e:
                self._logger.error(f"Error processing batch queue: {e}")
                await asyncio.sleep(1.0)

    async def _process_batch(self, batch: EpisodeBatch) -> None:
        """Process a single batch of episodes."""
        self._logger.info(f"Processing batch {batch.batch_id} with {len(batch.episode_ids)} episodes")
        
        # Create workers for the batch episodes
        batch_workers = []
        episodes_per_worker = max(1, len(batch.episode_ids) // batch.max_parallel)
        
        for i in range(0, len(batch.episode_ids), episodes_per_worker):
            episode_chunk = batch.episode_ids[i:i + episodes_per_worker]
            worker_id = f"{batch.batch_id}_worker_{i // episodes_per_worker}"
            
            worker_task = asyncio.create_task(
                self._run_episode_worker(worker_id, episode_chunk, batch)
            )
            batch_workers.append(worker_task)
            self.active_workers[worker_id] = worker_task
        
        # Wait for all workers to complete
        try:
            await asyncio.gather(*batch_workers, return_exceptions=True)
        finally:
            # Clean up workers
            for worker_task in batch_workers:
                worker_id = None
                for wid, wtask in self.active_workers.items():
                    if wtask == worker_task:
                        worker_id = wid
                        break
                if worker_id:
                    del self.active_workers[worker_id]
        
        # Check if batch is complete
        await self._check_batch_completion(batch.batch_id)

    async def _run_episode_worker(
        self,
        worker_id: str,
        episode_ids: List[str],
        batch: EpisodeBatch
    ) -> None:
        """Run episodes assigned to a worker."""
        self._logger.debug(f"Worker {worker_id} starting with {len(episode_ids)} episodes")
        
        for episode_id in episode_ids:
            if self._shutdown_event.is_set():
                break
            
            task = self.episode_tasks.get(episode_id)
            if not task or task.status != EpisodeStatus.QUEUED:
                continue
            
            # Acquire semaphore for resource control
            async with self.worker_semaphore:
                await self._run_single_episode(task, batch)
        
        self._logger.debug(f"Worker {worker_id} completed")

    async def _run_single_episode(self, task: EpisodeTask, batch: EpisodeBatch) -> None:
        """Run a single episode with retry logic."""
        while task.retry_count <= self.max_retries and task.status != EpisodeStatus.CANCELLED:
            try:
                task.status = EpisodeStatus.RUNNING
                task.start_time = datetime.now()
                
                # Create game client
                game_client = GameClient(self.ws_url, self.game_server_url)
                
                # Run episode with timeout
                result = await asyncio.wait_for(
                    self._execute_episode(game_client, task.episode_id, batch),
                    timeout=batch.timeout_seconds
                )
                
                # Episode completed successfully
                task.status = EpisodeStatus.COMPLETED
                task.end_time = datetime.now()
                task.result = result
                
                # Notify handlers
                for handler in self.episode_complete_handlers:
                    try:
                        await handler(result)
                    except Exception as e:
                        self._logger.error(f"Error in episode complete handler: {e}")
                
                self._logger.debug(f"Episode {task.episode_id} completed successfully")
                break
                
            except asyncio.TimeoutError:
                task.retry_count += 1
                task.error = f"Episode timed out after {batch.timeout_seconds} seconds"
                self._logger.warning(f"Episode {task.episode_id} timed out (attempt {task.retry_count})")
                
            except Exception as e:
                task.retry_count += 1
                task.error = str(e)
                self._logger.warning(f"Episode {task.episode_id} failed: {e} (attempt {task.retry_count})")
            
            # If we've exhausted retries, mark as failed
            if task.retry_count > self.max_retries:
                task.status = EpisodeStatus.FAILED
                task.end_time = datetime.now()
                
                # Notify failure handlers
                for handler in self.episode_failed_handlers:
                    try:
                        await handler(task.episode_id, task.error or "Unknown error")
                    except Exception as e:
                        self._logger.error(f"Error in episode failed handler: {e}")
                
                self._logger.error(f"Episode {task.episode_id} failed after {self.max_retries} retries")
                break
            
            # Wait before retry
            if task.retry_count <= self.max_retries:
                await asyncio.sleep(min(2 ** task.retry_count, 10))  # Exponential backoff

    async def _execute_episode(
        self,
        game_client: GameClient,
        episode_id: str,
        batch: EpisodeBatch
    ) -> EpisodeResult:
        """Execute a single episode."""
        start_time = datetime.now()
        
        try:
            # Connect to game room
            await game_client.connect(
                batch.room_code,
                f"batch_bot_{episode_id}",
                batch.room_password
            )
            
            # Simulate episode execution (placeholder)
            # This will be replaced by actual RL training logic
            await asyncio.sleep(0.1)  # Simulate episode duration
            
            # Clean up
            await game_client.exit_game()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return EpisodeResult(
                episode_id=episode_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                total_reward=10.0,  # Placeholder
                episode_length=100,  # Placeholder
                game_result="win",   # Placeholder
                final_state={"health": 100, "score": 150},
                error=None
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return EpisodeResult(
                episode_id=episode_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                total_reward=0.0,
                episode_length=0,
                game_result="error",
                final_state={},
                error=str(e)
            )

    async def _check_batch_completion(self, batch_id: str) -> None:
        """Check if a batch is complete and notify handlers."""
        batch = self.active_batches.get(batch_id)
        if not batch:
            return
        
        # Check if all episodes are complete or failed
        all_done = True
        results = []
        
        for episode_id in batch.episode_ids:
            task = self.episode_tasks.get(episode_id)
            if not task or task.status not in [EpisodeStatus.COMPLETED, EpisodeStatus.FAILED, EpisodeStatus.CANCELLED]:
                all_done = False
                break
            
            if task.result:
                results.append(task.result)
        
        if all_done:
            # Batch is complete
            del self.active_batches[batch_id]
            
            # Clean up episode tasks
            for episode_id in batch.episode_ids:
                if episode_id in self.episode_tasks:
                    del self.episode_tasks[episode_id]
            
            # Notify batch completion handlers
            for handler in self.batch_complete_handlers:
                try:
                    await handler(batch_id, results)
                except Exception as e:
                    self._logger.error(f"Error in batch complete handler: {e}")
            
            self._logger.info(f"Batch {batch_id} completed with {len(results)} successful episodes")

    async def _monitor_workers(self) -> None:
        """Monitor worker health and performance."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up completed workers
                completed_workers = []
                for worker_id, worker_task in self.active_workers.items():
                    if worker_task.done():
                        completed_workers.append(worker_id)
                
                for worker_id in completed_workers:
                    del self.active_workers[worker_id]
                    self._logger.debug(f"Cleaned up completed worker {worker_id}")
                
                # Log worker statistics
                if len(self.active_workers) > 0:
                    self._logger.debug(f"Active workers: {len(self.active_workers)}")
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self._logger.error(f"Error monitoring workers: {e}")
                await asyncio.sleep(10.0)