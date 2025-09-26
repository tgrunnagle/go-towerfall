"""
Unit tests for BatchEpisodeManager class.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from rl_bot_system.training.batch_episode_manager import (
    BatchEpisodeManager,
    EpisodeBatch,
    EpisodeTask,
    EpisodeStatus
)
from rl_bot_system.training.training_session import EpisodeResult


class TestBatchEpisodeManager:
    """Test cases for BatchEpisodeManager class."""

    @pytest.fixture
    def batch_manager(self):
        """Create a test batch episode manager."""
        return BatchEpisodeManager(
            max_parallel_episodes=4,
            max_retries=2,
            episode_timeout=60,
            game_server_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws"
        )

    def test_batch_manager_initialization(self, batch_manager):
        """Test batch manager initialization."""
        assert batch_manager.max_parallel_episodes == 4
        assert batch_manager.max_retries == 2
        assert batch_manager.episode_timeout == 60
        assert len(batch_manager.active_batches) == 0
        assert len(batch_manager.episode_tasks) == 0
        assert batch_manager.worker_semaphore._value == 4

    def test_episode_batch_creation(self):
        """Test EpisodeBatch data structure."""
        episode_ids = ["ep_001", "ep_002", "ep_003"]
        batch = EpisodeBatch(
            batch_id="test_batch",
            episode_ids=episode_ids,
            room_code="TR123456",
            room_password="secret",
            max_parallel=2,
            timeout_seconds=120,
            created_at=datetime.now()
        )
        
        assert batch.batch_id == "test_batch"
        assert batch.episode_ids == episode_ids
        assert batch.room_code == "TR123456"
        assert batch.room_password == "secret"
        assert batch.max_parallel == 2
        assert batch.timeout_seconds == 120

    def test_episode_task_creation(self):
        """Test EpisodeTask data structure."""
        task = EpisodeTask(
            episode_id="test_episode",
            batch_id="test_batch",
            status=EpisodeStatus.QUEUED
        )
        
        assert task.episode_id == "test_episode"
        assert task.batch_id == "test_batch"
        assert task.status == EpisodeStatus.QUEUED
        assert task.start_time is None
        assert task.retry_count == 0

    def test_episode_status_enum(self):
        """Test EpisodeStatus enum values."""
        assert EpisodeStatus.QUEUED.value == "queued"
        assert EpisodeStatus.RUNNING.value == "running"
        assert EpisodeStatus.COMPLETED.value == "completed"
        assert EpisodeStatus.FAILED.value == "failed"
        assert EpisodeStatus.CANCELLED.value == "cancelled"

    @pytest.mark.asyncio
    async def test_submit_batch(self, batch_manager):
        """Test batch submission."""
        episode_ids = ["ep_001", "ep_002", "ep_003"]
        
        await batch_manager.submit_batch(
            batch_id="test_batch",
            episode_ids=episode_ids,
            room_code="TR123456",
            room_password="secret",
            max_parallel=2,
            timeout_seconds=120
        )
        
        # Check batch was added
        assert "test_batch" in batch_manager.active_batches
        batch = batch_manager.active_batches["test_batch"]
        assert batch.batch_id == "test_batch"
        assert batch.episode_ids == episode_ids
        
        # Check episode tasks were created
        for episode_id in episode_ids:
            assert episode_id in batch_manager.episode_tasks
            task = batch_manager.episode_tasks[episode_id]
            assert task.status == EpisodeStatus.QUEUED
            assert task.batch_id == "test_batch"

    @pytest.mark.asyncio
    async def test_get_batch_status(self, batch_manager):
        """Test getting batch status."""
        # Submit a batch first
        episode_ids = ["ep_001", "ep_002"]
        await batch_manager.submit_batch(
            batch_id="test_batch",
            episode_ids=episode_ids,
            room_code="TR123456"
        )
        
        # Update some episode statuses
        batch_manager.episode_tasks["ep_001"].status = EpisodeStatus.COMPLETED
        batch_manager.episode_tasks["ep_001"].start_time = datetime.now()
        batch_manager.episode_tasks["ep_001"].end_time = datetime.now()
        
        batch_manager.episode_tasks["ep_002"].status = EpisodeStatus.RUNNING
        batch_manager.episode_tasks["ep_002"].start_time = datetime.now()
        
        status = await batch_manager.get_batch_status("test_batch")
        
        assert status is not None
        assert status["batchId"] == "test_batch"
        assert status["totalEpisodes"] == 2
        assert status["completed"] == 1
        assert status["running"] == 1
        assert status["queued"] == 0
        assert "episodes" in status

    @pytest.mark.asyncio
    async def test_get_batch_status_nonexistent(self, batch_manager):
        """Test getting status for non-existent batch."""
        status = await batch_manager.get_batch_status("nonexistent")
        assert status is None

    @pytest.mark.asyncio
    async def test_cancel_batch(self, batch_manager):
        """Test batch cancellation."""
        episode_ids = ["ep_001", "ep_002", "ep_003"]
        await batch_manager.submit_batch(
            batch_id="test_batch",
            episode_ids=episode_ids,
            room_code="TR123456"
        )
        
        # Set one episode as running
        batch_manager.episode_tasks["ep_001"].status = EpisodeStatus.RUNNING
        
        result = await batch_manager.cancel_batch("test_batch")
        
        assert result is True
        assert "test_batch" not in batch_manager.active_batches
        
        # Check episodes were cancelled
        for episode_id in episode_ids:
            task = batch_manager.episode_tasks[episode_id]
            if task.status in [EpisodeStatus.QUEUED, EpisodeStatus.RUNNING]:
                assert task.status == EpisodeStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_batch(self, batch_manager):
        """Test cancelling non-existent batch."""
        result = await batch_manager.cancel_batch("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_start_and_stop_manager(self, batch_manager):
        """Test starting and stopping the batch manager."""
        with patch.object(batch_manager, '_process_batch_queue', new_callable=AsyncMock) as mock_processor, \
             patch.object(batch_manager, '_monitor_workers', new_callable=AsyncMock) as mock_monitor:
            
            await batch_manager.start()
            
            assert batch_manager._batch_processor_task is not None
            assert batch_manager._worker_monitor_task is not None
            
            await batch_manager.stop()
            
            assert batch_manager._shutdown_event.is_set()

    def test_handler_registration(self, batch_manager):
        """Test event handler registration."""
        episode_handler = AsyncMock()
        batch_handler = AsyncMock()
        failed_handler = AsyncMock()
        
        batch_manager.register_episode_complete_handler(episode_handler)
        batch_manager.register_batch_complete_handler(batch_handler)
        batch_manager.register_episode_failed_handler(failed_handler)
        
        assert episode_handler in batch_manager.episode_complete_handlers
        assert batch_handler in batch_manager.batch_complete_handlers
        assert failed_handler in batch_manager.episode_failed_handlers

    @pytest.mark.asyncio
    async def test_execute_episode_success(self, batch_manager):
        """Test successful episode execution."""
        mock_game_client = AsyncMock()
        mock_game_client.connect = AsyncMock()
        mock_game_client.exit_game = AsyncMock()
        
        batch = EpisodeBatch(
            batch_id="test_batch",
            episode_ids=["test_episode"],
            room_code="TR123456",
            room_password=None,
            max_parallel=1,
            timeout_seconds=60,
            created_at=datetime.now()
        )
        
        result = await batch_manager._execute_episode(
            mock_game_client,
            "test_episode",
            batch
        )
        
        assert isinstance(result, EpisodeResult)
        assert result.episode_id == "test_episode"
        assert result.error is None
        assert result.total_reward == 10.0  # Placeholder value
        
        mock_game_client.connect.assert_called_once_with(
            "TR123456",
            "batch_bot_test_episode",
            None
        )
        mock_game_client.exit_game.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_episode_failure(self, batch_manager):
        """Test episode execution failure."""
        mock_game_client = AsyncMock()
        mock_game_client.connect.side_effect = Exception("Connection failed")
        
        batch = EpisodeBatch(
            batch_id="test_batch",
            episode_ids=["test_episode"],
            room_code="TR123456",
            room_password=None,
            max_parallel=1,
            timeout_seconds=60,
            created_at=datetime.now()
        )
        
        result = await batch_manager._execute_episode(
            mock_game_client,
            "test_episode",
            batch
        )
        
        assert isinstance(result, EpisodeResult)
        assert result.episode_id == "test_episode"
        assert result.error == "Connection failed"
        assert result.total_reward == 0.0
        assert result.game_result == "error"

    @pytest.mark.asyncio
    async def test_run_single_episode_with_retries(self, batch_manager):
        """Test single episode execution with retry logic."""
        task = EpisodeTask(
            episode_id="test_episode",
            batch_id="test_batch",
            status=EpisodeStatus.QUEUED
        )
        
        batch = EpisodeBatch(
            batch_id="test_batch",
            episode_ids=["test_episode"],
            room_code="TR123456",
            room_password=None,
            max_parallel=1,
            timeout_seconds=60,
            created_at=datetime.now()
        )
        
        # Mock execute_episode to fail first time, succeed second time
        call_count = 0
        async def mock_execute_episode(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt failed")
            return EpisodeResult(
                episode_id="test_episode",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration=1.0,
                total_reward=10.0,
                episode_length=100,
                game_result="win",
                final_state={}
            )
        
        with patch.object(batch_manager, '_execute_episode', side_effect=mock_execute_episode):
            await batch_manager._run_single_episode(task, batch)
        
        assert task.status == EpisodeStatus.COMPLETED
        assert task.retry_count == 1  # One retry after initial failure
        assert task.result is not None

    @pytest.mark.asyncio
    async def test_run_single_episode_max_retries_exceeded(self, batch_manager):
        """Test single episode execution when max retries are exceeded."""
        task = EpisodeTask(
            episode_id="test_episode",
            batch_id="test_batch",
            status=EpisodeStatus.QUEUED
        )
        
        batch = EpisodeBatch(
            batch_id="test_batch",
            episode_ids=["test_episode"],
            room_code="TR123456",
            room_password=None,
            max_parallel=1,
            timeout_seconds=60,
            created_at=datetime.now()
        )
        
        # Mock execute_episode to always fail
        async def mock_execute_episode(*args):
            raise Exception("Always fails")
        
        with patch.object(batch_manager, '_execute_episode', side_effect=mock_execute_episode):
            await batch_manager._run_single_episode(task, batch)
        
        assert task.status == EpisodeStatus.FAILED
        assert task.retry_count > batch_manager.max_retries
        assert task.error == "Always fails"

    @pytest.mark.asyncio
    async def test_run_single_episode_timeout(self, batch_manager):
        """Test single episode execution with timeout."""
        task = EpisodeTask(
            episode_id="test_episode",
            batch_id="test_batch",
            status=EpisodeStatus.QUEUED
        )
        
        batch = EpisodeBatch(
            batch_id="test_batch",
            episode_ids=["test_episode"],
            room_code="TR123456",
            room_password=None,
            max_parallel=1,
            timeout_seconds=1,  # Very short timeout
            created_at=datetime.now()
        )
        
        # Mock execute_episode to take longer than timeout
        async def mock_execute_episode(*args):
            await asyncio.sleep(2)  # Longer than timeout
            return EpisodeResult(
                episode_id="test_episode",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration=2.0,
                total_reward=10.0,
                episode_length=100,
                game_result="win",
                final_state={}
            )
        
        with patch.object(batch_manager, '_execute_episode', side_effect=mock_execute_episode):
            await batch_manager._run_single_episode(task, batch)
        
        assert task.status == EpisodeStatus.FAILED
        assert "timed out" in task.error

    @pytest.mark.asyncio
    async def test_check_batch_completion(self, batch_manager):
        """Test batch completion checking."""
        episode_ids = ["ep_001", "ep_002"]
        await batch_manager.submit_batch(
            batch_id="test_batch",
            episode_ids=episode_ids,
            room_code="TR123456"
        )
        
        # Mock batch completion handler
        batch_handler = AsyncMock()
        batch_manager.register_batch_complete_handler(batch_handler)
        
        # Mark all episodes as completed
        for episode_id in episode_ids:
            task = batch_manager.episode_tasks[episode_id]
            task.status = EpisodeStatus.COMPLETED
            task.result = EpisodeResult(
                episode_id=episode_id,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration=1.0,
                total_reward=10.0,
                episode_length=100,
                game_result="win",
                final_state={}
            )
        
        await batch_manager._check_batch_completion("test_batch")
        
        # Check batch was removed and handler was called
        assert "test_batch" not in batch_manager.active_batches
        batch_handler.assert_called_once()
        
        # Check episode tasks were cleaned up
        for episode_id in episode_ids:
            assert episode_id not in batch_manager.episode_tasks

    @pytest.mark.asyncio
    async def test_worker_semaphore_limits_parallel_execution(self, batch_manager):
        """Test that worker semaphore limits parallel episode execution."""
        # The semaphore should limit concurrent episodes to max_parallel_episodes
        assert batch_manager.worker_semaphore._value == batch_manager.max_parallel_episodes
        
        # Acquire all semaphore permits
        permits = []
        for _ in range(batch_manager.max_parallel_episodes):
            permit = await batch_manager.worker_semaphore.acquire()
            permits.append(permit)
        
        # Should not be able to acquire more
        try:
            await asyncio.wait_for(
                batch_manager.worker_semaphore.acquire(),
                timeout=0.1
            )
            assert False, "Should not have been able to acquire semaphore"
        except asyncio.TimeoutError:
            pass  # Expected
        
        # Release permits
        for _ in permits:
            batch_manager.worker_semaphore.release()