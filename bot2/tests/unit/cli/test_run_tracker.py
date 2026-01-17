"""Unit tests for the CLI run tracker."""

import tempfile
from collections.abc import Generator

import pytest

from bot.cli.run_tracker import RunTracker, TrainingRun


class TestTrainingRun:
    """Tests for the TrainingRun dataclass."""

    def test_new_creates_unique_id(self) -> None:
        """Test that new() creates runs with unique IDs."""
        run1 = TrainingRun.new()
        run2 = TrainingRun.new()
        assert run1.run_id != run2.run_id

    def test_new_sets_defaults(self) -> None:
        """Test that new() sets sensible defaults."""
        run = TrainingRun.new(total_timesteps=100000)
        assert run.state == "pending"
        assert run.timesteps == 0
        assert run.total_timesteps == 100000
        assert run.generation == 0
        assert run.start_time is None
        assert run.end_time is None

    def test_start_sets_running_state(self) -> None:
        """Test that start() transitions to running state."""
        run = TrainingRun.new()
        run.start(pid=12345)
        assert run.state == "running"
        assert run.start_time is not None
        assert run.pid == 12345

    def test_complete_sets_completed_state(self) -> None:
        """Test that complete() transitions to completed state."""
        run = TrainingRun.new()
        run.start()
        run.complete()
        assert run.state == "completed"
        assert run.end_time is not None
        assert run.pid is None

    def test_fail_sets_failed_state(self) -> None:
        """Test that fail() transitions to failed state."""
        run = TrainingRun.new()
        run.start()
        run.fail("Test error")
        assert run.state == "failed"
        assert run.error_message == "Test error"
        assert run.end_time is not None

    def test_stop_sets_stopped_state(self) -> None:
        """Test that stop() transitions to stopped state."""
        run = TrainingRun.new()
        run.start()
        run.stop()
        assert run.state == "stopped"
        assert run.end_time is not None

    def test_pause_and_resume(self) -> None:
        """Test pause and resume transitions."""
        run = TrainingRun.new()
        run.start()
        run.pause()
        assert run.state == "paused"
        run.resume()
        assert run.state == "running"

    def test_update_progress(self) -> None:
        """Test progress updates."""
        run = TrainingRun.new(total_timesteps=100000)
        run.update_progress(timesteps=50000, generation=1, checkpoint="ckpt.pt")
        assert run.timesteps == 50000
        assert run.generation == 1
        assert run.last_checkpoint == "ckpt.pt"

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        run = TrainingRun.new(
            config_path="config.yaml",
            total_timesteps=100000,
        )
        run.start()
        run.update_progress(timesteps=50000)

        data = run.to_dict()
        restored = TrainingRun.from_dict(data)

        assert restored.run_id == run.run_id
        assert restored.state == run.state
        assert restored.timesteps == run.timesteps
        assert restored.config_path == run.config_path


class TestRunTracker:
    """Tests for the RunTracker class."""

    @pytest.fixture
    def tracker(self) -> Generator[RunTracker, None, None]:
        """Create a tracker with a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield RunTracker(tmpdir)

    def test_create_run(self, tracker: RunTracker) -> None:
        """Test creating a new run."""
        run = tracker.create_run(
            config_path="config.yaml",
            total_timesteps=100000,
        )
        assert run.run_id is not None
        assert run.config_path == "config.yaml"
        assert run.total_timesteps == 100000

    def test_get_run(self, tracker: RunTracker) -> None:
        """Test retrieving a run by ID."""
        run = tracker.create_run()
        retrieved = tracker.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.run_id == run.run_id

    def test_get_run_not_found(self, tracker: RunTracker) -> None:
        """Test that get_run returns None for unknown IDs."""
        result = tracker.get_run("nonexistent")
        assert result is None

    def test_save_run_updates_existing(self, tracker: RunTracker) -> None:
        """Test that save_run updates existing runs."""
        run = tracker.create_run()
        run.start()
        run.update_progress(timesteps=50000)
        tracker.save_run(run)

        retrieved = tracker.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.state == "running"
        assert retrieved.timesteps == 50000

    def test_list_runs_empty(self, tracker: RunTracker) -> None:
        """Test listing runs when empty."""
        runs = tracker.list_runs()
        assert runs == []

    def test_list_runs(self, tracker: RunTracker) -> None:
        """Test listing all runs."""
        run1 = tracker.create_run()
        tracker.create_run()  # Second run created but not started
        run1.start()
        tracker.save_run(run1)

        runs = tracker.list_runs()
        assert len(runs) == 2

    def test_list_runs_filter_by_state(self, tracker: RunTracker) -> None:
        """Test filtering runs by state."""
        run1 = tracker.create_run()
        run2 = tracker.create_run()
        run1.start()
        run2.start()
        run2.complete()
        tracker.save_run(run1)
        tracker.save_run(run2)

        running = tracker.list_runs(state="running")
        completed = tracker.list_runs(state="completed")

        assert len(running) == 1
        assert len(completed) == 1
        assert running[0].run_id == run1.run_id
        assert completed[0].run_id == run2.run_id

    def test_list_runs_limit(self, tracker: RunTracker) -> None:
        """Test limiting the number of runs returned."""
        for _ in range(5):
            run = tracker.create_run()
            run.start()
            tracker.save_run(run)

        runs = tracker.list_runs(limit=3)
        assert len(runs) == 3

    def test_get_active_run(self, tracker: RunTracker) -> None:
        """Test getting the active run."""
        run = tracker.create_run()
        run.start()
        tracker.save_run(run)

        active = tracker.get_active_run()
        assert active is not None
        assert active.run_id == run.run_id

    def test_get_active_run_none_when_empty(self, tracker: RunTracker) -> None:
        """Test that get_active_run returns None when no runs."""
        active = tracker.get_active_run()
        assert active is None

    def test_delete_run(self, tracker: RunTracker) -> None:
        """Test deleting a run."""
        run = tracker.create_run()
        result = tracker.delete_run(run.run_id)
        assert result is True
        assert tracker.get_run(run.run_id) is None

    def test_delete_run_not_found(self, tracker: RunTracker) -> None:
        """Test deleting a nonexistent run."""
        result = tracker.delete_run("nonexistent")
        assert result is False

    def test_persistence_across_instances(self) -> None:
        """Test that data persists across tracker instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create run with first tracker
            tracker1 = RunTracker(tmpdir)
            run = tracker1.create_run(total_timesteps=100000)
            run_id = run.run_id

            # Load with second tracker
            tracker2 = RunTracker(tmpdir)
            loaded = tracker2.get_run(run_id)

            assert loaded is not None
            assert loaded.run_id == run_id
            assert loaded.total_timesteps == 100000
