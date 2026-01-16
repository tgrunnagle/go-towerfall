"""Training run tracking for the CLI.

This module provides persistent tracking of training runs, enabling
status queries, pause/resume, and run history.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TrainingRun:
    """Represents a training run with its metadata and state.

    Attributes:
        run_id: Unique identifier for this run
        state: Current state (pending, running, paused, completed, failed, stopped)
        config_path: Path to the configuration file used
        config_snapshot: Snapshot of configuration at run start
        checkpoint_dir: Directory where checkpoints are saved
        registry_path: Path to model registry
        start_time: When the run started
        end_time: When the run ended (if applicable)
        pid: Process ID if running in background
        generation: Current model generation
        timesteps: Current timesteps completed
        total_timesteps: Total timesteps to train
        last_checkpoint: Path to most recent checkpoint
        error_message: Error message if failed
    """

    run_id: str
    state: str = "pending"
    config_path: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    checkpoint_dir: str = "./checkpoints"
    registry_path: str = "./model_registry"
    start_time: str | None = None
    end_time: str | None = None
    pid: int | None = None
    generation: int = 0
    timesteps: int = 0
    total_timesteps: int = 0
    last_checkpoint: str | None = None
    error_message: str | None = None

    @classmethod
    def new(
        cls,
        config_path: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        checkpoint_dir: str = "./checkpoints",
        registry_path: str = "./model_registry",
        total_timesteps: int = 0,
    ) -> TrainingRun:
        """Create a new training run.

        Args:
            config_path: Path to configuration file
            config_snapshot: Configuration snapshot
            checkpoint_dir: Checkpoint directory
            registry_path: Model registry path
            total_timesteps: Total timesteps to train

        Returns:
            New TrainingRun instance
        """
        return cls(
            run_id=str(uuid.uuid4())[:8],
            config_path=config_path,
            config_snapshot=config_snapshot or {},
            checkpoint_dir=checkpoint_dir,
            registry_path=registry_path,
            total_timesteps=total_timesteps,
        )

    def start(self, pid: int | None = None) -> None:
        """Mark the run as started."""
        self.state = "running"
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.pid = pid

    def pause(self) -> None:
        """Mark the run as paused."""
        self.state = "paused"

    def resume(self) -> None:
        """Mark the run as resumed."""
        self.state = "running"

    def complete(self) -> None:
        """Mark the run as completed."""
        self.state = "completed"
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.pid = None

    def fail(self, error: str) -> None:
        """Mark the run as failed."""
        self.state = "failed"
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.error_message = error
        self.pid = None

    def stop(self) -> None:
        """Mark the run as stopped."""
        self.state = "stopped"
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.pid = None

    def update_progress(
        self,
        timesteps: int,
        generation: int | None = None,
        checkpoint: str | None = None,
    ) -> None:
        """Update training progress.

        Args:
            timesteps: Current timesteps
            generation: Current generation (optional)
            checkpoint: Latest checkpoint path (optional)
        """
        self.timesteps = timesteps
        if generation is not None:
            self.generation = generation
        if checkpoint is not None:
            self.last_checkpoint = checkpoint

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRun:
        """Create from dictionary."""
        return cls(**data)


class RunTracker:
    """Manages persistent storage and retrieval of training runs.

    Stores run information in a JSON file for persistence across
    CLI invocations.

    Example:
        tracker = RunTracker("./runs")
        run = tracker.create_run(config_path="config.yaml")
        run.start()
        tracker.save_run(run)

        # Later
        run = tracker.get_run(run_id)
        print(f"Status: {run.state}")
    """

    def __init__(self, runs_dir: str | Path = "./.training_runs") -> None:
        """Initialize the run tracker.

        Args:
            runs_dir: Directory to store run data
        """
        self.runs_dir = Path(runs_dir)
        self.runs_file = self.runs_dir / "runs.json"
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure the runs directory exists."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if not self.runs_file.exists():
            self._save_all({})

    def _load_all(self) -> dict[str, dict[str, Any]]:
        """Load all runs from file."""
        if not self.runs_file.exists():
            return {}
        with open(self.runs_file) as f:
            return json.load(f)

    def _save_all(self, runs: dict[str, dict[str, Any]]) -> None:
        """Save all runs to file."""
        with open(self.runs_file, "w") as f:
            json.dump(runs, f, indent=2)

    def create_run(
        self,
        config_path: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        checkpoint_dir: str = "./checkpoints",
        registry_path: str = "./model_registry",
        total_timesteps: int = 0,
    ) -> TrainingRun:
        """Create a new training run.

        Args:
            config_path: Path to configuration file
            config_snapshot: Configuration snapshot
            checkpoint_dir: Checkpoint directory
            registry_path: Model registry path
            total_timesteps: Total timesteps to train

        Returns:
            New TrainingRun instance
        """
        run = TrainingRun.new(
            config_path=config_path,
            config_snapshot=config_snapshot,
            checkpoint_dir=checkpoint_dir,
            registry_path=registry_path,
            total_timesteps=total_timesteps,
        )
        self.save_run(run)
        return run

    def save_run(self, run: TrainingRun) -> None:
        """Save a training run.

        Args:
            run: TrainingRun to save
        """
        runs = self._load_all()
        runs[run.run_id] = run.to_dict()
        self._save_all(runs)

    def get_run(self, run_id: str) -> TrainingRun | None:
        """Get a training run by ID.

        Args:
            run_id: Run identifier

        Returns:
            TrainingRun or None if not found
        """
        runs = self._load_all()
        if run_id not in runs:
            return None
        return TrainingRun.from_dict(runs[run_id])

    def list_runs(
        self,
        state: str | None = None,
        limit: int | None = None,
    ) -> list[TrainingRun]:
        """List training runs.

        Args:
            state: Filter by state (optional)
            limit: Maximum number to return (optional)

        Returns:
            List of TrainingRun objects
        """
        runs = self._load_all()
        result = [TrainingRun.from_dict(r) for r in runs.values()]

        # Filter by state if specified
        if state is not None:
            result = [r for r in result if r.state == state]

        # Sort by start time (most recent first)
        result.sort(
            key=lambda r: r.start_time or "0000-00-00",
            reverse=True,
        )

        # Apply limit if specified
        if limit is not None:
            result = result[:limit]

        return result

    def get_active_run(self) -> TrainingRun | None:
        """Get the currently running training run.

        Returns:
            Active TrainingRun or None if no run is active
        """
        runs = self.list_runs(state="running")
        return runs[0] if runs else None

    def delete_run(self, run_id: str) -> bool:
        """Delete a training run.

        Args:
            run_id: Run identifier

        Returns:
            True if deleted, False if not found
        """
        runs = self._load_all()
        if run_id not in runs:
            return False
        del runs[run_id]
        self._save_all(runs)
        return True

    def cleanup_stale_runs(self) -> int:
        """Clean up stale runs (running but process not alive).

        Returns:
            Number of runs cleaned up
        """
        runs = self._load_all()
        cleaned = 0

        for run_id, run_data in list(runs.items()):
            if run_data.get("state") == "running":
                pid = run_data.get("pid")
                if pid is not None:
                    # Check if process is still alive
                    try:
                        os.kill(pid, 0)
                    except (OSError, ProcessLookupError):
                        # Process not found - mark as failed
                        run = TrainingRun.from_dict(run_data)
                        run.fail("Process terminated unexpectedly")
                        runs[run_id] = run.to_dict()
                        cleaned += 1

        if cleaned > 0:
            self._save_all(runs)

        return cleaned
