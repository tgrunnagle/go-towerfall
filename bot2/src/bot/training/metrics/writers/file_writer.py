"""File-based metrics writer implementation.

This module provides a MetricsWriter that persists metrics to JSON or CSV files.
"""

import csv
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal

from bot.training.metrics.writers.base import MetricsWriter

logger = logging.getLogger(__name__)


class FileWriter(MetricsWriter):
    """Metrics writer that persists to JSON or CSV files.

    Supports two output formats:
    - JSON: Each metric is a JSON object per line (JSONL format)
    - CSV: Standard CSV with tag, value, step, timestamp columns

    Files are rotated when they exceed max_file_size_mb to prevent
    unbounded disk usage.

    Attributes:
        log_dir: Directory for output files
        format: Output format ("json" or "csv")
        max_file_size_mb: Maximum file size before rotation
        buffer_size: Number of entries to buffer before writing
    """

    def __init__(
        self,
        log_dir: str | Path,
        format: Literal["json", "csv"] = "json",
        max_file_size_mb: float = 100.0,
        buffer_size: int = 100,
    ) -> None:
        """Initialize the file writer.

        Args:
            log_dir: Directory for output files
            format: Output format ("json" or "csv")
            max_file_size_mb: Maximum file size in MB before rotation (default 100MB)
            buffer_size: Number of entries to buffer before writing (default 100)
        """
        self.log_dir = Path(log_dir)
        self.format = format
        self.max_file_size_mb = max_file_size_mb
        self.buffer_size = buffer_size

        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._file_counter = 0
        self._closed = False

        # Create output directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize file path
        self._current_file_path = self._get_file_path()

        # Write CSV header if needed
        if self.format == "csv" and not self._current_file_path.exists():
            self._write_csv_header()

        logger.info("FileWriter initialized: %s (format=%s)", self.log_dir, self.format)

    def _get_file_path(self) -> Path:
        """Get the current output file path.

        Returns:
            Path to the current output file
        """
        extension = "jsonl" if self.format == "json" else "csv"
        if self._file_counter == 0:
            return self.log_dir / f"metrics.{extension}"
        return self.log_dir / f"metrics_{self._file_counter}.{extension}"

    def _write_csv_header(self) -> None:
        """Write CSV header to the current file."""
        with open(self._current_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tag", "value", "step", "timestamp"])

    def _should_rotate(self) -> bool:
        """Check if file rotation is needed.

        Returns:
            True if current file exceeds max_file_size_mb
        """
        if not self._current_file_path.exists():
            return False
        size_mb = self._current_file_path.stat().st_size / (1024 * 1024)
        return size_mb >= self.max_file_size_mb

    def _rotate_file(self) -> None:
        """Rotate to a new output file."""
        self._file_counter += 1
        self._current_file_path = self._get_file_path()
        logger.info("Rotated to new file: %s", self._current_file_path)

        if self.format == "csv":
            self._write_csv_header()

    def write_scalar(self, tag: str, value: float, step: int) -> None:
        """Write a single scalar metric.

        Args:
            tag: Metric name/tag
            value: Scalar value to log
            step: Global step number
        """
        if self._closed:
            return

        entry = {
            "tag": tag,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }

        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()

    def write_scalars(
        self, main_tag: str, tag_scalar_dict: dict[str, float], step: int
    ) -> None:
        """Write multiple related scalar metrics.

        Args:
            main_tag: Group name for the metrics
            tag_scalar_dict: Dictionary mapping sub-tags to values
            step: Global step number
        """
        for sub_tag, value in tag_scalar_dict.items():
            self.write_scalar(f"{main_tag}/{sub_tag}", value, step)

    def _flush_buffer(self) -> None:
        """Flush buffered entries to file (caller must hold lock)."""
        if not self._buffer:
            return

        # Check for rotation
        if self._should_rotate():
            self._rotate_file()

        # Write entries
        if self.format == "json":
            self._write_json_entries()
        else:
            self._write_csv_entries()

        self._buffer.clear()

    def _write_json_entries(self) -> None:
        """Write buffered entries as JSONL."""
        with open(self._current_file_path, "a") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry) + "\n")

    def _write_csv_entries(self) -> None:
        """Write buffered entries as CSV rows."""
        with open(self._current_file_path, "a", newline="") as f:
            writer = csv.writer(f)
            for entry in self._buffer:
                writer.writerow(
                    [entry["tag"], entry["value"], entry["step"], entry["timestamp"]]
                )

    def flush(self) -> None:
        """Flush any buffered data to disk."""
        if self._closed:
            return

        with self._lock:
            self._flush_buffer()

    def close(self) -> None:
        """Close the writer and release resources."""
        if self._closed:
            return

        self.flush()
        self._closed = True
        logger.info("FileWriter closed: %s", self.log_dir)
