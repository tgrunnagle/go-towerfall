"""Unit tests for metrics writers."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bot.training.metrics.writers.base import MetricsWriter
from bot.training.metrics.writers.file_writer import FileWriter
from bot.training.metrics.writers.tensorboard_writer import TensorBoardWriter


class TestMetricsWriterInterface:
    """Tests for MetricsWriter abstract interface."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that MetricsWriter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MetricsWriter()  # type: ignore[abstract]


class TestFileWriter:
    """Tests for FileWriter class."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that initialization creates the log directory."""
        log_dir = tmp_path / "logs" / "nested"
        writer = FileWriter(log_dir=log_dir)
        assert log_dir.exists()
        writer.close()

    def test_init_default_format_json(self, tmp_path: Path) -> None:
        """Test that default format is JSON."""
        writer = FileWriter(log_dir=tmp_path)
        assert writer.format == "json"
        writer.close()

    def test_write_scalar_json(self, tmp_path: Path) -> None:
        """Test writing scalar to JSON file."""
        writer = FileWriter(log_dir=tmp_path, format="json", buffer_size=1)
        writer.write_scalar("train/loss", 0.5, 1)
        writer.close()

        # Read and verify
        output_file = tmp_path / "metrics.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            entry = json.loads(f.readline())

        assert entry["tag"] == "train/loss"
        assert entry["value"] == 0.5
        assert entry["step"] == 1
        assert "timestamp" in entry

    def test_write_scalar_csv(self, tmp_path: Path) -> None:
        """Test writing scalar to CSV file."""
        writer = FileWriter(log_dir=tmp_path, format="csv", buffer_size=1)
        writer.write_scalar("train/loss", 0.5, 1)
        writer.close()

        # Read and verify
        output_file = tmp_path / "metrics.csv"
        assert output_file.exists()

        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["tag"] == "train/loss"
        assert float(rows[0]["value"]) == 0.5
        assert int(rows[0]["step"]) == 1

    def test_write_scalars(self, tmp_path: Path) -> None:
        """Test writing multiple scalars."""
        writer = FileWriter(log_dir=tmp_path, format="json", buffer_size=10)
        writer.write_scalars("losses", {"policy": 0.1, "value": 0.2}, 1)
        writer.flush()
        writer.close()

        output_file = tmp_path / "metrics.jsonl"
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        tags = {json.loads(line)["tag"] for line in lines}
        assert tags == {"losses/policy", "losses/value"}

    def test_buffering(self, tmp_path: Path) -> None:
        """Test that data is buffered until buffer_size is reached."""
        writer = FileWriter(log_dir=tmp_path, format="json", buffer_size=5)

        # Write fewer than buffer_size entries
        for i in range(3):
            writer.write_scalar("metric", float(i), i)

        # File should not exist yet (or be empty)
        output_file = tmp_path / "metrics.jsonl"
        if output_file.exists():
            with open(output_file) as f:
                assert f.read() == ""

        # Write more to trigger flush
        for i in range(3, 6):
            writer.write_scalar("metric", float(i), i)

        # Now file should have content
        assert output_file.exists()
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) >= 5

        writer.close()

    def test_flush(self, tmp_path: Path) -> None:
        """Test explicit flush."""
        writer = FileWriter(log_dir=tmp_path, format="json", buffer_size=100)
        writer.write_scalar("metric", 1.0, 1)
        writer.flush()

        output_file = tmp_path / "metrics.jsonl"
        assert output_file.exists()
        with open(output_file) as f:
            assert len(f.readlines()) == 1

        writer.close()

    def test_file_rotation(self, tmp_path: Path) -> None:
        """Test file rotation when size limit is exceeded."""
        # Use tiny max file size for testing
        writer = FileWriter(
            log_dir=tmp_path,
            format="json",
            buffer_size=1,
            max_file_size_mb=0.0001,  # Very small
        )

        # Write enough to trigger rotation
        for i in range(100):
            writer.write_scalar(f"metric_{i}", float(i), i)

        writer.close()

        # Should have multiple files
        files = list(tmp_path.glob("metrics*.jsonl"))
        assert len(files) >= 2

    def test_close_flushes(self, tmp_path: Path) -> None:
        """Test that close flushes remaining buffer."""
        writer = FileWriter(log_dir=tmp_path, format="json", buffer_size=100)
        writer.write_scalar("metric", 1.0, 1)
        writer.close()

        output_file = tmp_path / "metrics.jsonl"
        with open(output_file) as f:
            assert len(f.readlines()) == 1

    def test_write_after_close_ignored(self, tmp_path: Path) -> None:
        """Test that writes after close are ignored."""
        writer = FileWriter(log_dir=tmp_path, format="json", buffer_size=1)
        writer.close()
        # Should not raise
        writer.write_scalar("metric", 1.0, 1)

    def test_csv_header(self, tmp_path: Path) -> None:
        """Test that CSV files have proper headers."""
        writer = FileWriter(log_dir=tmp_path, format="csv", buffer_size=1)
        writer.write_scalar("metric", 1.0, 1)
        writer.close()

        output_file = tmp_path / "metrics.csv"
        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == ["tag", "value", "step", "timestamp"]


class TestTensorBoardWriter:
    """Tests for TensorBoardWriter class."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that initialization creates the log directory."""
        log_dir = tmp_path / "tb_logs" / "nested"
        writer = TensorBoardWriter(log_dir=log_dir)
        assert log_dir.exists()
        writer.close()

    @patch("bot.training.metrics.writers.tensorboard_writer.SummaryWriter")
    def test_write_scalar(self, mock_summary_writer: MagicMock, tmp_path: Path) -> None:
        """Test writing scalar to TensorBoard."""
        mock_writer_instance = MagicMock()
        mock_summary_writer.return_value = mock_writer_instance

        writer = TensorBoardWriter(log_dir=tmp_path)
        writer.write_scalar("train/loss", 0.5, 1)

        mock_writer_instance.add_scalar.assert_called_once_with("train/loss", 0.5, 1)
        writer.close()

    @patch("bot.training.metrics.writers.tensorboard_writer.SummaryWriter")
    def test_write_scalars(
        self, mock_summary_writer: MagicMock, tmp_path: Path
    ) -> None:
        """Test writing multiple scalars to TensorBoard."""
        mock_writer_instance = MagicMock()
        mock_summary_writer.return_value = mock_writer_instance

        writer = TensorBoardWriter(log_dir=tmp_path)
        writer.write_scalars("losses", {"policy": 0.1, "value": 0.2}, 1)

        mock_writer_instance.add_scalars.assert_called_once_with(
            "losses", {"policy": 0.1, "value": 0.2}, 1
        )
        writer.close()

    @patch("bot.training.metrics.writers.tensorboard_writer.SummaryWriter")
    def test_flush(self, mock_summary_writer: MagicMock, tmp_path: Path) -> None:
        """Test flush calls underlying writer flush."""
        mock_writer_instance = MagicMock()
        mock_summary_writer.return_value = mock_writer_instance

        writer = TensorBoardWriter(log_dir=tmp_path)
        writer.flush()

        mock_writer_instance.flush.assert_called_once()
        writer.close()

    @patch("bot.training.metrics.writers.tensorboard_writer.SummaryWriter")
    def test_close(self, mock_summary_writer: MagicMock, tmp_path: Path) -> None:
        """Test close calls underlying writer close."""
        mock_writer_instance = MagicMock()
        mock_summary_writer.return_value = mock_writer_instance

        writer = TensorBoardWriter(log_dir=tmp_path)
        writer.close()

        mock_writer_instance.close.assert_called_once()

    @patch("bot.training.metrics.writers.tensorboard_writer.SummaryWriter")
    def test_write_after_close_ignored(
        self, mock_summary_writer: MagicMock, tmp_path: Path
    ) -> None:
        """Test that writes after close are ignored."""
        mock_writer_instance = MagicMock()
        mock_summary_writer.return_value = mock_writer_instance

        writer = TensorBoardWriter(log_dir=tmp_path)
        writer.close()
        writer.write_scalar("metric", 1.0, 1)

        # Should only have been called during init setup, not after close
        mock_writer_instance.add_scalar.assert_not_called()
