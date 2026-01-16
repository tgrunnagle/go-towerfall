"""Unit tests for CLI output utilities."""

from datetime import datetime

from rich.panel import Panel

from bot.cli.utils.output import (
    create_training_status_panel,
    format_duration,
    format_timestamp,
)


class TestFormatDuration:
    """Tests for the format_duration function."""

    def test_seconds(self) -> None:
        """Test formatting seconds."""
        assert format_duration(30.5) == "30.5s"
        assert format_duration(59.9) == "59.9s"

    def test_minutes(self) -> None:
        """Test formatting minutes."""
        assert format_duration(60) == "1.0m"
        assert format_duration(120) == "2.0m"
        assert format_duration(3599) == "60.0m"

    def test_hours(self) -> None:
        """Test formatting hours."""
        assert format_duration(3600) == "1.0h"
        assert format_duration(7200) == "2.0h"
        assert format_duration(36000) == "10.0h"


class TestFormatTimestamp:
    """Tests for the format_timestamp function."""

    def test_format(self) -> None:
        """Test timestamp formatting."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = format_timestamp(dt)
        assert result == "2024-01-15 10:30:45"


class TestCreateTrainingStatusPanel:
    """Tests for the create_training_status_panel function."""

    def test_creates_panel(self) -> None:
        """Test that a panel is created."""
        status = {
            "run_id": "abc123",
            "state": "running",
            "timesteps": 50000,
            "total_timesteps": 100000,
            "generation": 1,
            "elapsed_seconds": 3600,
            "fps": 500.0,
        }
        panel = create_training_status_panel(status)
        assert isinstance(panel, Panel)

    def test_handles_missing_fields(self) -> None:
        """Test that missing fields are handled gracefully."""
        status = {"run_id": "abc123", "state": "pending"}
        panel = create_training_status_panel(status)
        assert isinstance(panel, Panel)
