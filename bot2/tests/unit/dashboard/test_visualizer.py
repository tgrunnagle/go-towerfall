"""Unit tests for the dashboard visualizer."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bot.dashboard.data_aggregator import DataAggregator
from bot.dashboard.models import DashboardConfig, GenerationMetrics
from bot.dashboard.visualizer import DashboardVisualizer
from bot.training.registry import ModelRegistry


@pytest.fixture
def sample_generation_metrics() -> list[GenerationMetrics]:
    """Create sample generation metrics for testing."""
    base_time = datetime.now(timezone.utc)
    return [
        GenerationMetrics(
            generation_id=0,
            model_version="ppo_gen_000",
            opponent_type="baseline",
            total_episodes=1000,
            total_kills=1500,
            total_deaths=2000,
            kill_death_ratio=0.75,
            win_rate=0.4,
            avg_episode_reward=30.0,
            avg_episode_length=400.0,
            training_steps=100000,
            training_duration_seconds=3600.0,
            timestamp=base_time,
        ),
        GenerationMetrics(
            generation_id=1,
            model_version="ppo_gen_001",
            opponent_type="ppo_gen_000",
            total_episodes=1500,
            total_kills=3750,
            total_deaths=2250,
            kill_death_ratio=1.67,
            win_rate=0.6,
            avg_episode_reward=50.0,
            avg_episode_length=500.0,
            training_steps=150000,
            training_duration_seconds=5400.0,
            timestamp=base_time,
        ),
        GenerationMetrics(
            generation_id=2,
            model_version="ppo_gen_002",
            opponent_type="ppo_gen_001",
            total_episodes=2000,
            total_kills=7000,
            total_deaths=2000,
            kill_death_ratio=3.5,
            win_rate=0.8,
            avg_episode_reward=75.0,
            avg_episode_length=600.0,
            training_steps=200000,
            training_duration_seconds=7200.0,
            timestamp=base_time,
        ),
    ]


@pytest.fixture
def mock_aggregator(sample_generation_metrics: list[GenerationMetrics]):
    """Create a mock aggregator with sample metrics."""
    aggregator = MagicMock(spec=DataAggregator)
    aggregator.get_all_generation_metrics.return_value = sample_generation_metrics
    aggregator.get_summary_table_data.return_value = [
        {
            "Generation": m.generation_id,
            "Model": m.model_version,
            "Opponent": m.opponent_type,
            "K/D Ratio": f"{m.kill_death_ratio:.2f}",
            "Win Rate": f"{m.win_rate * 100:.1f}%",
            "Avg Reward": f"{m.avg_episode_reward:.1f}",
            "Episodes": m.total_episodes,
            "Training Time": "1h 0m",
        }
        for m in sample_generation_metrics
    ]
    return aggregator


class TestDashboardVisualizer:
    """Tests for DashboardVisualizer class."""

    def test_init_creates_output_dir(self, mock_aggregator: MagicMock):
        """Test that initializing visualizer creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "reports"
            assert not output_dir.exists()

            DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=output_dir,
            )

            assert output_dir.exists()

    def test_from_config(self):
        """Test creating visualizer from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry"
            ModelRegistry(registry_path)

            config = DashboardConfig(
                registry_path=str(registry_path),
                output_dir=str(Path(tmpdir) / "reports"),
                title="Test Dashboard",
            )

            visualizer = DashboardVisualizer.from_config(config)

            assert visualizer.title == "Test Dashboard"

    def test_generate_all_empty_metrics(self, mock_aggregator: MagicMock):
        """Test generating dashboard with no metrics."""
        mock_aggregator.get_all_generation_metrics.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
            )

            files = visualizer.generate_all()

            assert files == []

    def test_generate_all_html_format(self, mock_aggregator: MagicMock):
        """Test generating dashboard with HTML format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
            )

            files = visualizer.generate_all(output_format="html")

            # Should generate HTML files for each chart plus summary and dashboard
            html_files = [f for f in files if f.suffix == ".html"]
            assert len(html_files) >= 4  # 3 charts + summary + dashboard

            # Check main dashboard exists
            dashboard_path = Path(tmpdir) / "dashboard.html"
            assert dashboard_path.exists()

            # Check content
            content = dashboard_path.read_text()
            assert "Training Generation Comparison" in content
            assert "K/D Ratio" in content

    def test_generate_all_png_format(self, mock_aggregator: MagicMock):
        """Test generating dashboard with PNG format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
            )

            files = visualizer.generate_all(output_format="png")

            # Should generate PNG files for charts
            png_files = [f for f in files if f.suffix == ".png"]
            assert len(png_files) == 3  # K/D, win rate, reward

            # Check files exist and have content
            for png_file in png_files:
                assert png_file.exists()
                assert png_file.stat().st_size > 0

    def test_generate_all_both_formats(self, mock_aggregator: MagicMock):
        """Test generating dashboard with both formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
            )

            files = visualizer.generate_all(output_format="both")

            # Should have both HTML and PNG files
            png_files = [f for f in files if f.suffix == ".png"]
            html_files = [f for f in files if f.suffix == ".html"]

            assert len(png_files) == 3
            assert len(html_files) >= 4

    def test_generate_with_generation_range(self, mock_aggregator: MagicMock):
        """Test generating dashboard with generation range filter."""
        # Modify mock to return filtered results
        filtered_metrics = mock_aggregator.get_all_generation_metrics.return_value[:2]
        mock_aggregator.get_all_generation_metrics.side_effect = lambda r=None: (
            filtered_metrics
            if r == (0, 1)
            else mock_aggregator.get_all_generation_metrics.return_value
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
            )

            files = visualizer.generate_all(
                output_format="html",
                generation_range=(0, 1),
            )

            assert len(files) > 0

    def test_kd_ratio_chart_content(
        self,
        mock_aggregator: MagicMock,
        sample_generation_metrics: list[GenerationMetrics],
    ):
        """Test that K/D ratio chart has correct data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
            )

            # Generate just the K/D chart
            files = visualizer._generate_kd_ratio_chart(
                sample_generation_metrics,
                output_format="html",
            )

            assert len(files) == 1
            html_path = files[0]
            content = html_path.read_text()

            # Check it's valid HTML with embedded image
            assert "<!DOCTYPE html>" in content
            assert "data:image/png;base64," in content

    def test_win_rate_chart_content(
        self,
        mock_aggregator: MagicMock,
        sample_generation_metrics: list[GenerationMetrics],
    ):
        """Test that win rate chart has correct data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
            )

            files = visualizer._generate_win_rate_chart(
                sample_generation_metrics,
                output_format="html",
            )

            assert len(files) == 1
            html_path = files[0]
            content = html_path.read_text()

            assert "data:image/png;base64," in content

    def test_reward_chart_content(
        self,
        mock_aggregator: MagicMock,
        sample_generation_metrics: list[GenerationMetrics],
    ):
        """Test that reward chart has correct data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
            )

            files = visualizer._generate_reward_chart(
                sample_generation_metrics,
                output_format="html",
            )

            assert len(files) == 1
            html_path = files[0]
            content = html_path.read_text()

            assert "data:image/png;base64," in content

    def test_summary_table_content(
        self,
        mock_aggregator: MagicMock,
        sample_generation_metrics: list[GenerationMetrics],
    ):
        """Test that summary table has correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
            )

            path = visualizer._generate_summary_table(sample_generation_metrics)

            content = path.read_text()

            # Check table structure
            assert "<table>" in content
            assert "<th>" in content
            assert "<td>" in content

            # Check column headers
            assert "Generation" in content
            assert "K/D Ratio" in content
            assert "Win Rate" in content

    def test_combined_dashboard_structure(
        self,
        mock_aggregator: MagicMock,
        sample_generation_metrics: list[GenerationMetrics],
    ):
        """Test that combined dashboard has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
                title="Test Dashboard Title",
            )

            path = visualizer._generate_combined_dashboard(sample_generation_metrics)

            content = path.read_text()

            # Check structure
            assert "<!DOCTYPE html>" in content
            assert "<html>" in content
            assert "Test Dashboard Title" in content

            # Check sections
            assert "Performance Charts" in content
            assert "Generation Summary" in content

            # Check charts are embedded
            assert "data:image/png;base64," in content

            # Check table is included
            assert "<table>" in content


class TestBuildTableHtml:
    """Tests for HTML table building helper."""

    def test_build_empty_table(self, mock_aggregator: MagicMock):
        """Test building table with empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
            )

            html = visualizer._build_table_html([])

            assert "No data available" in html

    def test_build_table_with_data(self, mock_aggregator: MagicMock):
        """Test building table with data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = DashboardVisualizer(
                aggregator=mock_aggregator,
                output_dir=tmpdir,
            )

            data = [
                {"Col1": "A", "Col2": "B"},
                {"Col1": "C", "Col2": "D"},
            ]

            html = visualizer._build_table_html(data)

            assert "<table>" in html
            assert "<thead>" in html
            assert "<tbody>" in html
            assert "<th>Col1</th>" in html
            assert "<td>A</td>" in html
