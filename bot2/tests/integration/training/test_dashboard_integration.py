"""Integration tests for the dashboard module.

These tests verify the complete workflow of dashboard generation with
real model registry data, including file generation and content validation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from bot.agent.network import ActorCriticNetwork
from bot.dashboard.data_aggregator import DataAggregator
from bot.dashboard.visualizer import DashboardVisualizer
from bot.training.registry import ModelRegistry, TrainingMetrics


@pytest.fixture
def temp_registry_path():
    """Create a temporary directory for registry storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "registry"


@pytest.fixture
def temp_output_path():
    """Create a temporary directory for dashboard output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "reports"


@pytest.fixture
def temp_metrics_path():
    """Create a temporary directory for metrics logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "metrics"


@pytest.fixture
def populated_registry(temp_registry_path: Path) -> ModelRegistry:
    """Create a registry populated with multiple model generations."""
    registry = ModelRegistry(temp_registry_path)
    network = ActorCriticNetwork(observation_size=114, action_size=27)
    hyperparams = {"learning_rate": 3e-4, "gamma": 0.99}

    # Generation 0 - trained against baseline
    metrics_gen0 = TrainingMetrics(
        total_episodes=500,
        total_timesteps=50000,
        average_reward=25.0,
        average_episode_length=350.0,
        win_rate=0.35,
        average_kills=1.2,
        average_deaths=1.8,
        kills_deaths_ratio=0.67,
    )
    registry.register_model(
        model=network,
        generation=0,
        opponent_model_id=None,
        training_metrics=metrics_gen0,
        hyperparameters=hyperparams,
        training_duration_seconds=1800.0,
    )

    # Generation 1 - trained against gen 0
    metrics_gen1 = TrainingMetrics(
        total_episodes=750,
        total_timesteps=75000,
        average_reward=45.0,
        average_episode_length=450.0,
        win_rate=0.55,
        average_kills=2.0,
        average_deaths=1.5,
        kills_deaths_ratio=1.33,
    )
    registry.register_model(
        model=network,
        generation=1,
        opponent_model_id="ppo_gen_000",
        training_metrics=metrics_gen1,
        hyperparameters=hyperparams,
        training_duration_seconds=2700.0,
    )

    # Generation 2 - trained against gen 1
    metrics_gen2 = TrainingMetrics(
        total_episodes=1000,
        total_timesteps=100000,
        average_reward=65.0,
        average_episode_length=550.0,
        win_rate=0.72,
        average_kills=3.2,
        average_deaths=1.2,
        kills_deaths_ratio=2.67,
    )
    registry.register_model(
        model=network,
        generation=2,
        opponent_model_id="ppo_gen_001",
        training_metrics=metrics_gen2,
        hyperparameters=hyperparams,
        training_duration_seconds=3600.0,
    )

    return registry


class TestDashboardIntegration:
    """Integration tests for the complete dashboard workflow."""

    def test_full_dashboard_generation_workflow(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
        temp_output_path: Path,
    ):
        """Integration test: load registry -> aggregate data -> generate dashboard."""
        # Create aggregator with real registry
        aggregator = DataAggregator(registry_path=temp_registry_path)

        # Verify data aggregation
        metrics = aggregator.get_all_generation_metrics()
        assert len(metrics) == 3
        assert metrics[0].generation_id == 0
        assert metrics[1].generation_id == 1
        assert metrics[2].generation_id == 2

        # Verify K/D progression
        assert metrics[0].kill_death_ratio < metrics[1].kill_death_ratio
        assert metrics[1].kill_death_ratio < metrics[2].kill_death_ratio

        # Create visualizer and generate dashboard
        visualizer = DashboardVisualizer(
            aggregator=aggregator,
            output_dir=temp_output_path,
            title="Integration Test Dashboard",
        )

        files = visualizer.generate_all(output_format="html")

        # Verify files were created
        assert len(files) > 0
        assert temp_output_path.exists()

        # Verify main dashboard file
        dashboard_file = temp_output_path / "dashboard.html"
        assert dashboard_file.exists()

        # Verify dashboard content
        content = dashboard_file.read_text()
        assert "Integration Test Dashboard" in content
        assert "K/D Ratio" in content
        assert "Win Rate" in content
        assert "ppo_gen_000" in content
        assert "ppo_gen_001" in content
        assert "ppo_gen_002" in content

    def test_dashboard_with_png_output(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
        temp_output_path: Path,
    ):
        """Integration test: generate PNG chart files."""
        aggregator = DataAggregator(registry_path=temp_registry_path)
        visualizer = DashboardVisualizer(
            aggregator=aggregator,
            output_dir=temp_output_path,
        )

        files = visualizer.generate_all(output_format="png")

        # Verify PNG files were created
        png_files = [f for f in files if f.suffix == ".png"]
        assert len(png_files) == 3  # K/D, win rate, reward

        # Verify files have content (valid PNG)
        for png_file in png_files:
            assert png_file.exists()
            content = png_file.read_bytes()
            # PNG magic bytes
            assert content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_dashboard_with_both_formats(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
        temp_output_path: Path,
    ):
        """Integration test: generate both HTML and PNG outputs."""
        aggregator = DataAggregator(registry_path=temp_registry_path)
        visualizer = DashboardVisualizer(
            aggregator=aggregator,
            output_dir=temp_output_path,
        )

        files = visualizer.generate_all(output_format="both")

        # Verify both formats
        png_files = [f for f in files if f.suffix == ".png"]
        html_files = [f for f in files if f.suffix == ".html"]

        assert len(png_files) == 3
        assert len(html_files) >= 4  # 3 charts + summary + dashboard

    def test_dashboard_with_generation_range(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
        temp_output_path: Path,
    ):
        """Integration test: generate dashboard with filtered generations."""
        aggregator = DataAggregator(registry_path=temp_registry_path)
        visualizer = DashboardVisualizer(
            aggregator=aggregator,
            output_dir=temp_output_path,
        )

        visualizer.generate_all(
            output_format="html",
            generation_range=(0, 1),
        )

        # Verify dashboard was created
        dashboard_file = temp_output_path / "dashboard.html"
        assert dashboard_file.exists()

        content = dashboard_file.read_text()
        # Should include gen 0 and 1 in some form
        assert "ppo_gen_000" in content or "Gen 0" in content or "0.67" in content
        assert "ppo_gen_001" in content or "Gen 1" in content or "1.33" in content
        # Verify the charts are embedded (filtered data)
        assert "data:image/png;base64," in content

    def test_summary_table_data_integration(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
    ):
        """Integration test: verify summary table data format."""
        aggregator = DataAggregator(registry_path=temp_registry_path)

        table_data = aggregator.get_summary_table_data()

        assert len(table_data) == 3

        # Check first generation
        gen0 = table_data[0]
        assert gen0["Generation"] == 0
        assert gen0["Model"] == "ppo_gen_000"
        assert gen0["Opponent"] == "baseline"
        assert gen0["K/D Ratio"] == "0.67"
        assert gen0["Win Rate"] == "35.0%"

        # Check third generation
        gen2 = table_data[2]
        assert gen2["Generation"] == 2
        assert gen2["Model"] == "ppo_gen_002"
        assert gen2["Opponent"] == "ppo_gen_001"
        assert gen2["K/D Ratio"] == "2.67"
        assert gen2["Win Rate"] == "72.0%"


class TestDashboardWithMetricsLogs:
    """Integration tests for dashboard with training metrics logs."""

    def test_load_jsonl_metrics_integration(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
        temp_metrics_path: Path,
    ):
        """Integration test: load metrics from JSONL log files."""
        # Create metrics log file
        temp_metrics_path.mkdir(parents=True, exist_ok=True)
        jsonl_file = temp_metrics_path / "metrics.jsonl"

        metrics_entries = [
            {
                "tag": "episode/reward",
                "value": 10.0,
                "step": 1,
                "timestamp": "2025-01-18T10:00:00",
            },
            {
                "tag": "episode/reward",
                "value": 15.0,
                "step": 2,
                "timestamp": "2025-01-18T10:00:01",
            },
            {
                "tag": "episode/reward",
                "value": 20.0,
                "step": 3,
                "timestamp": "2025-01-18T10:00:02",
            },
            {
                "tag": "episode/length",
                "value": 450,
                "step": 1,
                "timestamp": "2025-01-18T10:00:00",
            },
            {
                "tag": "episode/kills",
                "value": 2,
                "step": 1,
                "timestamp": "2025-01-18T10:00:00",
            },
        ]

        with open(jsonl_file, "w") as f:
            for entry in metrics_entries:
                f.write(json.dumps(entry) + "\n")

        # Create aggregator with metrics directory
        aggregator = DataAggregator(
            registry_path=temp_registry_path,
            metrics_dir=temp_metrics_path,
        )

        # Load metrics from logs
        log_metrics = aggregator.load_episode_metrics_from_logs()

        assert len(log_metrics) == 5

        # Check reward progression
        reward_progression = aggregator.get_reward_progression()
        assert len(reward_progression) == 3
        assert reward_progression[0] == (1, 10.0)
        assert reward_progression[1] == (2, 15.0)
        assert reward_progression[2] == (3, 20.0)

    def test_load_csv_metrics_integration(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
        temp_metrics_path: Path,
    ):
        """Integration test: load metrics from CSV log files."""
        # Create metrics log file
        temp_metrics_path.mkdir(parents=True, exist_ok=True)
        csv_file = temp_metrics_path / "metrics.csv"

        with open(csv_file, "w") as f:
            f.write("tag,value,step,timestamp\n")
            f.write("episode/reward,12.5,1,2025-01-18T10:00:00\n")
            f.write("episode/reward,18.0,2,2025-01-18T10:00:01\n")
            f.write("episode/length,400,1,2025-01-18T10:00:00\n")

        aggregator = DataAggregator(
            registry_path=temp_registry_path,
            metrics_dir=temp_metrics_path,
        )

        log_metrics = aggregator.load_episode_metrics_from_logs()

        assert len(log_metrics) == 3
        assert log_metrics[0]["tag"] == "episode/reward"
        assert log_metrics[0]["value"] == 12.5


class TestDashboardCLIIntegration:
    """Integration tests for dashboard CLI commands."""

    def test_cli_summary_command(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
    ):
        """Integration test: run summary command programmatically."""
        from typer.testing import CliRunner

        from bot.dashboard.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["summary", "-r", str(temp_registry_path)],
        )

        assert result.exit_code == 0
        # Rich may truncate column names/values, so check for partial matches
        assert "Generation" in result.stdout or "Genera" in result.stdout
        # Check for K/D ratios which are unique to each generation
        assert "0.67" in result.stdout  # Gen 0
        assert "1.33" in result.stdout  # Gen 1
        assert "2.67" in result.stdout  # Gen 2

    def test_cli_generate_command(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
        temp_output_path: Path,
    ):
        """Integration test: run generate command programmatically."""
        from typer.testing import CliRunner

        from bot.dashboard.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate",
                "-r",
                str(temp_registry_path),
                "-o",
                str(temp_output_path),
                "-f",
                "html",
            ],
        )

        assert result.exit_code == 0
        assert "Generated" in result.stdout
        assert temp_output_path.exists()
        assert (temp_output_path / "dashboard.html").exists()

    def test_cli_compare_command(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
    ):
        """Integration test: run compare command programmatically."""
        from typer.testing import CliRunner

        from bot.dashboard.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["compare", "0", "2", "-r", str(temp_registry_path)],
        )

        assert result.exit_code == 0
        assert "Gen 0" in result.stdout
        assert "Gen 2" in result.stdout
        assert "K/D Ratio" in result.stdout

    def test_cli_generate_with_generation_range(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
        temp_output_path: Path,
    ):
        """Integration test: generate with generation range filter."""
        from typer.testing import CliRunner

        from bot.dashboard.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate",
                "-r",
                str(temp_registry_path),
                "-o",
                str(temp_output_path),
                "-g",
                "0-1",
            ],
        )

        assert result.exit_code == 0
        assert "Found 2 generation" in result.stdout

    def test_cli_error_missing_registry(self, temp_output_path: Path):
        """Integration test: error handling for missing registry."""
        from typer.testing import CliRunner

        from bot.dashboard.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["generate", "-r", "/nonexistent/path", "-o", str(temp_output_path)],
        )

        assert result.exit_code == 1
        assert "does not exist" in result.stdout

    def test_cli_error_invalid_generation_range(
        self,
        populated_registry: ModelRegistry,
        temp_registry_path: Path,
        temp_output_path: Path,
    ):
        """Integration test: error handling for invalid generation range."""
        from typer.testing import CliRunner

        from bot.dashboard.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate",
                "-r",
                str(temp_registry_path),
                "-o",
                str(temp_output_path),
                "-g",
                "invalid",
            ],
        )

        assert result.exit_code != 0


class TestEmptyRegistryHandling:
    """Integration tests for handling empty registry cases."""

    def test_empty_registry_summary(self, temp_registry_path: Path):
        """Integration test: summary with empty registry."""
        from typer.testing import CliRunner

        from bot.dashboard.cli import app

        # Create empty registry
        ModelRegistry(temp_registry_path)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["summary", "-r", str(temp_registry_path)],
        )

        assert result.exit_code == 0
        assert "No models found" in result.stdout

    def test_empty_registry_generate(
        self,
        temp_registry_path: Path,
        temp_output_path: Path,
    ):
        """Integration test: generate with empty registry."""
        from typer.testing import CliRunner

        from bot.dashboard.cli import app

        # Create empty registry
        ModelRegistry(temp_registry_path)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate",
                "-r",
                str(temp_registry_path),
                "-o",
                str(temp_output_path),
            ],
        )

        assert result.exit_code == 0
        assert "No models found" in result.stdout
