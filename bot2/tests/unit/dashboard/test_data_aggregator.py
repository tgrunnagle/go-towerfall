"""Unit tests for the dashboard data aggregator."""

import json
import tempfile
from pathlib import Path

import pytest

from bot.agent.network import ActorCriticNetwork
from bot.dashboard.data_aggregator import DataAggregator, _format_duration
from bot.dashboard.models import DashboardConfig
from bot.training.registry import ModelRegistry
from bot.training.registry.model_metadata import TrainingMetrics


@pytest.fixture
def sample_metrics() -> TrainingMetrics:
    """Create sample training metrics for testing."""
    return TrainingMetrics(
        total_episodes=1000,
        total_timesteps=100000,
        average_reward=50.0,
        average_episode_length=500.0,
        win_rate=0.6,
        average_kills=2.5,
        average_deaths=1.5,
        kills_deaths_ratio=1.67,
    )


@pytest.fixture
def sample_hyperparams() -> dict:
    """Create sample hyperparameters for testing."""
    return {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "clip_range": 0.2,
        "n_epochs": 10,
    }


@pytest.fixture
def sample_network() -> ActorCriticNetwork:
    """Create a sample network for testing."""
    return ActorCriticNetwork(observation_size=114, action_size=27, hidden_size=256)


@pytest.fixture
def temp_registry_path():
    """Create a temporary directory for registry storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "registry"


@pytest.fixture
def registry(temp_registry_path: Path) -> ModelRegistry:
    """Create a ModelRegistry with temporary storage."""
    return ModelRegistry(temp_registry_path)


@pytest.fixture
def populated_registry(
    registry: ModelRegistry,
    sample_network: ActorCriticNetwork,
    sample_hyperparams: dict,
) -> ModelRegistry:
    """Create a registry with multiple model generations."""
    # Generation 0 - trained against baseline
    metrics_gen0 = TrainingMetrics(
        total_episodes=1000,
        total_timesteps=100000,
        average_reward=30.0,
        average_episode_length=400.0,
        win_rate=0.4,
        average_kills=1.5,
        average_deaths=2.0,
        kills_deaths_ratio=0.75,
    )
    registry.register_model(
        model=sample_network,
        generation=0,
        opponent_model_id=None,
        training_metrics=metrics_gen0,
        hyperparameters=sample_hyperparams,
        training_duration_seconds=3600.0,
    )

    # Generation 1 - trained against gen 0
    metrics_gen1 = TrainingMetrics(
        total_episodes=1500,
        total_timesteps=150000,
        average_reward=50.0,
        average_episode_length=500.0,
        win_rate=0.6,
        average_kills=2.5,
        average_deaths=1.5,
        kills_deaths_ratio=1.67,
    )
    registry.register_model(
        model=sample_network,
        generation=1,
        opponent_model_id="ppo_gen_000",
        training_metrics=metrics_gen1,
        hyperparameters=sample_hyperparams,
        training_duration_seconds=5400.0,
    )

    # Generation 2 - trained against gen 1
    metrics_gen2 = TrainingMetrics(
        total_episodes=2000,
        total_timesteps=200000,
        average_reward=75.0,
        average_episode_length=600.0,
        win_rate=0.8,
        average_kills=3.5,
        average_deaths=1.0,
        kills_deaths_ratio=3.5,
    )
    registry.register_model(
        model=sample_network,
        generation=2,
        opponent_model_id="ppo_gen_001",
        training_metrics=metrics_gen2,
        hyperparameters=sample_hyperparams,
        training_duration_seconds=7200.0,
    )

    return registry


class TestDataAggregator:
    """Tests for DataAggregator class."""

    def test_init_with_paths(self, temp_registry_path: Path):
        """Test initializing aggregator with paths."""
        aggregator = DataAggregator(
            registry_path=temp_registry_path,
            metrics_dir=None,
        )

        assert aggregator.registry_path == temp_registry_path
        assert aggregator.metrics_dir is None

    def test_from_config(self, temp_registry_path: Path):
        """Test creating aggregator from config."""
        config = DashboardConfig(
            registry_path=str(temp_registry_path),
            metrics_dir="/some/metrics",
            output_dir="./output",
        )

        aggregator = DataAggregator.from_config(config)

        assert aggregator.registry_path == Path(temp_registry_path)
        assert aggregator.metrics_dir == Path("/some/metrics")

    def test_get_all_generation_metrics_empty_registry(
        self, temp_registry_path: Path
    ):
        """Test getting metrics from empty registry."""
        # Create empty registry
        ModelRegistry(temp_registry_path)
        aggregator = DataAggregator(registry_path=temp_registry_path)

        metrics = aggregator.get_all_generation_metrics()

        assert metrics == []

    def test_get_all_generation_metrics(
        self, populated_registry: ModelRegistry, temp_registry_path: Path
    ):
        """Test getting all generation metrics."""
        aggregator = DataAggregator(registry_path=temp_registry_path)

        metrics = aggregator.get_all_generation_metrics()

        assert len(metrics) == 3
        # Should be sorted by generation
        assert metrics[0].generation_id == 0
        assert metrics[1].generation_id == 1
        assert metrics[2].generation_id == 2

        # Check values for gen 0
        gen0 = metrics[0]
        assert gen0.model_version == "ppo_gen_000"
        assert gen0.opponent_type == "baseline"
        assert gen0.kill_death_ratio == 0.75
        assert gen0.win_rate == 0.4

        # Check values for gen 1
        gen1 = metrics[1]
        assert gen1.opponent_type == "ppo_gen_000"
        assert gen1.kill_death_ratio == 1.67

        # Check values for gen 2
        gen2 = metrics[2]
        assert gen2.opponent_type == "ppo_gen_001"
        assert gen2.kill_death_ratio == 3.5

    def test_get_all_generation_metrics_with_range(
        self, populated_registry: ModelRegistry, temp_registry_path: Path
    ):
        """Test getting metrics with generation range filter."""
        aggregator = DataAggregator(registry_path=temp_registry_path)

        # Filter to only generations 0-1
        metrics = aggregator.get_all_generation_metrics(generation_range=(0, 1))

        assert len(metrics) == 2
        assert metrics[0].generation_id == 0
        assert metrics[1].generation_id == 1

    def test_get_generation_metrics(
        self, populated_registry: ModelRegistry, temp_registry_path: Path
    ):
        """Test getting metrics for a specific generation."""
        aggregator = DataAggregator(registry_path=temp_registry_path)

        metrics = aggregator.get_generation_metrics(1)

        assert metrics is not None
        assert metrics.generation_id == 1
        assert metrics.model_version == "ppo_gen_001"

    def test_get_generation_metrics_not_found(
        self, populated_registry: ModelRegistry, temp_registry_path: Path
    ):
        """Test getting metrics for non-existent generation."""
        aggregator = DataAggregator(registry_path=temp_registry_path)

        metrics = aggregator.get_generation_metrics(99)

        assert metrics is None

    def test_get_summary_table_data(
        self, populated_registry: ModelRegistry, temp_registry_path: Path
    ):
        """Test getting summary table data."""
        aggregator = DataAggregator(registry_path=temp_registry_path)

        table_data = aggregator.get_summary_table_data()

        assert len(table_data) == 3
        assert "Generation" in table_data[0]
        assert "Model" in table_data[0]
        assert "K/D Ratio" in table_data[0]
        assert "Win Rate" in table_data[0]

        # Check formatting
        assert table_data[0]["Generation"] == 0
        assert table_data[0]["K/D Ratio"] == "0.75"
        assert table_data[0]["Win Rate"] == "40.0%"

    def test_get_summary_table_data_with_range(
        self, populated_registry: ModelRegistry, temp_registry_path: Path
    ):
        """Test getting summary table data with range filter."""
        aggregator = DataAggregator(registry_path=temp_registry_path)

        table_data = aggregator.get_summary_table_data(generation_range=(1, 2))

        assert len(table_data) == 2
        assert table_data[0]["Generation"] == 1
        assert table_data[1]["Generation"] == 2


class TestMetricsLogParsing:
    """Tests for metrics log file parsing."""

    def test_load_episode_metrics_no_dir(self, temp_registry_path: Path):
        """Test loading metrics with no metrics_dir configured."""
        ModelRegistry(temp_registry_path)
        aggregator = DataAggregator(
            registry_path=temp_registry_path,
            metrics_dir=None,
        )

        metrics = aggregator.load_episode_metrics_from_logs()

        assert metrics == []

    def test_load_episode_metrics_missing_dir(self, temp_registry_path: Path):
        """Test loading metrics when directory doesn't exist."""
        ModelRegistry(temp_registry_path)
        aggregator = DataAggregator(
            registry_path=temp_registry_path,
            metrics_dir="/nonexistent/path",
        )

        metrics = aggregator.load_episode_metrics_from_logs()

        assert metrics == []

    def test_parse_jsonl_file(self, temp_registry_path: Path):
        """Test parsing JSONL metrics file."""
        ModelRegistry(temp_registry_path)

        # Create temp metrics directory with JSONL file
        with tempfile.TemporaryDirectory() as metrics_dir:
            jsonl_path = Path(metrics_dir) / "metrics.jsonl"
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({"tag": "episode/reward", "value": 10.5, "step": 1, "timestamp": "2025-01-18T10:00:00"}) + "\n")
                f.write(json.dumps({"tag": "episode/length", "value": 500, "step": 1, "timestamp": "2025-01-18T10:00:00"}) + "\n")
                f.write(json.dumps({"tag": "episode/reward", "value": 15.0, "step": 2, "timestamp": "2025-01-18T10:00:01"}) + "\n")

            aggregator = DataAggregator(
                registry_path=temp_registry_path,
                metrics_dir=metrics_dir,
            )

            metrics = aggregator.load_episode_metrics_from_logs()

            assert len(metrics) == 3
            assert metrics[0]["tag"] == "episode/reward"
            assert metrics[0]["value"] == 10.5
            assert metrics[0]["step"] == 1

    def test_parse_csv_file(self, temp_registry_path: Path):
        """Test parsing CSV metrics file."""
        ModelRegistry(temp_registry_path)

        # Create temp metrics directory with CSV file
        with tempfile.TemporaryDirectory() as metrics_dir:
            csv_path = Path(metrics_dir) / "metrics.csv"
            with open(csv_path, "w") as f:
                f.write("tag,value,step,timestamp\n")
                f.write("episode/reward,10.5,1,2025-01-18T10:00:00\n")
                f.write("episode/length,500,1,2025-01-18T10:00:00\n")

            aggregator = DataAggregator(
                registry_path=temp_registry_path,
                metrics_dir=metrics_dir,
            )

            metrics = aggregator.load_episode_metrics_from_logs()

            assert len(metrics) == 2
            assert metrics[0]["tag"] == "episode/reward"
            assert metrics[0]["value"] == 10.5
            assert metrics[0]["step"] == 1

    def test_get_reward_progression(self, temp_registry_path: Path):
        """Test getting reward progression from logs."""
        ModelRegistry(temp_registry_path)

        with tempfile.TemporaryDirectory() as metrics_dir:
            jsonl_path = Path(metrics_dir) / "metrics.jsonl"
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({"tag": "episode/reward", "value": 10.0, "step": 1, "timestamp": "2025-01-18T10:00:00"}) + "\n")
                f.write(json.dumps({"tag": "episode/length", "value": 500, "step": 1, "timestamp": "2025-01-18T10:00:00"}) + "\n")
                f.write(json.dumps({"tag": "episode/reward", "value": 20.0, "step": 2, "timestamp": "2025-01-18T10:00:01"}) + "\n")
                f.write(json.dumps({"tag": "episode/reward", "value": 15.0, "step": 3, "timestamp": "2025-01-18T10:00:02"}) + "\n")

            aggregator = DataAggregator(
                registry_path=temp_registry_path,
                metrics_dir=metrics_dir,
            )

            progression = aggregator.get_reward_progression()

            assert len(progression) == 3
            # Should be sorted by step
            assert progression[0] == (1, 10.0)
            assert progression[1] == (2, 20.0)
            assert progression[2] == (3, 15.0)


class TestFormatDuration:
    """Tests for duration formatting helper."""

    def test_format_seconds(self):
        """Test formatting durations less than a minute."""
        assert _format_duration(30) == "30s"
        assert _format_duration(59) == "59s"

    def test_format_minutes(self):
        """Test formatting durations less than an hour."""
        assert _format_duration(60) == "1m"
        assert _format_duration(90) == "2m"
        assert _format_duration(3599) == "60m"

    def test_format_hours(self):
        """Test formatting durations of an hour or more."""
        assert _format_duration(3600) == "1h 0m"
        assert _format_duration(5400) == "1h 30m"
        assert _format_duration(7200) == "2h 0m"
        assert _format_duration(7320) == "2h 2m"
