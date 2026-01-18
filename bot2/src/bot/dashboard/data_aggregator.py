"""Data aggregation for the training dashboard.

This module provides the DataAggregator class that queries the model registry
and parses training metrics logs to produce generation-level statistics.
"""

import json
import logging
from pathlib import Path

from bot.dashboard.models import DashboardConfig, GenerationMetrics
from bot.training.registry import ModelRegistry
from bot.training.registry.model_metadata import ModelMetadata

logger = logging.getLogger(__name__)


class DataAggregator:
    """Aggregates training data from the model registry and metrics logs.

    The DataAggregator queries the model registry for model metadata and
    optionally parses detailed metrics from training log files to provide
    comprehensive per-generation statistics.

    Example:
        aggregator = DataAggregator(registry_path="./model_registry")
        metrics = aggregator.get_all_generation_metrics()
        for gen in metrics:
            print(f"Gen {gen.generation_id}: K/D = {gen.kill_death_ratio:.2f}")
    """

    def __init__(
        self,
        registry_path: str | Path,
        metrics_dir: str | Path | None = None,
    ) -> None:
        """Initialize the data aggregator.

        Args:
            registry_path: Path to the model registry directory
            metrics_dir: Optional path to training metrics log directory.
                        If provided, will attempt to load detailed metrics.
        """
        self.registry_path = Path(registry_path)
        self.metrics_dir = Path(metrics_dir) if metrics_dir else None
        self._registry: ModelRegistry | None = None

    @classmethod
    def from_config(cls, config: DashboardConfig) -> "DataAggregator":
        """Create a DataAggregator from a DashboardConfig.

        Args:
            config: Dashboard configuration

        Returns:
            Configured DataAggregator instance
        """
        return cls(
            registry_path=config.registry_path,
            metrics_dir=config.metrics_dir,
        )

    @property
    def registry(self) -> ModelRegistry:
        """Lazy-load the model registry.

        Returns:
            ModelRegistry instance
        """
        if self._registry is None:
            self._registry = ModelRegistry(self.registry_path)
        return self._registry

    def get_all_generation_metrics(
        self,
        generation_range: tuple[int, int] | None = None,
    ) -> list[GenerationMetrics]:
        """Get aggregated metrics for all model generations.

        Args:
            generation_range: Optional (start, end) tuple to filter generations.
                             Both bounds are inclusive.

        Returns:
            List of GenerationMetrics sorted by generation_id
        """
        all_metadata = self.registry.list_models()

        if not all_metadata:
            logger.warning("No models found in registry at %s", self.registry_path)
            return []

        # Filter by generation range if specified
        if generation_range is not None:
            start, end = generation_range
            all_metadata = [
                m for m in all_metadata if start <= m.generation <= end
            ]

        # Convert each model's metadata to GenerationMetrics
        generation_metrics: list[GenerationMetrics] = []
        for metadata in all_metadata:
            metrics = self._metadata_to_generation_metrics(metadata)
            generation_metrics.append(metrics)

        return sorted(generation_metrics, key=lambda g: g.generation_id)

    def _metadata_to_generation_metrics(
        self,
        metadata: ModelMetadata,
    ) -> GenerationMetrics:
        """Convert model metadata to GenerationMetrics.

        Args:
            metadata: ModelMetadata from the registry

        Returns:
            GenerationMetrics with data from the metadata
        """
        training = metadata.training_metrics

        # Determine opponent type description
        if metadata.opponent_model_id is None:
            opponent_type = "baseline"
        else:
            opponent_type = metadata.opponent_model_id

        # Calculate total kills/deaths from averages and episode count
        total_kills = int(training.average_kills * training.total_episodes)
        total_deaths = int(training.average_deaths * training.total_episodes)

        return GenerationMetrics(
            generation_id=metadata.generation,
            model_version=metadata.model_id,
            opponent_type=opponent_type,
            total_episodes=training.total_episodes,
            total_kills=total_kills,
            total_deaths=total_deaths,
            kill_death_ratio=training.kills_deaths_ratio,
            win_rate=training.win_rate,
            avg_episode_reward=training.average_reward,
            avg_episode_length=training.average_episode_length,
            training_steps=training.total_timesteps,
            training_duration_seconds=metadata.training_duration_seconds,
            timestamp=metadata.created_at,
        )

    def get_generation_metrics(self, generation: int) -> GenerationMetrics | None:
        """Get metrics for a specific generation.

        Args:
            generation: Generation number to retrieve

        Returns:
            GenerationMetrics for the generation, or None if not found
        """
        try:
            model_id = ModelRegistry.MODEL_ID_FORMAT.format(generation=generation)
            metadata = self.registry.get_metadata(model_id)
            return self._metadata_to_generation_metrics(metadata)
        except Exception:
            logger.debug("Generation %d not found in registry", generation)
            return None

    def load_episode_metrics_from_logs(self) -> list[dict]:
        """Load raw episode metrics from training log files.

        Parses JSONL or CSV metrics files to extract episode-level data.
        This provides more granular data than the aggregated registry metadata.

        Returns:
            List of episode metric dictionaries with keys:
            - tag: Metric tag (e.g., "episode/reward")
            - value: Metric value
            - step: Training step
            - timestamp: ISO timestamp string
        """
        if self.metrics_dir is None:
            logger.debug("No metrics_dir configured, skipping log loading")
            return []

        if not self.metrics_dir.exists():
            logger.warning("Metrics directory does not exist: %s", self.metrics_dir)
            return []

        all_metrics: list[dict] = []

        # Look for JSONL files
        jsonl_files = list(self.metrics_dir.glob("*.jsonl"))
        for jsonl_file in jsonl_files:
            all_metrics.extend(self._parse_jsonl_file(jsonl_file))

        # Look for CSV files
        csv_files = list(self.metrics_dir.glob("*.csv"))
        for csv_file in csv_files:
            all_metrics.extend(self._parse_csv_file(csv_file))

        logger.info(
            "Loaded %d metric entries from %d files",
            len(all_metrics),
            len(jsonl_files) + len(csv_files),
        )

        return all_metrics

    def _parse_jsonl_file(self, file_path: Path) -> list[dict]:
        """Parse a JSONL metrics file.

        Args:
            file_path: Path to the JSONL file

        Returns:
            List of metric dictionaries
        """
        metrics: list[dict] = []
        try:
            with open(file_path) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        metrics.append(entry)
                    except json.JSONDecodeError as e:
                        logger.debug(
                            "Skipping invalid JSON at %s:%d: %s",
                            file_path,
                            line_num,
                            e,
                        )
        except OSError as e:
            logger.warning("Failed to read %s: %s", file_path, e)
        return metrics

    def _parse_csv_file(self, file_path: Path) -> list[dict]:
        """Parse a CSV metrics file.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of metric dictionaries
        """
        import csv

        metrics: list[dict] = []
        try:
            with open(file_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert value to float
                    try:
                        row["value"] = float(row["value"])
                        row["step"] = int(row["step"])
                        metrics.append(row)
                    except (ValueError, KeyError) as e:
                        logger.debug("Skipping invalid CSV row: %s", e)
        except OSError as e:
            logger.warning("Failed to read %s: %s", file_path, e)
        return metrics

    def get_reward_progression(self) -> list[tuple[int, float]]:
        """Get episode reward progression from metrics logs.

        Returns:
            List of (step, reward) tuples sorted by step
        """
        all_metrics = self.load_episode_metrics_from_logs()

        rewards = [
            (m["step"], m["value"])
            for m in all_metrics
            if m.get("tag") == "episode/reward"
        ]

        return sorted(rewards, key=lambda x: x[0])

    def get_summary_table_data(
        self,
        generation_range: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Get data formatted for a summary table.

        Args:
            generation_range: Optional (start, end) generation range

        Returns:
            List of dictionaries with table-friendly keys
        """
        metrics = self.get_all_generation_metrics(generation_range)

        return [
            {
                "Generation": m.generation_id,
                "Model": m.model_version,
                "Opponent": m.opponent_type,
                "K/D Ratio": f"{m.kill_death_ratio:.2f}",
                "Win Rate": f"{m.win_rate * 100:.1f}%",
                "Avg Reward": f"{m.avg_episode_reward:.1f}",
                "Episodes": m.total_episodes,
                "Training Time": _format_duration(m.training_duration_seconds),
            }
            for m in metrics
        ]


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "1h 23m", "45m 12s")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.0f}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"
