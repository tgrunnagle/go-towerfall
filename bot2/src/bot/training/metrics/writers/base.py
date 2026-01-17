"""Abstract base class for metrics writers.

This module defines the MetricsWriter interface that all concrete writer
implementations must follow.
"""

from abc import ABC, abstractmethod


class MetricsWriter(ABC):
    """Abstract base class for metrics writers.

    Writers are responsible for persisting metrics to various destinations
    such as files, TensorBoard, or other logging backends.

    Implementations must be thread-safe if used concurrently.
    """

    @abstractmethod
    def write_scalar(self, tag: str, value: float, step: int) -> None:
        """Write a single scalar metric.

        Args:
            tag: Metric name/tag (e.g., "episode/reward", "train/policy_loss")
            value: Scalar value to log
            step: Global step number for this metric
        """
        ...

    @abstractmethod
    def write_scalars(
        self, main_tag: str, tag_scalar_dict: dict[str, float], step: int
    ) -> None:
        """Write multiple related scalar metrics.

        Args:
            main_tag: Group name for the metrics (e.g., "losses", "episode")
            tag_scalar_dict: Dictionary mapping sub-tags to values
            step: Global step number for these metrics
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered data to the underlying storage.

        Should be called periodically to ensure data is persisted.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the writer and release resources.

        Implementations should flush before closing.
        """
        ...
