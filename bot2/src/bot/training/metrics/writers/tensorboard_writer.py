"""TensorBoard metrics writer implementation.

This module provides a MetricsWriter that logs metrics to TensorBoard
for real-time visualization during training.
"""

import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from bot.training.metrics.writers.base import MetricsWriter

logger = logging.getLogger(__name__)


class TensorBoardWriter(MetricsWriter):
    """Metrics writer that logs to TensorBoard.

    Uses PyTorch's SummaryWriter to log scalar metrics that can be
    visualized with `tensorboard --logdir <path>`.

    Attributes:
        log_dir: Directory for TensorBoard event files
    """

    def __init__(
        self,
        log_dir: str | Path,
        flush_secs: int = 120,
    ) -> None:
        """Initialize the TensorBoard writer.

        Args:
            log_dir: Directory for TensorBoard event files
            flush_secs: Interval in seconds for automatic flushing (default 120)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_secs,
        )
        self._closed = False

        logger.info("TensorBoardWriter initialized: %s", self.log_dir)

    def write_scalar(self, tag: str, value: float, step: int) -> None:
        """Write a single scalar metric.

        Args:
            tag: Metric name/tag (e.g., "episode/reward")
            value: Scalar value to log
            step: Global step number
        """
        if self._closed:
            return

        self._writer.add_scalar(tag, value, step)

    def write_scalars(
        self, main_tag: str, tag_scalar_dict: dict[str, float], step: int
    ) -> None:
        """Write multiple related scalar metrics.

        Args:
            main_tag: Group name for the metrics
            tag_scalar_dict: Dictionary mapping sub-tags to values
            step: Global step number
        """
        if self._closed:
            return

        self._writer.add_scalars(main_tag, tag_scalar_dict, step)

    def flush(self) -> None:
        """Flush buffered data to TensorBoard files."""
        if self._closed:
            return

        self._writer.flush()

    def close(self) -> None:
        """Close the writer and release resources."""
        if self._closed:
            return

        self._writer.close()
        self._closed = True
        logger.info("TensorBoardWriter closed: %s", self.log_dir)
