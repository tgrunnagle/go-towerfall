"""Progress display utilities for training."""

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


class TrainingProgressDisplay:
    """Rich console display for training progress.

    Provides real-time updates with progress bars, metrics tables,
    and status information during training.

    Example:
        async with TrainingProgressDisplay(total_timesteps=1_000_000) as display:
            display.update_progress(timesteps=50000, metrics={...})
    """

    def __init__(
        self,
        total_timesteps: int,
        console: Console | None = None,
    ) -> None:
        """Initialize the progress display.

        Args:
            total_timesteps: Total timesteps for training run
            console: Rich console to use (defaults to new Console)
        """
        self.total_timesteps = total_timesteps
        self.console = console or Console()

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.fields[timesteps]:,}[/cyan] steps"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._task_id: TaskID | None = None
        self._live: Live | None = None
        self._latest_metrics: dict[str, Any] = {}

    def _build_layout(self) -> Panel:
        """Build the complete display layout."""
        # Create metrics table
        metrics_table = Table(show_header=False, box=None)
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", justify="right")

        # Add latest metrics
        if self._latest_metrics:
            metrics_table.add_row(
                "Policy Loss", f"{self._latest_metrics.get('policy_loss', 0):.4f}"
            )
            metrics_table.add_row(
                "Value Loss", f"{self._latest_metrics.get('value_loss', 0):.4f}"
            )
            metrics_table.add_row(
                "Entropy", f"{self._latest_metrics.get('entropy', 0):.4f}"
            )
            metrics_table.add_row("FPS", f"{self._latest_metrics.get('fps', 0):.1f}")
            if "avg_episode_reward" in self._latest_metrics:
                metrics_table.add_row(
                    "Avg Reward",
                    f"{self._latest_metrics.get('avg_episode_reward', 0):.2f}",
                )
            if "eval_kd_ratio" in self._latest_metrics:
                metrics_table.add_row(
                    "Eval K/D", f"{self._latest_metrics.get('eval_kd_ratio', 0):.2f}"
                )

        # Combine into panel
        from rich.layout import Layout

        layout = Layout()
        layout.split_column(
            Layout(self._progress, name="progress", size=3),
            Layout(metrics_table, name="metrics"),
        )

        return Panel(
            layout,
            title="[bold]Training Progress[/bold]",
            border_style="blue",
        )

    def start(self) -> None:
        """Start the live display."""
        self._task_id = self._progress.add_task(
            "[green]Training...",
            total=self.total_timesteps,
            timesteps=0,
        )
        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=4,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update_progress(
        self,
        timesteps: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Update the progress display.

        Args:
            timesteps: Current timestep count
            metrics: Optional dictionary of training metrics to display
        """
        if self._task_id is None:
            return

        self._progress.update(
            self._task_id,
            completed=timesteps,
            timesteps=timesteps,
        )

        if metrics:
            self._latest_metrics.update(metrics)

        if self._live:
            self._live.update(self._build_layout())

    def update_status(self, status: str) -> None:
        """Update the status description.

        Args:
            status: New status text
        """
        if self._task_id is not None:
            self._progress.update(self._task_id, description=f"[green]{status}")
            if self._live:
                self._live.update(self._build_layout())

    def show_evaluation(self, metrics: dict[str, Any]) -> None:
        """Display evaluation results.

        Args:
            metrics: Evaluation metrics dictionary
        """
        self._latest_metrics.update(metrics)
        if self._live:
            self._live.update(self._build_layout())

    def __enter__(self) -> "TrainingProgressDisplay":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.stop()


def create_simple_progress(description: str, total: int) -> Progress:
    """Create a simple progress bar for operations.

    Args:
        description: Description text
        total: Total steps

    Returns:
        Rich Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
