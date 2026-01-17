"""Rich console output formatting utilities."""

from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_error(message: str) -> None:
    """Print an error message in red."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_success(message: str) -> None:
    """Print a success message in green."""
    console.print(f"[bold green]Success:[/bold green] {message}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_timestamp(dt: datetime) -> str:
    """Format timestamp for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def create_model_table(models: list[Any]) -> Table:
    """Create a rich table displaying model information.

    Args:
        models: List of ModelMetadata objects

    Returns:
        Rich Table instance
    """
    table = Table(title="Registered Models")

    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Gen", justify="right", style="magenta")
    table.add_column("K/D Ratio", justify="right", style="green")
    table.add_column("Avg Reward", justify="right")
    table.add_column("Timesteps", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Created", no_wrap=True)

    for model in models:
        metrics = model.training_metrics
        table.add_row(
            model.model_id,
            str(model.generation),
            f"{metrics.kills_deaths_ratio:.2f}",
            f"{metrics.average_reward:.2f}",
            f"{metrics.total_timesteps:,}",
            format_duration(model.training_duration_seconds),
            format_timestamp(model.created_at),
        )

    return table


def create_model_detail_panel(model: Any) -> Panel:
    """Create a detailed panel for a single model.

    Args:
        model: ModelMetadata object

    Returns:
        Rich Panel instance
    """
    metrics = model.training_metrics
    arch = model.architecture

    content = f"""[bold]Model ID:[/bold] {model.model_id}
[bold]Generation:[/bold] {model.generation}
[bold]Created:[/bold] {format_timestamp(model.created_at)}
[bold]Training Duration:[/bold] {format_duration(model.training_duration_seconds)}
[bold]Opponent:[/bold] {model.opponent_model_id or "Rule-based bot"}

[bold cyan]Performance Metrics[/bold cyan]
  K/D Ratio: {metrics.kills_deaths_ratio:.3f}
  Win Rate: {metrics.win_rate:.1%}
  Avg Reward: {metrics.average_reward:.2f}
  Avg Episode Length: {metrics.average_episode_length:.1f}
  Avg Kills: {metrics.average_kills:.2f}
  Avg Deaths: {metrics.average_deaths:.2f}

[bold cyan]Training Stats[/bold cyan]
  Total Timesteps: {metrics.total_timesteps:,}
  Total Episodes: {metrics.total_episodes:,}

[bold cyan]Architecture[/bold cyan]
  Observation Size: {arch.observation_size}
  Action Size: {arch.action_size}
  Hidden Size: {arch.hidden_size}
  Actor Hidden: {arch.actor_hidden}
  Critic Hidden: {arch.critic_hidden}"""

    if model.notes:
        content += f"\n\n[bold]Notes:[/bold] {model.notes}"

    return Panel(content, title=f"[bold]{model.model_id}[/bold]", border_style="blue")


def create_training_status_panel(status: dict[str, Any]) -> Panel:
    """Create a panel showing current training status.

    Args:
        status: Dictionary with training status information

    Returns:
        Rich Panel instance
    """
    run_id = status.get("run_id", "N/A")
    state = status.get("state", "unknown")
    timesteps = status.get("timesteps", 0)
    total_timesteps = status.get("total_timesteps", 0)
    generation = status.get("generation", 0)
    start_time = status.get("start_time")
    elapsed = status.get("elapsed_seconds", 0)
    fps = status.get("fps", 0)

    # Color based on state
    state_colors = {
        "running": "green",
        "paused": "yellow",
        "completed": "blue",
        "failed": "red",
        "stopped": "red",
    }
    state_color = state_colors.get(state, "white")

    progress = (timesteps / total_timesteps * 100) if total_timesteps > 0 else 0

    content = f"""[bold]Run ID:[/bold] {run_id}
[bold]State:[/bold] [{state_color}]{state.upper()}[/{state_color}]
[bold]Generation:[/bold] {generation}

[bold cyan]Progress[/bold cyan]
  Timesteps: {timesteps:,} / {total_timesteps:,} ({progress:.1f}%)
  Elapsed: {format_duration(elapsed)}
  FPS: {fps:.1f}"""

    if start_time:
        content += f"\n  Started: {format_timestamp(start_time)}"

    return Panel(content, title="[bold]Training Status[/bold]", border_style="green")
