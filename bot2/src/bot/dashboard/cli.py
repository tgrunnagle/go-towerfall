"""CLI commands for the dashboard module.

This module provides Typer commands for generating training dashboards
from the command line.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="dashboard",
    help="Training metrics dashboard generation",
    no_args_is_help=True,
)

console = Console()


def _parse_generations(generations: str | None) -> tuple[int, int] | None:
    """Parse generation range string to tuple.

    Args:
        generations: String like "1-5" or "0-10"

    Returns:
        Tuple of (start, end) or None if not specified
    """
    if not generations:
        return None

    try:
        parts = generations.split("-")
        if len(parts) != 2:
            raise typer.BadParameter(
                f"Invalid generation range: {generations}. Expected format: START-END"
            )
        return (int(parts[0]), int(parts[1]))
    except ValueError as e:
        raise typer.BadParameter(f"Invalid generation range: {e}") from e


@app.command("generate")
def generate_dashboard(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for dashboard files",
        ),
    ] = Path("./reports"),
    registry_path: Annotated[
        Path,
        typer.Option(
            "--registry-path",
            "-r",
            help="Path to model registry directory",
        ),
    ] = Path("./model_registry"),
    metrics_dir: Annotated[
        Path | None,
        typer.Option(
            "--metrics-dir",
            "-m",
            help="Path to training metrics logs (optional)",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'html', 'png', or 'both'",
        ),
    ] = "html",
    generations: Annotated[
        str | None,
        typer.Option(
            "--generations",
            "-g",
            help="Generation range to include (e.g., '0-5')",
        ),
    ] = None,
    title: Annotated[
        str,
        typer.Option(
            "--title",
            "-t",
            help="Dashboard title",
        ),
    ] = "Training Generation Comparison",
) -> None:
    """Generate training comparison dashboard.

    Creates visualizations comparing model performance across generations,
    including K/D ratio progression, win rates, and reward curves.

    Examples:
        uv run python -m bot.dashboard.cli generate --output ./reports/

        uv run python -m bot.dashboard.cli generate -r ./model_registry -f both

        uv run python -m bot.dashboard.cli generate --generations 0-5
    """
    from bot.dashboard.data_aggregator import DataAggregator
    from bot.dashboard.visualizer import DashboardVisualizer

    # Validate output format
    if output_format not in ("html", "png", "both"):
        raise typer.BadParameter(
            f"Invalid format: {output_format}. Must be 'html', 'png', or 'both'"
        )

    # Parse generation range
    gen_range = _parse_generations(generations)

    # Check registry exists
    if not registry_path.exists():
        console.print(
            f"[red]Error:[/red] Registry path does not exist: {registry_path}"
        )
        raise typer.Exit(1)

    console.print(f"[blue]Loading models from:[/blue] {registry_path}")

    # Create aggregator and visualizer
    aggregator = DataAggregator(
        registry_path=registry_path,
        metrics_dir=metrics_dir,
    )

    # Check for models
    metrics = aggregator.get_all_generation_metrics(gen_range)
    if not metrics:
        console.print("[yellow]Warning:[/yellow] No models found in registry")
        raise typer.Exit(0)

    console.print(f"[green]Found {len(metrics)} generation(s)[/green]")

    # Generate dashboard
    visualizer = DashboardVisualizer(
        aggregator=aggregator,
        output_dir=output,
        title=title,
    )

    with console.status("[blue]Generating dashboard..."):
        generated_files = visualizer.generate_all(
            output_format=output_format,  # type: ignore[arg-type]
            generation_range=gen_range,
        )

    # Report results
    console.print()
    console.print(f"[green]Generated {len(generated_files)} file(s):[/green]")
    for file_path in generated_files:
        console.print(f"  - {file_path}")

    # Highlight main dashboard file
    dashboard_file = output / "dashboard.html"
    if dashboard_file.exists():
        console.print()
        console.print(f"[bold blue]Open dashboard:[/bold blue] {dashboard_file}")


@app.command("summary")
def show_summary(
    registry_path: Annotated[
        Path,
        typer.Option(
            "--registry-path",
            "-r",
            help="Path to model registry directory",
        ),
    ] = Path("./model_registry"),
    generations: Annotated[
        str | None,
        typer.Option(
            "--generations",
            "-g",
            help="Generation range to include (e.g., '0-5')",
        ),
    ] = None,
) -> None:
    """Show generation metrics summary in terminal.

    Displays a table of model performance metrics for all generations
    without generating any files.

    Examples:
        uv run python -m bot.dashboard.cli summary

        uv run python -m bot.dashboard.cli summary -r ./model_registry

        uv run python -m bot.dashboard.cli summary --generations 0-3
    """
    from bot.dashboard.data_aggregator import DataAggregator

    # Parse generation range
    gen_range = _parse_generations(generations)

    # Check registry exists
    if not registry_path.exists():
        console.print(
            f"[red]Error:[/red] Registry path does not exist: {registry_path}"
        )
        raise typer.Exit(1)

    # Load data
    aggregator = DataAggregator(registry_path=registry_path)
    table_data = aggregator.get_summary_table_data(gen_range)

    if not table_data:
        console.print("[yellow]No models found in registry[/yellow]")
        raise typer.Exit(0)

    # Build rich table
    table = Table(title="Generation Summary")

    # Add columns
    for header in table_data[0].keys():
        table.add_column(header, style="cyan" if header == "Generation" else None)

    # Add rows
    for row in table_data:
        table.add_row(*[str(v) for v in row.values()])

    console.print(table)


@app.command("compare")
def compare_generations(
    gen1: Annotated[int, typer.Argument(help="First generation to compare")],
    gen2: Annotated[int, typer.Argument(help="Second generation to compare")],
    registry_path: Annotated[
        Path,
        typer.Option(
            "--registry-path",
            "-r",
            help="Path to model registry directory",
        ),
    ] = Path("./model_registry"),
) -> None:
    """Compare two specific generations side-by-side.

    Shows detailed metrics comparison between two model generations.

    Examples:
        uv run python -m bot.dashboard.cli compare 0 1

        uv run python -m bot.dashboard.cli compare 2 5 -r ./model_registry
    """
    from bot.dashboard.data_aggregator import DataAggregator

    # Check registry exists
    if not registry_path.exists():
        console.print(
            f"[red]Error:[/red] Registry path does not exist: {registry_path}"
        )
        raise typer.Exit(1)

    # Load data
    aggregator = DataAggregator(registry_path=registry_path)

    metrics1 = aggregator.get_generation_metrics(gen1)
    metrics2 = aggregator.get_generation_metrics(gen2)

    if metrics1 is None:
        console.print(f"[red]Error:[/red] Generation {gen1} not found")
        raise typer.Exit(1)

    if metrics2 is None:
        console.print(f"[red]Error:[/red] Generation {gen2} not found")
        raise typer.Exit(1)

    # Build comparison table
    table = Table(title=f"Generation {gen1} vs Generation {gen2}")
    table.add_column("Metric", style="cyan")
    table.add_column(f"Gen {gen1}", justify="right")
    table.add_column(f"Gen {gen2}", justify="right")
    table.add_column("Delta", justify="right")

    # Add comparison rows
    comparisons = [
        ("K/D Ratio", metrics1.kill_death_ratio, metrics2.kill_death_ratio, ".2f"),
        ("Win Rate", metrics1.win_rate * 100, metrics2.win_rate * 100, ".1f%"),
        ("Avg Reward", metrics1.avg_episode_reward, metrics2.avg_episode_reward, ".1f"),
        (
            "Avg Episode Length",
            metrics1.avg_episode_length,
            metrics2.avg_episode_length,
            ".0f",
        ),
        ("Total Episodes", metrics1.total_episodes, metrics2.total_episodes, "d"),
        ("Total Kills", metrics1.total_kills, metrics2.total_kills, "d"),
        ("Total Deaths", metrics1.total_deaths, metrics2.total_deaths, "d"),
    ]

    for name, val1, val2, fmt in comparisons:
        delta = val2 - val1
        delta_str = f"{delta:+{fmt}}" if "%" not in fmt else f"{delta:+.1f}%"

        # Color delta based on whether higher is better
        if name in ("Total Deaths",):
            # Lower is better
            delta_style = "green" if delta < 0 else "red" if delta > 0 else ""
        else:
            # Higher is better
            delta_style = "green" if delta > 0 else "red" if delta < 0 else ""

        if "%" in fmt:
            val1_str = f"{val1:.1f}%"
            val2_str = f"{val2:.1f}%"
        elif fmt == "d":
            val1_str = str(int(val1))
            val2_str = str(int(val2))
        else:
            val1_str = f"{val1:{fmt}}"
            val2_str = f"{val2:{fmt}}"

        table.add_row(
            name,
            val1_str,
            val2_str,
            f"[{delta_style}]{delta_str}[/{delta_style}]" if delta_style else delta_str,
        )

    # Add opponent info
    table.add_row("Opponent", metrics1.opponent_type, metrics2.opponent_type, "-")

    console.print(table)


def main() -> None:
    """Entry point for the dashboard CLI."""
    app()


if __name__ == "__main__":
    main()
