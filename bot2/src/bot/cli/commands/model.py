"""Model subcommands for managing trained models."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from bot.agent.network import ActorCriticNetwork
from bot.cli.utils.output import (
    console,
    create_model_detail_panel,
    create_model_table,
    print_error,
    print_success,
)
from bot.training.registry import ModelNotFoundError, ModelRegistry

app = typer.Typer(no_args_is_help=True)

DEFAULT_REGISTRY_PATH = "./model_registry"


def get_registry(registry_path: str) -> ModelRegistry:
    """Get or create the model registry."""
    path = Path(registry_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return ModelRegistry(registry_path)


@app.command("list")
def list_models(
    registry_path: Annotated[
        str,
        typer.Option("--registry", "-r", help="Path to model registry"),
    ] = DEFAULT_REGISTRY_PATH,
    generation: Annotated[
        int | None,
        typer.Option("--generation", "-g", help="Filter by generation"),
    ] = None,
    best: Annotated[
        bool,
        typer.Option("--best", "-b", help="Show only the best model"),
    ] = False,
    latest: Annotated[
        bool,
        typer.Option("--latest", "-l", help="Show only the latest model"),
    ] = False,
) -> None:
    """List registered models in the registry.

    Shows all trained models with their performance metrics.
    Use filters to narrow down the list.

    Examples:
        uv run python -m bot.cli model list
        uv run python -m bot.cli model list --generation 3
        uv run python -m bot.cli model list --best
    """
    try:
        registry = get_registry(registry_path)
    except Exception as e:
        print_error(f"Failed to open registry: {e}")
        raise typer.Exit(1)

    if best:
        result = registry.get_best_model()
        if result is None:
            console.print("No models in registry")
            return
        _, metadata = result
        console.print(create_model_detail_panel(metadata))
        return

    if latest:
        result = registry.get_latest_model()
        if result is None:
            console.print("No models in registry")
            return
        _, metadata = result
        console.print(create_model_detail_panel(metadata))
        return

    models = registry.list_models()

    if not models:
        console.print("No models in registry")
        return

    if generation is not None:
        models = [m for m in models if m.generation == generation]
        if not models:
            console.print(f"No models found for generation {generation}")
            return

    console.print(create_model_table(models))


@app.command("show")
def show_model(
    model_id: Annotated[
        str,
        typer.Argument(help="Model ID to show details for"),
    ],
    registry_path: Annotated[
        str,
        typer.Option("--registry", "-r", help="Path to model registry"),
    ] = DEFAULT_REGISTRY_PATH,
) -> None:
    """Show detailed information about a specific model.

    Examples:
        uv run python -m bot.cli model show ppo_gen_003
    """
    try:
        registry = get_registry(registry_path)
        metadata = registry.get_metadata(model_id)
        console.print(create_model_detail_panel(metadata))
    except ModelNotFoundError:
        print_error(f"Model '{model_id}' not found in registry")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        raise typer.Exit(1)


@app.command("compare")
def compare_models(
    model_a: Annotated[
        str,
        typer.Argument(help="First model ID"),
    ],
    model_b: Annotated[
        str,
        typer.Argument(help="Second model ID"),
    ],
    registry_path: Annotated[
        str,
        typer.Option("--registry", "-r", help="Path to model registry"),
    ] = DEFAULT_REGISTRY_PATH,
) -> None:
    """Compare two models based on their metrics.

    Examples:
        uv run python -m bot.cli model compare ppo_gen_002 ppo_gen_003
    """
    try:
        registry = get_registry(registry_path)
        meta_a = registry.get_metadata(model_a)
        meta_b = registry.get_metadata(model_b)
    except ModelNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Create comparison table
    from rich.table import Table

    table = Table(title=f"Model Comparison: {model_a} vs {model_b}")
    table.add_column("Metric", style="bold")
    table.add_column(model_a, justify="right", style="cyan")
    table.add_column(model_b, justify="right", style="magenta")
    table.add_column("Winner", justify="center")

    metrics = [
        ("Generation", "generation", lambda x: str(x)),
        ("K/D Ratio", "kills_deaths_ratio", lambda x: f"{x:.3f}"),
        ("Win Rate", "win_rate", lambda x: f"{x:.1%}"),
        ("Avg Reward", "average_reward", lambda x: f"{x:.2f}"),
        ("Avg Kills", "average_kills", lambda x: f"{x:.2f}"),
        ("Avg Deaths", "average_deaths", lambda x: f"{x:.2f}"),
        ("Timesteps", "total_timesteps", lambda x: f"{x:,}"),
    ]

    for label, attr, fmt in metrics:
        if attr == "generation":
            val_a = getattr(meta_a, attr)
            val_b = getattr(meta_b, attr)
        else:
            val_a = getattr(meta_a.training_metrics, attr)
            val_b = getattr(meta_b.training_metrics, attr)

        # Determine winner (higher is better except for deaths)
        if attr == "average_deaths":
            winner = model_a if val_a < val_b else (model_b if val_b < val_a else "-")
        elif attr in ("generation", "total_timesteps"):
            winner = "-"
        else:
            winner = model_a if val_a > val_b else (model_b if val_b > val_a else "-")

        # Color the winner
        if winner == model_a:
            winner = f"[cyan]{winner}[/cyan]"
        elif winner == model_b:
            winner = f"[magenta]{winner}[/magenta]"

        table.add_row(label, fmt(val_a), fmt(val_b), winner)

    console.print(table)

    # Overall comparison
    is_a_better = registry.is_better_than(model_a, model_b)
    if is_a_better:
        console.print(f"\n[cyan]{model_a}[/cyan] is better overall (higher K/D ratio)")
    else:
        console.print(
            f"\n[magenta]{model_b}[/magenta] is better overall (higher K/D ratio)"
        )


@app.command("evaluate")
def evaluate_model(
    model_id: Annotated[
        str,
        typer.Argument(help="Model ID to evaluate"),
    ],
    episodes: Annotated[
        int,
        typer.Option("--episodes", "-e", help="Number of evaluation episodes"),
    ] = 10,
    opponent: Annotated[
        str | None,
        typer.Option("--opponent", "-o", help="Opponent model ID (or 'rule-based')"),
    ] = None,
    server_url: Annotated[
        str,
        typer.Option("--server", "-s", help="Game server URL"),
    ] = "http://localhost:4000",
    registry_path: Annotated[
        str,
        typer.Option("--registry", "-r", help="Path to model registry"),
    ] = DEFAULT_REGISTRY_PATH,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", help="Maximum steps per episode"),
    ] = 10000,
) -> None:
    """Evaluate a model against an opponent.

    Runs evaluation episodes and reports performance metrics.

    Examples:
        uv run python -m bot.cli model evaluate ppo_gen_003 --episodes 100
        uv run python -m bot.cli model evaluate ppo_gen_003 --opponent ppo_gen_002
    """
    try:
        registry = get_registry(registry_path)
        network, metadata = registry.get_model(model_id)
    except ModelNotFoundError:
        print_error(f"Model '{model_id}' not found in registry")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        raise typer.Exit(1)

    console.print(f"[bold]Evaluating:[/bold] {model_id}")
    console.print(f"[bold]Opponent:[/bold] {opponent or 'rule-based bot'}")
    console.print(f"[bold]Episodes:[/bold] {episodes}")
    console.print()

    try:
        metrics = asyncio.run(
            _run_evaluation(
                network=network,
                episodes=episodes,
                server_url=server_url,
                max_steps=max_steps,
            )
        )
    except Exception as e:
        print_error(f"Evaluation failed: {e}")
        raise typer.Exit(1)

    # Display results
    from rich.table import Table

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("Episodes", str(episodes))
    table.add_row("Avg Reward", f"{metrics.get('avg_reward', 0):.2f}")
    table.add_row("Std Reward", f"{metrics.get('std_reward', 0):.2f}")
    table.add_row("Avg Episode Length", f"{metrics.get('avg_length', 0):.1f}")
    table.add_row("Avg Kills", f"{metrics.get('avg_kills', 0):.2f}")
    table.add_row("Avg Deaths", f"{metrics.get('avg_deaths', 0):.2f}")
    table.add_row("K/D Ratio", f"{metrics.get('kd_ratio', 0):.2f}")

    console.print(table)


async def _run_evaluation(
    network: ActorCriticNetwork,
    episodes: int,
    server_url: str,
    max_steps: int,
) -> dict[str, float]:
    """Run evaluation episodes."""
    import numpy as np
    import torch

    from bot.gym.vectorized_env import VectorizedTowerfallEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)

    # Create evaluation environment
    env = VectorizedTowerfallEnv(
        num_envs=1,
        http_url=server_url,
        ws_url=server_url.replace("http", "ws") + "/ws",
        player_name="EvalBot",
        room_name_prefix="Evaluation",
        tick_rate_multiplier=10.0,
        max_episode_steps=max_steps,
    )

    rewards: list[float] = []
    lengths: list[int] = []
    kills: list[float] = []
    deaths: list[float] = []

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running evaluation...", total=episodes)

        for ep in range(episodes):
            obs_array, _ = env.reset()
            obs = torch.as_tensor(obs_array, dtype=torch.float32, device=device)

            episode_reward = 0.0
            episode_length = 0
            episode_kills = 0.0
            episode_deaths = 0.0
            done = False

            while not done and episode_length < max_steps:
                with torch.no_grad():
                    action, _, _, _ = network.get_action_and_value(
                        obs, deterministic=True
                    )

                next_obs, reward, terminated, truncated, info = env.step(
                    action.cpu().numpy()
                )

                episode_reward += float(reward.sum())
                episode_length += 1
                done = bool(np.any(terminated | truncated))
                obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

            rewards.append(episode_reward)
            lengths.append(episode_length)
            kills.append(episode_kills)
            deaths.append(episode_deaths)

            progress.update(task, advance=1)

    env.close()

    avg_deaths = float(np.mean(deaths)) if deaths else 1.0
    avg_kills = float(np.mean(kills)) if kills else 0.0

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_length": float(np.mean(lengths)),
        "avg_kills": avg_kills,
        "avg_deaths": avg_deaths,
        "kd_ratio": avg_kills / max(avg_deaths, 1.0),
    }


@app.command("delete")
def delete_model(
    model_id: Annotated[
        str,
        typer.Argument(help="Model ID to delete"),
    ],
    registry_path: Annotated[
        str,
        typer.Option("--registry", "-r", help="Path to model registry"),
    ] = DEFAULT_REGISTRY_PATH,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation"),
    ] = False,
) -> None:
    """Delete a model from the registry.

    Examples:
        uv run python -m bot.cli model delete ppo_gen_000
        uv run python -m bot.cli model delete ppo_gen_000 --force
    """
    try:
        registry = get_registry(registry_path)
        metadata = registry.get_metadata(model_id)
    except ModelNotFoundError:
        print_error(f"Model '{model_id}' not found in registry")
        raise typer.Exit(1)

    if not force:
        console.print(f"[bold]Model:[/bold] {model_id}")
        console.print(f"[bold]Generation:[/bold] {metadata.generation}")
        console.print(
            f"[bold]K/D Ratio:[/bold] {metadata.training_metrics.kills_deaths_ratio:.2f}"
        )
        console.print()

        confirm = typer.confirm("Are you sure you want to delete this model?")
        if not confirm:
            console.print("Cancelled")
            return

    if registry.delete_model(model_id):
        print_success(f"Model '{model_id}' deleted")
    else:
        print_error(f"Failed to delete model '{model_id}'")
        raise typer.Exit(1)


@app.command("export")
def export_model(
    model_id: Annotated[
        str,
        typer.Argument(help="Model ID to export"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ],
    registry_path: Annotated[
        str,
        typer.Option("--registry", "-r", help="Path to model registry"),
    ] = DEFAULT_REGISTRY_PATH,
) -> None:
    """Export a model to a standalone checkpoint file.

    Examples:
        uv run python -m bot.cli model export ppo_gen_003 --output exported_model.pt
    """
    import shutil

    try:
        registry = get_registry(registry_path)
        metadata = registry.get_metadata(model_id)
    except ModelNotFoundError:
        print_error(f"Model '{model_id}' not found in registry")
        raise typer.Exit(1)

    # Get the checkpoint path from the registry
    source_path = Path(registry_path) / metadata.checkpoint_path

    if not source_path.exists():
        print_error(f"Checkpoint file not found: {source_path}")
        raise typer.Exit(1)

    # Copy to output location
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output)

    print_success(f"Model exported to {output}")
