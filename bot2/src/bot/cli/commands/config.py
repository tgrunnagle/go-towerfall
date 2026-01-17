"""Config subcommands for configuration management."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

from bot.cli.utils.output import console, print_error, print_success, print_warning
from bot.training.orchestrator_config import OrchestratorConfig

app = typer.Typer(no_args_is_help=True)


@app.command("validate")
def validate(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to configuration file to validate",
            exists=True,
            dir_okay=False,
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed configuration"),
    ] = False,
) -> None:
    """Validate a training configuration file.

    Checks that the YAML file is valid and all configuration values
    are within acceptable ranges.

    Examples:
        uv run python -m bot.cli config validate config/training.yaml
        uv run python -m bot.cli config validate config/training.yaml --verbose
    """
    # First, check if it's valid YAML
    try:
        with open(config_path) as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print_error(f"Invalid YAML syntax: {e}")
        raise typer.Exit(1)

    if raw_data is None:
        print_error("Configuration file is empty")
        raise typer.Exit(1)

    if not isinstance(raw_data, dict):
        print_error("Configuration must be a YAML mapping (dictionary)")
        raise typer.Exit(1)

    # Now validate as OrchestratorConfig
    errors: list[str] = []
    warnings: list[str] = []

    try:
        config = OrchestratorConfig.from_dict(raw_data)
    except TypeError as e:
        print_error(f"Invalid configuration: {e}")
        raise typer.Exit(1)
    except ValueError as e:
        print_error(f"Invalid configuration value: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to parse configuration: {e}")
        raise typer.Exit(1)

    # Additional validation checks
    if config.total_timesteps < 10000:
        warnings.append(
            f"total_timesteps ({config.total_timesteps:,}) is very low - "
            "consider at least 100,000 for meaningful training"
        )

    if config.num_envs < 2:
        warnings.append(
            f"num_envs ({config.num_envs}) is low - "
            "consider 4-8 for better training throughput"
        )

    if config.checkpoint_interval > config.total_timesteps:
        warnings.append(
            "checkpoint_interval is greater than total_timesteps - "
            "no checkpoints will be saved during training"
        )

    if config.eval_interval > config.total_timesteps:
        warnings.append(
            "eval_interval is greater than total_timesteps - "
            "no evaluations will run during training"
        )

    # Check PPO config reasonableness
    ppo = config.ppo_config
    if ppo.learning_rate > 0.01:
        warnings.append(
            f"learning_rate ({ppo.learning_rate}) is high - "
            "typical values are 1e-4 to 3e-4"
        )

    if ppo.clip_range > 0.5:
        warnings.append(f"clip_range ({ppo.clip_range}) is high - typical value is 0.2")

    # Print results
    if errors:
        for error in errors:
            print_error(error)
        raise typer.Exit(1)

    print_success(f"Configuration is valid: {config_path}")

    if warnings:
        console.print()
        for warning in warnings:
            print_warning(warning)

    if verbose:
        console.print()
        console.print("[bold]Configuration Summary:[/bold]")
        console.print(f"  Total Timesteps: {config.total_timesteps:,}")
        console.print(f"  Parallel Environments: {config.num_envs}")
        console.print(f"  Game Server: {config.game_server_url}")
        console.print(f"  Checkpoint Interval: {config.checkpoint_interval:,}")
        console.print(f"  Evaluation Interval: {config.eval_interval:,}")
        console.print(f"  Checkpoint Dir: {config.checkpoint_dir}")
        console.print(f"  Registry Path: {config.registry_path}")

        console.print()
        console.print("[bold]PPO Hyperparameters:[/bold]")
        console.print(f"  Learning Rate: {ppo.learning_rate}")
        console.print(f"  Clip Range: {ppo.clip_range}")
        console.print(f"  Num Steps: {ppo.num_steps}")
        console.print(f"  Num Epochs: {ppo.num_epochs}")
        console.print(f"  Minibatch Size: {ppo.minibatch_size}")
        console.print(f"  Gamma: {ppo.gamma}")
        console.print(f"  GAE Lambda: {ppo.gae_lambda}")
        console.print(f"  Value Coef: {ppo.value_coef}")
        console.print(f"  Entropy Coef: {ppo.entropy_coef}")

        console.print()
        console.print("[bold]Game Config:[/bold]")
        game = config.game_config
        console.print(f"  Room Name: {game.room_name}")
        console.print(f"  Map Type: {game.map_type}")
        console.print(f"  Tick Multiplier: {game.tick_multiplier}")
        console.print(f"  Max Game Duration: {game.max_game_duration_sec}s")


@app.command("generate")
def generate(
    output: Annotated[
        Path,
        typer.Argument(help="Output file path"),
    ],
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            "-p",
            help="Configuration preset (default, quick, full)",
        ),
    ] = "default",
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing file"),
    ] = False,
) -> None:
    """Generate a sample configuration file.

    Creates a YAML configuration file with reasonable defaults.
    Use presets to generate different starting configurations.

    Presets:
        default: Balanced settings for typical training
        quick: Fast training for testing (fewer timesteps)
        full: Extended training for production

    Examples:
        uv run python -m bot.cli config generate config/training.yaml
        uv run python -m bot.cli config generate config/quick.yaml --preset quick
    """
    if output.exists() and not force:
        print_error(f"File already exists: {output}")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)

    # Create config based on preset
    if preset == "quick":
        config = OrchestratorConfig(
            num_envs=2,
            total_timesteps=50_000,
            checkpoint_interval=10_000,
            eval_interval=25_000,
            eval_episodes=5,
        )
    elif preset == "full":
        config = OrchestratorConfig(
            num_envs=8,
            total_timesteps=5_000_000,
            checkpoint_interval=100_000,
            eval_interval=250_000,
            eval_episodes=20,
        )
    else:  # default
        config = OrchestratorConfig(
            num_envs=4,
            total_timesteps=1_000_000,
            checkpoint_interval=50_000,
            eval_interval=100_000,
            eval_episodes=10,
        )

    # Save configuration
    output.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(output)

    print_success(f"Generated configuration: {output}")
    console.print(f"[dim]Preset: {preset}[/dim]")


@app.command("show")
def show(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to configuration file",
            exists=True,
            dir_okay=False,
        ),
    ],
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (yaml, json)"),
    ] = "yaml",
) -> None:
    """Display a configuration file with syntax highlighting.

    Examples:
        uv run python -m bot.cli config show config/training.yaml
        uv run python -m bot.cli config show config/training.yaml --format json
    """
    try:
        config = OrchestratorConfig.from_yaml(config_path)
    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        raise typer.Exit(1)

    if output_format == "json":
        import json

        output = json.dumps(config.to_dict(), indent=2)
        from rich.syntax import Syntax

        syntax = Syntax(output, "json", theme="monokai", line_numbers=True)
    else:
        output = yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False)
        from rich.syntax import Syntax

        syntax = Syntax(output, "yaml", theme="monokai", line_numbers=True)

    console.print(syntax)


@app.command("diff")
def diff(
    config_a: Annotated[
        Path,
        typer.Argument(
            help="First configuration file",
            exists=True,
            dir_okay=False,
        ),
    ],
    config_b: Annotated[
        Path,
        typer.Argument(
            help="Second configuration file",
            exists=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """Compare two configuration files.

    Shows differences between two configuration files.

    Examples:
        uv run python -m bot.cli config diff config/old.yaml config/new.yaml
    """
    try:
        cfg_a = OrchestratorConfig.from_yaml(config_a)
        cfg_b = OrchestratorConfig.from_yaml(config_b)
    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        raise typer.Exit(1)

    dict_a = cfg_a.to_dict()
    dict_b = cfg_b.to_dict()

    # Find differences
    from rich.table import Table

    table = Table(title=f"Configuration Diff: {config_a.name} vs {config_b.name}")
    table.add_column("Setting", style="bold")
    table.add_column(config_a.name, justify="right", style="cyan")
    table.add_column(config_b.name, justify="right", style="magenta")

    def flatten_dict(d: dict, parent_key: str = "") -> dict:
        """Flatten nested dictionary."""
        items: list[tuple[str, str]] = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_a = flatten_dict(dict_a)
    flat_b = flatten_dict(dict_b)

    all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))
    has_diff = False

    for key in all_keys:
        val_a = flat_a.get(key, "[dim]N/A[/dim]")
        val_b = flat_b.get(key, "[dim]N/A[/dim]")

        if val_a != val_b:
            has_diff = True
            table.add_row(key, str(val_a), str(val_b))

    if has_diff:
        console.print(table)
    else:
        console.print("[green]Configurations are identical[/green]")
