"""Train subcommands for managing training runs."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from bot.cli.run_tracker import RunTracker, TrainingRun
from bot.cli.utils.output import (
    console,
    create_training_status_panel,
    format_duration,
    format_timestamp,
    print_error,
    print_success,
    print_warning,
)
from bot.cli.utils.progress import TrainingProgressDisplay
from bot.training.orchestrator import TrainingOrchestrator
from bot.training.orchestrator_config import OrchestratorConfig

app = typer.Typer(no_args_is_help=True)

# Default paths
DEFAULT_RUNS_DIR = "./.training_runs"
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_REGISTRY_PATH = "./model_registry"


def get_tracker() -> RunTracker:
    """Get the run tracker instance."""
    return RunTracker(DEFAULT_RUNS_DIR)


def _spawn_background_training(run_id: str) -> int:
    """Spawn training in a background process.

    Args:
        run_id: The run ID to resume training for

    Returns:
        Process ID of the spawned background process
    """
    # Build command to resume the run (which has config already saved)
    cmd = [
        sys.executable,
        "-m",
        "bot.cli",
        "train",
        "run-background",
        "--run-id",
        run_id,
    ]

    # Spawn detached process
    if sys.platform == "win32":
        # Windows: use CREATE_NEW_PROCESS_GROUP and DETACHED_PROCESS
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    else:
        # Unix: use start_new_session to detach
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    return process.pid


@app.command("start")
def start(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    num_envs: Annotated[
        int | None,
        typer.Option("--num-envs", "-n", help="Number of parallel environments"),
    ] = None,
    total_timesteps: Annotated[
        int | None,
        typer.Option("--timesteps", "-t", help="Total training timesteps"),
    ] = None,
    server_url: Annotated[
        str | None,
        typer.Option("--server", "-s", help="Game server URL"),
    ] = None,
    opponent: Annotated[
        str | None,
        typer.Option("--opponent", "-o", help="Opponent model ID (or 'rule-based')"),
    ] = None,
    checkpoint_dir: Annotated[
        str,
        typer.Option("--checkpoint-dir", help="Directory for checkpoints"),
    ] = DEFAULT_CHECKPOINT_DIR,
    registry_path: Annotated[
        str,
        typer.Option("--registry", help="Path to model registry"),
    ] = DEFAULT_REGISTRY_PATH,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
    background: Annotated[
        bool,
        typer.Option("--background", "-b", help="Run training in background"),
    ] = False,
) -> None:
    """Start a new training run.

    Load configuration from a YAML file and/or override with command-line options.
    CLI options take precedence over config file values.

    Examples:
        uv run python -m bot.cli train start --config config/training.yaml
        uv run python -m bot.cli train start --timesteps 500000 --num-envs 4
    """
    # Load configuration
    if config is not None:
        try:
            orch_config = OrchestratorConfig.from_yaml(config)
            console.print(f"Loaded configuration from [cyan]{config}[/cyan]")
        except Exception as e:
            print_error(f"Failed to load config: {e}")
            raise typer.Exit(1)
    else:
        orch_config = OrchestratorConfig()

    # Apply CLI overrides by building a config dict and using from_dict
    config_dict = orch_config.to_dict()

    if num_envs is not None:
        config_dict["num_envs"] = num_envs
    if total_timesteps is not None:
        config_dict["total_timesteps"] = total_timesteps
    if server_url is not None:
        config_dict["game_server_url"] = server_url
    if opponent is not None and opponent != "rule-based":
        config_dict["opponent_model_id"] = opponent
    if seed is not None:
        config_dict["seed"] = seed

    # Update paths
    config_dict["checkpoint_dir"] = checkpoint_dir
    config_dict["registry_path"] = registry_path

    orch_config = OrchestratorConfig.from_dict(config_dict)

    # Create run tracking
    tracker = get_tracker()
    run = tracker.create_run(
        config_path=str(config) if config else None,
        config_snapshot=orch_config.to_dict(),
        checkpoint_dir=checkpoint_dir,
        registry_path=registry_path,
        total_timesteps=orch_config.total_timesteps,
    )

    console.print(f"\n[bold]Training Run:[/bold] {run.run_id}")
    console.print(f"[bold]Timesteps:[/bold] {orch_config.total_timesteps:,}")
    console.print(f"[bold]Environments:[/bold] {orch_config.num_envs}")
    console.print(f"[bold]Server:[/bold] {orch_config.game_server_url}")
    console.print()

    if background:
        # Spawn training in a background process
        pid = _spawn_background_training(run.run_id)
        run.start(pid=pid)
        tracker.save_run(run)

        print_success("Training started in background!")
        console.print(f"  Run ID: [cyan]{run.run_id}[/cyan]")
        console.print(f"  Process ID: [dim]{pid}[/dim]")
        console.print()
        console.print("[bold]Check status:[/bold]")
        console.print(f"  uv run python -m bot.cli train status --run-id {run.run_id}")
        console.print()
        console.print("[bold]Stop training:[/bold]")
        console.print(f"  uv run python -m bot.cli train stop --run-id {run.run_id}")
        return

    # Run training in foreground
    try:
        asyncio.run(_run_training(orch_config, run, tracker))
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        run.stop()
        tracker.save_run(run)
        raise typer.Exit(0)


async def _run_training(
    config: OrchestratorConfig,
    run: TrainingRun,
    tracker: RunTracker,
) -> None:
    """Run the training loop with progress display."""
    run.start()
    tracker.save_run(run)

    orchestrator: TrainingOrchestrator | None = None
    display: TrainingProgressDisplay | None = None

    # Setup signal handler for graceful shutdown
    shutdown_requested = False

    def handle_signal(signum: int, frame: Any) -> None:
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            console.print(
                "\n[yellow]Shutdown requested, finishing current step...[/yellow]"
            )
            if orchestrator:
                orchestrator.stop()

    # Register signal handlers
    original_sigint = signal.signal(signal.SIGINT, handle_signal)
    if sys.platform != "win32":
        original_sigterm = signal.signal(signal.SIGTERM, handle_signal)

    try:
        display = TrainingProgressDisplay(config.total_timesteps, console)

        # Callback to update progress display
        def progress_callback(event: dict[str, Any]) -> None:
            if display is None:
                return

            event_type = event.get("type")
            metrics = event.get("metrics", {})

            if event_type == "progress":
                timesteps = metrics.get("timesteps", 0)
                display.update_progress(timesteps, metrics)

                # Update run tracking
                run.update_progress(
                    timesteps=timesteps,
                    generation=metrics.get("generation"),
                )
                tracker.save_run(run)

            elif event_type == "evaluation":
                display.show_evaluation(metrics)

            elif event_type == "checkpoint":
                checkpoint_path = event.get("path")
                if checkpoint_path:
                    run.update_progress(
                        timesteps=run.timesteps,
                        checkpoint=checkpoint_path,
                    )
                    tracker.save_run(run)

        async with TrainingOrchestrator(config) as orch:
            orchestrator = orch
            orchestrator.register_callback(progress_callback)

            with display:
                metadata = await orchestrator.train()

        # Training completed successfully
        run.complete()
        tracker.save_run(run)

        console.print()
        print_success(f"Training completed! Model: [cyan]{metadata.model_id}[/cyan]")
        console.print(f"  Generation: {metadata.generation}")
        console.print(
            f"  K/D Ratio: {metadata.training_metrics.kills_deaths_ratio:.2f}"
        )
        console.print(f"  Avg Reward: {metadata.training_metrics.average_reward:.2f}")

    except Exception as e:
        run.fail(str(e))
        tracker.save_run(run)
        print_error(f"Training failed: {e}")
        raise typer.Exit(1)

    finally:
        # Restore signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, original_sigterm)


@app.command("resume")
def resume(
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", "-r", help="Run ID to resume"),
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            "-c",
            help="Checkpoint file to resume from",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
) -> None:
    """Resume a paused or stopped training run.

    Either specify a run ID to resume from the last checkpoint,
    or specify a checkpoint file directly.

    Examples:
        uv run python -m bot.cli train resume --run-id abc123
        uv run python -m bot.cli train resume --checkpoint checkpoints/checkpoint_50000.pt
    """
    tracker = get_tracker()

    if run_id is not None:
        # Resume by run ID
        run = tracker.get_run(run_id)
        if run is None:
            print_error(f"Run '{run_id}' not found")
            raise typer.Exit(1)

        if run.state not in ("paused", "stopped", "failed"):
            print_error(f"Run '{run_id}' is not resumable (state: {run.state})")
            raise typer.Exit(1)

        if run.last_checkpoint is None:
            print_error(f"No checkpoint found for run '{run_id}'")
            raise typer.Exit(1)

        checkpoint_path = Path(run.last_checkpoint)
        config = OrchestratorConfig.from_dict(run.config_snapshot)

    elif checkpoint is not None:
        # Resume from checkpoint file
        checkpoint_path = checkpoint
        config = OrchestratorConfig()
        run = tracker.create_run(
            config_snapshot=config.to_dict(),
            total_timesteps=config.total_timesteps,
        )

    else:
        print_error("Must specify either --run-id or --checkpoint")
        raise typer.Exit(1)

    if not checkpoint_path.exists():
        print_error(f"Checkpoint not found: {checkpoint_path}")
        raise typer.Exit(1)

    console.print(f"[bold]Resuming from:[/bold] {checkpoint_path}")

    try:
        asyncio.run(_resume_training(config, checkpoint_path, run, tracker))
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        run.stop()
        tracker.save_run(run)
        raise typer.Exit(0)


async def _resume_training(
    config: OrchestratorConfig,
    checkpoint_path: Path,
    run: TrainingRun,
    tracker: RunTracker,
) -> None:
    """Resume training from a checkpoint."""
    run.resume()
    tracker.save_run(run)

    try:
        display = TrainingProgressDisplay(config.total_timesteps, console)

        def progress_callback(event: dict[str, Any]) -> None:
            event_type = event.get("type")
            metrics = event.get("metrics", {})

            if event_type == "progress":
                timesteps = metrics.get("timesteps", 0)
                display.update_progress(timesteps, metrics)
                run.update_progress(timesteps=timesteps)
                tracker.save_run(run)

        async with TrainingOrchestrator(config) as orchestrator:
            orchestrator.register_callback(progress_callback)
            orchestrator.load_checkpoint(str(checkpoint_path))

            console.print(
                f"[green]Resumed at timestep {orchestrator.total_timesteps:,}[/green]"
            )

            with display:
                display.update_progress(orchestrator.total_timesteps)
                metadata = await orchestrator.train()

        run.complete()
        tracker.save_run(run)

        console.print()
        print_success(f"Training completed! Model: [cyan]{metadata.model_id}[/cyan]")

    except Exception as e:
        run.fail(str(e))
        tracker.save_run(run)
        print_error(f"Training failed: {e}")
        raise typer.Exit(1)


@app.command("status")
def status(
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", "-r", help="Specific run ID to check"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed information"),
    ] = False,
) -> None:
    """Check status of training runs.

    Without arguments, shows the status of the most recent run.
    Use --run-id to check a specific run.

    Examples:
        uv run python -m bot.cli train status
        uv run python -m bot.cli train status --run-id abc123 --verbose
    """
    tracker = get_tracker()
    tracker.cleanup_stale_runs()

    if run_id is not None:
        run = tracker.get_run(run_id)
        if run is None:
            print_error(f"Run '{run_id}' not found")
            raise typer.Exit(1)
        runs = [run]
    else:
        runs = tracker.list_runs(limit=1)
        if not runs:
            console.print("No training runs found")
            return

    run = runs[0]

    # Calculate elapsed time
    elapsed = 0.0
    if run.start_time:
        start = datetime.fromisoformat(run.start_time)
        if run.end_time:
            end = datetime.fromisoformat(run.end_time)
        else:
            end = datetime.now(timezone.utc)
        elapsed = (end - start).total_seconds()

    fps = run.timesteps / elapsed if elapsed > 0 else 0

    status_data = {
        "run_id": run.run_id,
        "state": run.state,
        "timesteps": run.timesteps,
        "total_timesteps": run.total_timesteps,
        "generation": run.generation,
        "start_time": datetime.fromisoformat(run.start_time)
        if run.start_time
        else None,
        "elapsed_seconds": elapsed,
        "fps": fps,
    }

    console.print(create_training_status_panel(status_data))

    if verbose:
        console.print()
        console.print("[bold]Configuration:[/bold]")
        if run.config_path:
            console.print(f"  Config file: {run.config_path}")
        console.print(f"  Checkpoint dir: {run.checkpoint_dir}")
        console.print(f"  Registry path: {run.registry_path}")
        if run.last_checkpoint:
            console.print(f"  Last checkpoint: {run.last_checkpoint}")
        if run.error_message:
            console.print(f"  [red]Error: {run.error_message}[/red]")


@app.command("stop")
def stop(
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", "-r", help="Run ID to stop"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force stop without graceful shutdown"),
    ] = False,
) -> None:
    """Stop a running training run.

    Sends a stop signal to the training process. Use --force to kill immediately.

    Examples:
        uv run python -m bot.cli train stop --run-id abc123
    """
    tracker = get_tracker()

    if run_id is not None:
        run = tracker.get_run(run_id)
    else:
        run = tracker.get_active_run()

    if run is None:
        if run_id:
            print_error(f"Run '{run_id}' not found")
        else:
            print_error("No active training run found")
        raise typer.Exit(1)

    if run.state != "running":
        print_warning(f"Run '{run.run_id}' is not running (state: {run.state})")
        return

    if run.pid is not None:
        import os

        try:
            if force:
                if sys.platform == "win32":
                    os.kill(run.pid, signal.SIGTERM)
                else:
                    os.kill(run.pid, signal.SIGKILL)
                console.print(f"[red]Force killed run {run.run_id}[/red]")
            else:
                os.kill(run.pid, signal.SIGINT)
                console.print(f"[yellow]Stop signal sent to run {run.run_id}[/yellow]")
        except (OSError, ProcessLookupError):
            print_warning("Process not found - may have already stopped")

    run.stop()
    tracker.save_run(run)
    print_success(f"Run {run.run_id} marked as stopped")


@app.command("running")
def running() -> None:
    """List all currently running training sessions.

    Shows detailed information about active training runs including
    process ID, progress, and elapsed time.

    Examples:
        uv run python -m bot.cli train running
    """
    tracker = get_tracker()
    tracker.cleanup_stale_runs()

    runs = tracker.list_runs(state="running")

    if not runs:
        console.print("No training runs are currently running")
        console.print()
        console.print("[dim]Start a new training run with:[/dim]")
        console.print("  uv run python -m bot.cli train start --background")
        return

    console.print(f"[bold green]{len(runs)} running training session(s)[/bold green]")
    console.print()

    for run in runs:
        # Calculate elapsed time and progress
        elapsed = 0.0
        if run.start_time:
            start = datetime.fromisoformat(run.start_time)
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()

        fps = run.timesteps / elapsed if elapsed > 0 else 0
        progress_pct = (
            (run.timesteps / run.total_timesteps * 100)
            if run.total_timesteps > 0
            else 0
        )

        # Display run info
        console.print(f"[cyan bold]{run.run_id}[/cyan bold]")
        console.print(f"  Progress: {run.timesteps:,} / {run.total_timesteps:,} ({progress_pct:.1f}%)")
        console.print(f"  Generation: {run.generation}")
        console.print(f"  Elapsed: {format_duration(elapsed)}")
        console.print(f"  Speed: {fps:.1f} steps/sec")
        if run.pid:
            console.print(f"  PID: {run.pid}")
        if run.last_checkpoint:
            console.print(f"  Last checkpoint: {run.last_checkpoint}")
        console.print()

    console.print("[dim]Stop a run with:[/dim]")
    console.print("  uv run python -m bot.cli train stop --run-id <run_id>")


@app.command("list")
def list_runs(
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum number of runs to show"),
    ] = 10,
    state: Annotated[
        str | None,
        typer.Option(
            "--status",
            "-s",
            help="Filter by state (running, completed, failed, stopped)",
        ),
    ] = None,
) -> None:
    """List training runs.

    Shows recent training runs with their status and progress.

    Examples:
        uv run python -m bot.cli train list
        uv run python -m bot.cli train list --status completed --limit 5
    """
    tracker = get_tracker()
    tracker.cleanup_stale_runs()

    runs = tracker.list_runs(state=state, limit=limit)

    if not runs:
        console.print("No training runs found")
        return

    table = Table(title="Training Runs")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("State", justify="center")
    table.add_column("Progress", justify="right")
    table.add_column("Gen", justify="right", style="magenta")
    table.add_column("Started", no_wrap=True)
    table.add_column("Duration", justify="right")

    state_styles = {
        "running": "[green]RUNNING[/green]",
        "paused": "[yellow]PAUSED[/yellow]",
        "completed": "[blue]COMPLETED[/blue]",
        "failed": "[red]FAILED[/red]",
        "stopped": "[red]STOPPED[/red]",
        "pending": "[dim]PENDING[/dim]",
    }

    for run in runs:
        # Calculate progress percentage
        if run.total_timesteps > 0:
            progress = (
                f"{run.timesteps:,} ({run.timesteps / run.total_timesteps * 100:.1f}%)"
            )
        else:
            progress = f"{run.timesteps:,}"

        # Calculate duration
        duration = ""
        if run.start_time:
            start = datetime.fromisoformat(run.start_time)
            if run.end_time:
                end = datetime.fromisoformat(run.end_time)
            else:
                end = datetime.now(timezone.utc)
            duration = format_duration((end - start).total_seconds())
            started = format_timestamp(start)
        else:
            started = "-"

        state_display = state_styles.get(run.state, run.state)

        table.add_row(
            run.run_id,
            state_display,
            progress,
            str(run.generation),
            started,
            duration,
        )

    console.print(table)


@app.command("run-background", hidden=True)
def run_background(
    run_id: Annotated[
        str,
        typer.Option("--run-id", "-r", help="Run ID to execute"),
    ],
) -> None:
    """Internal command to run training in background.

    This command is invoked by the background spawning mechanism and
    should not be called directly by users.
    """
    tracker = get_tracker()
    run = tracker.get_run(run_id)

    if run is None:
        # Can't use print_error since we're headless
        raise typer.Exit(1)

    if run.state != "running":
        # Run was stopped or modified before background process started
        raise typer.Exit(1)

    # Load config from the run's snapshot
    config = OrchestratorConfig.from_dict(run.config_snapshot)

    # Update PID to current process (the spawned one)
    run.pid = os.getpid()
    tracker.save_run(run)

    # Run training (headless - no console output)
    try:
        asyncio.run(_run_training_background(config, run, tracker))
    except Exception as e:
        run.fail(str(e))
        tracker.save_run(run)
        raise typer.Exit(1)


async def _run_training_background(
    config: OrchestratorConfig,
    run: TrainingRun,
    tracker: RunTracker,
) -> None:
    """Run training in background mode (no console output)."""
    orchestrator: TrainingOrchestrator | None = None
    shutdown_requested = False

    def handle_signal(signum: int, frame: Any) -> None:
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            if orchestrator:
                orchestrator.stop()

    # Register signal handlers
    original_sigint = signal.signal(signal.SIGINT, handle_signal)
    if sys.platform != "win32":
        original_sigterm = signal.signal(signal.SIGTERM, handle_signal)

    try:
        def progress_callback(event: dict[str, Any]) -> None:
            event_type = event.get("type")
            metrics = event.get("metrics", {})

            if event_type == "progress":
                timesteps = metrics.get("timesteps", 0)
                run.update_progress(
                    timesteps=timesteps,
                    generation=metrics.get("generation"),
                )
                tracker.save_run(run)

            elif event_type == "checkpoint":
                checkpoint_path = event.get("path")
                if checkpoint_path:
                    run.update_progress(
                        timesteps=run.timesteps,
                        checkpoint=checkpoint_path,
                    )
                    tracker.save_run(run)

        async with TrainingOrchestrator(config) as orch:
            orchestrator = orch
            orchestrator.register_callback(progress_callback)
            await orchestrator.train()

        # Training completed successfully
        run.complete()
        tracker.save_run(run)

    except Exception as e:
        run.fail(str(e))
        tracker.save_run(run)
        raise

    finally:
        signal.signal(signal.SIGINT, original_sigint)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, original_sigterm)
