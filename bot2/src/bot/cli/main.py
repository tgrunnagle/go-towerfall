"""Main CLI application and entry point.

This module defines the main Typer application and aggregates all
command groups (train, model, config, dashboard).
"""

import typer

from bot.cli.commands import config as config_commands
from bot.cli.commands import model as model_commands
from bot.cli.commands import train as train_commands
from bot.dashboard import cli as dashboard_commands

app = typer.Typer(
    name="towerfall-train",
    help="ML Bot Training CLI for TowerFall",
    no_args_is_help=True,
    pretty_exceptions_enable=True,
)

# Add command groups
app.add_typer(train_commands.app, name="train", help="Training run management")
app.add_typer(model_commands.app, name="model", help="Model registry operations")
app.add_typer(config_commands.app, name="config", help="Configuration utilities")
app.add_typer(dashboard_commands.app, name="dashboard", help="Training metrics dashboard")


@app.callback()
def main_callback() -> None:
    """TowerFall ML Bot Training CLI.

    Use the subcommands to manage training runs, inspect models,
    and validate configuration files.
    """
    pass


if __name__ == "__main__":
    app()
