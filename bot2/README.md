# bot2 - ML Reinforcement Learning Bot

A PPO-based reinforcement learning agent for the go-towerfall game. This module provides the training infrastructure, gymnasium environment, and neural network architecture for training ML bots to play TowerFall through the game server API.

For detailed architecture information, see [docs/architecture/bot.md](../docs/architecture/bot.md).

## Quick Start

### Installation

```bash
task bot2:install
```

### Running Training

First, start the game server:

```bash
task be:run
```

Then start training using the CLI:

```bash
# Validate configuration
uv run python -m bot.cli config validate config/training.yaml

# Start training
uv run python -m bot.cli train start --config config/training.yaml

# Start training in background
uv run python -m bot.cli train start --background --config config/training.yaml

# Check training status
uv run python -m bot.cli train status

# Stop training gracefully
uv run python -m bot.cli train stop --run-id <run-id>
```

### Running Tests

```bash
# Run unit tests (no server required)
task bot2:test:unit

# Run integration tests (requires running server)
task bot2:test:integration

# Run all tests including slow tests
task bot2:test:all
```

### Running Checks

```bash
# Run all checks (lint, format, typecheck, unit tests)
task bot2:check

# Individual checks
task bot2:lint        # Linting with ruff
task bot2:lint:fix    # Auto-fix lint issues
task bot2:format      # Format code with ruff
task bot2:typecheck   # Type checking with ty
```

## CLI Commands

### Training

```bash
uv run python -m bot.cli train start --config config/training.yaml  # Start training
uv run python -m bot.cli train resume --run-id <id>                 # Resume from checkpoint
uv run python -m bot.cli train status                               # Check status
uv run python -m bot.cli train stop --run-id <id>                   # Stop gracefully
uv run python -m bot.cli train list                                 # List all runs
```

### Models

```bash
uv run python -m bot.cli model list                     # List all models
uv run python -m bot.cli model show <model-id>          # Show model details
uv run python -m bot.cli model compare <id1> <id2>      # Compare models
uv run python -m bot.cli model evaluate <id>            # Evaluate model
uv run python -m bot.cli model export <id> --output model.pt  # Export model
```

### Configuration

```bash
uv run python -m bot.cli config validate config/training.yaml    # Validate config
uv run python -m bot.cli config generate config/new.yaml --preset quick  # Generate config
uv run python -m bot.cli config show config/training.yaml        # Show config
```

### Dashboard

```bash
uv run python -m bot.cli dashboard summary                       # Show generation metrics in terminal
uv run python -m bot.cli dashboard generate --output ./reports   # Generate HTML dashboard
uv run python -m bot.cli dashboard generate --format both        # Generate HTML and PNG files
uv run python -m bot.cli dashboard compare 0 1                   # Compare two generations side-by-side
```
