# bot2 - ML Reinforcement Learning Bot

A PPO-based reinforcement learning agent for the go-towerfall game. This module provides the training infrastructure, gymnasium environment, and neural network architecture for training ML bots to play TowerFall through the game server API.

## Installation

```bash
# Install dependencies with uv
cd bot2
uv sync --dev
```

## Training CLI

The CLI provides commands to manage training runs, inspect models, and validate configuration.

### Quick Start

```bash
# Validate your configuration
uv run python -m bot.cli config validate config/training.yaml

# Start training
uv run python -m bot.cli train start --config config/training.yaml

# Check training status
uv run python -m bot.cli train status

# List trained models
uv run python -m bot.cli model list
```

### Train Commands

```bash
# Start a new training run
uv run python -m bot.cli train start --config config/training.yaml
uv run python -m bot.cli train start --timesteps 500000 --num-envs 4

# Resume from checkpoint
uv run python -m bot.cli train resume --run-id abc123
uv run python -m bot.cli train resume --checkpoint checkpoints/checkpoint_50000.pt

# Check status
uv run python -m bot.cli train status
uv run python -m bot.cli train status --run-id abc123 --verbose

# Stop training gracefully
uv run python -m bot.cli train stop --run-id abc123

# List all runs
uv run python -m bot.cli train list --limit 10 --status completed
```

### Background Training

Run training in the background to free up your terminal for other work. The CLI saves the run ID so you can check status or stop training later.

```bash
# Start training in background
uv run python -m bot.cli train start --background --config config/training.yaml

# Output:
#   Training started in background!
#   Run ID: abc12345
#   Process ID: 54321
#
#   Check status:
#     uv run python -m bot.cli train status --run-id abc12345
#
#   Stop training:
#     uv run python -m bot.cli train stop --run-id abc12345

# List all running training sessions
uv run python -m bot.cli train running

# Check status of a specific run
uv run python -m bot.cli train status --run-id abc12345

# Stop a background run gracefully
uv run python -m bot.cli train stop --run-id abc12345

# Force stop (kills immediately)
uv run python -m bot.cli train stop --run-id abc12345 --force
```

The background training process:
1. Saves training progress to `.training_runs/runs.json`
2. Updates progress as training continues
3. Can be stopped gracefully (saves checkpoint) or force-killed
4. Detects if a background process crashes and marks it as failed

### Model Commands

```bash
# List all models
uv run python -m bot.cli model list
uv run python -m bot.cli model list --best
uv run python -m bot.cli model list --generation 3

# Show model details
uv run python -m bot.cli model show ppo_gen_003

# Compare models
uv run python -m bot.cli model compare ppo_gen_002 ppo_gen_003

# Evaluate against opponent
uv run python -m bot.cli model evaluate ppo_gen_003 --episodes 100

# Export model
uv run python -m bot.cli model export ppo_gen_003 --output exported_model.pt
```

### Config Commands

```bash
# Validate configuration
uv run python -m bot.cli config validate config/training.yaml --verbose

# Generate config from preset
uv run python -m bot.cli config generate config/new.yaml --preset quick  # 50k steps
uv run python -m bot.cli config generate config/new.yaml --preset default  # 1M steps
uv run python -m bot.cli config generate config/new.yaml --preset full  # 5M steps

# Show config with syntax highlighting
uv run python -m bot.cli config show config/training.yaml

# Compare two configs
uv run python -m bot.cli config diff config/old.yaml config/new.yaml
```

## Running Training Manually

### Prerequisites

1. Start the go-towerfall game server:
   ```bash
   cd backend && go run main.go
   ```

2. (Optional) Start the frontend for visualization:
   ```bash
   task compose  # Uses docker-compose
   ```

### Basic Training Loop

```python
import torch
from bot.gym import TowerfallEnv, VectorizedTowerfallEnv, RewardConfig
from bot.agent import ActorCriticNetwork, PPOTrainer, PPOConfig

# Create vectorized environment for parallel training
env = VectorizedTowerfallEnv(
    num_envs=4,
    http_base_url="http://localhost:4000",
    tick_rate_multiplier=10.0,  # Speed up training
)

# Initialize network and trainer
network = ActorCriticNetwork()
trainer = PPOTrainer(network, PPOConfig())

# Training loop
obs, _ = env.reset()
obs = torch.as_tensor(obs, dtype=torch.float32)

for update in range(1000):
    metrics, obs = trainer.train_step(env, obs)
    print(f"Update {update}: policy_loss={metrics['policy_loss']:.4f}")

env.close()
```

### Using the Model Registry

```python
from bot.training import ModelRegistry, TrainingMetrics

registry = ModelRegistry("/path/to/registry")

# Register trained model
model_id = registry.register_model(
    model=network,
    generation=0,
    training_metrics=TrainingMetrics(
        total_episodes=1000,
        total_timesteps=1_000_000,
        average_reward=1.5,
        average_episode_length=200.0,
        win_rate=0.65,
        average_kills=2.3,
        average_deaths=1.8,
    ),
    hyperparameters={"learning_rate": 3e-4},
    training_duration_seconds=3600.0,
)

# Load best model for evaluation
network, metadata = registry.get_best_model()
```

### Configuration via YAML

```python
from bot.config import PPOConfig

# Load from file
config = PPOConfig.from_yaml("config.yaml")

# Save configuration
config.to_yaml("config_backup.yaml")
```

Example `config.yaml`:
```yaml
core:
  learning_rate: 0.0003
  clip_range: 0.2
  n_epochs: 10
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95

network:
  hidden_sizes: [64, 64]
  activation: tanh

training:
  total_timesteps: 1000000
  device: auto
```

## Development

```bash
# Run all checks
task bot2:check

# Individual commands
task bot2:lint       # Linting with ruff
task bot2:typecheck  # Type checking with ty
task bot2:test:unit  # Unit tests (no server needed)
task bot2:test       # All tests (requires server)
```

## Architecture

```
bot2/src/bot/
├── cli/             # Command-line interface
│   ├── main.py          # Main Typer app entry point
│   ├── run_tracker.py   # Persistent training run tracking
│   ├── commands/        # Subcommand implementations
│   │   ├── train.py     # train start|resume|status|stop|list
│   │   ├── model.py     # model list|show|compare|evaluate|export
│   │   └── config.py    # config validate|generate|show|diff
│   └── utils/           # Output formatting and progress display
├── agent/           # Neural network and PPO training
│   ├── network.py       # ActorCriticNetwork (shared features, actor/critic heads)
│   ├── ppo_trainer.py   # PPOTrainer with rollout collection and updates
│   ├── rollout_buffer.py# Experience buffer with GAE computation
│   └── serialization.py # Model save/load utilities
├── gym/             # Gymnasium environment
│   ├── towerfall_env.py    # Main TowerfallEnv class
│   ├── vectorized_env.py   # Parallel environment wrapper
│   ├── reward.py           # Configurable reward function
│   ├── termination.py      # Episode termination logic
│   └── opponent_manager.py # Opponent handling (rule-based, none)
├── training/        # Training infrastructure
│   ├── orchestrator.py     # Training pipeline coordinator
│   ├── server_manager.py   # Game server lifecycle management
│   └── registry/           # Model versioning and storage
├── config/          # Hyperparameter configuration
│   └── ppo_config.py       # Pydantic-validated PPO config
├── models/          # Game state data models
├── client/          # HTTP/WebSocket game client
├── observation/     # Observation space encoding
├── bots/            # Rule-based bot implementations
└── actions.py       # Discrete action space (27 actions)
```

### Key Components

**ActorCriticNetwork**: Shared-feature MLP with separate actor (policy) and critic (value) heads. Default architecture: 414 inputs → 256 → 256 → (128 → 27 actions | 128 → 1 value).

**TowerfallEnv**: Gymnasium-compatible environment wrapping the game server. Supports configurable rewards, termination conditions, and opponent types.

**PPOTrainer**: Complete PPO implementation with:
- GAE advantage estimation
- Clipped surrogate objective
- Value function clipping (optional)
- Gradient norm clipping
- Minibatch training

**ModelRegistry**: Stores trained models with metadata, supporting:
- Generation-based versioning (`ppo_gen_000`, `ppo_gen_001`, ...)
- Performance comparison (kills/deaths ratio)
- Checkpoint data for training resumption
- Thread-safe file operations

### Action Space

27 discrete actions organized into categories:
- **Movement (8)**: Press/release for left, right, jump, dive
- **Aim Direction (16)**: 16 directional buckets (22.5° each)
- **Shooting (2)**: Start drawing bow, release arrow
- **No-Op (1)**: Do nothing

### Observation Space

414-dimensional normalized vector including:
- Player state (position, velocity, health, arrows)
- Enemy positions and states
- Projectile tracking
- Pickup locations
- Map encoding

### Reward Function

Default reward shaping:
- **Kill**: +1.0 per kill
- **Death**: -1.0 per death
- **Timestep**: -0.001 per step (encourages efficiency)

Configurable via `RewardConfig`:
```python
from bot.gym import RewardConfig, TowerfallEnv

env = TowerfallEnv(
    reward_config=RewardConfig(
        kill_reward=2.0,
        death_penalty=-1.5,
        timestep_penalty=-0.002,
    )
)
```
