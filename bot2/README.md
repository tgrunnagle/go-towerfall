# bot2 - ML Reinforcement Learning Bot

A PPO-based reinforcement learning agent for the go-towerfall game. This module provides the training infrastructure, gymnasium environment, and neural network architecture for training ML bots to play TowerFall through the game server API.

## Installation

```bash
# Install dependencies with uv
cd bot2
uv sync --dev
```

## Running Training

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

## Dependencies

- Python ≥3.11
- PyTorch ≥2.7.0 (with CUDA support)
- Gymnasium ≥0.29.0
- Pydantic ≥2.5.0
- httpx, websockets, numpy, pyyaml
