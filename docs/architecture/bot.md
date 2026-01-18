# bot2/ Architecture - ML Training Framework

Python-based reinforcement learning framework for training TowerFall bots using PPO (Proximal Policy Optimization).

## Directory Structure

```
bot2/src/bot/
├── agent/           # Neural network & PPO training
├── bots/            # Bot implementations (rule-based, trained)
├── cli/             # Typer CLI commands
├── client/          # Game server communication (HTTP/WebSocket)
├── config/          # Configuration models
├── gym/             # Gymnasium RL environments
├── models/          # Pydantic data models
├── observation/     # State-to-observation conversion
└── training/        # Orchestration, metrics, model registry
```

## Core Components

### Data Models (`models/`)

Pydantic v2 models for type-safe game state serialization.

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| [game_state.py](../../bot2/src/bot/models/game_state.py) | Game state parsing | `GameState`, `GameUpdate` |
| [game_objects.py](../../bot2/src/bot/models/game_objects.py) | Entity states | `PlayerState`, `ArrowState`, `BlockState` |
| [api.py](../../bot2/src/bot/models/api.py) | HTTP API models | `CreateGameRequest`, `GetGameStateResponse` |
| [websocket.py](../../bot2/src/bot/models/websocket.py) | WebSocket messages | `KeyStatusRequest`, `SpectatorUpdate` |

**Pattern**: All models use `model_config = ConfigDict(populate_by_name=True)` with field aliases for snake_case/camelCase mapping.

### Game Client (`client/`)

Async communication with the Go backend.

- [game_client.py](../../bot2/src/bot/client/game_client.py) - Unified client with two modes:
  - **REST mode**: Synchronous state polling for training
  - **WebSocket mode**: Real-time updates for live play
- [http_client.py](../../bot2/src/bot/client/http_client.py) - Async HTTP with retry logic

**Key endpoints**: `/api/createGame`, `/api/joinGame`, `/api/rooms/{id}/state`, `/api/rooms/{id}/players/{pid}/actions`

### Action Space (`actions.py`)

[actions.py](../../bot2/src/bot/actions.py) defines 27 discrete actions:

- **Movement (8)**: MOVE_LEFT/RIGHT PRESS/RELEASE, JUMP PRESS/RELEASE, DIVE PRESS/RELEASE
- **Aiming (16)**: AIM_0 through AIM_15 (16 angular buckets)
- **Shooting (2)**: SHOOT_START, SHOOT_RELEASE
- **No-op (1)**: NO_OP

Key functions: `execute_action()`, `aim_action_to_radians()`, `radians_to_aim_action()`

### Neural Network (`agent/`)

| Module | Purpose |
|--------|---------|
| [network.py](../../bot2/src/bot/agent/network.py) | `ActorCriticNetwork` - shared features + actor/critic heads |
| [ppo_trainer.py](../../bot2/src/bot/agent/ppo_trainer.py) | `PPOTrainer` - PPO algorithm with clipped objectives |
| [rollout_buffer.py](../../bot2/src/bot/agent/rollout_buffer.py) | `RolloutBuffer` - stores trajectories, computes GAE advantages |
| [serialization.py](../../bot2/src/bot/agent/serialization.py) | Model checkpoint save/load |

**Network architecture** (default):
```
Input (414) → FC(256) → FC(256) → [Actor(128→27), Critic(128→1)]
```

### Gymnasium Environments (`gym/`)

| Module | Purpose |
|--------|---------|
| [towerfall_env.py](../../bot2/src/bot/gym/towerfall_env.py) | `TowerfallEnv` - single-agent Gymnasium env |
| [vectorized_env.py](../../bot2/src/bot/gym/vectorized_env.py) | `VectorizedTowerfallEnv` - parallel envs for efficient training |
| [reward.py](../../bot2/src/bot/gym/reward.py) | `StandardRewardFunction` - kill/death/timestep rewards |
| [termination.py](../../bot2/src/bot/gym/termination.py) | `TerminationTracker` - episode end conditions |
| [opponent_manager.py](../../bot2/src/bot/gym/opponent_manager.py) | `OpponentProtocol`, `RuleBasedOpponent` |
| [model_opponent.py](../../bot2/src/bot/gym/model_opponent.py) | `ModelOpponent` - trained model as opponent |

**Gym registration**: `Towerfall-v0` registered in [gym/__init__.py](../../bot2/src/bot/gym/__init__.py)

### Observation Builder (`observation/`)

[observation_space.py](../../bot2/src/bot/observation/observation_space.py) - Converts `GameState` to normalized observation vector:

```
Own player (14) + Other players (3×12) + Arrows (8×8) + Map grid (20×15) = 414 features
```

All values normalized to [-1, 1].

### Training Orchestration (`training/`)

| Module | Purpose |
|--------|---------|
| [orchestrator.py](../../bot2/src/bot/training/orchestrator.py) | `TrainingOrchestrator` - main training loop |
| [successive_trainer.py](../../bot2/src/bot/training/successive_trainer.py) | `SuccessiveTrainer` - multi-generation self-play |
| [evaluation.py](../../bot2/src/bot/training/evaluation.py) | `EvaluationManager` - model comparison |
| [server_manager.py](../../bot2/src/bot/training/server_manager.py) | Game server lifecycle |

**Model Registry** (`training/registry/`):
- [model_registry.py](../../bot2/src/bot/training/registry/model_registry.py) - versioned model storage
- [model_metadata.py](../../bot2/src/bot/training/registry/model_metadata.py) - training metrics, hyperparameters

**Metrics** (`training/metrics/`):
- [logger.py](../../bot2/src/bot/training/metrics/logger.py) - `MetricsLogger` with pluggable writers
- Writers: TensorBoard, JSON/CSV files

### Bot Implementations (`bots/`)

| Module | Purpose |
|--------|---------|
| [base_bot.py](../../bot2/src/bot/bots/base_bot.py) | `BaseBot` abstract class |
| [rule_based_bot.py](../../bot2/src/bot/bots/rule_based_bot.py) | `RuleBasedBot` - heuristic opponent for initial training |
| [shooting_utils.py](../../bot2/src/bot/bots/shooting_utils.py) | Arrow trajectory prediction |

## Data Flow

### Training Loop

```
OrchestratorConfig
    ↓
TrainingOrchestrator.setup()
    ├── VectorizedTowerfallEnv (N parallel games)
    ├── ActorCriticNetwork
    ├── PPOTrainer
    └── MetricsLogger
    ↓
For each iteration:
    1. PPOTrainer.collect_rollout() → RolloutBuffer
    2. RolloutBuffer.compute_advantages() (GAE)
    3. PPOTrainer.train() → gradient updates
    4. MetricsLogger.log_*() → TensorBoard/files
    ↓
ModelRegistry.register_model()
```

### Environment Step

```
VectorizedTowerfallEnv.step(actions)
    ├── execute_action(client, action)  # Send to game server
    ├── client.get_game_state()         # Poll new state
    ├── ObservationBuilder.build()      # State → observation
    ├── RewardFunction.calculate()      # Compute reward
    └── TerminationTracker.check()      # Check if done
    ↓
Returns: (obs, rewards, terminated, truncated, info)
```

### Successive Self-Play

```
Gen 0: Train against RuleBasedBot
    ↓ (evaluate K/D ratio)
Gen 1: Train against Gen-0 model
    ↓ (evaluate improvement)
Gen N: Train against Gen-(N-1) model
```

Promotion criteria in [successive_config.py](../../bot2/src/bot/training/successive_config.py): K/D ratio improvement threshold.

## Configuration

Hierarchical dataclass configs:

```
SuccessiveTrainingConfig
└── OrchestratorConfig
    ├── PPOConfig (lr, clip_range, epochs, etc.)
    ├── MetricsLoggerConfig
    └── TrainingGameConfig
```

## CLI Entry Points

```bash
python -m bot train start --config config.yaml
python -m bot model list
python -m bot config validate config.yaml
```

CLI defined in [cli/main.py](../../bot2/src/bot/cli/main.py) using Typer.

## Key Patterns

1. **Pydantic v2**: `model_config = ConfigDict(...)`, `Field(alias="...")`, `model_validate()`
2. **Async/await**: All I/O operations async, `async with` context managers
3. **Protocols**: `OpponentProtocol`, `VectorizedEnvironment` for duck typing
4. **Frozen dataclasses**: Immutable configuration objects

## Dependencies

- **torch**: Neural networks, GPU training
- **gymnasium**: RL environment interface
- **httpx**: Async HTTP client
- **pydantic**: Data validation
- **typer/rich**: CLI framework
- **tensorboard**: Metrics visualization

## Extending

| To add... | Implement/Modify |
|-----------|------------------|
| New reward function | Subclass in `gym/reward.py` |
| New opponent type | Implement `OpponentProtocol` |
| New observation features | Modify `ObservationBuilder` |
| New termination condition | Modify `TerminationTracker` |
| New metrics writer | Subclass `MetricsWriter` in `training/metrics/writers/` |
