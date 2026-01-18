# bot2/ Architecture - ML Training Framework

Python-based reinforcement learning framework for training TowerFall bots using PPO (Proximal Policy Optimization).

## Project Summary

This module provides infrastructure for training neural network agents to play TowerFall via the game server API. It implements the PPO algorithm with a vectorized Gymnasium environment, supports multi-generation self-play training, and includes comprehensive metrics logging and model versioning.

**Key capabilities:**
- PPO-based RL training with configurable hyperparameters
- Parallel environment execution for efficient data collection
- Multi-generation successive training with automatic opponent promotion
- Model registry with versioning and metadata tracking
- TensorBoard integration and dashboard visualization

## Directory Structure

```
bot2/src/bot/
├── actions.py              # Discrete action space (27 actions)
├── agent/                  # Neural network and PPO training
├── bots/                   # Bot implementations (rule-based, trained)
├── cli/                    # Typer CLI commands
├── client/                 # Game server communication (HTTP/WebSocket)
├── config/                 # Configuration models (Pydantic/YAML)
├── dashboard/              # Training metrics visualization
├── gym/                    # Gymnasium RL environments
├── models/                 # Pydantic data models for game state
├── observation/            # State-to-observation conversion
└── training/               # Orchestration, metrics, model registry
    ├── metrics/            # Logging infrastructure
    └── registry/           # Model persistence and versioning
```

## Core Components

### Actions (`actions.py`)

Defines the 27 discrete actions available to agents.

| Reference | Description |
|-----------|-------------|
| [actions.py:31-78](../../bot2/src/bot/actions.py#L31-L78) | `Action` enum: 27 discrete actions |
| [actions.py:82](../../bot2/src/bot/actions.py#L82) | `ACTION_SPACE_SIZE = 27` |
| [actions.py:91-120](../../bot2/src/bot/actions.py#L91-L120) | `aim_action_to_radians()` - Convert aim action to radians |
| [actions.py:123-146](../../bot2/src/bot/actions.py#L123-L146) | `radians_to_aim_action()` - Convert radians to nearest action |
| [actions.py:201-247](../../bot2/src/bot/actions.py#L201-L247) | `execute_action()` - Execute action via GameClient |

**Action categories:**
- Movement (0-7): MOVE_LEFT/RIGHT, JUMP, DIVE (press/release pairs)
- Aim (8-23): 16 directional buckets (π/8 radians each)
- Shooting (24-25): SHOOT_START, SHOOT_RELEASE
- No-op (26): NO_OP

---

### Neural Network (`agent/`)

PPO implementation with actor-critic architecture.

#### Network Architecture (`agent/network.py`)

| Reference | Description |
|-----------|-------------|
| [network.py:17-210](../../bot2/src/bot/agent/network.py#L17-L210) | `ActorCriticNetwork` - Main neural network class |
| [network.py:45-102](../../bot2/src/bot/agent/network.py#L45-L102) | `__init__()` - Initialize shared + separate heads |
| [network.py:127-141](../../bot2/src/bot/agent/network.py#L127-L141) | `forward()` - Returns action logits and value |
| [network.py:143-178](../../bot2/src/bot/agent/network.py#L143-L178) | `get_action_and_value()` - Sample actions with entropy |

**Default architecture:** Input(414) → FC(256) → FC(256) → [Actor(128→27), Critic(128→1)]

#### PPO Trainer (`agent/ppo_trainer.py`)

| Reference | Description |
|-----------|-------------|
| [ppo_trainer.py:52-93](../../bot2/src/bot/agent/ppo_trainer.py#L52-L93) | `PPOConfig` dataclass - Hyperparameters |
| [ppo_trainer.py:96-378](../../bot2/src/bot/agent/ppo_trainer.py#L96-L378) | `PPOTrainer` - Main training class |
| [ppo_trainer.py:140-209](../../bot2/src/bot/agent/ppo_trainer.py#L140-L209) | `collect_rollout()` - Collect experience from environment |
| [ppo_trainer.py:211-323](../../bot2/src/bot/agent/ppo_trainer.py#L211-L323) | `update()` - PPO update step (multiple epochs) |
| [ppo_trainer.py:19-49](../../bot2/src/bot/agent/ppo_trainer.py#L19-L49) | `VectorizedEnvironment` Protocol - Environment interface |

#### Experience Buffer (`agent/rollout_buffer.py`)

| Reference | Description |
|-----------|-------------|
| [rollout_buffer.py:13-198](../../bot2/src/bot/agent/rollout_buffer.py#L13-L198) | `RolloutBuffer` - Trajectory storage |
| [rollout_buffer.py:41-69](../../bot2/src/bot/agent/rollout_buffer.py#L41-L69) | `create()` - Pre-allocate tensors |
| [rollout_buffer.py:71-116](../../bot2/src/bot/agent/rollout_buffer.py#L71-L116) | `compute_advantages()` - GAE advantage calculation |
| [rollout_buffer.py:148-193](../../bot2/src/bot/agent/rollout_buffer.py#L148-L193) | `get_minibatches()` - Shuffle and batch |

#### Serialization (`agent/serialization.py`)

| Reference | Description |
|-----------|-------------|
| [serialization.py:19-47](../../bot2/src/bot/agent/serialization.py#L19-L47) | `CheckpointMetadata` - Training metadata |
| [serialization.py:50-68](../../bot2/src/bot/agent/serialization.py#L50-L68) | `ModelCheckpoint` - Complete checkpoint structure |
| [serialization.py:71-95](../../bot2/src/bot/agent/serialization.py#L71-L95) | `save_model()` - Save network to disk |

---

### Data Models (`models/`)

Pydantic v2 models for type-safe game state serialization. All models use `model_config = ConfigDict(populate_by_name=True)` with field aliases for snake_case/camelCase mapping.

#### Game Constants (`models/constants.py`)

| Reference | Description |
|-----------|-------------|
| [constants.py:10-67](../../bot2/src/bot/models/constants.py#L10-L67) | `GameConstants` - Physics and game parameters |
| [constants.py:70](../../bot2/src/bot/models/constants.py#L70) | `GAME_CONSTANTS` - Module-level instance |
| [constants.py:74-80](../../bot2/src/bot/models/constants.py#L74-L80) | `ObjectTypes` - Type constants |
| [constants.py:83-108](../../bot2/src/bot/models/constants.py#L83-L108) | `StateKeys` - Short names from server |

**Key constants:** Tick rate 50 Hz, gravity 20 m/s², player speed 15 m/s, max 4 arrows, room 800×600 px

#### Game Objects (`models/game_objects.py`)

| Reference | Description |
|-----------|-------------|
| [game_objects.py:17-37](../../bot2/src/bot/models/game_objects.py#L17-L37) | `PlayerState` - Player character state |
| [game_objects.py:40-57](../../bot2/src/bot/models/game_objects.py#L40-L57) | `ArrowState` - Arrow projectile state |
| [game_objects.py:60-94](../../bot2/src/bot/models/game_objects.py#L60-L94) | `BlockState` - Static platform with bounds |

#### Game State (`models/game_state.py`)

| Reference | Description |
|-----------|-------------|
| [game_state.py:22-39](../../bot2/src/bot/models/game_state.py#L22-L39) | `GameUpdate` - WebSocket update message |
| [game_state.py:42-130](../../bot2/src/bot/models/game_state.py#L42-L130) | `GameState` - High-level state container |
| [game_state.py:65-83](../../bot2/src/bot/models/game_state.py#L65-L83) | `GameState.from_update()` - Parse GameUpdate |

#### API Models (`models/api.py`)

Request/response models for REST API: `CreateGameRequest`, `JoinGameRequest`, `GetGameStateResponse`, `BotActionRequest`, etc.

#### WebSocket Models (`models/websocket.py`)

Message types for real-time communication: `KeyStatusRequest`, `SpectatorUpdate`, `CreateGameWSRequest/Response`, etc.

---

### Game Client (`client/`)

Async communication with the Go backend.

| Reference | Description |
|-----------|-------------|
| [game_client.py:34-38](../../bot2/src/bot/client/game_client.py#L34-L38) | `ClientMode` enum - WEBSOCKET or REST mode |
| [game_client.py:47-201](../../bot2/src/bot/client/game_client.py#L47-L201) | `GameClient` - Unified HTTP/WebSocket client |
| [http_client.py](../../bot2/src/bot/client/http_client.py) | `GameHTTPClient` - Low-level REST wrapper |

**Key methods:** `create_game()`, `join_game()`, `get_game_state()`, `step()`, `send_keyboard_input()`, `send_direction()`

**Key endpoints:** `/api/createGame`, `/api/joinGame`, `/api/rooms/{id}/state`, `/api/rooms/{id}/players/{pid}/actions`

---

### Gymnasium Environments (`gym/`)

Standard RL environment interface with parallel execution support.

#### Single Environment (`gym/towerfall_env.py`)

| Reference | Description |
|-----------|-------------|
| [towerfall_env.py:27-280](../../bot2/src/bot/gym/towerfall_env.py#L27-L280) | `TowerfallEnv` - Single-agent Gymnasium env |
| [towerfall_env.py:46-141](../../bot2/src/bot/gym/towerfall_env.py#L46-L141) | `__init__()` - Configure environment |

**Spaces:** `action_space = Discrete(27)`, `observation_space = Box([-1,1], shape=(414,))`

#### Vectorized Environment (`gym/vectorized_env.py`)

| Reference | Description |
|-----------|-------------|
| [vectorized_env.py:33-350](../../bot2/src/bot/gym/vectorized_env.py#L33-L350) | `VectorizedTowerfallEnv` - Parallel environments |

Manages N parallel game instances using asyncio for concurrent execution.

#### Reward Function (`gym/reward.py`)

| Reference | Description |
|-----------|-------------|
| [reward.py:35-47](../../bot2/src/bot/gym/reward.py#L35-L47) | `RewardConfig` - Reward parameters |
| [reward.py:50-74](../../bot2/src/bot/gym/reward.py#L50-L74) | `RewardFunction` Protocol |
| [reward.py:77-130](../../bot2/src/bot/gym/reward.py#L77-L130) | `StandardRewardFunction` - Default implementation |

**Defaults:** kill_reward=1.0, death_penalty=-1.0, timestep_penalty=-0.001

#### Termination (`gym/termination.py`)

| Reference | Description |
|-----------|-------------|
| [termination.py:36-86](../../bot2/src/bot/gym/termination.py#L36-L86) | `TerminationConfig` - Episode end conditions |
| [termination.py:89-200](../../bot2/src/bot/gym/termination.py#L89-L200) | `TerminationTracker` - Track and check conditions |

**Defaults:** max_timesteps=10000, max_deaths=5, use_game_over_signal=True

#### Opponents (`gym/opponent_manager.py`, `gym/model_opponent.py`)

| Reference | Description |
|-----------|-------------|
| [opponent_manager.py:27-57](../../bot2/src/bot/gym/opponent_manager.py#L27-L57) | `OpponentProtocol` - Interface for opponents |
| [opponent_manager.py:60-200](../../bot2/src/bot/gym/opponent_manager.py#L60-L200) | `RuleBasedOpponent` - Heuristic opponent |
| [model_opponent.py](../../bot2/src/bot/gym/model_opponent.py) | `ModelOpponent` - Trained model as opponent |

**Factory:** `create_opponent()` instantiates appropriate opponent type

---

### Observation Builder (`observation/`)

Converts `GameState` to normalized observation vector.

| Reference | Description |
|-----------|-------------|
| [observation_space.py:35-95](../../bot2/src/bot/observation/observation_space.py#L35-L95) | `ObservationConfig` - Builder configuration |
| [observation_space.py:98-250](../../bot2/src/bot/observation/observation_space.py#L98-L250) | `ObservationBuilder` - State→observation conversion |
| [map_encoder.py:15-44](../../bot2/src/bot/observation/map_encoder.py#L15-L44) | `MapEncodingConfig` - Grid encoding config |
| [map_encoder.py:51-130](../../bot2/src/bot/observation/map_encoder.py#L51-L130) | `MapEncoder` - Block→occupancy grid |
| [normalizer.py](../../bot2/src/bot/observation/normalizer.py) | Normalization utilities (position, velocity, angle) |

**Observation structure (414 dims):**
- Own player: 14 dims (position, velocity, direction, health, state flags)
- Other players: 3 × 12 = 36 dims
- Arrows: 8 × 8 = 64 dims
- Map grid: 20 × 15 = 300 dims (optional occupancy grid)

All values normalized to [-1, 1].

---

### Training Orchestration (`training/`)

High-level training coordination and multi-generation self-play.

#### Main Orchestrator (`training/orchestrator.py`)

| Reference | Description |
|-----------|-------------|
| [orchestrator.py:32-400](../../bot2/src/bot/training/orchestrator.py#L32-L400) | `TrainingOrchestrator` - Main training loop |

**Key methods:** `setup()`, `train()`, `evaluate()`, `add_callback()`

#### Configuration (`training/orchestrator_config.py`)

| Reference | Description |
|-----------|-------------|
| [orchestrator_config.py:18-52](../../bot2/src/bot/training/orchestrator_config.py#L18-L52) | `MetricsLoggerConfig` |
| [orchestrator_config.py:55-130](../../bot2/src/bot/training/orchestrator_config.py#L55-L130) | `OrchestratorConfig` - Full training config |

#### Successive Training (`training/successive_trainer.py`)

| Reference | Description |
|-----------|-------------|
| [successive_trainer.py:27-52](../../bot2/src/bot/training/successive_trainer.py#L27-L52) | `GenerationResult` - Generation outcome |
| [successive_trainer.py:55-250](../../bot2/src/bot/training/successive_trainer.py#L55-L250) | `SuccessiveTrainer` - Multi-generation coordinator |
| [successive_config.py](../../bot2/src/bot/training/successive_config.py) | `SuccessiveTrainingConfig`, `PromotionCriteria` |

**Training progression:**
1. Gen 0: Train vs RuleBasedBot
2. Evaluate K/D ratio
3. If meets criteria, promote model as new opponent
4. Gen N: Train vs Gen N-1 model

#### Evaluation (`training/evaluation.py`)

| Reference | Description |
|-----------|-------------|
| [evaluation.py:16-130](../../bot2/src/bot/training/evaluation.py#L16-L130) | `EvaluationResult` - Performance metrics |
| [evaluation.py](../../bot2/src/bot/training/evaluation.py) | `EvaluationManager` - Run evaluation episodes |

#### Model Registry (`training/registry/`)

| Reference | Description |
|-----------|-------------|
| [model_registry.py:34-250](../../bot2/src/bot/training/registry/model_registry.py#L34-L250) | `ModelRegistry` - Versioned model storage |
| [model_metadata.py](../../bot2/src/bot/training/registry/model_metadata.py) | `ModelMetadata`, `NetworkArchitecture`, `TrainingMetrics` |
| [storage_backend.py](../../bot2/src/bot/training/registry/storage_backend.py) | `StorageBackend` - File-based persistence |

**Key methods:** `register_model()`, `get_model()`, `get_best_model()`, `list_models()`, `get_generation()`

**Model ID format:** `ppo_gen_XXX`

#### Metrics Logging (`training/metrics/`)

| Reference | Description |
|-----------|-------------|
| [logger.py:23-250](../../bot2/src/bot/training/metrics/logger.py#L23-L250) | `MetricsLogger` - Main logging class |
| [models.py](../../bot2/src/bot/training/metrics/models.py) | `EpisodeMetrics`, `TrainingStepMetrics`, `AggregateMetrics` |
| [aggregators.py](../../bot2/src/bot/training/metrics/aggregators.py) | `RollingAggregator` - Rolling statistics |
| [writers/base.py](../../bot2/src/bot/training/metrics/writers/base.py) | `MetricsWriter` Protocol |
| [writers/file_writer.py](../../bot2/src/bot/training/metrics/writers/file_writer.py) | `FileWriter` - JSON/CSV output |
| [writers/tensorboard_writer.py](../../bot2/src/bot/training/metrics/writers/tensorboard_writer.py) | `TensorBoardWriter` |

---

### Bot Implementations (`bots/`)

| Reference | Description |
|-----------|-------------|
| [base_bot.py:20-80](../../bot2/src/bot/bots/base_bot.py#L20-L80) | `BaseBot` ABC - Abstract base class |
| [rule_based_bot.py](../../bot2/src/bot/bots/rule_based_bot.py) | `RuleBasedBot` - Heuristic opponent for initial training |
| [shooting_utils.py](../../bot2/src/bot/bots/shooting_utils.py) | Arrow trajectory prediction utilities |

**BaseBot methods:** `update_state()`, `decide_actions()` (abstract), `get_own_player()`, `get_enemies()`

---

### Configuration (`config/`)

| Reference | Description |
|-----------|-------------|
| [ppo_config.py:14-80](../../bot2/src/bot/config/ppo_config.py#L14-L80) | `PPOCoreConfig` - Algorithm parameters |
| [ppo_config.py:83-108](../../bot2/src/bot/config/ppo_config.py#L83-L108) | `NetworkConfig` - Architecture config |
| [ppo_config.py:111-142](../../bot2/src/bot/config/ppo_config.py#L111-L142) | `TrainingConfig` - Training parameters |
| [ppo_config.py:173-236](../../bot2/src/bot/config/ppo_config.py#L173-L236) | `PPOConfig` - Complete config with YAML support |

**Configuration hierarchy:**
```
SuccessiveTrainingConfig
└── OrchestratorConfig
    ├── PPOConfig (lr, clip_range, epochs, etc.)
    ├── MetricsLoggerConfig
    └── TrainingGameConfig
```

---

### CLI (`cli/`)

| Reference | Description |
|-----------|-------------|
| [main.py:14-35](../../bot2/src/bot/cli/main.py#L14-L35) | Main Typer application entry point |
| [commands/train.py](../../bot2/src/bot/cli/commands/train.py) | Training commands (start, resume, status, stop) |
| [commands/model.py](../../bot2/src/bot/cli/commands/model.py) | Model registry commands (list, info, export) |
| [commands/config.py](../../bot2/src/bot/cli/commands/config.py) | Config commands (validate, generate, show) |
| [run_tracker.py](../../bot2/src/bot/cli/run_tracker.py) | Training run status tracking |

**Entry point:** `python -m bot.cli`

---

### Dashboard (`dashboard/`)

| Reference | Description |
|-----------|-------------|
| [cli.py](../../bot2/src/bot/dashboard/cli.py) | Dashboard CLI commands |
| [data_aggregator.py](../../bot2/src/bot/dashboard/data_aggregator.py) | Metrics aggregation from files |
| [models.py](../../bot2/src/bot/dashboard/models.py) | Dashboard data structures |
| [visualizer.py](../../bot2/src/bot/dashboard/visualizer.py) | Chart generation (Plotly-based) |

---

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
    3. PPOTrainer.update() → gradient updates
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

---

## Key Defaults

| Parameter | Value | Location |
|-----------|-------|----------|
| Tick rate | 50 Hz | [constants.py:12](../../bot2/src/bot/models/constants.py#L12) |
| Observation size | 414 dims | [observation_space.py:35](../../bot2/src/bot/observation/observation_space.py#L35) |
| Action space | 27 | [actions.py:82](../../bot2/src/bot/actions.py#L82) |
| PPO steps/rollout | 2048 | [ppo_trainer.py:72](../../bot2/src/bot/agent/ppo_trainer.py#L72) |
| PPO epochs | 10 | [ppo_trainer.py:79](../../bot2/src/bot/agent/ppo_trainer.py#L79) |
| Learning rate | 3e-4 | [ppo_config.py](../../bot2/src/bot/config/ppo_config.py) |
| Discount (γ) | 0.99 | [ppo_trainer.py:75](../../bot2/src/bot/agent/ppo_trainer.py#L75) |
| GAE lambda | 0.95 | [ppo_trainer.py:76](../../bot2/src/bot/agent/ppo_trainer.py#L76) |
| Clip range (ε) | 0.2 | [ppo_trainer.py:82](../../bot2/src/bot/agent/ppo_trainer.py#L82) |
| Num environments | 4 | [orchestrator_config.py](../../bot2/src/bot/training/orchestrator_config.py) |
| Max timesteps/episode | 10000 | [termination.py:36](../../bot2/src/bot/gym/termination.py#L36) |

---

## Key Patterns

1. **Pydantic v2**: `model_config = ConfigDict(...)`, `Field(alias="...")`, `model_validate()`
2. **Async/await**: All I/O operations async, `async with` context managers
3. **Protocols**: `OpponentProtocol`, `VectorizedEnvironment`, `RewardFunction`, `MetricsWriter` for duck typing
4. **Frozen dataclasses**: Immutable configuration objects
5. **Factory functions**: `create_opponent()`, config `from_yaml()` methods

---

## Dependencies

- **torch**: Neural networks, GPU training
- **gymnasium**: RL environment interface
- **httpx**: Async HTTP client
- **pydantic**: Data validation
- **typer/rich**: CLI framework
- **tensorboard**: Metrics visualization
- **plotly**: Dashboard charts

---

## Extension Points

| To add... | Implement/Modify |
|-----------|------------------|
| New reward function | Implement `RewardFunction` Protocol in [gym/reward.py](../../bot2/src/bot/gym/reward.py) |
| New opponent type | Implement `OpponentProtocol` in [gym/opponent_manager.py](../../bot2/src/bot/gym/opponent_manager.py) |
| New observation features | Modify `ObservationBuilder` in [observation/observation_space.py](../../bot2/src/bot/observation/observation_space.py) |
| New termination condition | Modify `TerminationTracker` in [gym/termination.py](../../bot2/src/bot/gym/termination.py) |
| New metrics writer | Implement `MetricsWriter` Protocol in [training/metrics/writers/](../../bot2/src/bot/training/metrics/writers/) |
| New CLI command | Add to appropriate module in [cli/commands/](../../bot2/src/bot/cli/commands/) |
