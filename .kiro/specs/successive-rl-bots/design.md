# Design Document

## Overview

The successive RL bot system will build upon the existing game client infrastructure to create a reinforcement learning framework that trains increasingly sophisticated bot players. The system uses a generational approach where each new RL model learns from and improves upon previous generations, creating a progressive improvement in bot performance.

The architecture leverages the existing `GameClient` class for game interaction and introduces new components for RL model training, evaluation, and management. The system will support multiple RL algorithms (DQN, PPO, A3C) and provide a unified interface for training successive generations of bots.

## Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Training Environment"
        TE[Training Engine]
        MM[Model Manager]
        EM[Evaluation Manager]
    end
    
    subgraph "RL Models"
        M1[Generation 1 Model]
        M2[Generation 2 Model]
        MN[Generation N Model]
    end
    
    subgraph "Game Interface"
        GC[Game Client]
        GE[Game Environment]
    end
    
    subgraph "Storage & Analytics"
        MS[Model Storage]
        MD[Metrics Database]
        RL[Replay Logger]
    end
    
    TE --> MM
    MM --> M1
    MM --> M2
    MM --> MN
    TE --> EM
    EM --> GC
    GC --> GE
    MM --> MS
    EM --> MD
    GC --> RL
```

### Component Architecture

The system consists of several key components:

1. **Rules-Based Bot Foundation**: Provides baseline behavior and initial training opponents
2. **Game Speed Controller**: Accelerates game simulation for faster training
3. **Training Spectator Interface**: Enables browser-based observation of training sessions
4. **Player Bot Integration**: Allows players to add bots to games through browser UI
5. **RL Training Engine**: Orchestrates the training process for successive model generations
6. **Model Manager**: Handles model lifecycle, versioning, and knowledge transfer between generations
7. **Game Environment Wrapper**: Adapts the existing GameClient for RL training
8. **Evaluation Framework**: Compares model performance across generations
9. **Replay System**: Records and analyzes game episodes for training and evaluation

## Components and Interfaces

### 1. Rules-Based Bot Foundation

**Purpose**: Provides intelligent baseline behavior and serves as initial training opponents

**Key Methods**:
- `analyze_game_state(state)`: Processes current game situation
- `select_action(state, available_actions)`: Chooses actions based on predefined rules
- `update_strategy(game_result)`: Adjusts rule priorities based on outcomes
- `get_difficulty_level()`: Returns current bot difficulty setting

**Rule Categories**:
- **Survival Rules**: Avoid projectiles, stay in bounds, maintain health
- **Combat Rules**: Target enemies, aim projectiles, use cover
- **Strategic Rules**: Control territory, collect power-ups, time attacks
- **Adaptive Rules**: Counter opponent patterns, adjust aggression

**Difficulty Scaling**:
- **Beginner**: Basic movement and shooting with reaction delays
- **Intermediate**: Improved aim, basic strategy, moderate reaction time
- **Advanced**: Complex strategies, quick reactions, pattern recognition
- **Expert**: Near-optimal play, advanced tactics, minimal delays

### 2. Game Speed Controller

**Purpose**: Accelerates game simulation for faster RL training while maintaining game physics accuracy

**Key Methods**:
- `set_simulation_speed(multiplier)`: Sets game speed multiplier (1x to 100x)
- `enable_headless_mode()`: Disables rendering for maximum speed
- `batch_simulate_episodes(episodes, speed)`: Runs multiple episodes in parallel
- `sync_with_realtime()`: Returns to normal speed for human evaluation
- `create_spectator_room(training_session)`: Creates spectatable training room
- `get_spectator_info(session_id)`: Returns room code and connection details for spectators

**Speed Modes**:
- **Real-time (1x)**: Normal game speed for human players and final evaluation
- **Training (10-50x)**: Accelerated speed for RL training with visual feedback
- **Headless (50-100x)**: Maximum speed with no rendering for bulk training
- **Batch Mode**: Parallel episode execution for distributed training
- **Spectator Mode**: Real-time or slowed-down playback for human observation

**Implementation Approach**:

*Client-Side Components*:
- **Training Mode Interface**: API to request accelerated game sessions
- **Direct State Access**: Bypass WebSocket for faster state retrieval
- **Batch Episode Management**: Queue and manage multiple concurrent episodes

*Server-Side Components*:
- **Game Loop Acceleration**: Configurable tick rate and frame timing
- **Headless Game Instances**: Game sessions without rendering pipeline
- **Training Room Management**: Dedicated rooms for bot training with speed controls
- **Physics Time Scaling**: Maintain accurate physics at accelerated speeds
- **Memory-Optimized Game State**: Efficient state representation for rapid access
- **Training API Endpoints**: REST/WebSocket APIs for training session management

**Server-Side Modifications Required**:
- **Backend Game Engine**: Modify game loop to support variable tick rates
- **Training Room Types**: New room type specifically for bot training
- **Headless Mode**: Game instances that run without graphics rendering
- **State Serialization**: Optimized game state format for ML consumption
- **Resource Management**: Handle multiple concurrent training sessions
- **Training APIs**: New endpoints for training session control and monitoring
- **Spectator Integration**: Enable browser-based spectating of training sessions

**Configuration**:
- Maximum safe speed multiplier based on system capabilities
- Automatic speed adjustment based on training performance
- Fallback to slower speeds if physics accuracy degrades
- Speed ramping for gradual acceleration during training
- Server resource limits for concurrent training sessions

### 3. Training Spectator Interface

**Purpose**: Enables real-time observation of bot training and evaluation through browser-based game client

**Key Methods**:
- `create_spectator_session(training_session, speed_mode)`: Creates spectatable training room
- `get_room_info(session_id)`: Returns room code, password, and connection details
- `set_spectator_speed(room_id, speed)`: Adjusts playback speed for spectators
- `broadcast_training_metrics(room_id, metrics)`: Sends training data to spectator UI
- `enable_replay_mode(episode_data)`: Allows spectating of recorded episodes

**Spectator Features**:
- **Live Training Observation**: Watch bots train in real-time or accelerated speed
- **Training Metrics Overlay**: Display current reward, episode count, model generation
- **Model Comparison Mode**: Side-by-side comparison of different model generations
- **Episode Replay**: Review specific training episodes with pause/rewind controls
- **Performance Graphs**: Real-time charts of training progress and metrics

**Integration with Existing Spectator Mode**:
- **Room Code Generation**: Training sessions automatically generate spectator room codes
- **Access Control**: Optional password protection for training sessions
- **Spectator Limits**: Configurable maximum number of concurrent spectators
- **Bandwidth Optimization**: Reduced update frequency for spectators during high-speed training

**Training Session Information Display**:
- Current model generation and algorithm
- Training episode count and duration
- Real-time performance metrics (reward, win rate, etc.)
- Bot decision-making visualization (action probabilities, state values)
- Comparison with previous generations

### 4. Player Bot Integration

**Purpose**: Enables players to add AI bots to regular game sessions through the browser-based game client

**Key Methods**:

*Game Server APIs*:
- `get_available_bots()`: Returns list of available bot types and difficulty levels
- `add_bot_to_room(room_id, bot_config)`: Requests bot addition to game room
- `remove_bot_from_room(room_id, bot_id)`: Removes a bot from the game
- `get_bot_status(room_id)`: Returns information about active bots in the room

*Python Bot Server APIs*:
- `spawn_bot(bot_type, difficulty, room_info)`: Creates new bot instance
- `configure_bot_difficulty(bot_id, difficulty)`: Adjusts bot skill level mid-game
- `terminate_bot(bot_id)`: Cleanly shuts down bot instance
- `get_bot_health()`: Returns bot server status and resource usage
- `handle_player_disconnect(room_id, player_id)`: Processes player leave events
- `cleanup_empty_rooms()`: Removes bots from rooms with no human players

**Browser UI Integration**:
- **Bot Selection Menu**: Dropdown/grid showing available bot types and generations
- **Difficulty Slider**: Easy adjustment of bot skill level (Beginner to Expert)
- **Bot Management Panel**: Add/remove bots, view bot status, adjust settings
- **Bot Information Display**: Show bot generation, algorithm, win rate statistics
- **Quick Add Buttons**: One-click addition of common bot configurations

**Bot Types Available to Players**:
- **Rules-Based Bots**: Traditional AI with configurable difficulty levels
- **RL Generation Bots**: Trained models from different generations (Gen 1, Gen 2, etc.)
- **Hybrid Bots**: Combination of rules-based and RL behaviors
- **Specialized Bots**: Bots trained for specific playstyles (aggressive, defensive, etc.)

**Game Integration Features**:
- **Seamless Join/Leave**: Bots can join ongoing games without disruption
- **Player Replacement**: Bots can replace disconnected players
- **Balanced Teams**: Automatic team balancing when adding/removing bots
- **Custom Bot Names**: Players can name their bot opponents
- **Bot Performance Tracking**: Track wins/losses against specific bot types
- **Auto-Cleanup**: Bots automatically leave when all human players disconnect
- **Resource Conservation**: Prevents orphaned bot-only games from consuming resources

**Architecture Requirements**:

*Python Bot Server*:
- **Bot Instance Management**: Spawn and manage individual bot processes
- **Model Loading**: Load and cache trained RL models and rules-based bots
- **Game Client Pool**: Maintain pool of GameClient connections for bot players
- **Resource Management**: Monitor CPU/memory usage, scale bot instances
- **Bot Lifecycle**: Handle bot creation, configuration, and cleanup
- **Auto-Cleanup**: Automatically disconnect bots when all human players leave

*Game Server Integration*:
- **Bot Registration API**: Register available bots with the main game server
- **Room Assignment**: Coordinate bot placement in game rooms
- **Bot Status Tracking**: Monitor which bots are active in which rooms
- **Health Monitoring**: Ensure bot server connectivity and responsiveness

*Communication Flow*:
```
Browser UI → Game Server → Python Bot Server → GameClient → Game Server
```

**Server-Side Requirements**:
- **Bot Pool Management**: Maintain pool of available bot instances
- **Room Bot Tracking**: Track which bots are active in which rooms
- **Resource Allocation**: Manage computational resources for bot players
- **Bot State Synchronization**: Ensure bot actions are properly synchronized with game state
- **Python Bot Server**: Dedicated service for running RL models and managing bot instances

### 5. RL Training Engine

**Purpose**: Central orchestrator for training successive RL models

**Key Methods**:
- `train_next_generation(previous_model, training_config)`: Trains a new model generation
- `evaluate_generation(model, evaluation_episodes)`: Evaluates model performance
- `should_promote_model(current_best, candidate)`: Determines if a new model should replace the current best
- `initialize_from_rules_bot(rules_bot)`: Creates first RL generation using rules-based behavior as guidance
- `select_training_cohort(generation, cohort_size)`: Selects opponent bots for training
- `configure_enemy_count(min_enemies, max_enemies)`: Sets variable opponent numbers per episode

**Training Progression**:
- **Generation 0**: Rules-based bot serves as baseline and initial opponent
- **Generation 1**: RL model trained against rules-based bot with behavior cloning initialization
- **Generation N**: Each successive model trains against a cohort of previous generations + rules-based opponents

**Cohort Training Configuration**:
- **Opponent Pool Size**: Configurable number of previous bot generations to include (e.g., last 3-5 generations)
- **Enemy Count**: Variable number of opponents per training episode (1v1, 1v2, 1v3, etc.)
- **Opponent Selection**: Random sampling from cohort, weighted by performance, or round-robin
- **Difficulty Progression**: Start with easier opponents, gradually include stronger ones
- **Multi-Agent Scenarios**: Train against teams of mixed bot generations

**Configuration**:
- Training algorithms (DQN, PPO, A3C)
- Hyperparameters per generation
- Training episode limits
- Evaluation criteria
- Cohort size (number of previous generations to train against)
- Enemy count range (min/max opponents per episode)
- Opponent selection strategy (random, weighted, round-robin)

### 6. Model Manager

**Purpose**: Manages model lifecycle, storage, and knowledge transfer

**Key Methods**:
- `save_model(model, generation, metadata)`: Persists trained models with versioning
- `load_model(generation)`: Loads a specific model generation
- `transfer_knowledge(source_model, target_model)`: Transfers learned weights/policies
- `get_best_model()`: Returns the current best-performing model

**Storage Structure**:
```
models/
├── generation_1/
│   ├── model.pth
│   ├── config.json
│   └── metrics.json
├── generation_2/
│   ├── model.pth
│   ├── config.json
│   └── metrics.json
└── current_best/
    └── symlink to best generation
```

### 7. Game Environment Wrapper

**Purpose**: Adapts the existing GameClient for RL training

**Key Methods**:
- `reset()`: Starts a new game episode
- `step(action)`: Executes an action and returns (state, reward, done, info)
- `get_state()`: Extracts game state features for the RL model
- `calculate_reward(game_state, action, result)`: Computes reward signals using configurable reward functions
- `set_reward_function(reward_type, parameters)`: Configures which reward function to use
- `get_available_reward_functions()`: Returns list of supported reward function types
- `set_training_mode(enabled)`: Switches between training (accelerated) and evaluation (real-time) modes
- `set_state_representation(representation_type)`: Configures how game state is encoded for the model
- `register_state_processor(processor)`: Adds custom state processing functions
- `get_available_representations()`: Returns list of supported state representation types

**State Representation Options**:

*Raw Coordinate Representation*:
- Player position (x, y) and velocity (vx, vy)
- Enemy positions and velocities
- Projectile locations, velocities, and types
- Game boundaries and platform positions
- Power-up locations and types

*Grid-Based Representation*:
- Discretized game world as 2D grid
- Cell values indicating player, enemies, projectiles, platforms
- Multi-channel representation for different entity types
- Spatial relationships preserved in grid format

*Feature Vector Representation*:
- Distances to nearest enemies, projectiles, platforms
- Relative positions and velocities
- Health, ammunition, power-up status
- Tactical features (cover availability, escape routes)

*Hybrid Representations*:
- Combination of coordinate and feature data
- Multi-modal inputs (coordinate + tactical features)
- Hierarchical state encoding (local + global features)
- Custom domain-specific representations

**State Processing Pipeline**:
- **Raw State Extraction**: Get complete game state from server
- **Representation Selection**: Apply chosen state encoding method
- **Normalization**: Scale features to appropriate ranges
- **History Integration**: Include temporal information from previous frames
- **Custom Processing**: Apply domain-specific transformations

**Action Space Options**:

*Available Game Inputs* (from GameClient):
- **Keyboard**: Press/release W, A, S, D keys
- **Mouse Left**: Click/hold/release for shooting at (x, y) coordinates (hold to charge shot)
- **Mouse Right**: Click to cancel currently charging shot (only relevant when left-click is held)
- **Timing**: Actions can be held for variable durations

*Discrete Action Space*:
- **Basic Actions**: {press_A, release_A, press_D, release_D, press_W, release_W, press_S, release_S, no_action}
- **Mouse Actions**: {left_click, hold_left_click, release_left_click, right_click_cancel, no_mouse_action}
- **Combined Actions**: {press_A_and_left_click, press_D_and_hold_left_click, move_and_cancel_shot, etc.}
- **Simplified Actions**: {move_left, move_right, jump, shoot, no_action} (mapped to appropriate key combinations)

*Continuous Action Space*:
- **Movement**: [-1.0, 1.0] mapped to A/D key press intensity and duration
- **Mouse Position**: (x, y) coordinates for aiming [0.0, 1.0] normalized screen space
- **Action Duration**: [0, 1.0] for how long to hold keys/mouse buttons

*Hybrid Action Space*:
- **Discrete Movement + Continuous Aiming**: W/A/S/D keys + continuous mouse (x, y)
- **Discrete Actions + Continuous Timing**: Fixed actions with variable hold durations
- **Hierarchical Actions**: High-level intentions mapped to low-level input sequences

**Reward Function Options**:

*Sparse Reward Functions*:
- **Win/Loss Only**: +1 for win, -1 for loss, 0 otherwise
- **Survival Based**: +1 for surviving, -1 for death, 0 for ongoing
- **Objective Based**: Rewards only for achieving specific game objectives

*Dense Reward Functions*:
- **Health Differential**: Reward based on health gained/lost relative to opponents
- **Damage Dealt**: Positive reward for dealing damage to enemies
- **Positional Advantage**: Rewards for controlling strategic positions
- **Resource Collection**: Rewards for collecting power-ups and ammunition

*Shaped Reward Functions*:
- **Distance to Enemy**: Negative reward for being too far from combat
- **Aim Accuracy**: Rewards for shots that hit vs miss
- **Movement Efficiency**: Penalties for excessive or inefficient movement
- **Tactical Positioning**: Rewards for using cover, high ground, etc.

*Horizon-Based Rewards*:
- **Short-Term (1-5 steps)**: Immediate action consequences
- **Medium-Term (10-50 steps)**: Tactical sequence rewards
- **Long-Term (100+ steps)**: Strategic game-level outcomes
- **Multi-Horizon**: Weighted combination of different time scales

*Curriculum Reward Functions*:
- **Progressive Complexity**: Start with simple rewards, add complexity over time
- **Difficulty Scaling**: Adjust reward sensitivity based on bot skill level
- **Opponent-Adaptive**: Modify rewards based on opponent strength

**Reward Function Configuration**:
```python
reward_config = {
    "primary_function": "health_differential",
    "secondary_functions": ["aim_accuracy", "survival"],
    "weights": [0.7, 0.2, 0.1],
    "horizon_weights": {
        "short_term": 0.5,
        "medium_term": 0.3, 
        "long_term": 0.2
    },
    "curriculum_stage": "intermediate",
    "normalization": "z_score"
}
```

**Reward Engineering Features**:
- **Multi-Objective**: Combine multiple reward signals with configurable weights
- **Temporal Discounting**: Apply different discount factors for different reward types
- **Normalization**: Standardize reward scales across different function types
- **Clipping**: Prevent extreme reward values from destabilizing training
- **Exploration Bonuses**: Additional rewards for novel states or actions

**Experimental Configuration**:
- **A/B Testing Framework**: Compare different representations and reward functions on same models
- **Performance Metrics**: Track learning speed, final performance, stability
- **Representation Switching**: Change representations between training generations
- **Reward Function Evolution**: Modify reward functions as bots become more sophisticated
- **Custom Processors**: Plugin architecture for new state representations and reward functions

### 8. Evaluation Framework

**Purpose**: Compares and validates model performance across generations

**Key Methods**:
- `run_evaluation(model, opponent_models, episodes)`: Runs evaluation matches
- `calculate_metrics(game_results)`: Computes performance metrics
- `compare_generations(model_a, model_b)`: Statistical comparison of models
- `generate_report(evaluation_results)`: Creates performance reports

**Metrics Tracked**:
- Win rate against previous generations
- Average episode reward
- Game completion time
- Strategic diversity (action entropy)
- Learning stability (reward variance)

### 9. Replay System

**Purpose**: Records, stores, and analyzes game episodes

**Key Methods**:
- `record_episode(states, actions, rewards)`: Stores episode data
- `get_training_batch(batch_size)`: Retrieves training samples
- `analyze_behavior(model, episodes)`: Analyzes decision patterns
- `export_replay(episode_id, format)`: Exports episode for analysis

## Data Models

### RLModel
```python
class RLModel:
    generation: int
    algorithm: str  # 'DQN', 'PPO', 'A3C'
    network_architecture: dict
    hyperparameters: dict
    training_episodes: int
    performance_metrics: dict
    parent_generation: Optional[int]
    created_at: datetime
    model_path: str
```

### TrainingSession
```python
class TrainingSession:
    session_id: str
    model_generation: int
    start_time: datetime
    end_time: Optional[datetime]
    episodes_completed: int
    current_reward: float
    best_reward: float
    training_config: dict
    status: str  # 'running', 'completed', 'failed'
```

### GameEpisode
```python
class GameEpisode:
    episode_id: str
    model_generation: int
    states: List[dict]
    actions: List[int]
    rewards: List[float]
    total_reward: float
    episode_length: int
    game_result: str  # 'win', 'loss', 'draw'
    opponent_info: dict
```

### EvaluationResult
```python
class EvaluationResult:
    model_generation: int
    opponent_generations: List[int]
    total_games: int
    wins: int
    losses: int
    draws: int
    average_reward: float
    win_rate: float
    performance_metrics: dict
    evaluation_date: datetime
```

## Error Handling

### Training Failures
- **Model Convergence Issues**: Implement early stopping and hyperparameter adjustment
- **Resource Exhaustion**: Queue management and resource monitoring
- **Game Connection Failures**: Retry logic and fallback to simulation mode

### Model Performance Degradation
- **Performance Regression**: Automatic rollback to previous best model
- **Overfitting Detection**: Validation against diverse opponent sets
- **Training Instability**: Gradient clipping and learning rate scheduling

### System Failures
- **Storage Failures**: Model backup and recovery mechanisms
- **Memory Leaks**: Automatic cleanup and resource monitoring
- **Concurrent Training**: Lock management and resource allocation

## Testing Strategy

### Unit Testing
- **Model Components**: Test individual RL algorithms and neural networks
- **Environment Wrapper**: Validate state extraction and reward calculation
- **Evaluation Metrics**: Test performance calculation accuracy
- **Storage Operations**: Test model save/load functionality

### Integration Testing
- **End-to-End Training**: Complete training pipeline from start to finish
- **Model Succession**: Test knowledge transfer between generations
- **Game Integration**: Validate bot behavior in actual game sessions
- **Performance Comparison**: Test evaluation framework accuracy

### Performance Testing
- **Training Speed**: Benchmark training time per episode and generation
- **Memory Usage**: Monitor memory consumption during training
- **Concurrent Training**: Test multiple simultaneous training sessions
- **Model Inference**: Measure bot response time during gameplay

### Validation Testing
- **Model Quality**: Validate that successive models actually improve
- **Behavioral Analysis**: Ensure bots exhibit intelligent gameplay
- **Stability Testing**: Long-running training sessions without degradation
- **Cross-Generation Compatibility**: Ensure models can compete across generations