# RL Bot System Components

Technical documentation for the reinforcement learning bot system components and architecture.

> **Note:** For setup instructions, run scripts usage, and general system overview, see the main [bot/README.md](../README.md).

## Architecture Overview

This system implements a comprehensive RL training framework that creates increasingly sophisticated bot players through successive generations. Each new generation learns from and improves upon previous models, creating a progressive improvement in bot performance.

## Core Features

- **Successive Learning**: Train multiple RL model generations where each improves upon previous ones
- **Multiple Algorithms**: Support for DQN, PPO, A3C, and SAC algorithms
- **Flexible State Representations**: Raw coordinates, grid-based, feature vectors, and hybrid representations
- **Configurable Action Spaces**: Discrete, continuous, and hybrid action spaces
- **Advanced Reward Functions**: Multiple reward function types with multi-objective optimization
- **Cohort Training**: Train against multiple previous generations simultaneously
- **Rules-Based Foundation**: Start with intelligent rules-based bots as baseline opponents
- **Comprehensive Evaluation**: Statistical comparison and performance tracking across generations
- **Real-time Spectator System**: Live training observation with metrics overlay

## Component Architecture

```
rl_bot_system/
├── config/                  # Configuration management system
│   ├── config_manager.py    # YAML configuration loading and validation
│   ├── templates/           # Configuration templates
│   └── schemas/             # Configuration validation schemas
├── models/                  # RL model implementations
│   ├── dqn.py              # Deep Q-Network implementation
│   ├── ppo.py              # Proximal Policy Optimization
│   ├── a3c.py              # Asynchronous Actor-Critic
│   └── base_model.py       # Base model interface
├── training/                # Training engine and algorithms
│   ├── training_engine.py   # Core training orchestration
│   ├── model_manager.py     # Model lifecycle management
│   ├── cohort_training.py   # Multi-generation training
│   └── hyperparameter_tuning.py # Automated hyperparameter optimization
├── evaluation/              # Model evaluation and comparison
│   ├── evaluator.py         # Performance evaluation framework
│   └── metrics/             # Evaluation metrics and statistics
├── spectator/               # Real-time training observation
│   ├── spectator_manager.py # Spectator session management
│   ├── training_metrics_overlay.py # Real-time metrics display
│   └── room_manager.py      # Spectator room access control
├── server/                  # Training metrics and API server
│   ├── training_metrics_server.py # FastAPI server implementation
│   ├── data_models.py       # Pydantic data models
│   ├── websocket_manager.py # Real-time WebSocket connections
│   └── integration.py       # Training engine integration
├── replay/                  # Experience replay and analysis
│   ├── replay_manager.py    # Replay buffer management
│   ├── experience_buffer.py # Experience storage and sampling
│   └── episode_recorder.py  # Game episode recording
└── rules_based/             # Rules-based bot foundation
    └── rules_based_bot.py   # Intelligent baseline bots
```

## Configuration System

The system uses YAML configuration files to manage all aspects of training and evaluation:

### Key Configuration Sections

- **game**: Game connection and environment settings
- **state_representation**: How game state is encoded for the RL model
- **action_space**: Available actions and their encoding
- **reward**: Reward function configuration and multi-objective optimization
- **training**: RL algorithm parameters and training settings
- **cohort_training**: Settings for training against previous generations
- **evaluation**: Model evaluation and comparison settings
- **model**: Neural network architecture and model management
- **speed_control**: Game speed acceleration for faster training
- **logging**: Logging, monitoring, and visualization settings

### Example Configuration Usage

```bash
# Use default configuration
python rl_bot_cli.py --config default train start

# Use quick training configuration for experimentation
python rl_bot_cli.py --config quick_training train start

# Create and use custom configuration
python rl_bot_cli.py config create my_config --algorithm PPO
python rl_bot_cli.py --config my_config train start
```

### GPU Configuration

The system automatically detects and uses GPU acceleration when available. You can control GPU usage through configuration:

```yaml
training:
  device: "auto"  # "auto", "cpu", "cuda", "cuda:0"
  # auto: automatically use GPU if available, fallback to CPU
  # cpu: force CPU training
  # cuda: use first available GPU
  # cuda:0: use specific GPU device
```

**GPU-Optimized Settings:**
```bash
# Create GPU-optimized configuration
python rl_bot_cli.py config create gpu_training --template default

# Then edit config/gpu_training.yaml to increase batch sizes:
# training:
#   batch_size: 256        # Increase from 64
#   buffer_size: 1000000   # Increase from 100000
#   n_steps: 4096          # Increase from 2048 (for PPO)
```

## Core Components

### Training Engine (`training/`)

The training engine orchestrates the entire RL training process:

- **TrainingEngine**: Main training loop with successive learning
- **ModelManager**: Handles model lifecycle, checkpointing, and knowledge transfer
- **CohortTraining**: Manages training against multiple previous generations
- **HyperparameterTuning**: Automated hyperparameter optimization

### Model Implementations (`models/`)

Support for multiple RL algorithms:

- **DQN**: Deep Q-Network with experience replay and target networks
- **PPO**: Proximal Policy Optimization with clipped surrogate objective
- **A3C**: Asynchronous Actor-Critic with parallel workers
- **Base Model**: Common interface for all RL algorithms

### Spectator System (`spectator/`)

Real-time training observation and metrics:

- **SpectatorManager**: Session management and access control
- **TrainingMetricsOverlay**: Real-time metrics display and visualization
- **RoomManager**: Spectator room creation and access control

### Server Components (`server/`)

FastAPI server for training metrics and real-time data:

- **TrainingMetricsServer**: REST API and WebSocket endpoints
- **WebSocketManager**: Real-time connection management
- **DataModels**: Pydantic models for API data validation
- **Integration**: Training engine and spectator system integration

### Evaluation Framework (`evaluation/`)

Comprehensive model evaluation and comparison:

- **Evaluator**: Performance evaluation against various opponents
- **Metrics**: Statistical analysis and performance tracking
- **Comparison**: Multi-generation performance comparison

### Replay System (`replay/`)

Experience replay and game analysis:

- **ReplayManager**: Replay buffer management and optimization
- **ExperienceBuffer**: Efficient experience storage and sampling
- **EpisodeRecorder**: Game episode recording and analysis

## Technical Specifications

### Supported RL Algorithms

**Deep Q-Network (DQN)**
- Experience replay with prioritized sampling
- Target network for stable training
- Double DQN and Dueling DQN variants
- Configurable network architectures

**Proximal Policy Optimization (PPO)**
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Adaptive KL penalty
- Parallel environment support

**Asynchronous Actor-Critic (A3C)**
- Parallel worker processes
- Shared global network
- Entropy regularization
- Gradient accumulation

### State Representations

**Raw Coordinates**
- Direct position and velocity data
- Minimal preprocessing
- Fast computation
- Suitable for simple environments

**Grid-Based**
- Discretized spatial representation
- Configurable grid resolution
- Spatial relationship encoding
- Good for navigation tasks

**Feature Vectors**
- Hand-crafted feature extraction
- Domain-specific knowledge
- Reduced dimensionality
- Interpretable representations

**Hybrid Representations**
- Combination of multiple representations
- Multi-modal input processing
- Flexible architecture adaptation
- Enhanced learning capability

### Action Spaces

**Discrete Actions**
- Fixed set of available actions
- Categorical action selection
- Simple policy representation
- Fast action sampling

**Continuous Actions**
- Real-valued action parameters
- Gaussian policy distributions
- Fine-grained control
- Suitable for complex movements

**Hybrid Actions**
- Combination of discrete and continuous
- Multi-dimensional action spaces
- Complex behavior modeling
- Advanced control strategies

### Performance Optimization

**GPU Acceleration**
- CUDA-optimized PyTorch operations
- Batch processing for efficiency
- Memory-efficient data loading
- Parallel environment execution

**Memory Management**
- Efficient experience replay buffers
- Gradient checkpointing
- Model compression techniques
- Dynamic batch sizing

**Training Acceleration**
- Vectorized environment execution
- Asynchronous data collection
- Optimized neural network architectures
- Early stopping and convergence detection

## API Integration

### Training Metrics Server Integration

The server components provide REST and WebSocket APIs for real-time training monitoring:

```python
# Start training metrics server
from rl_bot_system.server.integration import TrainingEngineIntegration

# Create integration
integration = TrainingEngineIntegration(metrics_server)

# Register training session
session_id = await integration.register_training_session(
    training_id="dqn_generation_3",
    model_generation=3,
    algorithm="DQN",
    total_episodes=1000
)

# Update metrics during training
await integration.update_training_metrics(
    training_id="dqn_generation_3",
    episode=150,
    current_reward=25.5,
    average_reward=18.2,
    win_rate=72.0
)
```

### Spectator System Integration

Real-time training observation with access control:

```python
# Create spectator session
from rl_bot_system.spectator.spectator_manager import SpectatorManager

spectator_manager = SpectatorManager()
session = await spectator_manager.create_spectator_session(
    training_session_id="dqn_generation_3",
    max_spectators=10,
    enable_metrics_overlay=True,
    enable_decision_visualization=True
)

print(f"Spectator room: {session.room_code}")
print(f"Password: {session.room_password}")
```

### Model Management API

Programmatic model lifecycle management:

```python
# Load and manage models
from rl_bot_system.training.model_manager import ModelManager

model_manager = ModelManager()

# Load previous generation
previous_model = model_manager.load_model("dqn_generation_2")

# Create new generation with knowledge transfer
new_model = model_manager.create_next_generation(
    previous_model=previous_model,
    algorithm="DQN",
    transfer_method="fine_tuning"
)

# Save trained model
model_manager.save_model(new_model, "dqn_generation_3")
```

## Development Guidelines

### Adding New RL Algorithms

1. **Inherit from BaseModel**: Implement the common interface
2. **Configuration Schema**: Add algorithm-specific config validation
3. **Integration Tests**: Test with training engine and evaluation framework
4. **Documentation**: Update algorithm comparison and usage examples

### Extending State Representations

1. **State Processor**: Implement state transformation logic
2. **Configuration Options**: Add representation-specific parameters
3. **Performance Testing**: Benchmark against existing representations
4. **Compatibility**: Ensure compatibility with all RL algorithms

### Custom Reward Functions

1. **Reward Interface**: Implement the reward calculation interface
2. **Multi-objective Support**: Handle multiple reward components
3. **Configuration**: Add reward function parameters to config system
4. **Validation**: Test reward function behavior and convergence

## Performance Benchmarks

### Training Speed (Episodes/Hour)

| Algorithm | CPU (8 cores) | GPU (RTX 3080) | GPU (RTX 4090) |
|-----------|---------------|----------------|----------------|
| DQN       | 1,200         | 8,500          | 12,000         |
| PPO       | 800           | 6,200          | 9,500          |
| A3C       | 1,500         | 10,200         | 15,000         |

### Memory Usage

| Component           | Typical Usage | Peak Usage |
|--------------------|---------------|------------|
| Experience Buffer  | 2-4 GB        | 8 GB       |
| Model Training     | 1-2 GB        | 4 GB       |
| Spectator System   | 100-500 MB    | 1 GB       |
| Metrics Server     | 50-200 MB     | 500 MB     |

### Scalability Limits

- **Maximum Concurrent Spectators**: 100 per training session
- **Maximum Training Sessions**: 10 concurrent sessions
- **Experience Buffer Size**: Up to 10M transitions (limited by RAM)
- **Model Generations**: No practical limit (disk space dependent)

## Contributing

### Code Standards

1. **Type Hints**: Use comprehensive type annotations
2. **Documentation**: Docstrings for all public methods
3. **Error Handling**: Comprehensive exception handling
4. **Logging**: Structured logging with appropriate levels
5. **Testing**: Unit tests for all new functionality

### Performance Considerations

1. **Memory Efficiency**: Minimize memory allocations in training loops
2. **GPU Utilization**: Optimize batch sizes and data loading
3. **Async Operations**: Use async/await for I/O operations
4. **Caching**: Cache expensive computations where appropriate

### Integration Testing

1. **End-to-End Tests**: Full training pipeline validation
2. **Performance Tests**: Benchmark critical paths
3. **Compatibility Tests**: Test across different hardware configurations
4. **Stress Tests**: Validate system behavior under load

## License

This project is part of the larger game system and follows the same licensing terms.