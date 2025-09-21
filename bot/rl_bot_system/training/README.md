# Training Session Management

This module provides comprehensive training session management for RL bot training with accelerated game instances, headless mode configuration, and batch episode management.

## Overview

The training module consists of four main components:

1. **TrainingSession** - Manages individual training sessions with accelerated game instances
2. **SessionManager** - Coordinates multiple concurrent training sessions with resource management
3. **BatchEpisodeManager** - Handles parallel execution of training episodes
4. **TrainingAPI** - Provides HTTP-style API endpoints for training management

## Key Features

- **Accelerated Training**: Support for game speed multipliers (1x to 100x)
- **Headless Mode**: Maximum speed training without rendering
- **Parallel Episodes**: Concurrent episode execution for faster training
- **Resource Management**: Automatic resource allocation and monitoring
- **Session Queuing**: Priority-based session scheduling
- **Real-time Metrics**: Live training progress and performance tracking
- **Spectator Support**: Browser-based training observation
- **Retry Logic**: Automatic episode retry with exponential backoff
- **Event Handlers**: Extensible event system for custom integrations

## Components

### TrainingSession

Manages individual training sessions with configurable speed and parallelism.

```python
from bot.rl_bot_system.training import TrainingSession, TrainingConfig, TrainingMode

# Create configuration
config = TrainingConfig(
    speed_multiplier=10.0,
    headless_mode=True,
    max_episodes=1000,
    parallel_episodes=4,
    training_mode=TrainingMode.TRAINING,
    spectator_enabled=False
)

# Create and run session
session = TrainingSession(config=config)
await session.initialize()
await session.start_training()
```

### SessionManager

Coordinates multiple training sessions with resource limits and queuing.

```python
from bot.rl_bot_system.training import SessionManager, ResourceLimits

# Configure resource limits
limits = ResourceLimits(
    max_concurrent_sessions=5,
    max_parallel_episodes_per_session=4,
    max_total_parallel_episodes=16
)

# Create and start manager
manager = SessionManager(resource_limits=limits)
await manager.start()

# Create sessions
session_id = await manager.create_session(config=config, priority=1)
```

### BatchEpisodeManager

Handles parallel execution of training episodes with load balancing.

```python
from bot.rl_bot_system.training import BatchEpisodeManager

# Create batch manager
batch_manager = BatchEpisodeManager(
    max_parallel_episodes=8,
    max_retries=3,
    episode_timeout=300
)

await batch_manager.start()

# Submit episode batch
await batch_manager.submit_batch(
    batch_id="training_batch_001",
    episode_ids=["ep_001", "ep_002", "ep_003"],
    room_code="TR123456",
    max_parallel=4
)
```

### TrainingAPI

Provides HTTP-style API endpoints for training management.

```python
from bot.rl_bot_system.training import TrainingAPI

# Create API
api = TrainingAPI()
await api.start()

# Create session via API
result = await api.create_training_session({
    "config": {
        "speedMultiplier": 15.0,
        "headlessMode": True,
        "maxEpisodes": 500
    },
    "priority": 1
})

# Get system status
status = await api.get_system_status()
```

## Configuration Options

### TrainingConfig

- `speed_multiplier`: Game speed multiplier (1.0 to 100.0)
- `headless_mode`: Enable headless mode for maximum speed
- `max_episodes`: Maximum number of episodes per session
- `episode_timeout`: Timeout for individual episodes (seconds)
- `parallel_episodes`: Number of parallel episodes
- `training_mode`: Training mode (REALTIME, TRAINING, HEADLESS, BATCH)
- `room_password`: Optional room password
- `spectator_enabled`: Enable spectator room creation
- `auto_cleanup`: Automatic resource cleanup

### ResourceLimits

- `max_concurrent_sessions`: Maximum concurrent training sessions
- `max_parallel_episodes_per_session`: Max parallel episodes per session
- `max_total_parallel_episodes`: Global parallel episode limit
- `memory_limit_mb`: Memory limit in MB
- `cpu_limit_percent`: CPU usage limit percentage

## Training Modes

1. **REALTIME** (1x speed) - Normal game speed for human evaluation
2. **TRAINING** (10-50x speed) - Accelerated speed with visual feedback
3. **HEADLESS** (50-100x speed) - Maximum speed without rendering
4. **BATCH** - Parallel episode execution for distributed training

## Event Handling

The system provides extensive event handling capabilities:

```python
# Episode completion handler
async def on_episode_complete(result):
    print(f"Episode {result.episode_id}: reward={result.total_reward}")

session.register_episode_handler(on_episode_complete)

# Metrics update handler
async def on_metrics_update(metrics):
    print(f"Progress: {metrics.episodes_completed} episodes")

session.register_metrics_handler(on_metrics_update)

# Session lifecycle handlers
manager.register_session_start_handler(on_session_start)
manager.register_session_complete_handler(on_session_complete)
manager.register_resource_alert_handler(on_resource_alert)
```

## Metrics and Monitoring

Real-time metrics are available for all training sessions:

- Episodes completed/failed
- Average/best/worst rewards
- Success rate
- Episodes per minute
- Resource utilization
- Training duration

```python
# Get session metrics
info = await session.get_session_info()
metrics = info["metrics"]

# Get global metrics
global_metrics = await manager.get_global_metrics()
```

## Error Handling and Retry Logic

The system includes comprehensive error handling:

- Automatic episode retry with exponential backoff
- Session failure recovery
- Resource exhaustion handling
- Network connection retry
- Graceful degradation

## Integration with Game Server

The training system integrates with the game server through:

- Training room creation with speed multipliers
- Headless mode configuration
- Spectator room management
- Bot player coordination
- Resource monitoring

## Server-Side Requirements

For full functionality, the game server needs:

- Training room support with variable tick rates
- Headless game instances
- Training API endpoints
- Spectator integration
- Resource management

## Example Usage

See `example_usage.py` for comprehensive examples of all components.

## Testing

Run the test suite:

```bash
python run_test.py
```

The module includes comprehensive unit tests for all components:

- TrainingSession functionality
- SessionManager resource management
- BatchEpisodeManager parallel execution
- TrainingAPI endpoint behavior
- Error handling and edge cases

## Performance Considerations

- Use headless mode for maximum training speed
- Configure parallel episodes based on system resources
- Monitor memory usage with large batch sizes
- Adjust episode timeouts based on game complexity
- Use appropriate speed multipliers for stable physics

## Future Enhancements

Planned improvements include:

- GPU acceleration support
- Distributed training across multiple machines
- Advanced scheduling algorithms
- Performance profiling and optimization
- Integration with cloud training platforms