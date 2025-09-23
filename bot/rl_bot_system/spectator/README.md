# Spectator System for RL Bot Training

The spectator system provides real-time observation capabilities for RL bot training sessions. It enables users to watch bot training progress, view performance metrics, and analyze bot decision-making through a browser-based interface.

## Features

### 🎯 Core Functionality
- **Real-time Training Observation**: Watch bots train live with configurable speed controls
- **Training Metrics Overlay**: Display real-time training progress, rewards, and performance statistics
- **Bot Decision Visualization**: View action probabilities, state values, and decision-making process
- **Performance Graphs**: Real-time charts showing training progress and metrics over time

### 🔐 Access Control
- **Multiple Access Levels**: Public, password-protected, invite-only, and private rooms
- **Approval Workflows**: Optional approval process for spectator access
- **Room Management**: Create, join, and manage spectator rooms with unique codes
- **User Permissions**: Fine-grained control over who can access training sessions

### 📊 Metrics and Analytics
- **Comprehensive Metrics**: Episode progress, rewards, win rates, learning metrics
- **Historical Data**: Access to training history and performance trends
- **Custom Callbacks**: Register callbacks for training progress notifications
- **Export Capabilities**: Export training data and metrics for analysis

## Quick Start

### Basic Spectator Session

```python
import asyncio
from bot.rl_bot_system.spectator import SpectatorManager, SpectatorMode

async def create_basic_session():
    # Create spectator manager
    manager = SpectatorManager(game_server_url="http://localhost:4000")
    
    # Create a spectator session
    session = await manager.create_spectator_session(
        training_session_id="my_training_session",
        spectator_mode=SpectatorMode.LIVE_TRAINING,
        max_spectators=10,
        enable_metrics_overlay=True,
        enable_performance_graphs=True
    )
    
    print(f"Room code: {session.room_code}")
    print(f"Session ID: {session.session_id}")
    
    # Cleanup
    await manager.cleanup()

asyncio.run(create_basic_session())
```

### Password-Protected Room

```python
from bot.rl_bot_system.spectator import SpectatorManager

async def create_protected_session():
    manager = SpectatorManager()
    
    session = await manager.create_spectator_session(
        training_session_id="private_training",
        password_protected=True,  # Generates secure password
        max_spectators=5
    )
    
    print(f"Room code: {session.room_code}")
    print(f"Password: {session.room_password}")
    
    await manager.cleanup()
```

### Room Access Control

```python
from bot.rl_bot_system.spectator import SpectatorRoomManager, RoomAccessControl, AccessLevel

async def create_controlled_room():
    room_manager = SpectatorRoomManager()
    
    # Create private room with specific users
    access_control = RoomAccessControl(
        access_level=AccessLevel.PRIVATE,
        allowed_users={"user1", "user2", "researcher3"},
        max_spectators=3
    )
    
    room = await room_manager.create_spectator_room(
        training_session_id="research_training",
        creator_id="lead_researcher",
        access_control=access_control
    )
    
    print(f"Private room created: {room.room_code}")
    
    await room_manager.cleanup()
```

## Architecture

### Core Components

1. **SpectatorManager**: Main orchestrator for spectator sessions
2. **TrainingMetricsOverlay**: Handles real-time metrics display and visualization
3. **SpectatorRoomManager**: Manages room creation, access control, and user permissions
4. **GameClient Integration**: Seamless integration with existing game client infrastructure

### Data Flow

```
Training Session → Metrics Updates → Spectator Manager → WebSocket Broadcast → Browser UI
                                  ↓
                            Metrics Overlay → Performance Graphs → Real-time Display
```

## API Reference

### SpectatorManager

#### `create_spectator_session()`
Creates a new spectator session for training observation.

**Parameters:**
- `training_session_id` (str): ID of the training session to observe
- `spectator_mode` (SpectatorMode): Type of viewing mode (LIVE_TRAINING, REPLAY, COMPARISON)
- `max_spectators` (int): Maximum concurrent spectators (default: 10)
- `session_duration_hours` (int): Session lifetime in hours (default: 24)
- `password_protected` (bool): Enable password protection (default: False)
- `enable_metrics_overlay` (bool): Enable training metrics display (default: True)
- `enable_performance_graphs` (bool): Enable real-time graphs (default: True)
- `enable_decision_visualization` (bool): Enable bot decision visualization (default: True)

**Returns:** `SpectatorSession` object with room details

#### `join_spectator_session()`
Join an existing spectator session.

**Parameters:**
- `session_id` (str): ID of the spectator session
- `spectator_name` (str): Display name for the spectator
- `password` (Optional[str]): Password if session is protected
- `user_id` (Optional[str]): User ID for access control

**Returns:** Connection information dictionary

#### `update_training_metrics()`
Update training metrics for real-time display.

**Parameters:**
- `session_id` (str): ID of the spectator session
- `metrics_data` (MetricsData): Updated training metrics

### TrainingMetricsOverlay

#### `update_metrics()`
Update and broadcast training metrics to spectators.

#### `send_bot_decision_data()`
Send bot decision visualization data to spectators.

**Parameters:**
- `action_probabilities` (Dict[str, float]): Action probability distribution
- `state_values` (Optional[float]): Current state value estimate
- `q_values` (Optional[List[float]]): Q-values for each action
- `selected_action` (Optional[str]): The selected action

### SpectatorRoomManager

#### `create_spectator_room()`
Create a new spectator room with access control.

#### `join_room_request()`
Request to join a spectator room (handles access control).

#### `approve_join_request()`
Approve or deny a pending join request.

## Configuration

### Access Levels

- **PUBLIC**: Anyone can join with room code
- **PASSWORD**: Requires password for access
- **INVITE_ONLY**: Only explicitly invited users can join
- **PRIVATE**: Creator and invited users only

### Metrics Configuration

```python
from bot.rl_bot_system.spectator import MetricsData

metrics = MetricsData(
    timestamp=datetime.now(),
    episode=100,
    total_episodes=1000,
    current_reward=75.5,
    average_reward=65.3,
    best_reward=100.0,
    episode_length=450,
    win_rate=0.68,
    loss_value=0.045,
    learning_rate=0.001,
    epsilon=0.15,
    model_generation=3,
    algorithm="DQN",
    training_time_elapsed=1800.0,
    # Bot decision data
    action_probabilities={"move_left": 0.25, "move_right": 0.35, "shoot": 0.4},
    state_values=0.82,
    selected_action="shoot"
)
```

## Integration with Training Engine

### Automatic Metrics Updates

```python
from bot.rl_bot_system.training import TrainingEngine
from bot.rl_bot_system.spectator import SpectatorManager

class SpectatorIntegratedTraining(TrainingEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectator_manager = SpectatorManager()
        self.spectator_session = None
    
    async def start_training_with_spectators(self, enable_spectators=True):
        if enable_spectators:
            self.spectator_session = await self.spectator_manager.create_spectator_session(
                training_session_id=self.session_id,
                enable_metrics_overlay=True,
                enable_performance_graphs=True
            )
            print(f"Spectator room: {self.spectator_session.room_code}")
    
    async def on_episode_complete(self, episode_data):
        # Update spectators with training progress
        if self.spectator_session:
            metrics_data = self._create_metrics_data(episode_data)
            await self.spectator_manager.update_training_metrics(
                session_id=self.spectator_session.session_id,
                metrics_data=metrics_data
            )
```

## Browser Integration

The spectator system integrates with the existing browser-based game client to provide:

- **Real-time Training View**: Watch training sessions in the browser
- **Metrics Dashboard**: Live performance graphs and statistics
- **Decision Visualization**: Bot action probabilities and decision trees
- **Room Management UI**: Join rooms, manage access, view spectator lists

### WebSocket Messages

The system sends various message types to browser clients:

```javascript
// Training metrics update
{
  "type": "training_metrics",
  "data": {
    "episode": 150,
    "current_reward": 85.5,
    "win_rate": 0.68,
    "algorithm": "DQN",
    // ... more metrics
  }
}

// Bot decision visualization
{
  "type": "bot_decision",
  "action_probabilities": {
    "move_left": 0.25,
    "move_right": 0.35,
    "shoot": 0.4
  },
  "selected_action": "shoot",
  "state_values": 0.82
}

// Performance graph update
{
  "type": "graph_update",
  "graphs": [
    {
      "graph_id": "reward_progress",
      "timestamp": "2024-01-15T10:30:00Z",
      "data_points": {
        "current_reward": 85.5,
        "average_reward": 72.3
      }
    }
  ]
}
```

## Testing

Run the test suite:

```bash
# Run all spectator tests
python -m pytest bot/rl_bot_system/spectator/tests/ -v

# Run specific test file
python -m pytest bot/rl_bot_system/spectator/tests/test_spectator_manager.py -v

# Run with coverage
python -m pytest bot/rl_bot_system/spectator/tests/ --cov=bot.rl_bot_system.spectator
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `example_spectator_usage.py`: Complete examples of all spectator functionality
- Basic session creation
- Password protection
- Access control workflows
- Metrics overlay usage
- Room management

## Security Considerations

- **Password Protection**: Secure password generation for protected rooms
- **Access Control**: Fine-grained permissions and user validation
- **Session Expiration**: Automatic cleanup of expired sessions
- **Rate Limiting**: Protection against spam and abuse
- **Input Validation**: Sanitization of all user inputs

## Performance

- **Efficient Broadcasting**: Optimized WebSocket message distribution
- **Memory Management**: Automatic cleanup of disconnected clients
- **Scalable Architecture**: Support for multiple concurrent sessions
- **Resource Monitoring**: Built-in resource usage tracking

## Troubleshooting

### Common Issues

1. **Connection Failures**: Check game server URL and network connectivity
2. **Permission Denied**: Verify user permissions and access control settings
3. **Session Expired**: Check session duration and expiration times
4. **WebSocket Errors**: Ensure proper WebSocket configuration and error handling

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- **Multi-Session Comparison**: Side-by-side comparison of different training sessions
- **Advanced Analytics**: Statistical analysis and trend detection
- **Mobile Support**: Mobile-optimized spectator interface
- **Recording and Playback**: Save and replay training sessions
- **Integration APIs**: REST APIs for external tool integration