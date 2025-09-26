# Training Metrics Server

A FastAPI-based server for real-time training metrics collection, WebSocket communication, and spectator functionality for RL bot training sessions.

## Overview

The Training Metrics Server provides a comprehensive solution for collecting, storing, and broadcasting training metrics from RL bot training sessions. It includes:

- **FastAPI REST API** for training session management and historical data retrieval
- **WebSocket connections** for real-time metrics broadcasting
- **Integration utilities** for connecting with existing training engines and spectator systems
- **Comprehensive data models** for training metrics, bot decisions, and performance graphs
- **Configuration management** for flexible deployment scenarios

## Features

### Core Functionality

- **Real-time Metrics Broadcasting**: WebSocket-based real-time updates for training metrics, bot decisions, and performance graphs
- **Training Session Management**: Create, update, and manage training sessions with metadata
- **Historical Data API**: Retrieve historical training data with filtering and pagination
- **Multi-client Support**: Handle multiple spectator connections per training session
- **Subscription Management**: Clients can subscribe to specific message types

### Integration Support

- **Training Engine Integration**: Adapters for connecting with existing RL training engines
- **Spectator Manager Integration**: Bridge with existing spectator systems
- **Metrics Collection Utilities**: Helper classes for common metrics calculations
- **Flexible Configuration**: Environment-based and file-based configuration options

### Data Models

- **TrainingMetricsData**: Comprehensive training metrics including rewards, win rates, loss values
- **BotDecisionData**: Bot decision visualization data with action probabilities and Q-values
- **PerformanceGraphData**: Time-series data for performance graphs and charts
- **WebSocket Messages**: Structured message format for real-time communication

## Quick Start

### Basic Server Setup

```python
from bot.rl_bot_system.server import TrainingMetricsServer, ServerConfig

# Create configuration
config = ServerConfig(
    host="localhost",
    port=8000,
    cors_origins=["http://localhost:3000"],
    max_connections_per_session=50
)

# Create and run server
server = TrainingMetricsServer(config)
server.run()
```

### Training Engine Integration

```python
from bot.rl_bot_system.server.integration import TrainingEngineIntegration, MetricsCollector

# Create integration
integration = TrainingEngineIntegration(server)
collector = MetricsCollector(integration)

# Register training session
session_id = await integration.register_training_session(
    training_id="my_training_001",
    model_generation=2,
    algorithm="DQN",
    total_episodes=1000
)

# Record training episodes
await collector.record_episode_end(
    training_id="my_training_001",
    reward=15.5,
    episode_length=200,
    won=True,
    total_episodes=1000
)
```

### WebSocket Client Connection

```javascript
// Frontend JavaScript example
const ws = new WebSocket('ws://localhost:8000/ws/my_session?user_name=Spectator1');

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    switch (message.type) {
        case 'training_metrics':
            updateMetricsDisplay(message.data);
            break;
        case 'bot_decision':
            updateDecisionVisualization(message.data);
            break;
        case 'graph_update':
            updatePerformanceGraphs(message.data);
            break;
    }
};
```

## API Reference

### REST Endpoints

#### Training Sessions

- `POST /api/training/sessions` - Create a new training session
- `GET /api/training/sessions` - List all active training sessions
- `GET /api/training/sessions/{session_id}` - Get session information
- `PUT /api/training/sessions/{session_id}` - Update session status
- `DELETE /api/training/sessions/{session_id}` - Delete a session

#### Metrics Updates

- `POST /api/training/sessions/{session_id}/metrics` - Update training metrics
- `POST /api/training/sessions/{session_id}/bot_decision` - Update bot decision data
- `POST /api/training/sessions/{session_id}/graph_update` - Update performance graphs

#### Historical Data

- `GET /api/training/sessions/{session_id}/history` - Get historical training data

#### Server Status

- `GET /health` - Health check endpoint
- `GET /status` - Server status and statistics

### WebSocket Endpoints

- `WS /ws/{session_id}` - Real-time training metrics connection

#### WebSocket Message Types

- `training_metrics` - Training progress updates
- `bot_decision` - Bot decision visualization data
- `graph_update` - Performance graph updates
- `training_status` - Training session status changes
- `connection_status` - Connection status updates
- `error` - Error messages

## Configuration

### Environment Variables

```bash
# Server settings
TRAINING_METRICS_HOST=localhost
TRAINING_METRICS_PORT=8000
TRAINING_METRICS_DEBUG=false

# Database settings
DATABASE_URL=sqlite:///data/training_metrics.db

# Redis settings (optional)
REDIS_URL=redis://localhost:6379

# Security settings
JWT_SECRET_KEY=your-secret-key
CORS_ORIGINS=http://localhost:3000,http://localhost:4000

# Integration settings
GAME_SERVER_URL=http://localhost:4000

# Performance settings
MAX_CONNECTIONS_PER_SESSION=50
DATA_STORAGE_PATH=data/training_metrics
```

### Configuration File

```json
{
  "host": "localhost",
  "port": 8000,
  "debug": false,
  "database": {
    "enabled": true,
    "url": "sqlite:///data/training_metrics.db"
  },
  "redis": {
    "enabled": false,
    "host": "localhost",
    "port": 6379
  },
  "security": {
    "enable_auth": false,
    "cors_origins": ["http://localhost:3000"]
  },
  "integration": {
    "game_server_url": "http://localhost:4000",
    "enable_spectator_integration": true
  },
  "performance": {
    "max_connections_per_session": 50,
    "metrics_history_size": 10000,
    "cleanup_interval_seconds": 300
  }
}
```

## Data Models

### TrainingMetricsData

```python
{
    "timestamp": "2024-01-01T12:00:00Z",
    "episode": 100,
    "total_episodes": 1000,
    "current_reward": 15.5,
    "average_reward": 12.3,
    "best_reward": 20.1,
    "episode_length": 250,
    "win_rate": 75.5,
    "loss_value": 0.05,
    "learning_rate": 0.001,
    "epsilon": 0.1,
    "model_generation": 2,
    "algorithm": "DQN",
    "training_time_elapsed": 3600.0,
    "actions_per_second": 30.5,
    "frames_per_second": 60.0,
    "memory_usage_mb": 512.0
}
```

### BotDecisionData

```python
{
    "timestamp": "2024-01-01T12:00:00Z",
    "action_probabilities": {
        "move_left": 0.3,
        "move_right": 0.2,
        "jump": 0.1,
        "shoot": 0.4
    },
    "state_values": 5.5,
    "q_values": [1.2, 2.3, 0.8, 3.1],
    "selected_action": "shoot",
    "confidence_score": 0.85
}
```

### PerformanceGraphData

```python
{
    "graph_id": "reward_progress",
    "title": "Reward Progress",
    "y_label": "Reward",
    "metrics": ["current_reward", "average_reward"],
    "data_points": {
        "current_reward": [
            {"timestamp": "2024-01-01T12:00:00Z", "value": 10.0},
            {"timestamp": "2024-01-01T12:01:00Z", "value": 12.5}
        ]
    },
    "max_points": 1000
}
```

## Integration Examples

### Training Engine Integration

```python
from bot.rl_bot_system.server.integration import TrainingEngineIntegration

class MyTrainingEngine:
    def __init__(self):
        self.metrics_integration = TrainingEngineIntegration(metrics_server)
    
    async def start_training(self, config):
        # Register training session
        session_id = await self.metrics_integration.register_training_session(
            training_id=config.training_id,
            model_generation=config.generation,
            algorithm=config.algorithm,
            total_episodes=config.episodes
        )
        
        for episode in range(config.episodes):
            # Train episode
            reward, length, won = await self.train_episode()
            
            # Update metrics
            await self.metrics_integration.update_training_metrics(
                training_id=config.training_id,
                episode=episode,
                current_reward=reward,
                episode_length=length,
                # ... other metrics
            )
```

### Spectator Manager Integration

```python
from bot.rl_bot_system.server.integration import SpectatorManagerIntegration

# Create integrated session
integration = SpectatorManagerIntegration(metrics_server, spectator_manager)

session_info = await integration.create_integrated_session(
    training_session_id="training_001",
    model_generation=2,
    algorithm="PPO",
    total_episodes=1000,
    # Spectator-specific options
    max_spectators=20,
    password_protected=True
)

print(f"Room code: {session_info['room_code']}")
print(f"Password: {session_info['room_password']}")
```

## Testing

### Running Tests

```bash
# Run all server tests
python -m pytest bot/rl_bot_system/server/tests/

# Run specific test modules
python -m pytest bot/rl_bot_system/server/tests/test_data_models.py
python -m pytest bot/rl_bot_system/server/tests/test_websocket_manager.py

# Run with coverage
python -m pytest bot/rl_bot_system/server/tests/ --cov=bot.rl_bot_system.server
```

### Test Coverage

The test suite covers:

- **Data Models**: Pydantic model validation and serialization
- **WebSocket Manager**: Connection management and message broadcasting
- **Training Metrics Server**: API endpoints and session management
- **Integration Utilities**: Training engine and spectator manager integration
- **Configuration Management**: Configuration loading and validation

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY bot/ ./bot/
EXPOSE 8000

CMD ["python", "-m", "bot.rl_bot_system.server.training_metrics_server"]
```

### Production Configuration

```python
# production_config.py
from bot.rl_bot_system.server.config import ServerConfiguration

config = ServerConfiguration(
    host="0.0.0.0",
    port=8000,
    debug=False,
    database=DatabaseConfig(
        enabled=True,
        url="postgresql://user:pass@db:5432/training_metrics"
    ),
    redis=RedisConfig(
        enabled=True,
        host="redis",
        port=6379
    ),
    security=SecurityConfig(
        enable_auth=True,
        jwt_secret_key="production-secret-key",
        cors_origins=["https://yourdomain.com"]
    ),
    performance=PerformanceConfig(
        max_connections_per_session=100,
        max_total_connections=1000,
        enable_rate_limiting=True
    )
)
```

## Monitoring and Logging

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed server status
curl http://localhost:8000/status
```

### Logging Configuration

```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_metrics_server.log'),
        logging.StreamHandler()
    ]
)

# Set specific log levels
logging.getLogger('bot.rl_bot_system.server').setLevel(logging.DEBUG)
logging.getLogger('uvicorn').setLevel(logging.INFO)
```

## Performance Considerations

### Scaling

- **Horizontal Scaling**: Use multiple server instances behind a load balancer
- **Database Scaling**: Use PostgreSQL with connection pooling for production
- **Redis Caching**: Enable Redis for session data and message caching
- **WebSocket Scaling**: Consider using Redis pub/sub for multi-instance WebSocket support

### Optimization

- **Connection Limits**: Configure appropriate connection limits per session
- **Memory Management**: Set metrics history size limits to prevent memory growth
- **Cleanup Tasks**: Regular cleanup of expired sessions and old data
- **Rate Limiting**: Enable rate limiting to prevent abuse

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failures**
   - Check CORS configuration
   - Verify session exists before connecting
   - Check connection limits

2. **High Memory Usage**
   - Reduce metrics_history_size
   - Enable periodic cleanup
   - Check for connection leaks

3. **Slow Performance**
   - Enable Redis caching
   - Optimize database queries
   - Reduce message broadcast frequency

### Debug Mode

```python
# Enable debug mode for detailed logging
config = ServerConfig(debug=True, log_level="DEBUG")
server = TrainingMetricsServer(config)
```

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest bot/rl_bot_system/server/tests/

# Run linting
flake8 bot/rl_bot_system/server/
black bot/rl_bot_system/server/

# Type checking
mypy bot/rl_bot_system/server/
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public methods
- Include unit tests for new functionality

## License

This project is part of the RL Bot System and follows the same license terms.