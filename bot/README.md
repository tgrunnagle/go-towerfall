# RL Bot System

A comprehensive reinforcement learning bot system for game AI development, training, and evaluation.

## Quick Start

### 1. Setup
```bash
# Install dependencies and configure the environment
python run_setup.py
```

### 2. Run RL Bot Training Server
```bash
# Start the unified server with training metrics and episode replay
python run_server.py
```

### 3. Run Examples
```bash
# Run example scripts with proper path setup
python run_example.py rl_bot_system/training/examples/example_training_engine_usage.py
```

### 4. Run Tests
```bash
# Run all tests or specific test files
python run_test.py
python run_test.py rl_bot_system/server/tests/
```

## Run Scripts Overview

The bot directory contains several `run_*` scripts that provide convenient entry points for different aspects of the system:

**Available Scripts:**
- `run_server.py` - Unified RL Bot Training Server with metrics and replay
- `run_setup.py` - Environment setup and dependency installation  
- `run_example.py` - Example script runner with proper imports
- `run_test.py` - Test runner with pytest integration

### 🚀 `run_server.py` - RL Bot Training Server

**Purpose:** Starts a unified FastAPI server with training metrics, episode replay, and spectator functionality.

**Usage:**
```bash
python run_server.py [options]
```

**Options:**
- `--host HOST` - Server host (default: localhost)
- `--port PORT` - Server port (default: 4002)
- `--log-level LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `--config CONFIG` - Configuration file path (JSON format)
- `--simple` - Force use of simple server (basic metrics only)

**Features:**
- **Training Metrics API:** Real-time training metrics and session management
- **Episode Replay API:** Browse, replay, and compare recorded episodes
- **WebSocket Support:** Real-time updates for spectator interfaces
- **Spectator Integration:** Coordinate with spectator management system
- **Data Persistence:** Store training data and replay sessions

**Configuration File Example:**
```json
{
  "host": "localhost",
  "port": 4002,
  "cors_origins": ["http://localhost:3000", "http://localhost:4000"],
  "max_connections_per_session": 50,
  "metrics_history_size": 10000,
  "cleanup_interval_seconds": 300,
  "log_level": "INFO",
  "game_server_url": "http://localhost:4000",
  "enable_spectator_integration": true,
  "enable_replay_integration": true,
  "data_storage_path": "data/training_metrics",
  "replay_storage_path": "data/replays",
  "enable_data_persistence": true
}
```

**API Endpoints:**
- `GET /health` - Health check
- `GET /status` - Server status and metrics
- `GET /info` - Server information and available features
- `POST /api/training/sessions` - Create training session
- `GET /api/training/sessions` - List active sessions
- `POST /api/training/sessions/{id}/metrics` - Update training metrics
- `GET /api/replay/sessions` - List replay sessions
- `POST /api/replay/start` - Start episode replay
- `WS /ws/{session_id}` - Real-time metrics WebSocket

**Example:**
```bash
# Start server with default settings
python run_server.py

# Start server on custom port
python run_server.py --port 4003 --log-level DEBUG

# Start with configuration file
python run_server.py --config server_config.json

# Force simple server (basic metrics only)
python run_server.py --simple
```

### 🔧 `run_setup.py` - Environment Setup

**Purpose:** Installs dependencies, configures the environment, and sets up the RL bot system.

**Usage:**
```bash
python run_setup.py [options]
```

**What it does:**
- ✅ Installs Python dependencies from requirements.txt
- ✅ Sets up virtual environment if needed
- ✅ Configures logging directories
- ✅ Validates system requirements
- ✅ Creates necessary data directories

**Example:**
```bash
# Basic setup
python run_setup.py

# Setup with specific options (see setup/setup.py for details)
python run_setup.py --verbose
```

### 🎯 `run_example.py` - Example Script Runner

**Purpose:** Runs example scripts with proper Python path setup and import resolution.

**Usage:**
```bash
python run_example.py <example_script_path>
```

**Available Examples:**
- `rl_bot_system/training/examples/example_training_engine_usage.py`
- `rl_bot_system/evaluation/examples/simple_evaluation_demo.py`
- `rl_bot_system/models/examples/example_dqn_usage.py`
- `rl_bot_system/spectator/examples/example_spectator_usage.py`
- `rl_bot_system/replay/examples/example_replay_usage.py`

**Features:**
- ✅ Automatic Python path configuration
- ✅ Import resolution for RL bot system modules
- ✅ Error handling and helpful error messages

**Example:**
```bash
# Run training engine example
python run_example.py rl_bot_system/training/examples/example_training_engine_usage.py

# Run evaluation demo
python run_example.py rl_bot_system/evaluation/examples/simple_evaluation_demo.py
```

### 🧪 `run_test.py` - Test Runner

**Purpose:** Runs tests with proper path setup and pytest configuration.

**Usage:**
```bash
python run_test.py [target] [options]
```

**Parameters:**
- `target` - Test file, directory, or "." for all tests (default: ".")
- `-v, --verbose` - Verbose output (default: True)
- `-q, --quiet` - Quiet output (overrides verbose)
- `-c, --capture` - Capture output (don't show print statements)

**Features:**
- ✅ Automatic Python path setup
- ✅ Integration with pytest
- ✅ Recursive test discovery
- ✅ Proper import resolution
- ✅ Virtual environment support (uses `uv run pytest`)

**Examples:**
```bash
# Run all tests
python run_test.py

# Run specific test file
python run_test.py rl_bot_system/server/tests/test_data_models.py

# Run tests in a directory
python run_test.py rl_bot_system/training/tests/

# Run tests quietly
python run_test.py rl_bot_system/server/tests/ -q

# Run with output capture
python run_test.py rl_bot_system/server/tests/ -c
```

## System Architecture

```
bot/
├── run_server.py          # Unified RL Bot Training Server
├── run_setup.py           # Environment setup and configuration
├── run_example.py         # Example script runner
├── run_test.py            # Test runner with pytest integration
├── rl_bot_system/         # Core RL bot system modules
│   ├── server/            # Server components (APIs, WebSocket, data models)
│   ├── training/          # Training engines and algorithms
│   ├── models/            # RL model implementations (DQN, PPO, A2C)
│   ├── evaluation/        # Model evaluation and testing
│   ├── spectator/         # Real-time training observation
│   └── replay/            # Experience replay and analysis
├── config/                # Configuration files
├── data/                  # Training data and metrics storage
├── logs/                  # Log files
└── setup/                 # Setup scripts and utilities
```

## Development Workflow

### 1. Initial Setup
```bash
# Clone repository and navigate to bot directory
cd bot/

# Run setup to install dependencies and configure environment
python run_setup.py
```

### 2. Development
```bash
# Start the RL bot training server for development
python run_server.py --log-level DEBUG

# Run examples to test functionality
python run_example.py rl_bot_system/training/examples/example_training_engine_usage.py

# Run tests during development
python run_test.py rl_bot_system/server/tests/
```

### 3. Testing
```bash
# Run all tests
python run_test.py

# Run specific component tests
python run_test.py rl_bot_system/training/tests/
python run_test.py rl_bot_system/server/tests/
python run_test.py rl_bot_system/models/tests/
```

## Integration with Frontend

The RL Bot Training Server (`run_server.py`) provides comprehensive API endpoints that integrate with the frontend spectator UI:

### WebSocket Connection
```javascript
// Connect to real-time metrics (default port 4002)
const ws = new WebSocket('ws://localhost:4002/ws/training_session_123?user_name=Spectator');

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    switch (message.type) {
        case 'training_metrics':
            updateTrainingMetrics(message.data);
            break;
        case 'bot_decision':
            updateBotDecisionVisualization(message.data);
            break;
        case 'graph_update':
            updatePerformanceGraphs(message.data);
            break;
        case 'replay_frame':
            updateReplayVisualization(message.data);
            break;
    }
};
```

### REST API Usage
```javascript
// Create training session
const response = await fetch('http://localhost:4002/api/training/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        training_session_id: 'session_123',
        model_generation: 2,
        algorithm: 'DQN',
        total_episodes: 1000,
        room_code: 'ABC123'
    })
});

// Start episode replay
const replayResponse = await fetch('http://localhost:4002/api/replay/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        session_id: 'session_123',
        episode_id: 'episode_456',
        controls: {
            playback_speed: 1.0,
            show_frame_info: true
        }
    })
});

// Get server status and features
const status = await fetch('http://localhost:4002/status').then(r => r.json());
const info = await fetch('http://localhost:4002/info').then(r => r.json());
```

## Configuration

### Server Configuration
Create a `server_config.json` file (example provided):
```json
{
  "host": "localhost",
  "port": 4002,
  "cors_origins": [
    "http://localhost:3000",
    "http://localhost:4000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:4000"
  ],
  "max_connections_per_session": 50,
  "metrics_history_size": 10000,
  "cleanup_interval_seconds": 300,
  "log_level": "INFO",
  "game_server_url": "http://localhost:4000",
  "enable_spectator_integration": true,
  "enable_replay_integration": true,
  "data_storage_path": "data/training_metrics",
  "replay_storage_path": "data/replays",
  "enable_data_persistence": true
}
```

### Environment Variables
```bash
# Server settings
export RL_BOT_SERVER_HOST=localhost
export RL_BOT_SERVER_PORT=4002
export RL_BOT_SERVER_DEBUG=false

# Integration settings
export GAME_SERVER_URL=http://localhost:4000
export DATA_STORAGE_PATH=data/training_metrics
export REPLAY_STORAGE_PATH=data/replays
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Make sure you're running from the bot/ directory
cd bot/
python run_server.py
```

**Port Already in Use:**
```bash
# Use a different port
python run_server.py --port 4003
```

**Missing Dependencies:**
```bash
# Re-run setup
python run_setup.py
```

**Test Failures:**
```bash
# Run tests with verbose output to see details
python run_test.py -v
```

### Getting Help

- Check the logs in `logs/` directory
- Run with `--log-level DEBUG` for detailed output
- Use `python run_test.py` to verify system integrity
- Check `requirements.txt` for dependency versions

## Server Architecture

The RL Bot Training Server uses a modular architecture with separate API modules:

### Unified Server (`rl_bot_system/server/server.py`)
- **Main server class:** `UnifiedServer` - Combines all functionality
- **Configuration:** `ServerConfig` - Centralized configuration management
- **Features:** Training metrics, episode replay, WebSocket communication

### API Modules
- **Training Metrics API** (`training_metrics_api.py`): Session management, metrics updates, historical data
- **Replay API** (`replay_api.py`): Episode browsing, replay controls, comparison features
- **WebSocket Manager** (`websocket_manager.py`): Real-time communication with frontend

### Fallback Support
- **Simple Server:** Basic training metrics functionality when full features unavailable
- **Graceful Degradation:** Automatically falls back if dependencies missing
- **Standalone Operation:** Can run without complex integrations

### Integration Points
- **Spectator Manager:** Coordinate spectator sessions and room management
- **Replay Manager:** Handle episode storage and retrieval
- **Training Engine:** Automatic metrics collection during training

### Episode Replay Features
- **Episode Browser:** Browse and filter recorded training episodes
- **Replay Controls:** Play, pause, rewind, step-by-step navigation
- **Speed Control:** Variable playback speed (0.25x to 4x)
- **Episode Comparison:** Side-by-side comparison of multiple episodes
- **Metrics Analysis:** Detailed performance metrics and statistics
- **WebSocket Integration:** Real-time replay updates to frontend

## Contributing

1. Run setup: `python run_setup.py`
2. Make changes to the codebase
3. Run tests: `python run_test.py`
4. Test examples: `python run_example.py <example_path>`
5. Test server: `python run_server.py --log-level DEBUG`
6. Submit pull request

## Quick Reference

### Common Commands
```bash
# Setup and start server
python run_setup.py
python run_server.py

# Run with custom configuration
python run_server.py --config server_config.json

# Development mode with debug logging
python run_server.py --log-level DEBUG

# Run tests
python run_test.py

# Run examples
python run_example.py rl_bot_system/training/examples/example_training_engine_usage.py
```

### Default Ports and URLs
- **Server:** http://localhost:4002
- **WebSocket:** ws://localhost:4002/ws/{session_id}
- **Health Check:** http://localhost:4002/health
- **Server Status:** http://localhost:4002/status
- **API Documentation:** http://localhost:4002/docs (when server is running)

## License

This project is part of the RL Bot System and follows the same license terms.