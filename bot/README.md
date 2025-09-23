# RL Bot System

A comprehensive reinforcement learning bot system for game AI development, training, and evaluation.

## Quick Start

### 1. Setup
```bash
# Install dependencies and configure the environment
python run_setup.py
```

### 2. Run Training Metrics Server
```bash
# Start the FastAPI server for real-time training metrics
python run_server.py --port 8001
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

### 🚀 `run_server.py` - Training Metrics Server

**Purpose:** Starts a FastAPI server for real-time training metrics and spectator functionality.

**Usage:**
```bash
python run_server.py [options]
```

**Options:**
- `--host HOST` - Server host (default: localhost)
- `--port PORT` - Server port (default: 4002)
- `--log-level LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `--config CONFIG` - Configuration file path

**Features:**
- ✅ REST API endpoints for training session management
- ✅ WebSocket endpoints for real-time metrics streaming
- ✅ Health check and server status endpoints
- ✅ CORS support for frontend integration
- ✅ Standalone operation without complex dependencies

**API Endpoints:**
- `GET /health` - Health check
- `GET /status` - Server status and metrics
- `POST /api/training/sessions` - Create training session
- `GET /api/training/sessions` - List active sessions
- `WS /ws/{session_id}` - Real-time metrics WebSocket

**Example:**
```bash
# Start server on custom port
python run_server.py --port 8001 --log-level DEBUG

# Start with configuration file
python run_server.py --config config/server.json
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
├── run_server.py          # FastAPI training metrics server
├── run_setup.py           # Environment setup and configuration
├── run_example.py         # Example script runner
├── run_test.py            # Test runner with pytest integration
├── rl_bot_system/         # Core RL bot system modules
│   ├── server/            # Training metrics server components
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
# Start the training metrics server for development
python run_server.py --port 8001 --log-level DEBUG

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

The training metrics server (`run_server.py`) provides API endpoints that integrate with the frontend spectator UI:

### WebSocket Connection
```javascript
// Connect to real-time metrics
const ws = new WebSocket('ws://localhost:8001/ws/training_session_123?user_name=Spectator');

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
    }
};
```

### REST API Usage
```javascript
// Create training session
const response = await fetch('http://localhost:8001/api/training/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        training_session_id: 'session_123',
        model_generation: 2,
        algorithm: 'DQN',
        total_episodes: 1000
    })
});

// Get server status
const status = await fetch('http://localhost:8001/status').then(r => r.json());
```

## Configuration

### Server Configuration
Create a `config/server.json` file:
```json
{
    "host": "localhost",
    "port": 8001,
    "cors_origins": ["http://localhost:3000", "http://localhost:4000"],
    "max_connections_per_session": 50,
    "log_level": "INFO",
    "data_storage_path": "data/training_metrics"
}
```

### Environment Variables
```bash
# Server settings
export TRAINING_METRICS_HOST=localhost
export TRAINING_METRICS_PORT=8001
export TRAINING_METRICS_DEBUG=false

# Integration settings
export GAME_SERVER_URL=http://localhost:4000
export DATA_STORAGE_PATH=data/training_metrics
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
python run_server.py --port 8002
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

## Contributing

1. Run setup: `python run_setup.py`
2. Make changes to the codebase
3. Run tests: `python run_test.py`
4. Test examples: `python run_example.py <example_path>`
5. Test server: `python run_server.py --port 8001`
6. Submit pull request

## License

This project is part of the RL Bot System and follows the same license terms.