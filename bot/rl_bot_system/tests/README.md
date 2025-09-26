# Bot-Game Server Integration Tests

This directory contains comprehensive integration tests that validate communication between the Python bot server and Go game server.

## Overview

The integration tests cover:

- **Bot Spawning**: Testing bot creation through API calls and WebSocket connections
- **Action Execution**: Validating bot actions and game state responses
- **Message Serialization**: Ensuring proper message format between servers
- **Error Handling**: Testing network failures and server disconnections
- **Authentication**: Validating bot authentication and room access control
- **Resource Management**: Testing server limits and resource cleanup

## Requirements

### Dependencies

The tests require the following Python packages:
- `pytest` - Test framework
- `aiohttp` - HTTP client for API calls
- `websockets` - WebSocket client for real-time communication
- `asyncio` - Asynchronous programming support

Install with:
```bash
pip install pytest aiohttp websockets
```

### Go Server

The tests require a working Go game server running on port 4000.

## Running Tests

### Prerequisites

**IMPORTANT**: Before running integration tests, ensure the Go game server is running:

1. Navigate to the `backend/` directory
2. Build and start the Go server:
   ```bash
   go build -o go-ws-server.exe .
   ./go-ws-server.exe
   ```
3. Verify the server is running by visiting `http://localhost:4000/api/maps`

### Quick Start

From the `bot/` directory, run all integration tests:
```bash
python run_test.py rl_bot_system/tests/test_bot_game_server_communication.py
```

### Using the project test runner

```bash
# Run all integration tests
python run_test.py rl_bot_system/tests/test_bot_game_server_communication.py

# Run all tests in the tests directory
python run_test.py rl_bot_system/tests/

# Run with verbose output (default)
python run_test.py rl_bot_system/tests/test_bot_game_server_communication.py -v

# Run with quiet output
python run_test.py rl_bot_system/tests/test_bot_game_server_communication.py -q
```

### Using pytest directly

From the `bot/` directory:
```bash
# Run all integration tests
python -m pytest rl_bot_system/tests/test_bot_game_server_communication.py -v -s

# Run specific test
python -m pytest rl_bot_system/tests/test_bot_game_server_communication.py::TestBotGameServerCommunication::test_bot_spawning_via_api -v -s

# Run standalone tests (no game server required)
python -m pytest rl_bot_system/tests/test_bot_game_server_communication.py::TestBotGameServerCommunication::test_go_server_accessibility -v -s
python -m pytest rl_bot_system/tests/test_bot_game_server_communication.py::TestBotGameServerCommunication::test_bot_configuration_validation -v -s

# Run with integration marker
python -m pytest -m integration -v -s
```

## Test Configuration

### Environment Variables

- `SKIP_INTEGRATION_TESTS=true` - Skip integration tests entirely
- `PYTHONPATH` - Automatically set by test runners

### Test Markers

- `@pytest.mark.integration` - Marks tests as integration tests
- `@pytest.mark.slow` - Marks tests as slow-running

### Server Configuration

Tests automatically:
1. Start Go game server on port 4000
2. Start Python bot server with test configuration
3. Create test rooms as needed
4. Clean up resources after tests

## Test Structure

### TestBotGameServerCommunication

Main test class containing:

#### Server Management Tests
- `test_server_startup_and_health()` - Verify both servers start correctly
- `test_server_resource_limits()` - Test server behavior under limits

#### Bot Communication Tests
- `test_bot_spawning_via_api()` - Test bot creation through API
- `test_bot_websocket_connection()` - Test WebSocket connections
- `test_bot_action_execution_and_state_validation()` - Test action/state flow

#### Message Handling Tests
- `test_message_serialization_deserialization()` - Test message formats
- `test_multiple_bots_same_room()` - Test concurrent bot communication

#### Error Handling Tests
- `test_network_failure_handling()` - Test connection failures
- `test_invalid_server_connection()` - Test invalid connections
- `test_bot_reconnection_after_disconnect()` - Test reconnection logic

#### Authentication Tests
- `test_bot_authentication_and_room_access()` - Test access control

#### Integration Tests
- `test_bot_server_integration_with_game_server()` - Full integration test

## Test Utilities

### ServerManager
Manages test server lifecycle:
- Starts/stops Go and Python servers
- Creates test rooms
- Monitors server health

### EnhancedGameClient
Enhanced game client that extends the original GameClient with testing utilities:
- **Inherits all GameClient functionality**: Full compatibility with the real client
- **Message Capture**: Automatically captures all received messages for validation
- **Test-friendly Methods**: Returns boolean success/failure for easy assertions
- **Message Filtering**: Get messages by type or wait for specific message types
- **Action Sequences**: Send complex test sequences with timing control
- **Connection Info**: Get detailed connection status for debugging

### ServerHealthChecker
Utilities for server health monitoring:
- Check HTTP server availability
- Check WebSocket server connectivity
- Wait for servers to be ready

## Troubleshooting

### Common Issues

1. **Go server fails to start**
   - Ensure Go is installed and in PATH
   - Check if port 4000 is available
   - Verify backend directory structure

2. **Tests timeout**
   - Increase timeout values in test configuration
   - Check server logs for errors
   - Ensure sufficient system resources

3. **Import errors**
   - Verify PYTHONPATH includes bot directory
   - Check all dependencies are installed
   - Run from correct working directory

4. **WebSocket connection failures**
   - Verify Go server WebSocket endpoint is working
   - Check firewall/network settings
   - Ensure proper server startup sequence

### Debug Mode

Run tests with debug logging by setting environment variable:
```bash
export PYTHONPATH=$(pwd)
python run_test.py rl_bot_system/tests/test_bot_game_server_communication.py
```

### Skipping Integration Tests

To skip integration tests during development:
```bash
export SKIP_INTEGRATION_TESTS=true
python -m pytest
```

## Test Coverage

The integration tests validate:

✅ **Server Communication** (11/14 tests passing)
- HTTP API calls between servers
- WebSocket message exchange with correct mouse input format
- Message serialization/deserialization
- Bot authentication and room access control
- Game state validation and action execution

✅ **Basic Bot Operations**
- Bot spawning through API calls
- WebSocket connections to game server
- Network failure handling
- Multiple bots in same room

⚠️ **Known Issues** (3/14 tests failing)
- **Bot AI Loop Integration**: Bot initialization fails with mysterious `DifficultyLevel` exception

## Recent Fixes

✅ **Mouse Input Format**: Fixed button field format (string → integer) for Go server compatibility  
✅ **Game State Validation**: Updated to handle Go server's `objectStates` format  
✅ **Pytest Collection Warnings**: Renamed utility classes to avoid test collection conflicts  
✅ **Enhanced Test Client**: Improved `EnhancedGameClient` with better testing utilities

## Current Status

**Working Tests (11/14):**
- `test_go_server_accessibility` ✅ (Standalone server health check)
- `test_bot_configuration_validation` ✅ (Unit test for bot config)
- `test_server_startup_and_health` ✅
- `test_bot_spawning_via_api` ✅  
- `test_bot_websocket_connection` ✅
- `test_bot_action_execution_and_state_validation` ✅ (Fixed with mouse input correction)
- `test_message_serialization_deserialization` ✅
- `test_network_failure_handling` ✅
- `test_invalid_server_connection` ✅
- `test_bot_authentication_and_room_access` ✅
- `test_multiple_bots_same_room` ✅

**Failing Tests (3/14):**
- `test_bot_server_integration_with_game_server` ❌ (Bot initialization error)
- `test_bot_reconnection_after_disconnect` ❌ (Bot initialization error)
- `test_server_resource_limits` ❌ (Bot initialization error)

**Note:** All core communication validation tests are passing. The 3 failing tests are due to a bot server initialization issue where `DifficultyLevel` enum values are being raised as exceptions instead of proper errors.

## Usage Examples

### Using EnhancedGameClient

```python
from bot.rl_bot_system.tests.test_utils import EnhancedGameClient

# Create enhanced test client
client = EnhancedGameClient(ws_url="ws://localhost:4000/ws", http_url="http://localhost:4000")

# Connect with test-friendly return value
success = await client.connect_to_room("ROOM123", "TestBot", "password")
assert success, "Failed to connect"

# Send action sequences
actions = [
    ("keyboard", ("W", True)),
    ("wait", 0.2),
    ("mouse", ("left", True, 100, 100)),  # Left click (button: "left" or "right")
    ("wait", 0.1),
    ("keyboard", ("W", False)),
]
await client.send_test_sequence(actions)

# Validate received messages
game_states = client.get_messages_by_type("GameState")
assert len(game_states) > 0, "No game states received"

# Get connection info for debugging
info = client.get_connection_info()
print(f"Connected: {info['connected']}, Messages: {info['messages_captured']}")
```

## Contributing

When adding new integration tests:

1. Follow existing test patterns
2. Use appropriate fixtures and utilities
3. Use `EnhancedGameClient` for tests that need message validation
4. Add proper cleanup in finally blocks
5. Document test purpose and requirements
6. Update this README if needed

## Performance Considerations

Integration tests are slower than unit tests because they:
- Start actual server processes
- Make real network connections
- Wait for server initialization
- Perform cleanup operations

Typical test run time: 2-5 minutes depending on system performance.