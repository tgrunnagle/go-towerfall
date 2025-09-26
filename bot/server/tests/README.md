# Bot Server Integration Tests

This directory contains integration tests for the bot server components that run against a deployed game server.

## Prerequisites

Before running these tests, you need to have the Go game server running:

1. Start the Go game server on port 4000:
   ```bash
   cd backend
   go run main.go
   ```

2. Verify the server is accessible:
   ```bash
   curl http://localhost:4000/api/maps
   ```

## Running Tests

### Using the project test runner (Recommended)

From the `bot/` directory:

```bash
# Run all server integration tests
python run_test.py rl_bot_system/server/tests/

# Run specific test files
python run_test.py rl_bot_system/server/tests/test_bot_lifecycle.py
python run_test.py rl_bot_system/server/tests/test_bot_server.py
python run_test.py rl_bot_system/server/tests/test_bot_lifecycle_api.py

# Run with verbose output (default)
python run_test.py rl_bot_system/server/tests/test_bot_lifecycle.py -v

# Run with quiet output
python run_test.py rl_bot_system/server/tests/test_bot_lifecycle.py -q

# Run specific test method
python run_test.py rl_bot_system/server/tests/test_bot_lifecycle.py::test_bot_spawning_and_termination
```

### Using pytest directly

```bash
cd bot
# Run all server integration tests
python -m pytest rl_bot_system/server/tests/ -v -s

# Run specific test files
python -m pytest rl_bot_system/server/tests/test_bot_lifecycle.py -v -s

# Run specific test method
python -m pytest rl_bot_system/server/tests/test_bot_lifecycle.py::test_bot_spawning_and_termination -v -s
```

### Skip integration tests:
If you want to skip tests that require a deployed server:
```bash
SKIP_INTEGRATION_TESTS=true python run_test.py rl_bot_system/server/tests/
```

## Test Structure

- **test_bot_lifecycle.py**: Tests bot spawning, termination, difficulty configuration, health monitoring, and cleanup (6 tests)
- **test_bot_lifecycle_api.py**: Tests the API layer integration with bot lifecycle management (4 tests)
- **test_bot_server.py**: Tests the core BotServer functionality including client pool management (25 tests)
- **test_data_models.py**: Unit tests for data models (no server required) (18 tests)
- **test_websocket_manager.py**: Unit tests for WebSocket management (no server required) (12 tests)

**Total: 65 tests - All passing ✅**

## Test Markers

Tests are marked with the following pytest markers:

- `@pytest.mark.server_integration`: Tests that require a deployed game server
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.integration`: General integration tests

## Test Results

✅ **All 65 tests passing**
- **Bot Lifecycle Tests**: 6/6 passing - Real bot spawning, termination, difficulty changes, health monitoring
- **API Integration Tests**: 4/4 passing - Bot server API functionality and integration
- **Bot Server Tests**: 25/25 passing - Core server functionality, client pool management, resource limits
- **Data Models Tests**: 18/18 passing - Pydantic model validation and serialization
- **WebSocket Manager Tests**: 12/12 passing - WebSocket connection management and message broadcasting

## Notes

- Integration tests automatically create and clean up game rooms
- Each test class uses fixtures to manage server lifecycle
- Tests wait for bot initialization before making assertions
- Failed tests will attempt to clean up spawned bots
- Integration tests are skipped if the game server is not accessible
- Unit tests (data models, websocket manager) run without requiring external servers

## Troubleshooting

If tests fail:

1. Ensure the Go game server is running on port 4000
2. Check that no other processes are using the required ports
3. Verify network connectivity to localhost
4. Check the test logs for specific error messages

For more detailed logging, run tests with:
```bash
# Using run_test.py (recommended)
python run_test.py rl_bot_system/server/tests/ -v

# Using pytest directly
python -m pytest rl_bot_system/server/tests/ -v -s --log-cli-level=INFO
```