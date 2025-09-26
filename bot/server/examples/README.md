# Bot Server Examples

This directory contains example scripts demonstrating how to use the bot server components.

## Available Examples

### bot_server_example.py

A comprehensive example showing how to use the BotServer to manage bot instances.

**Features demonstrated:**
- Bot server setup and configuration
- Spawning rules-based bots
- Real-time difficulty configuration
- Bot health monitoring
- Server status monitoring
- Bot termination and cleanup

**Prerequisites:**
- Go game server running on localhost:4000
- Python dependencies installed

**Usage:**
```bash
cd bot
python -m rl_bot_system.server.examples.bot_server_example
```

**What it does:**
1. Checks if the game server is running
2. Creates a test room
3. Starts the bot server
4. Spawns multiple bots with different configurations
5. Demonstrates bot management operations
6. Monitors bot health and performance
7. Cleans up all resources

## Running Examples

### Prerequisites

1. **Start the Go game server:**
   ```bash
   cd backend
   go run main.go
   ```

2. **Verify server is running:**
   ```bash
   curl http://localhost:4000/api/maps
   ```

### Run Examples

From the `bot/` directory:

```bash
# Run the bot server example
python -m rl_bot_system.server.examples.bot_server_example
```

## Example Output

When running successfully, you should see output like:

```
INFO - Checking if game server is running...
INFO - Game server is running ✓
INFO - Creating test room...
INFO - Created test room: ABC123 (ID: room_xyz)
INFO - Bot server started ✓
INFO - Spawning rules-based bot...
INFO - Spawned bot with ID: bot_001
INFO - Bot status: active
INFO - Difficulty changed successfully ✓
INFO - Server status: 3 bots active
INFO - Example completed successfully! ✓
```

## Troubleshooting

**Game server not running:**
- Ensure the Go server is started on port 4000
- Check that no other process is using port 4000

**Import errors:**
- Run from the `bot/` directory
- Ensure all dependencies are installed

**Bot spawn failures:**
- Check game server logs for errors
- Verify room creation succeeded
- Ensure sufficient system resources

## Adding New Examples

When creating new examples:

1. Follow the existing pattern in `bot_server_example.py`
2. Include proper error handling and cleanup
3. Add comprehensive logging
4. Check prerequisites before running
5. Update this README with the new example

## Integration with Tests

These examples can be used as reference for:
- Integration test patterns
- Bot server usage patterns
- Error handling approaches
- Resource cleanup strategies