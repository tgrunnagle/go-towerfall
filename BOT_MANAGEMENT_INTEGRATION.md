# Bot Management Integration

This document describes the bot management API integration between the Go game server and Python bot server.

## Overview

The bot management system allows players to add, remove, and configure AI bots in game rooms through a browser-based interface. The system consists of:

1. **Go Game Server**: Provides REST API endpoints for bot management
2. **Python Bot Server**: Manages bot instances and AI logic
3. **React Frontend**: Browser UI for bot selection and management

## Architecture

```
Browser UI → Go Game Server → Python Bot Server → Game Client → Game Room
```

## API Endpoints

### Go Game Server Endpoints

#### Get Available Bot Types
- **GET** `/api/bots/available`
- Returns list of available bot types and configurations

#### Get Bot Server Status
- **GET** `/api/bots/status`
- Returns bot server health and statistics

#### Add Bot to Room
- **POST** `/api/rooms/{roomId}/bots`
- **Headers**: `Authorization: Bearer {playerToken}`
- **Body**:
  ```json
  {
    "botType": "rules_based",
    "difficulty": "intermediate",
    "botName": "MyBot",
    "generation": 1,
    "trainingMode": false
  }
  ```

#### Get Room Bots
- **GET** `/api/rooms/{roomId}/bots`
- **Headers**: `Authorization: Bearer {playerToken}`
- Returns list of bots in the specified room

#### Remove Bot from Room
- **DELETE** `/api/rooms/{roomId}/bots/{botId}`
- **Headers**: `Authorization: Bearer {playerToken}`
- Removes the specified bot from the room

#### Configure Bot Difficulty
- **PUT** `/api/rooms/{roomId}/bots/{botId}/difficulty`
- **Headers**: `Authorization: Bearer {playerToken}`
- **Body**:
  ```json
  {
    "difficulty": "advanced"
  }
  ```

### Python Bot Server Endpoints

#### Get Available Bot Types
- **GET** `/api/bots/available`
- Returns bot types with metadata

#### Spawn Bot
- **POST** `/api/bots/spawn`
- **Body**:
  ```json
  {
    "bot_type": "rules_based",
    "difficulty": "intermediate",
    "bot_name": "MyBot",
    "room_code": "ABC123",
    "room_password": "",
    "generation": 1,
    "training_mode": false
  }
  ```

#### Terminate Bot
- **POST** `/api/bots/{botId}/terminate`
- Terminates the specified bot instance

#### Configure Bot
- **PUT** `/api/bots/{botId}/configure`
- **Body**:
  ```json
  {
    "difficulty": "advanced"
  }
  ```

#### Get Room Bots
- **GET** `/api/rooms/{roomId}/bots`
- Returns bots in the specified room

#### Get Server Status
- **GET** `/api/bots/status`
- Returns server health and statistics

## Frontend Integration

### Bot Management Panel

The `BotManagementPanel` React component provides:

- **Bot Selection**: Dropdown to choose bot type and difficulty
- **Bot Configuration**: Options for generation selection (RL bots)
- **Current Bots List**: Shows active bots with status and performance
- **Real-time Updates**: Refreshes bot list after operations
- **Error Handling**: Displays user-friendly error messages

### Keyboard Shortcuts

- **B**: Toggle bot management panel (non-spectators only)
- **H**: Toggle training metrics overlay (training rooms)

### Integration Points

1. **GameWrapper Component**: Includes bot management panel
2. **API Layer**: Frontend API functions for bot operations
3. **Game Page**: Instructions for bot management shortcuts

## Bot Types

### Rules-Based Bot
- **Type**: `rules_based`
- **Description**: Traditional AI with configurable difficulty levels
- **Difficulties**: beginner, intermediate, advanced, expert
- **Features**: Survival rules, combat rules, strategic rules

### RL Generation Bot
- **Type**: `rl_generation`
- **Description**: Reinforcement learning trained models
- **Difficulties**: beginner, intermediate, advanced, expert
- **Generations**: 1, 2, 3, ... (based on available models)
- **Features**: Successive learning, model evolution

## Setup Instructions

### 1. Start the Go Game Server
```bash
cd backend
go run main.go
```

### 2. Start the Python Server
```bash
cd bot
python run_server.py
```

### 3. Start the Frontend
```bash
cd frontend
npm start
```

### 4. Test Integration
```bash
python test_bot_integration.py
```

## Configuration

### Bot Server Configuration
- **Host**: localhost
- **Port**: 8001
- **Max Bots**: 50 total, 8 per room
- **Timeout**: 10 seconds for requests

### Game Server Configuration
- **Bot Server URL**: http://localhost:8001
- **Request Timeout**: 10 seconds
- **CORS**: Enabled for all origins (development)

## Error Handling

### Bot Server Unavailable
- Go server returns static bot types as fallback
- Frontend shows appropriate error messages
- Operations gracefully degrade

### Bot Spawning Failures
- Detailed error messages returned to frontend
- Room limits enforced (max 8 bots per room)
- Authorization validation for all operations

### Network Issues
- Request timeouts handled gracefully
- Retry logic for critical operations
- User feedback for all error conditions

## Security

### Authorization
- Player tokens required for all bot operations
- Room-specific authorization validation
- Bot operations limited to room participants

### Input Validation
- Bot names limited to 20 characters
- Difficulty levels validated against available options
- Bot type validation against supported types

## Performance Considerations

### Connection Pooling
- Python bot server uses GameClient connection pool
- Efficient resource management for multiple bots
- Automatic cleanup of inactive connections

### Request Optimization
- Batch operations where possible
- Caching of bot type information
- Minimal API calls for UI updates

## Monitoring and Logging

### Bot Server Logs
- Bot lifecycle events (spawn, terminate, configure)
- Performance metrics and statistics
- Error conditions and recovery attempts

### Game Server Logs
- API request/response logging
- Bot integration status
- Room bot tracking

## Future Enhancements

### Planned Features
1. **Bot Profiles**: Save and load bot configurations
2. **Team Management**: Assign bots to specific teams
3. **Performance Analytics**: Detailed bot performance tracking
4. **Custom Bot Types**: User-defined bot behaviors
5. **Tournament Mode**: Automated bot tournaments

### Scalability Improvements
1. **Distributed Bot Server**: Multiple bot server instances
2. **Load Balancing**: Distribute bots across servers
3. **Database Integration**: Persistent bot configurations
4. **Metrics Collection**: Comprehensive performance monitoring

## Troubleshooting

### Common Issues

#### Bot Server Not Starting
- Check Python dependencies: `pip install -r bot/requirements.txt`
- Verify port 8001 is available
- Check for import errors in bot modules

#### Bots Not Appearing in Game
- Verify bot server is running and accessible
- Check room codes and passwords match
- Ensure WebSocket connections are established

#### Frontend Errors
- Check browser console for JavaScript errors
- Verify API endpoints are accessible
- Check CORS configuration

### Debug Commands

```bash
# Check bot server health
curl http://localhost:8001/health

# Get available bot types
curl http://localhost:4000/api/bots/available

# Get bot server status
curl http://localhost:4000/api/bots/status
```

## Contributing

When adding new bot types or features:

1. Update bot server API endpoints
2. Add corresponding Go server handlers
3. Update frontend components and API calls
4. Add tests for new functionality
5. Update documentation