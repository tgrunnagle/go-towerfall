# Training Mode Server-Side Requirements

This document outlines the server-side modifications and requirements needed to support training mode functionality for RL bot training.

## Overview

The training mode extensions enable accelerated game simulation, direct state access, and variable tick rates to support efficient reinforcement learning bot training. This document describes the server-side components that have been implemented and any additional requirements.

## Implemented Components

### 1. Training Room API Endpoints

The following new API endpoints have been added to support training mode:

#### POST /api/training/createRoom
Creates a new training room with speed control capabilities.

**Request Body:**
```json
{
  "playerName": "RL Bot",
  "roomName": "Training Room",
  "mapType": "default",
  "trainingConfig": {
    "speedMultiplier": 10.0,
    "headlessMode": false,
    "trainingMode": true,
    "sessionId": "optional-session-id",
    "directStateAccess": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "playerId": "uuid",
  "playerToken": "uuid",
  "roomId": "uuid",
  "roomCode": "ABC123",
  "roomName": "Training Room",
  "canvasSizeX": 800,
  "canvasSizeY": 600,
  "speedMultiplier": 10.0,
  "headlessMode": false
}
```

#### POST /api/training/joinRoom
Joins an existing training room with training capabilities.

**Request Body:**
```json
{
  "playerName": "RL Bot",
  "roomCode": "ABC123",
  "roomPassword": "PASSWORD",
  "enableDirectAccess": true
}
```

#### POST /api/training/rooms/{roomId}/speed
Sets the speed multiplier for a training room.

**Request Body:**
```json
{
  "speedMultiplier": 15.0
}
```

**Authorization:** Requires `Authorization: Bearer {playerToken}` header.

#### GET /api/training/rooms/{roomId}/state
Gets the current game state directly without WebSocket communication.

**Authorization:** Requires `Authorization: Bearer {playerToken}` header.

**Response:**
```json
{
  "success": true,
  "state": {
    "objects": {...},
    "room": {...},
    "players": [...],
    "map": {...}
  },
  "timestamp": 1634567890123
}
```

#### POST /api/training/rooms/{roomId}/configure
Configures training-specific settings for a room.

**Request Body:**
```json
{
  "trainingMode": "headless",
  "speedMultiplier": 25.0,
  "directStateAccess": true
}
```

### 2. GameRoom Extensions

The `GameRoom` struct has been extended with training-specific fields and methods:

#### New Fields
- `IsTraining bool` - Indicates if room supports training features
- `SpeedMultiplier float64` - Current speed multiplier (1.0 = normal speed)
- `HeadlessMode bool` - Enables headless mode for maximum speed
- `DirectStateAccess bool` - Enables direct state access API
- `TrainingSessionID string` - Optional training session identifier
- `CustomTickRate time.Duration` - Custom tick rate based on speed multiplier

#### New Methods
- `IsTrainingRoom() bool` - Check if room supports training
- `GetSpeedMultiplier() float64` - Get current speed multiplier
- `SetSpeedMultiplier(float64) error` - Set speed multiplier
- `ConfigureTraining(string, float64, bool) error` - Configure training settings
- `GetDirectGameState() map[string]interface{}` - Get comprehensive game state
- `ValidatePlayerToken(string) bool` - Validate player authorization
- `GetCustomTickRate() time.Duration` - Get custom tick rate

### 3. Variable Tick Rate System

The server's game tick system has been modified to support variable tick rates per room:

- **Base Tick Rate:** 20ms (50 FPS) for normal rooms
- **Training Tick Rate:** Calculated as `baseTickRate / speedMultiplier`
- **Per-Room Tickers:** Each room gets its own ticker with appropriate rate
- **Dynamic Updates:** Tick rates update automatically when speed multiplier changes

#### Implementation Details
- `runGameTick()` - Main loop managing room-specific tickers
- `runRoomTicker(roomID, ticker)` - Individual room ticker goroutine
- Automatic cleanup of tickers for removed rooms
- Support for speed multipliers from 0.1x to 100x

### 4. Training Room Factory Functions

New factory functions for creating training rooms:

- `NewTrainingGameRoom()` - Creates a room with training capabilities
- `NewTrainingGameWithPlayer()` - Creates training room with initial player

## Server-Side Requirements

### 1. Game Engine Modifications

The following modifications are required in the game engine:

#### Variable Tick Rate Support ✅ IMPLEMENTED
- Modified `runGameTick()` to support per-room tick rates
- Each training room gets its own ticker based on speed multiplier
- Automatic tick rate updates when speed changes

#### Headless Mode Support ⚠️ PARTIALLY IMPLEMENTED
- **Current Status:** Room flag exists, but rendering pipeline not modified
- **Required:** Disable rendering pipeline for headless rooms
- **Implementation:** Modify game state update logic to skip rendering operations
- **Benefits:** Significant performance improvement for high-speed training

#### State Serialization Optimization ⚠️ NEEDS OPTIMIZATION
- **Current Status:** Basic state extraction implemented
- **Required:** Optimized state format for ML consumption
- **Implementation:** Create lightweight state representation
- **Benefits:** Faster state access and reduced memory usage

### 2. Resource Management

#### Memory Management ⚠️ NEEDS IMPLEMENTATION
- **Required:** Monitor memory usage for high-speed training rooms
- **Implementation:** Add memory limits and cleanup for training sessions
- **Benefits:** Prevent memory leaks during extended training

#### CPU Throttling ⚠️ NEEDS IMPLEMENTATION
- **Required:** Limit concurrent high-speed training rooms
- **Implementation:** Add resource limits and queuing system
- **Benefits:** Prevent server overload

### 3. Network Optimization

#### WebSocket Bypass ✅ IMPLEMENTED
- Direct HTTP API for state access bypasses WebSocket overhead
- Reduces latency for training state retrieval
- Maintains WebSocket for real-time updates when needed

#### Batch State Updates ⚠️ FUTURE ENHANCEMENT
- **Optional:** Support for batch state requests
- **Benefits:** Further reduce network overhead for parallel training

## Configuration

### Environment Variables

The following environment variables can be used to configure training mode:

```bash
# Maximum speed multiplier allowed (default: 100.0)
MAX_SPEED_MULTIPLIER=100.0

# Maximum concurrent training rooms (default: 10)
MAX_TRAINING_ROOMS=10

# Enable headless mode optimizations (default: true)
ENABLE_HEADLESS_OPTIMIZATIONS=true

# Training room timeout in minutes (default: 60)
TRAINING_ROOM_TIMEOUT=60
```

### Server Configuration

Add to server configuration:

```go
type ServerConfig struct {
    MaxSpeedMultiplier    float64
    MaxTrainingRooms      int
    HeadlessOptimizations bool
    TrainingRoomTimeout   time.Duration
}
```

## Performance Considerations

### Speed Multiplier Limits
- **1x - 10x:** Safe for all scenarios
- **10x - 50x:** Recommended for training with visual feedback
- **50x - 100x:** Headless mode recommended
- **100x+:** May require additional optimizations

### Resource Usage
- **CPU:** Scales linearly with speed multiplier
- **Memory:** Minimal increase for training features
- **Network:** Reduced due to WebSocket bypass

### Monitoring
- Track training room count and resource usage
- Monitor tick rate performance and accuracy
- Log speed multiplier changes and performance impact

## Testing

### Unit Tests
- Training room creation and configuration
- Speed multiplier validation and updates
- Direct state access functionality
- Player token validation

### Integration Tests
- End-to-end training room workflow
- Variable tick rate accuracy
- State consistency at high speeds
- Resource cleanup and management

### Performance Tests
- Speed multiplier performance impact
- Memory usage during extended training
- Concurrent training room limits
- State access latency measurements

## Future Enhancements

### 1. Advanced State Representations
- Compressed state formats for faster transmission
- Configurable state filtering for specific training needs
- Binary state formats for maximum performance

### 2. Training Analytics
- Real-time training metrics collection
- Performance profiling and optimization suggestions
- Training session replay and analysis

### 3. Distributed Training Support
- Multi-server training coordination
- Load balancing for training rooms
- Shared state synchronization

### 4. Advanced Speed Control
- Dynamic speed adjustment based on training progress
- Automatic speed optimization for hardware capabilities
- Speed ramping for gradual acceleration

## Conclusion

The training mode server-side implementation provides a solid foundation for RL bot training with:

- ✅ Complete API endpoints for training room management
- ✅ Variable tick rate system for speed control
- ✅ Direct state access bypassing WebSocket overhead
- ✅ Comprehensive game state extraction
- ⚠️ Partial headless mode support (rendering optimizations needed)
- ⚠️ Basic resource management (monitoring and limits needed)

The implementation supports speed multipliers up to 100x and provides the necessary infrastructure for efficient RL training while maintaining game physics accuracy.