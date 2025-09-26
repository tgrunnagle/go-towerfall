# Design Document

## Overview

The bot activity diagnosis system addresses the issue where bots are successfully added through the frontend but remain inactive in the game. The current architecture has the frontend communicating directly with the Python bot server, which then spawns bots that should connect to the Go game server via WebSocket. The issue lies in the communication chain between bot spawning and actual bot activity in the game.

## Architecture

The current system has the following communication flow:

```
Frontend (React) → Python Bot Server → Game Client → WebSocket → Go Game Server
```

### Current State Analysis

1. **Frontend**: Successfully sends bot management requests to Python bot server
2. **Python Bot Server**: Receives requests and attempts to spawn bots
3. **Game Client**: Should connect spawned bots to game via WebSocket
4. **Go Game Server**: Should receive bot connections and integrate them into game

### Root Cause Investigation Areas

The bot inactivity could be caused by failures at several points:

1. **Bot Spawning**: Python bot server may fail to properly initialize bot instances
2. **WebSocket Connection**: Bots may fail to establish WebSocket connections to game server
3. **Game Integration**: Bots may connect but fail to be recognized as active players
4. **AI Execution**: Bot AI logic may not be executing or sending game actions
5. **Game State Synchronization**: Bots may not be receiving or processing game state updates

## Components and Interfaces

### 1. Bot Spawning Diagnostics

Enhanced logging and monitoring in the Python bot server to track:
- Bot instance creation success/failure
- Game client initialization status
- WebSocket connection attempts and results
- Bot AI initialization and execution status

### 2. WebSocket Connection Monitoring

Diagnostic tools to monitor bot WebSocket connections:
- Connection establishment tracking
- Message send/receive logging on errors
- Connection health monitoring
- Reconnection attempt tracking

### 3. Game Integration Verification

Tools to verify bots are properly integrated into the game:
- Player registration confirmation
- Game state synchronization status
- Action command processing verification
- Bot visibility in game world

### 4. Selective Enhanced Logging

Implement targeted logging focused on problem identification:
- Bot lifecycle critical events (spawn, connect, disconnect, error states)
- WebSocket connection failures and reconnection attempts
- Bot AI initialization failures and error conditions
- Unusual bot behavior patterns (no actions sent, connection drops)
- Avoid logging routine operations like normal game state updates or successful actions

## Data Models

### Bot Diagnostic Information

```python
@dataclass
class BotDiagnosticInfo:
    bot_id: str
    status: BotStatus
    connection_status: str
    last_activity: datetime
    error_messages: List[str]
    game_client_status: str
    websocket_connected: bool
    ai_initialized: bool
    actions_sent: int
    game_state_updates_received: int
```

### Connection Health Status

```python
@dataclass
class ConnectionHealth:
    websocket_connected: bool
    last_ping: Optional[datetime]
    connection_errors: List[str]
    reconnection_attempts: int
    message_queue_size: int
```

### Bot Activity Metrics

```python
@dataclass
class BotActivityMetrics:
    decisions_made: int
    actions_executed: int
    game_state_updates: int
    errors_encountered: int
    uptime_seconds: float
    last_decision_time: Optional[datetime]
```

## Error Handling

### 1. Bot Spawning Failures
- Detailed logging of bot initialization steps
- Capture and report specific failure points
- Implement retry mechanisms for transient failures
- Validate room existence and capacity before spawning

### 2. WebSocket Connection Issues
- Monitor connection establishment process
- Log connection failures with specific error codes
- Implement exponential backoff for reconnection attempts
- Track connection health and stability

### 3. Game Integration Problems
- Verify bot player registration in game server
- Monitor game state synchronization
- Track bot action processing and responses
- Detect and report bot invisibility issues

### 4. AI Execution Failures
- Monitor bot decision-making processes
- Log AI initialization and execution errors
- Track bot response times and performance
- Implement fallback behaviors for AI failures

## Testing Strategy

### 1. Integration Testing
- Test complete flow: Frontend → Go Server → Python Bot Server → Game
- Verify bot appears in game and performs actions
- Test error scenarios and recovery

### 2. Real Component Testing
- Integration tests using actual Python bot server instances
- End-to-end testing with real WebSocket connections
- Validate request/response formats with live services

### 3. Bot Behavior Validation
- Verify bots connect to WebSocket and send game actions
- Test bot AI decision-making and movement
- Validate bot cleanup when players leave

### 4. Load Testing
- Test multiple bots in single room
- Test bot server capacity limits
- Verify performance under load

## Diagnostic Tools

### 1. Selective Logging
- Structured logging for critical bot operations and failures only
- Error-focused logging with correlation IDs for troubleshooting
- Bot lifecycle event tracking for state changes and problems

### 2. Health Check Endpoints
- Bot server connectivity status
- Individual bot health status
- Room bot statistics

### 3. Debug Information
- Bot decision-making logs
- WebSocket connection status
- Game state synchronization status

### 4. Frontend Status Display
- Real-time bot status indicators
- Error message display
- Bot performance metrics

## Implementation Phases

### Phase 1: Diagnostic Infrastructure
- Implement comprehensive logging throughout bot lifecycle
- Add bot status tracking and health monitoring
- Create diagnostic endpoints for troubleshooting

### Phase 2: Connection Monitoring
- Monitor WebSocket connection establishment and health
- Track message flow between bots and game server
- Implement connection recovery mechanisms

### Phase 3: Activity Verification
- Verify bot AI execution and decision-making
- Monitor bot actions and game state synchronization
- Implement automated bot behavior validation

## Security Considerations

### 1. Diagnostic Data Protection
- Ensure diagnostic logs don't expose sensitive information
- Implement appropriate access controls for diagnostic endpoints
- Sanitize bot configuration data in logs

### 2. Resource Monitoring
- Monitor bot resource usage and prevent abuse
- Implement safeguards against bot spawning attacks
- Track and limit diagnostic data collection

## Performance Considerations

### 1. Efficient Diagnostics
- Focus logging on errors and state changes, not routine operations
- Use asynchronous logging to avoid blocking bot operations
- Implement configurable diagnostic levels (ERROR, WARN, INFO)
- Avoid logging expected events like successful game state updates

### 2. Resource Management
- Monitor bot memory and CPU usage
- Implement cleanup for diagnostic data
- Optimize WebSocket connection handling

### 3. Scalable Monitoring
- Design diagnostic systems to scale with bot count
- Use efficient data structures for tracking bot status
- Implement periodic cleanup of old diagnostic data