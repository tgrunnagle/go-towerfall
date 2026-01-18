# Backend Architecture - Go Game Server

Go-based authoritative game server for a Towerfall-inspired arena combat game. Handles real-time physics simulation, WebSocket communication, and HTTP APIs for game management and ML bot integration.

## Project Summary

The backend is a multiplayer game server that:
- Manages game rooms with independent tick loops (default 50 Hz)
- Processes player inputs via WebSocket and HTTP (for bots)
- Runs physics simulation with gravity, collision detection, and screen wrapping
- Supports training mode with accelerated tick rates (up to 20x) for ML bot training
- Tracks player statistics (kills/deaths) per session

## Tech Stack

- **Go 1.24+** with standard library HTTP server
- **gorilla/websocket** - WebSocket support
- **google/uuid** - Unique identifiers
- **gonum** - Numerical/geometry computations

## Directory Structure

```
backend/
├── main.go                           # Entry point, HTTP route registration
├── go.mod / go.sum                   # Go module (go-ws-server)
├── Dockerfile                        # Container configuration
└── pkg/
    ├── server/
    │   ├── server.go                 # WebSocket server, update queues, connection mgmt
    │   ├── game_room.go              # Room lifecycle, tick loop, training mode
    │   ├── http_handlers.go          # REST API endpoint implementations
    │   ├── http_types.go             # HTTP request/response DTOs
    │   ├── message_handlers.go       # WebSocket message routing & event processing
    │   ├── room_manager.go           # Room registry and lookup
    │   ├── object_manager.go         # Game object lifecycle management
    │   ├── event_manager.go          # Event subscription and dispatch
    │   │
    │   ├── constants/
    │   │   └── constants.go          # Physics constants, object types, state keys
    │   │
    │   ├── game_objects/
    │   │   ├── base_game_object.go   # GameObject interface + base implementation
    │   │   ├── player_game_object.go # Player entity with movement/combat
    │   │   ├── arrow_game_object.go  # Arrow projectile with physics
    │   │   ├── bullet_game_object.go # Fast bullet projectile
    │   │   ├── block_game_object.go  # Static collision geometry
    │   │   ├── game_event.go         # Event type definitions
    │   │   └── util.go               # Physics extrapolation utilities
    │   │
    │   ├── game_maps/
    │   │   ├── map.go                # Map interface and BaseMap implementation
    │   │   ├── map_maker.go          # Map factory from JSON metadata
    │   │   ├── map_metadata.go       # Map metadata parsing
    │   │   └── meta/                 # JSON metadata + layout files per map
    │   │
    │   ├── geo/
    │   │   └── shape.go              # Geometric primitives (Circle, Line, Polygon), collision detection
    │   │
    │   └── types/
    │       └── api_types.go          # WebSocket message types, game state DTOs
    │
    └── util/
        └── util.go                   # Password/code generation, helpers
```

## Core Components

### Server

Central coordinator managing WebSocket connections and game state broadcasting.

| Item | Location |
|------|----------|
| `Server` struct | [server.go:55-74](../../backend/pkg/server/server.go#L55-L74) |
| `NewServer()` | [server.go:77-101](../../backend/pkg/server/server.go#L77-L101) |
| `HandleWebSocket()` | [server.go:104-121](../../backend/pkg/server/server.go#L104-L121) |
| WebSocket message routing | [server.go:148-191](../../backend/pkg/server/server.go#L148-L191) |
| `sendGameUpdate()` | [server.go:226-292](../../backend/pkg/server/server.go#L226-L292) |

**Background goroutines** (started on `NewServer()`):
- `runProcessGameUpdateQueue()` - Broadcasts game state to room connections
- `runProcessSpectatorUpdateQueue()` - Broadcasts spectator list updates
- `runPeriodicUpdates()` - Sends full state snapshot every 10 seconds
- `runCleanupInactiveRooms()` - Removes rooms inactive >10 minutes

**Key fields:**
- `gameStateUpdateQueue` (chan, buffer 100) - Queues game state broadcasts
- `spectatorUpdateQueue` (chan, buffer 100) - Queues spectator list broadcasts
- `connectionsByRoom` (map[string]map[string]*Connection) - Room → Connection tracking
- `roomManager` (*RoomManager) - All active game rooms

### GameRoom

Each room runs an independent tick loop with its own game state.

| Item | Location |
|------|----------|
| `GameRoom` struct | [game_room.go:76-118](../../backend/pkg/server/game_room.go#L76-L118) |
| `NewGameRoomWithTrainingConfig()` | [game_room.go:137-201](../../backend/pkg/server/game_room.go#L137-L201) |
| `StartTickLoop()` | [game_room.go:406-435](../../backend/pkg/server/game_room.go#L406-L435) |
| `Handle()` (event processing) | [game_room.go:277-308](../../backend/pkg/server/game_room.go#L277-L308) |
| `Reset()` (training episode reset) | [game_room.go:580-624](../../backend/pkg/server/game_room.go#L580-L624) |
| `TrainingOptions` struct | [game_room.go:54-65](../../backend/pkg/server/game_room.go#L54-L65) |

**Tick configuration:**
- Default: 20ms (50 Hz)
- Min: 1ms (1000 Hz) - for 20x training acceleration
- Max: 1000ms (1 Hz)

### Event System

Subscription-based event model with priority queue processing.

| Item | Location |
|------|----------|
| `GameEventManager` struct | [event_manager.go:17-21](../../backend/pkg/server/event_manager.go#L17-L21) |
| `Handle()` (event dispatch) | [event_manager.go:88-158](../../backend/pkg/server/event_manager.go#L88-L158) |
| `Subscribe()` | [event_manager.go:31-42](../../backend/pkg/server/event_manager.go#L31-L42) |
| Event types | [game_event.go:7-20](../../backend/pkg/server/game_objects/game_event.go#L7-L20) |
| `GameEvent` struct | [game_event.go:23-29](../../backend/pkg/server/game_objects/game_event.go#L23-L29) |

**Event types:**
| Event | Purpose |
|-------|---------|
| `game_tick` | Physics simulation step (fired by room tick loop) |
| `player_key_input` | WASD key press/release |
| `player_click_input` | Mouse click for shooting |
| `player_direction` | Aim direction update |
| `collision` | Object collision detected |
| `object_created` / `object_destroyed` | Lifecycle events |
| `player_died` | Player death (triggers respawn, stats update) |

### Object Management

| Item | Location |
|------|----------|
| `GameObjectManager` struct | [object_manager.go:19-23](../../backend/pkg/server/object_manager.go#L19-L23) |
| `HandleEvent()` (create/destroy) | [object_manager.go:150-194](../../backend/pkg/server/object_manager.go#L150-L194) |
| `ClearNonPlayerObjects()` | [object_manager.go:135-148](../../backend/pkg/server/object_manager.go#L135-L148) |

## Game Objects

### GameObject Interface

All game entities implement this interface.

| Item | Location |
|------|----------|
| `GameObject` interface | [base_game_object.go:32-59](../../backend/pkg/server/game_objects/base_game_object.go#L32-L59) |
| `BaseGameObject` struct | [base_game_object.go:62-67](../../backend/pkg/server/game_objects/base_game_object.go#L62-L67) |
| `GameObjectHandleEventResult` | [base_game_object.go:9-12](../../backend/pkg/server/game_objects/base_game_object.go#L9-L12) |

### Player

| Item | Location |
|------|----------|
| `PlayerGameObject` | [player_game_object.go](../../backend/pkg/server/game_objects/player_game_object.go) |
| Player constants | [constants.go:60-70](../../backend/pkg/server/constants/constants.go#L60-L70) |

**Properties:**
- Speed: 15 m/s horizontal, 20 m/s jump
- Max jumps: 2 (double jump)
- Starting arrows: 4 (max 4)
- Respawn time: 5s (0 in training with instant respawn)
- Bounding shape: Circle (radius 20px)

**State keys:** `x`, `y`, `dx`, `dy`, `dir`, `h` (health), `dead`, `sht` (shooting), `ac` (arrow count), `jc` (jump count)

### Arrow

| Item | Location |
|------|----------|
| `ArrowGameObject` | [arrow_game_object.go](../../backend/pkg/server/game_objects/arrow_game_object.go) |
| Arrow constants | [constants.go:79-89](../../backend/pkg/server/constants/constants.go#L79-L89) |

**States:**
- Flying: Line-segment collision, affected by gravity
- Grounded (`ag=true`): Circle collision (10px radius), can be picked up
- Destroyed (`d=true`): Removed from game

**Physics:** Power-based velocity from charge time (max 2 seconds, 100N max power)

### Block

| Item | Location |
|------|----------|
| `BlockGameObject` | [block_game_object.go](../../backend/pkg/server/game_objects/block_game_object.go) |

Static polygon geometry from map layout. Grid-based: 20px = 1 meter.

### Bullet

| Item | Location |
|------|----------|
| `BulletGameObject` | [bullet_game_object.go](../../backend/pkg/server/game_objects/bullet_game_object.go) |
| Bullet constants | [constants.go:73-76](../../backend/pkg/server/constants/constants.go#L73-L76) |

Fast projectile with short lifetime (0.1s, 1024px range).

## HTTP API

Entry point: [main.go:11-48](../../backend/main.go#L11-L48)

Handlers: [http_handlers.go](../../backend/pkg/server/http_handlers.go)

| Endpoint | Method | Handler | Purpose |
|----------|--------|---------|---------|
| `/ws` | GET | `HandleWebSocket` | WebSocket upgrade |
| `/api/maps` | GET | `HandleGetMaps` | List available maps |
| `/api/createGame` | POST | `HandleCreateGame` | Create new room |
| `/api/joinGame` | POST | `HandleJoinGame` | Join existing room |
| `/api/rooms/{id}/state` | GET | `HandleGetRoomState` | Get current game state |
| `/api/rooms/{id}/reset` | POST | `HandleResetGame` | Reset game (training) |
| `/api/rooms/{id}/stats` | GET | `HandleGetRoomStats` | Get kill/death stats |
| `/api/rooms/{id}/players/{pid}/action` | POST | `HandleBotAction` | Submit bot actions |
| `/api/training/sessions` | GET | `HandleGetTrainingSessions` | List training sessions |

**Authentication:** Player token via `X-Player-Token` header, query param `playerToken`, or Bearer token.

**Request/Response types:** [http_handlers.go:75-150](../../backend/pkg/server/http_handlers.go#L75-L150)

## WebSocket Protocol

Message handlers: [message_handlers.go](../../backend/pkg/server/message_handlers.go)

**Client -> Server:**
| Type | Handler | Purpose |
|------|---------|---------|
| `RejoinGame` | `handleRejoinGame` | Reconnect to room |
| `Key` | `handleKeyStatus` | Movement input (WASD) |
| `ClientState` | `handleClientState` | Aim direction (radians) |
| `Click` | `handlePlayerClick` | Mouse input for shooting |
| `ExitGame` | `handleExitGame` | Leave room |

**Server -> Client:**
| Type | Purpose |
|------|---------|
| `GameState` | Object states + events + training info |
| `Spectators` | Spectator list update |
| `RejoinGameResponse` | Reconnection confirmation |
| `ErrorMessage` | Error notification |

**Event processing flow:** [message_handlers.go:205-255](../../backend/pkg/server/message_handlers.go#L205-L255)

## Bot Action API

For ML bot integration via HTTP POST to `/api/rooms/{id}/players/{pid}/action`:

```json
{
  "actions": [
    {"type": "key", "key": "W", "isDown": true},
    {"type": "click", "x": 400, "y": 300, "isDown": true, "button": 0},
    {"type": "direction", "direction": 1.57}
  ]
}
```

Handler: [http_handlers.go](../../backend/pkg/server/http_handlers.go) (search for `HandleBotAction`)

## Physics System

### Constants

| Constant | Location |
|----------|----------|
| All physics constants | [constants.go:6-11](../../backend/pkg/server/constants/constants.go#L6-L11) |
| Room constants | [constants.go:14-21](../../backend/pkg/server/constants/constants.go#L14-L21) |
| Object state keys | [constants.go:33-57](../../backend/pkg/server/constants/constants.go#L33-L57) |

**Key values:**
- Gravity: 20 m/s² (heavier than real)
- Max velocity: 30 m/s
- Pixels per meter: 20
- Room size: 800x800 px (40x40 m)
- World wrap: 2m beyond edges

### Collision Detection

| Item | Location |
|------|----------|
| Shape interface | [geo/shape.go](../../backend/pkg/server/geo/shape.go) |

Supported shapes: Circle, Line, Polygon

### Extrapolation

Physics utilities: [game_objects/util.go](../../backend/pkg/server/game_objects/util.go)

Each tick:
1. Update position: `pos += velocity * deltaTime`
2. Apply gravity: `dy += gravity * deltaTime`
3. Cap velocity to max
4. Check collisions
5. Apply world wrapping

## Training Mode

### Configuration

| Item | Location |
|------|----------|
| `TrainingOptions` struct | [game_room.go:54-65](../../backend/pkg/server/game_room.go#L54-L65) |
| `IsTrainingComplete()` | [game_room.go:470-494](../../backend/pkg/server/game_room.go#L470-L494) |
| Spectator throttling | [game_room.go:639-650](../../backend/pkg/server/game_room.go#L639-L650) |

**Options:**
- `TickMultiplier` (1.0-20.0x) - Game speed acceleration
- `MaxGameDurationSec` - Auto-terminate after N seconds
- `DisableRespawnTimer` - Instant respawn (0s instead of 5s)
- `MaxKills` - End episode after N total kills

### Spectator Throttling

At tick rates >1x, spectator updates are throttled to ~60 FPS (16ms minimum interval) to prevent overwhelming browser clients.

| Item | Location |
|------|----------|
| `SpectatorThrottler` struct | [game_room.go:68-73](../../backend/pkg/server/game_room.go#L68-L73) |
| `ShouldThrottleSpectatorUpdate()` | [game_room.go:639-650](../../backend/pkg/server/game_room.go#L639-L650) |

## Concurrency Model

| Scope | Lock | Location | Protects |
|-------|------|----------|----------|
| Server | `serverLock` | [server.go:70](../../backend/pkg/server/server.go#L70) | Connection maps, activity tracking |
| Room | `LockObject` | [game_room.go:82](../../backend/pkg/server/game_room.go#L82) | Room state modifications |
| Connection | `WriteMutex` | [server.go:51](../../backend/pkg/server/server.go#L51) | Concurrent WebSocket writes |
| GameObject | `Mutex` | [base_game_object.go:66](../../backend/pkg/server/game_objects/base_game_object.go#L66) | State read/write |
| ObjectManager | `Mutex` | [object_manager.go:21](../../backend/pkg/server/object_manager.go#L21) | Object collection |
| EventManager | `mutex` | [event_manager.go:20](../../backend/pkg/server/event_manager.go#L20) | Subscription maps |

**Pattern:** Per-room tick goroutine + per-connection read goroutine + worker goroutines for update queues.

## Maps

| Item | Location |
|------|----------|
| `Map` interface | [game_maps/map.go](../../backend/pkg/server/game_maps/map.go) |
| `CreateMap()` factory | [game_maps/map_maker.go](../../backend/pkg/server/game_maps/map_maker.go) |
| Map metadata parsing | [game_maps/map_metadata.go](../../backend/pkg/server/game_maps/map_metadata.go) |
| Map definition files | [game_maps/meta/](../../backend/pkg/server/game_maps/meta/) |

Maps are defined as JSON metadata + ASCII layout files. Layout format: 'B' = block, '.' or space = empty.

## Extension Points

| To add... | Modify |
|-----------|--------|
| New game object type | Implement `GameObject` interface, add to `game_objects/` |
| New HTTP endpoint | Add handler in `http_handlers.go`, register in `main.go` |
| New WebSocket message | Add type to `types/api_types.go`, handler in `message_handlers.go` |
| New event type | Add to `game_event.go`, subscribe in relevant objects |
| New map | Add JSON + layout file to `game_maps/meta/` |
| New physics constant | Add to `constants/constants.go` |
