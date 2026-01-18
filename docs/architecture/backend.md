# Backend Architecture - Go Game Server

Go-based authoritative game server for a Towerfall-inspired arena combat game. Handles real-time physics simulation, WebSocket communication, and HTTP APIs for game management and ML bot integration.

## Tech Stack

- **Go 1.21+** with standard library HTTP server
- **gorilla/websocket** - WebSocket support
- **google/uuid** - Unique identifiers

## Directory Structure

```
backend/
├── main.go                     # Entry point, HTTP route registration
├── go.mod / go.sum             # Go module dependencies
├── Dockerfile                  # Container configuration
└── pkg/
    ├── server/
    │   ├── server.go           # WebSocket handling, update queues, connection mgmt
    │   ├── game_room.go        # Room lifecycle, tick loop, training mode
    │   ├── http_handlers.go    # REST API endpoint implementations
    │   ├── message_handlers.go # WebSocket message routing
    │   ├── room_manager.go     # Room registry and lookup
    │   ├── object_manager.go   # Game object lifecycle management
    │   ├── event_manager.go    # Event subscription and dispatch
    │   ├── http_types.go       # HTTP request/response DTOs
    │   │
    │   ├── constants/
    │   │   └── constants.go    # Physics constants, object types, state keys
    │   │
    │   ├── game_objects/
    │   │   ├── base_game_object.go   # GameObject interface + base implementation
    │   │   ├── player_game_object.go # Player entity with movement/combat
    │   │   ├── arrow_game_object.go  # Projectile with physics
    │   │   ├── block_game_object.go  # Static collision geometry
    │   │   └── game_event.go         # Event type definitions
    │   │
    │   ├── game_maps/
    │   │   ├── map.go          # Map interface
    │   │   ├── map_maker.go    # Map factory
    │   │   └── meta/           # JSON metadata + layout files per map
    │   │
    │   ├── geo/
    │   │   └── shape.go        # Geometric primitives, collision detection
    │   │
    │   └── types/
    │       └── api_types.go    # WebSocket message types, game state DTOs
    │
    └── util/
        └── util.go             # Password generation, helpers
```

## Core Architecture

### Server ([server.go](../../backend/pkg/server/server.go))

Central coordinator managing connections and message routing:

```go
type Server struct {
    gameStateUpdateQueue chan GameUpdateQueueItem  // Buffered (100) for broadcasts
    spectatorUpdateQueue chan SpectatorUpdateQueueItem
    connectionsByRoom    map[string]map[string]*Connection  // roomID -> connID -> conn
    lastActivity         map[string]time.Time
    serverLock           sync.Mutex
    roomManager          *RoomManager
}
```

**Background goroutines** (started on `NewServer()`):
- `runProcessGameUpdateQueue()` - Broadcasts game state to room connections
- `runProcessSpectatorUpdateQueue()` - Broadcasts spectator list updates
- `runPeriodicUpdates()` - Sends full state snapshot every 10 seconds
- `runCleanupInactiveRooms()` - Removes rooms inactive >10 minutes

### GameRoom ([game_room.go](../../backend/pkg/server/game_room.go))

Each room runs an independent tick loop:

```go
type GameRoom struct {
    ID, Name, Password, RoomCode string
    EventManager    *GameEventManager   // Event routing
    ObjectManager   *GameObjectManager  // Object lifecycle
    Players         map[string]*ConnectedPlayer
    PlayerStats     map[string]*PlayerStats  // Kill/death tracking
    Map             game_maps.Map
    TickInterval    time.Duration  // Default: 20ms (50 Hz)
    TickMultiplier  float64        // Training acceleration (1.0-20.0x)
    TrainingOptions *TrainingOptions
}
```

### Event-Driven Design

Subscription-based event model:

1. **GameObjects** register for event types via `GetEventTypes()`
2. **EventManager** routes events to subscribed objects
3. **Objects** return `GameObjectHandleEventResult`:
   - `StateChanged bool` - Whether to broadcast update
   - `RaisedEvents []*GameEvent` - New events to process

**Event types** ([game_event.go](../../backend/pkg/server/game_objects/game_event.go)):

| Event | Purpose |
|-------|---------|
| `game_tick` | Physics simulation step |
| `player_key_input` | WASD key press/release |
| `player_click_input` | Mouse click for shooting |
| `player_direction` | Aim direction update |
| `collision` | Object collision detected |
| `object_created` / `object_destroyed` | Lifecycle events |

## Domain Models

### GameObject Interface ([base_game_object.go](../../backend/pkg/server/game_objects/base_game_object.go))

All game entities implement:

```go
type GameObject interface {
    GetID() string
    GetObjectType() string
    Handle(event *GameEvent, roomObjects map[string]GameObject) *GameObjectHandleEventResult
    GetState() map[string]interface{}
    GetBoundingShape() geo.Shape
    GetProperty(key GameObjectProperty) (interface{}, bool)
}
```

### Player ([player_game_object.go](../../backend/pkg/server/game_objects/player_game_object.go))

| Property | State Key | Default |
|----------|-----------|---------|
| Position | `x`, `y` | Spawn point |
| Velocity | `dx`, `dy` | 0 |
| Direction | `dir` | radians (0 = right) |
| Health | `h` | 100 (instant death on hit) |
| Arrows | `ac` | 4 (max 4) |
| Dead | `dead` | false |
| Shooting | `sht` | false |

Movement: 15 m/s horizontal, 20 m/s jump, 2 double-jumps

### Arrow ([arrow_game_object.go](../../backend/pkg/server/game_objects/arrow_game_object.go))

States:
- **Flying**: Line-segment collision, affected by gravity
- **Grounded** (`ag=true`): Circle collision (10px radius), can be picked up
- **Destroyed** (`d=true`): Removed from game

Physics: Power-based velocity from charge time (max 2 seconds)

### Block ([block_game_object.go](../../backend/pkg/server/game_objects/block_game_object.go))

Static polygon geometry. Grid-based: 20px = 1 meter.

## Communication Layer

### HTTP Endpoints ([main.go](../../backend/main.go), [http_handlers.go](../../backend/pkg/server/http_handlers.go))

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ws` | GET | WebSocket upgrade |
| `/api/maps` | GET | List available maps |
| `/api/createGame` | POST | Create new game room |
| `/api/joinGame` | POST | Join existing room |
| `/api/rooms/{id}/state` | GET | Get current game state |
| `/api/rooms/{id}/reset` | POST | Reset game (training) |
| `/api/rooms/{id}/stats` | GET | Get kill/death stats |
| `/api/rooms/{id}/players/{pid}/action` | POST | Submit bot actions |
| `/api/training/sessions` | GET | List training sessions |

Authentication: Player token via `X-Player-Token` header, query param, or Bearer token.

### WebSocket Messages ([api_types.go](../../backend/pkg/server/types/api_types.go))

**Client -> Server:**

| Type | Payload | Purpose |
|------|---------|---------|
| `RejoinGame` | `{roomId, playerId, playerToken}` | Reconnect to room |
| `Key` | `{key: "W"/"A"/"S"/"D", isDown: bool}` | Movement input |
| `ClientState` | `{dir: float64}` | Aim direction (radians) |
| `Click` | `{x, y, isDown, button}` | Mouse input |
| `ExitGame` | `{}` | Leave room |

**Server -> Client:**

| Type | Purpose |
|------|---------|
| `GameState` | Object states + events + training info |
| `Spectators` | Spectator list update |
| `ErrorMessage` | Error notification |

### Bot Action API

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

## Game Loop & Physics

### Tick Loop

Default: 20ms interval (50 Hz). Each tick:
1. Emit `game_tick` event to subscribed objects
2. Objects update positions: `pos += velocity * deltaTime`
3. Apply gravity: `dy += gravity * deltaTime`
4. Collision detection via `geo` package
5. Queue state changes for broadcast

### Physics Constants ([constants.go](../../backend/pkg/server/constants/constants.go))

```go
AccelerationDueToGravityMetersPerSec2 = 20.0  // Heavy feel
MaxVelocityMetersPerSec = 30.0
PxPerMeter = 20.0  // 20 pixels = 1 meter

PlayerSpeedXMetersPerSec = 15.0
PlayerJumpSpeedMetersPerSec = 20.0
PlayerMaxJumps = 2
PlayerRespawnTimeSec = 5.0

ArrowMaxPowerNewton = 100.0
ArrowMaxPowerTimeSec = 2.0
```

### Collision Detection ([geo/shape.go](../../backend/pkg/server/geo/shape.go))

Supported shapes:
- **Circle** - Players, grounded arrows
- **Line** - Flying arrows
- **Polygon** - Blocks

Collision response: 8-direction push-back with 45-degree angle thresholds.

### World Wrapping

Players wrap horizontally at room edges (+/- 2 meters beyond boundary).

## Training Mode

### Configuration ([game_room.go](../../backend/pkg/server/game_room.go))

```go
type TrainingOptions struct {
    Enabled             bool
    TickMultiplier      float64  // 1.0-20.0x speed
    MaxGameDurationSec  int      // Auto-terminate
    DisableRespawnTimer bool     // Instant respawn
    MaxKills            int      // End after N kills
}
```

### Spectator Throttling

At tick rates >1x, spectator updates throttled to ~60 FPS (`SpectatorMinUpdateInterval = 16ms`) to prevent overwhelming browser clients.

### Training State Info

Included in `GameState` messages when training mode enabled:

```go
type TrainingStateInfo struct {
    Episode        int      // Current episode number
    TotalKills     int      // Kills this episode
    ElapsedTime    float64  // Seconds since episode start
    TickMultiplier float64  // Current speed
}
```

## Concurrency Model

| Scope | Lock | Protects |
|-------|------|----------|
| Server | `serverLock` | Connection maps, activity tracking |
| Room | `LockObject` | Room state modifications |
| Connection | `WriteMutex` | Concurrent WebSocket writes |
| GameObject | `Mutex` | State read/write |

Pattern: Per-room tick goroutine + per-connection read goroutine.

## Key Code Pointers

| Area | File | Notes |
|------|------|-------|
| Server init | [server.go:77-101](../../backend/pkg/server/server.go#L77-L101) | NewServer, goroutine setup |
| WebSocket routing | [server.go:148-191](../../backend/pkg/server/server.go#L148-L191) | Message type switch |
| Room creation | [game_room.go:137-199](../../backend/pkg/server/game_room.go#L137-L199) | NewGameRoomWithTrainingConfig |
| Player physics | [player_game_object.go:79-250](../../backend/pkg/server/game_objects/player_game_object.go#L79-L250) | Handle() method |
| HTTP handlers | [http_handlers.go](../../backend/pkg/server/http_handlers.go) | All REST endpoints |
| Collision shapes | [geo/shape.go](../../backend/pkg/server/geo/shape.go) | Circle, Line, Polygon |
| Constants | [constants/constants.go](../../backend/pkg/server/constants/constants.go) | All physics values |

## Extending

| To add... | Modify |
|-----------|--------|
| New game object type | Implement `GameObject` interface, add to `game_objects/` |
| New HTTP endpoint | Add handler in `http_handlers.go`, register in `main.go` |
| New WebSocket message | Add type to `api_types.go`, handler in `message_handlers.go` |
| New event type | Add to `game_event.go`, subscribe in relevant objects |
| New map | Add JSON + layout file to `game_maps/meta/` |
