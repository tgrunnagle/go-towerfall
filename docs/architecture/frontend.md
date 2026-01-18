# Frontend Architecture - React + Canvas Game Client

Hybrid web client combining React for UI/routing with a vanilla JavaScript game engine for real-time canvas rendering and WebSocket communication.

## Tech Stack

- **React 18** with functional components and hooks
- **React Router 6** for client-side routing
- **Axios** for HTTP API calls
- **Canvas 2D API** for game rendering
- **Create React App + Craco** for build tooling

## Directory Structure

```
frontend/
├── public/                         # Game engine (vanilla JS)
│   ├── index.html                  # Entry HTML, loads config + game module
│   ├── config.js                   # Generated from .env (backend URLs)
│   ├── GameModule.js               # WebSocket, input handling, animation loop
│   ├── GameStateManager.js         # Object lifecycle, canvas rendering
│   ├── AnimationsManager.js        # Short-lived visual effects
│   ├── Constants.js                # Game timing/rendering constants
│   └── game_objects/
│       ├── GameObject.js           # Base class with state management
│       ├── PlayerGameObject.js     # Player rendering + interpolation
│       ├── BulletGameObject.js     # Bullet projectiles
│       ├── ArrowGameObject.js      # Collectable arrows
│       └── BlockGameObject.js      # Static platforms
│
├── src/                            # React application
│   ├── index.js                    # React mount point
│   ├── App.js                      # Router (/, /game routes)
│   ├── Api.js                      # Axios HTTP client
│   ├── components/
│   │   └── GameWrapper.js          # Canvas element + game engine bridge
│   └── pages/
│       ├── LandingPage.js          # Create/join game lobby
│       ├── LandingPage.css
│       ├── GamePage.js             # Game container, URL param extraction
│       └── GamePage.css
│
├── scripts/
│   └── generate-config.js          # .env -> public/config.js generator
├── craco.config.js                 # Path alias configuration (@/ -> src/)
└── package.json                    # Dependencies and scripts
```

## Core Architecture

### Hybrid Design

React handles static UI (lobby, forms, routing) while vanilla JS handles real-time game rendering. They connect via a global singleton:

```
┌─────────────────────────────────────────────────────────────────┐
│  React App (src/)                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ LandingPage  │───>│   GamePage   │───>│ GameWrapper  │      │
│  │ (lobby UI)   │    │ (container)  │    │ (canvas ref) │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│         │                                        │              │
│         v                                        v              │
│  ┌──────────────┐                    window.gameInstance        │
│  │   Api.js     │                    ┌──────────────────┐      │
│  │  (REST API)  │                    │ initGame(canvas) │      │
│  └──────────────┘                    └────────┬─────────┘      │
├────────────────────────────────────────────────┼────────────────┤
│  Game Engine (public/)                         │                │
│  ┌─────────────────────────────────────────────v───────────────┐│
│  │                     GameModule.js                           ││
│  │  - WebSocket connection to backend /ws                      ││
│  │  - Keyboard (WASD) + mouse input handling                   ││
│  │  - requestAnimationFrame render loop                        ││
│  └─────────────────────────────────────────────┬───────────────┘│
│  ┌─────────────────────────────────────────────v───────────────┐│
│  │                  GameStateManager.js                        ││
│  │  - Manages gameObjects map (players, bullets, arrows)       ││
│  │  - Processes server state updates (full + partial)          ││
│  │  - Renders all objects to canvas each frame                 ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
         │                              │
         v                              v
    HTTP REST API                 WebSocket /ws
    (createGame, joinGame)        (GameState, Key, Click)
```

## Game Engine

### GameModule ([GameModule.js](../../frontend/public/GameModule.js))

Singleton `Game` class exposed as `window.gameInstance`. Manages WebSocket connection, input handling, and animation loop.

**Key properties** (lines 17-74):
- `canvas`, `ctx` - Canvas element and 2D context
- `gameStateManager` - GameStateManager instance
- `keysPressed` - Set of currently pressed keys (WASD)
- `socket`, `wsConnected` - WebSocket connection state
- `playerInfo`, `roomInfo` - Current session data

**Key methods:**

| Method | Location | Purpose |
|--------|----------|---------|
| `initConnection()` | L76-79 | Opens WebSocket to backend |
| `initGame()` | L82-144 | Binds canvas, sets up input handlers, starts render loop |
| `connect()` | L210-259 | WebSocket connection with auto-reconnect (2s backoff) |
| `handleMessage()` | L291-316 | Routes incoming messages by type |
| `animate()` | L472-477 | requestAnimationFrame render loop |
| `exitGame()` | L261-267 | Cleanup and disconnect |
| `rejoinGame()` | L388-407 | Send RejoinGame message to server |

**Input handling** (lines 409-469):
- `handleKeyDown/Up` (L410-428) - WASD keys -> `Key` WebSocket message
- `handleCanvasMouseMove` (L430-434) - Updates local aim direction
- `handleCanvasMouseDown/Up` (L436-453) - `Click` WebSocket message with position and button

### GameStateManager ([GameStateManager.js](../../frontend/public/GameStateManager.js))

Manages game object instances, processes server updates, and renders to canvas.

**Key properties** (lines 8-26):
- `gameObjects` - Map of objectId -> GameObject instance
- `spectators` - List of spectator names
- `trainingInfo` - Training mode overlay data
- `currentPlayerObjectId` - Player's own object ID
- `animationManager` - AnimationsManager instance

**Key methods:**

| Method | Location | Purpose |
|--------|----------|---------|
| `handleGameStateUpdate()` | L47-151 | Processes full/partial state updates, creates/updates/destroys objects |
| `render()` | L159-190 | Clears canvas, draws grid, renders objects and overlays |
| `handleMouseMove()` | L250-256 | Updates current player's aim direction |
| `drawSpectators()` | L192-206 | Renders spectator list (top-left) |
| `drawTrainingInfo()` | L208-248 | Renders training mode overlay (top-right) |

**State update flow** (`handleGameStateUpdate`, L47-151):
1. For `fullUpdate`: Remove objects not in payload (L50-56)
2. Create new `GameObject` instances by `objectType` (L59-98)
3. Update existing objects with `setServerState()` (L99-105)
4. Process events: object_created, object_destroyed, player_died, collision (L121-145)
5. Store training info for overlay (L147-150)

### AnimationsManager ([AnimationsManager.js](../../frontend/public/AnimationsManager.js))

Manages short-lived animations using a callback pattern.

**Key methods:**

| Method | Location | Purpose |
|--------|----------|---------|
| `registerAnimation()` | L17-20 | Add animation function to active set |
| `createCollisionAnimation()` | L22-37 | Creates collision effect at position |
| `createBlinkAnimation()` | L45-70 | Creates blinking effect for object |
| `render()` | L76-85 | Renders active animations, removes completed ones |

### GameObject Classes ([game_objects/](../../frontend/public/game_objects/))

Base class hierarchy with dual state management:

```
GameObject                        # Server/client state separation (L10-61)
└── GameObjectWithPosition        # Position + direction interpolation (L63-126)
    ├── PlayerGameObject          # Circle + name + arrow count
    ├── BulletGameObject          # Fast projectile with trajectory
    ├── ArrowGameObject           # Grounded/flying arrow
    └── BlockGameObject           # Static polygon
```

**State management pattern** ([GameObject.js](../../frontend/public/game_objects/GameObject.js)):
- `serverState` (L12) - Authoritative state from server
- `clientState` (L13) - Local predictions and UI state (initialized via `defaultClientState()` L2-8)
- `getStatePreferClient()` (L20-22) - Returns client state if set, else server state
- `setServerState()` (L24-38) - Updates server state, resets interpolation counters
- `setClientState()` (L40-48) - Updates client state

**Position interpolation** ([GameObject.js:82-101](../../frontend/public/game_objects/GameObject.js#L82-L101)):
- Lerps toward server position with configurable speed
- Applies velocity prediction for smooth motion between updates
- Returns `{predictedX, predictedY}` for rendering

**Direction interpolation** ([GameObject.js:103-125](../../frontend/public/game_objects/GameObject.js#L103-L125)):
- Current player uses client-side direction (immediate mouse response)
- Other players interpolate toward server direction

### Individual Game Objects

| Class | File | Purpose |
|-------|------|---------|
| **PlayerGameObject** | [PlayerGameObject.js](../../frontend/public/game_objects/PlayerGameObject.js) | Renders as colored circle (L48-51), shows name above (L54-57), arrow count indicators (L60-69), velocity vector (L72-82), direction indicator (L85-96), green highlight for current player (L99-105). Death animation with blinking (L110-135). |
| **BulletGameObject** | [BulletGameObject.js](../../frontend/public/game_objects/BulletGameObject.js) | Create animation shows predicted trajectory (L6-24). Destroy animation shows expanding red circle (L26-44). |
| **ArrowGameObject** | [ArrowGameObject.js](../../frontend/public/game_objects/ArrowGameObject.js) | Renders as triangle when flying (L40-58), 'X' when grounded (L23-38). Destroy animation shows expanding 'X' (L63-90). |
| **BlockGameObject** | [BlockGameObject.js](../../frontend/public/game_objects/BlockGameObject.js) | Static green polygon from points array (L9-21). No animations. |

### Constants ([Constants.js](../../frontend/public/Constants.js))

Game timing and rendering constants:
- `BULLET_SPEED_PX_SEC = 1024.0` (L2)
- `BULLET_LIFETIME_SEC = 0.05` (L4)
- `PLAYER_DIED_ANIMATION_TIME_SEC = 3.0` (L7)
- `COLLISION_ANIMATION_TIME_SEC = 0.5` (L10)
- Spectator text settings (L12-16)
- Training overlay settings (L18-23)

## React Application

### Routing ([App.js](../../frontend/src/App.js))

Simple two-route setup using BrowserRouter (L9-16):
- `/` -> LandingPage (lobby)
- `/game` -> GamePage (active game)

### LandingPage ([pages/LandingPage.js](../../frontend/src/pages/LandingPage.js))

Game lobby with create/join forms.

**State** (L9-18): activeOption, roomName, roomCode, playerName, roomPassword, isSpectator, error, isLoading, maps, selectedMap

**Key functions:**
- `loadMaps()` (L20-35) - Fetches available maps on mount
- `handleCreateGame()` (L37-61) - Validates fields, calls API, navigates to /game with query params
- `handleJoinGame()` (L63-97) - Validates fields, calls API, includes training mode params if applicable

**Form fields:**

| Create Game | Join Game |
|-------------|-----------|
| Room Name | Room Code |
| Player Name | Room Password |
| Map Selection | Player Name |
| | Join as Spectator checkbox |

### GamePage ([pages/GamePage.js](../../frontend/src/pages/GamePage.js))

Container component that extracts URL parameters and renders GameWrapper.

**URL query parameters** (L14-25):
- `roomId`, `playerId`, `playerToken` - Session identifiers (required)
- `canvasSizeX`, `canvasSizeY` - Canvas dimensions (required)
- `roomCode`, `isSpectator` - Optional
- Training mode: `trainingMode`, `tickMultiplier`, `maxGameDurationSec`, `maxKills`

Redirects to `/` if required parameters missing (L28-32).

### GameWrapper ([components/GameWrapper.js](../../frontend/src/components/GameWrapper.js))

Bridge between React and game engine.

**Props** (L3-14): roomId, playerId, playerToken, canvasSizeX, canvasSizeY, setPlayerName, setRoomName, setRoomCode, setRoomPassword, onExitGame

**Initialization** (L20-71):
- Creates canvas ref
- Calls `window.gameInstance.initGame()` with callbacks for connection status, game info, and errors
- Cleans up on unmount

**Callbacks:**
- `onConnectionChange` (L41-43) - Updates connected state
- `onGameInfoChange` (L44-49) - Propagates room/player info to parent
- `onError` (L50-52) - Sets error state

## API Client ([Api.js](../../frontend/src/Api.js))

Axios instance with 5s timeout. Base URL from `window.APP_CONFIG.BACKEND_API_URL`.

| Function | Endpoint | Request | Response |
|----------|----------|---------|----------|
| `getMaps()` | GET `/api/maps` | - | `{maps: [{type, name}]}` |
| `createGame()` | POST `/api/createGame` | `{playerName, roomName, mapType}` | `{roomId, playerId, playerToken, roomCode, canvasSizeX, canvasSizeY}` |
| `joinGame()` | POST `/api/joinGame` | `{playerName, roomCode, roomPassword, isSpectator}` | `{roomId, playerId, playerToken, roomCode, isSpectator, canvasSizeX, canvasSizeY, trainingMode?, tickMultiplier?, maxGameDurationSec?, maxKills?}` |
| `getTrainingSessions()` | GET `/api/training/sessions` | - | `{sessions: [...]}` |

## WebSocket Communication

### Message Types

**Client -> Server:**

| Type | Payload | Location |
|------|---------|----------|
| `RejoinGame` | `{roomId, playerId, playerToken}` | GameModule.js:402-406 |
| `Key` | `{key: "W"/"A"/"S"/"D", isDown: bool}` | GameModule.js:413-416 |
| `Click` | `{x, y, isDown, button}` (0=left, 2=right) | GameModule.js:443, 452 |
| `ExitGame` | `{}` | GameModule.js:264 |

**Server -> Client:**

| Type | Handler | Purpose |
|------|---------|---------|
| `RejoinGameResponse` | L318-350 | Auth confirmation, room/player info |
| `GameState` | L352-354 | Object states + events + training info |
| `Spectators` | L356-358 | Spectator list update |
| `ExitGameResponse` | L360-362 | Exit confirmation |
| `Error` | L364-370 | Error message |

### Session State via URL

Game session data passed as query parameters (survives page refresh):

```
/game?roomId=...&playerId=...&playerToken=...&roomCode=...
      &canvasSizeX=...&canvasSizeY=...&isSpectator=...
      &trainingMode=...&tickMultiplier=...
```

## Rendering

### Render Loop

60 FPS via `requestAnimationFrame` ([GameModule.js:472-477](../../frontend/public/GameModule.js#L472-L477)):

```javascript
animate(timestamp) {
    if (!this.isRunning) return;
    this.gameStateManager.render(this.ctx, timestamp);
    this.animationFrame = requestAnimationFrame(this.animate);
}
```

### Render Order ([GameStateManager.js:159-190](../../frontend/public/GameStateManager.js#L159-L190))

1. Clear canvas (L161)
2. Draw grid lines - 64px spacing (L164-181)
3. Render all game objects (L183-185)
4. Render animations (L186)
5. Draw spectator list - top-left (L188)
6. Draw training info overlay - top-right (L189)

## Build Configuration

### Scripts (package.json)

```json
"prestart": "node scripts/generate-config.js",
"start": "cross-env PORT=4001 craco start",
"build": "craco build"
```

### Path Alias (craco.config.js)

`@/` -> `src/`:
```javascript
import Api from '@/Api';  // resolves to src/Api.js
```

### Environment Config

`scripts/generate-config.js` reads `.env` and writes `public/config.js`:

```javascript
window.APP_CONFIG = {
    BACKEND_API_URL: "http://localhost:4000",
    BACKEND_WS_URL: "ws://localhost:4000"
};
```

## Quick Reference

### Key Code Locations

| Area | File | Lines | Notes |
|------|------|-------|-------|
| Game singleton | GameModule.js | 17-74 | Constructor, properties |
| WebSocket connect | GameModule.js | 210-259 | Connection with auto-reconnect |
| Message routing | GameModule.js | 291-316 | handleMessage switch |
| State updates | GameStateManager.js | 47-151 | handleGameStateUpdate |
| Canvas rendering | GameStateManager.js | 159-190 | render method |
| Position interp | GameObject.js | 82-101 | interpPosition |
| Direction interp | GameObject.js | 103-125 | interpDirection |
| React-game bridge | GameWrapper.js | 20-54 | useEffect init |
| Create game form | LandingPage.js | 37-61 | handleCreateGame |
| Join game form | LandingPage.js | 63-97 | handleJoinGame |
| URL param extraction | GamePage.js | 14-25 | Query params parsing |

### Extension Points

| To add... | Modify |
|-----------|--------|
| New game object type | Create class in `public/game_objects/`, add case in GameStateManager.js:65-97 |
| New WebSocket message | Add case in GameModule.js:handleMessage (L294-312), add handler method |
| New REST endpoint | Add function to src/Api.js |
| New page/route | Add component to src/pages/, add Route in App.js:11-14 |
| Game constants | Update public/Constants.js |
| New animation type | Add method to AnimationsManager.js |
