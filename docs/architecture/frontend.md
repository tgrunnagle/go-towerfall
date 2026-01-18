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
│       └── GamePage.js             # Game container, URL param extraction
│
├── scripts/
│   └── generate-config.js          # .env → public/config.js generator
├── craco.config.js                 # Path alias configuration (@/ → src/)
└── package.json                    # Dependencies and scripts
```

## Core Architecture

### Hybrid Design

React handles static UI (lobby, forms, routing) while vanilla JS handles real-time game rendering. They connect via a global singleton:

```
┌─────────────────────────────────────────────────────────────────┐
│  React App (src/)                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ LandingPage  │───►│   GamePage   │───►│ GameWrapper  │      │
│  │ (lobby UI)   │    │ (container)  │    │ (canvas ref) │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│         │                                        │              │
│         ▼                                        ▼              │
│  ┌──────────────┐                    window.gameInstance        │
│  │   Api.js     │                    ┌──────────────────┐      │
│  │  (REST API)  │                    │ initGame(canvas) │      │
│  └──────────────┘                    └────────┬─────────┘      │
├────────────────────────────────────────────────┼────────────────┤
│  Game Engine (public/)                         │                │
│  ┌─────────────────────────────────────────────▼───────────────┐│
│  │                     GameModule.js                           ││
│  │  • WebSocket connection to backend /ws                      ││
│  │  • Keyboard (WASD) + mouse input handling                   ││
│  │  • requestAnimationFrame render loop                        ││
│  └─────────────────────────────────────────────┬───────────────┘│
│  ┌─────────────────────────────────────────────▼───────────────┐│
│  │                  GameStateManager.js                        ││
│  │  • Manages gameObjects map (players, bullets, arrows)       ││
│  │  • Processes server state updates (full + partial)          ││
│  │  • Renders all objects to canvas each frame                 ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
    HTTP REST API                 WebSocket /ws
    (createGame, joinGame)        (GameState, Key, Click)
```

### Game Engine ([GameModule.js](../../frontend/public/GameModule.js))

Singleton `Game` class exposed as `window.gameInstance`:

```javascript
class Game {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.socket = null;
        this.gameStateManager = new GameStateManager();
        this.keysPressed = new Set();
        this.isRunning = false;
    }
}
```

**Key methods:**

| Method | Purpose |
|--------|---------|
| `initConnection()` | Opens WebSocket to backend |
| `initGame(canvas, roomId, playerId, token, ...)` | Binds canvas, sets up input handlers, starts render loop |
| `connect()` | WebSocket connection with auto-reconnect |
| `handleMessage(event)` | Routes incoming messages by type |
| `animate(timestamp)` | requestAnimationFrame render loop |
| `exitGame()` | Cleanup and disconnect |

**Input handling:**

- WASD keys → `Key` WebSocket message with `{key, isDown}`
- Mouse move → Updates local aim direction (client-side prediction)
- Mouse click → `Click` WebSocket message with `{x, y, isDown, button}`

### GameStateManager ([GameStateManager.js](../../frontend/public/GameStateManager.js))

Manages game object instances and rendering:

```javascript
class GameStateManager {
    constructor() {
        this.gameObjects = {};           // objectId -> GameObject instance
        this.spectators = [];            // Spectator names
        this.trainingInfo = null;        // Training mode overlay data
        this.currentPlayerObjectId = null;
        this.animationManager = new AnimationsManager();
    }
}
```

**State update flow** (`handleGameStateUpdate`):

1. For `fullUpdate`: Remove objects not in payload
2. Create new `GameObject` instances by `objectType`
3. Update existing objects with `setServerState()`
4. Process events (object_created, object_destroyed, collision)
5. Store training info for overlay

### GameObject Classes ([game_objects/](../../frontend/public/game_objects/))

Base class hierarchy:

```
GameObject                    # Server/client state separation
└── GameObjectWithPosition    # Position + direction interpolation
    ├── PlayerGameObject      # Circle + name + arrow count
    ├── BulletGameObject      # Fast projectile
    └── ArrowGameObject       # Grounded/flying arrow
BlockGameObject (extends GameObject)  # Static polygon
```

**State management pattern:**

```javascript
// Server state: authoritative, updated on WebSocket message
this.serverState = { x, y, dx, dy, dir, ... };

// Client state: local predictions and UI state
this.clientState = { x, y, dir, isCurrentPlayer, ... };

// Prefer client state if available (for smooth interpolation)
getStatePreferClient(key) {
    return this.clientState[key] ?? this.serverState[key];
}
```

**Position interpolation** (`interpPosition`):

```javascript
// Lerp toward server position
this.clientState.x += (this.serverState.x - this.clientState.x) * 0.2;

// Predict using velocity for smooth motion between updates
const timeDelta = (timestamp - this.clientState.lastUpdateFromServer) / 1000;
const predictedX = this.clientState.x + this.serverState.dx * timeDelta;
```

## React Application

### Routing ([App.js](../../frontend/src/App.js))

```javascript
<Routes>
  <Route path="/" element={<LandingPage />} />
  <Route path="/game" element={<GamePage />} />
</Routes>
```

### LandingPage ([pages/LandingPage.js](../../frontend/src/pages/LandingPage.js))

Game lobby with two forms:

| Form | Fields | API Call |
|------|--------|----------|
| Create Game | roomName, playerName, mapType | `Api.createGame()` |
| Join Game | roomCode, roomPassword, playerName, isSpectator | `Api.joinGame()` |

On success, navigates to `/game` with session params in URL query string.

### GameWrapper ([components/GameWrapper.js](../../frontend/src/components/GameWrapper.js))

Bridge between React and game engine:

```javascript
useEffect(() => {
    window.gameInstance.initGame(
        canvasRef.current,
        roomId, playerId, playerToken,
        canvasSizeX, canvasSizeY,
        { onConnectionChange, onGameInfoChange, onError }
    );
}, [roomId, playerId, ...]);
```

### API Client ([Api.js](../../frontend/src/Api.js))

Axios instance with endpoints:

| Function | Endpoint | Purpose |
|----------|----------|---------|
| `getMaps()` | GET `/api/maps` | List available maps |
| `createGame({playerName, roomName, mapType})` | POST `/api/createGame` | Create room |
| `joinGame({playerName, roomCode, roomPassword, isSpectator})` | POST `/api/joinGame` | Join room |
| `getTrainingSessions()` | GET `/api/training/sessions` | List training sessions |

Base URL from `window.APP_CONFIG.BACKEND_API_URL` (generated from `.env`).

## Communication Layer

### WebSocket Messages

**Client → Server:**

| Type | Payload | Purpose |
|------|---------|---------|
| `RejoinGame` | `{roomId, playerId, playerToken}` | Reconnect to room |
| `Key` | `{key: "W"/"A"/"S"/"D", isDown: bool}` | Movement input |
| `Click` | `{x, y, isDown, button}` | Mouse input (0=left, 2=right) |
| `ExitGame` | `{}` | Leave room |

**Server → Client:**

| Type | Purpose |
|------|---------|
| `RejoinGameResponse` | Auth confirmation, room/player info |
| `GameState` | Object states + events + training info |
| `Spectators` | Spectator list update |
| `Error` | Error message |

### Session State via URL

Game session data passed as query parameters (survives page refresh):

```
/game?roomId=...&playerId=...&playerToken=...&roomCode=...
      &canvasSizeX=...&canvasSizeY=...&isSpectator=...
      &trainingMode=...&tickMultiplier=...
```

## Rendering

### Render Loop

60 FPS via `requestAnimationFrame`:

```javascript
animate(timestamp) {
    this.gameStateManager.render(this.ctx, timestamp);
    this.animationFrame = requestAnimationFrame(this.animate);
}
```

### Render Order

1. Clear canvas
2. Draw grid lines (64px spacing)
3. Render all game objects (blocks, arrows, players, bullets)
4. Render animations (collisions, death effects)
5. Draw spectator list (top-left)
6. Draw training info overlay (top-right, if training mode)

### Constants ([Constants.js](../../frontend/public/Constants.js))

```javascript
BULLET_SPEED_PX_SEC = 1024.0
PLAYER_DIED_ANIMATION_TIME_SEC = 3.0
SPECTATOR_TEXT_FONT = '20px Arial'
TRAINING_TEXT_FONT = '16px Arial'
```

## Build Configuration

### Scripts

```json
"prestart": "node scripts/generate-config.js",
"start": "cross-env PORT=4001 craco start",
"build": "craco build"
```

### Path Alias

`craco.config.js` configures `@/` → `src/`:

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

## Key Code Pointers

| Area | File | Notes |
|------|------|-------|
| Game singleton | [GameModule.js:17-74](../../frontend/public/GameModule.js#L17-L74) | Constructor, options |
| WebSocket connect | [GameModule.js:210-259](../../frontend/public/GameModule.js#L210-L259) | Connection with reconnect |
| Message routing | [GameModule.js:291-316](../../frontend/public/GameModule.js#L291-L316) | handleMessage switch |
| State updates | [GameStateManager.js:47-151](../../frontend/public/GameStateManager.js#L47-L151) | handleGameStateUpdate |
| Canvas rendering | [GameStateManager.js:159-190](../../frontend/public/GameStateManager.js#L159-L190) | render method |
| Position interpolation | [GameObject.js:82-101](../../frontend/public/game_objects/GameObject.js#L82-L101) | interpPosition |
| React-game bridge | [GameWrapper.js:20-54](../../frontend/src/components/GameWrapper.js#L20-L54) | useEffect init |
| Lobby forms | [LandingPage.js:37-97](../../frontend/src/pages/LandingPage.js#L37-L97) | Form handlers |

## Extending

| To add... | Modify |
|-----------|--------|
| New game object type | Create class in `public/game_objects/`, add case in `GameStateManager.js:65-97` |
| New WebSocket message | Add case in `GameModule.js:handleMessage`, update handler |
| New REST endpoint | Add function to `src/Api.js` |
| New page/route | Add component to `src/pages/`, add Route in `App.js` |
| Game constants | Update `public/Constants.js` |
| New animation type | Add method to `AnimationsManager.js` |
