import GameStateManager from './GameStateManager.js';
import { Constants } from './Constants.js';

const defaultOptions = () => {
  return {
    serverUrl: 'ws://localhost:4000/ws',
    interpolationSpeed: 0.2,
    onConnectionChange: null,
    onGameInfoChange: null,
    onPlayerInfoChange: null,
    onRoomInfoChange: null,
    onError: null,
    clientStateUpdateInterval: 5000,
  };
}

class Game {
  constructor(options = {}) {
    this.gameInitialized = false;

    // Options and callbacks
    this.options = {
      ...defaultOptions(),
      ...options
    };

    // Canvas and rendering
    this.canvas = null;
    this.ctx = null;

    // Game state
    this.playerInfo = null;
    this.roomInfo = null;
    this.keysPressed = new Set();
    this.gameStateManager = new GameStateManager();

    // Animation
    this.animationFrame = null;
    this.isRunning = false;

    // WebSocket
    this.wsConnected = false;
    this.socket = null;
    this.reconnectTimeout = null;
    this.isConnecting = false;

    // Client state update interval
    this.clientStateUpdateInterval = null;

    // Bind methods
    this.initConnection = this.initConnection.bind(this);
    this.initGame = this.initGame.bind(this);
    this.cleanupGame = this.cleanupGame.bind(this);
    this.subscribeToConnectionReadyStateOpen = this.subscribeToConnectionReadyStateOpen.bind(this);
    this.connect = this.connect.bind(this);
    this.exitGame = this.exitGame.bind(this);
    this.sendMessage = this.sendMessage.bind(this);
    this.handleMessage = this.handleMessage.bind(this);
    this.handleCreateGameResponse = this.handleCreateGameResponse.bind(this);
    this.handleJoinGameResponse = this.handleJoinGameResponse.bind(this);
    this.handleRejoinGameResponse = this.handleRejoinGameResponse.bind(this);
    this.handleGameStateUpdate = this.handleGameStateUpdate.bind(this);
    this.handleExitGameResponse = this.handleExitGameResponse.bind(this);
    this.handleErrorMessage = this.handleErrorMessage.bind(this);
    this.createGame = this.createGame.bind(this);
    this.joinGame = this.joinGame.bind(this);
    this.rejoinGame = this.rejoinGame.bind(this);
    this.handleKeyDown = this.handleKeyDown.bind(this);
    this.handleKeyUp = this.handleKeyUp.bind(this);
    this.sendKeyStatus = this.sendKeyStatus.bind(this);
    this.handleCanvasMouseMove = this.handleCanvasMouseMove.bind(this);
    this.handleCanvasMouseDown = this.handleCanvasMouseDown.bind(this);
    this.handleCanvasMouseUp = this.handleCanvasMouseUp.bind(this);
    this.animate = this.animate.bind(this);
    this.setClientStateUpdateInterval = this.setClientStateUpdateInterval.bind(this);
  }

  initConnection() {
    this.connect();
    return this; // For chaining
  }

  // Initialize the game with a canvas element
  initGame(
    canvasElement,
    roomId,
    playerId,
    playerToken,
    callbacks = {}
  ) {
    // Update canvas reference
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');

    // Set up canvas
    this.canvas.width = Constants.CANVAS_SIZE_X;
    this.canvas.height = Constants.CANVAS_SIZE_Y;

    // Prevent context menu on right click
    this.canvas.addEventListener('contextmenu', e => e.preventDefault());

    // Update callbacks
    if (callbacks.onConnectionChange) {
      this.options.onConnectionChange = callbacks.onConnectionChange;
      this.subscribeToConnectionReadyStateOpen(() => this.options.onConnectionChange(true));
    }
    if (callbacks.onGameInfoChange) this.options.onGameInfoChange = callbacks.onGameInfoChange;
    if (callbacks.onError) this.options.onError = callbacks.onError;

    // Start animation loop if not already running
    if (!this.isRunning) {
      this.isRunning = true;
      this.animationFrame = requestAnimationFrame(this.animate);
    }

    if (this.gameInitialized) {
      if (callbacks.onInitComplete) {
        callbacks.onInitComplete();
      }
      return this;
    }

    this.gameInitialized = true;
    console.log("Initializing Game...");

    // Set up keyboard controls
    window.addEventListener('keydown', this.handleKeyDown);
    window.addEventListener('keyup', this.handleKeyUp);
    document.addEventListener('mousemove', this.handleCanvasMouseMove);
    document.addEventListener('mousedown', this.handleCanvasMouseDown);
    document.addEventListener('mouseup', this.handleCanvasMouseUp);

    // Rejoin game when the connection is ready
    this.subscribeToConnectionReadyStateOpen(() => {
      this.rejoinGame(roomId, playerId, playerToken)

      if (callbacks.onInitComplete) {
        callbacks.onInitComplete();
      }
    });

    this.setClientStateUpdateInterval();

    return this; // For chaining
  }

  // Clean up resources
  cleanupGame() {
    console.log("Cleaning up Game...")
    this.gameInitialized = false;

    // Remove event listeners
    window.removeEventListener('keydown', this.handleKeyDown);
    window.removeEventListener('keyup', this.handleKeyUp);
    document.removeEventListener('mousemove', this.handleCanvasMouseMove);
    document.removeEventListener('mousedown', this.handleCanvasMouseDown);
    document.removeEventListener('mouseup', this.handleCanvasMouseUp);

    // Reset canvas
    this.canvas = null;
    this.ctx = null;

    // Reset game state
    this.playerInfo = null;
    this.roomInfo = null;
    this.keysPressed.clear();
    this.gameStateManager.reset();

    // Reset animation
    this.isRunning = false;
    this.animationFrame = null;

    // Do not close ws connection

    // Options
    this.options = defaultOptions();

    // Intervals
    if (this.clientStateUpdateInterval) {
      clearInterval(this.clientStateUpdateInterval);
    }
    this.clientStateUpdateInterval = null;

    return this; // For chaining
  }

  subscribeToConnectionReadyStateOpen(callback) {
    let readyCheck = null;
    const func = () => {
      if (this.socket.readyState === WebSocket.OPEN) {
        if (readyCheck) {
          clearInterval(readyCheck);
          readyCheck = null;
        }
      }
      else {
        console.log("Waiting on connection ready state OPEN (1)", this.socket.readyState)
        return;
      }

      if (callback) {
        callback();
      }
    };

    // Poll for ready state
    readyCheck = setInterval(func, 100);
  }

  setClientStateUpdateInterval() {
    this.clientStateUpdateInterval = setInterval(() => {
      const clientState = this.gameStateManager.getCurrentPlayerClientState();
      if (!clientState) return;
      this.sendMessage('ClientState', clientState);
    }, this.options.clientStateUpdateInterval);
  }

  // WebSocket connection management
  connect() {
    if (this.socket) {
      return;
    }

    // Set connecting flag and lock
    const ws = new WebSocket(this.options.serverUrl);
    this.socket = ws;

    ws.onopen = () => {
      console.log('WebSocket Connected');
      this.wsConnected = true;
    };

    ws.onclose = (event) => {
      console.log(`WebSocket Disconnected: Code ${event.code}, Reason: ${event.reason}`);
      this.wsConnected = false;

      // Notify connection change
      if (this.options.onConnectionChange) {
        this.options.onConnectionChange(false);
      }

      // Only try to reconnect if this wasn't an intentional close or browser navigation
      // 1000: Normal closure, 1001: Going away (page navigation/refresh)
      if (event.code !== 1000 && event.code !== 1001) {
        console.log('Scheduling reconnect attempt...');

        // Clear any existing timeout
        if (this.reconnectTimeout) {
          clearTimeout(this.reconnectTimeout);
        }

        // Set new timeout
        this.reconnectTimeout = setTimeout(() => {
          this.reconnectTimeout = null;
          this.connect();
        }, 2000);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
      if (this.options.onError) {
        this.options.onError('Connection error');
      }
    };

    ws.onmessage = this.handleMessage;
  }

  exitGame() {
    // Send exit game message if we're in a game
    if (this.wsConnected && this.playerInfo && this.roomInfo) {
      this.sendMessage('ExitGame', {});
    }
    this.cleanupGame();
  }

  // Send a message to the server
  sendMessage(type, payload) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.error('Cannot send message, socket not connected', this.socket, this.socket?.readyState);
      return false;
    }

    const message = {
      type,
      payload
    };

    try {
      this.socket.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error sending message:', error);
      return false;
    }
  }

  // Handle incoming WebSocket messages
  handleMessage(event) {
    try {
      const message = JSON.parse(event.data);
      switch (message.type) {
        case 'CreateGameResponse':
          this.handleCreateGameResponse(message.payload);
          break;
        case 'JoinGameResponse':
          this.handleJoinGameResponse(message.payload);
          break;
        case 'RejoinGameResponse':
          this.handleRejoinGameResponse(message.payload);
          break;
        case 'GameState':
          this.handleGameStateUpdate(message.payload);
          break;
        case 'ExitGameResponse':
          this.handleExitGameResponse(message.payload);
          break;
        case 'Error':
          this.handleErrorMessage(message.payload);
          break;
        default:
          console.log('Unhandled message type:', message.type);
      }
    } catch (error) {
      console.error('Error parsing message:', error);
    }
  }

  // Handle specific message types
  handleCreateGameResponse(payload) {
    this.playerInfo = {
      playerId: payload.playerId,
      playerToken: payload.playerToken
    };

    this.gameStateManager.setCurrentPlayerObjectId(payload.playerId);

    this.roomInfo = {
      roomId: payload.roomId,
      roomName: payload.roomName,
      roomPassword: payload.roomPassword,
      roomCode: payload.roomCode
    };

    // Notify room info change
    if (this.options.onRoomInfoChange) {
      this.options.onRoomInfoChange(this.roomInfo);
    }

    // Notify player info change
    if (this.options.onPlayerInfoChange) {
      this.options.onPlayerInfoChange(this.playerInfo);
    }
  }

  handleJoinGameResponse(payload) {
    if (!payload.success && payload.error) {
      if (this.options.onError) {
        this.options.onError(payload.error);
      }
      return;
    }

    this.playerInfo = {
      playerId: payload.playerId,
      playerToken: payload.playerToken
    };

    this.gameStateManager.setCurrentPlayerObjectId(payload.playerId);

    this.roomInfo = {
      roomId: payload.roomId,
      roomName: payload.roomName,
      roomCode: payload.roomCode
    };

    // Notify room info change
    if (this.options.onRoomInfoChange) {
      this.options.onRoomInfoChange(this.roomInfo);
    }

    // Notify player info change
    if (this.options.onPlayerInfoChange) {
      this.options.onPlayerInfoChange(this.playerInfo);
    }
  }

  handleRejoinGameResponse(payload) {
    if (!payload.success) {
      if (this.options.onError) {
        this.options.onError(payload.error || 'Failed to rejoin game');
      }
      return;
    }

    console.log('Successfully rejoined game');

    // Update room info
    this.roomInfo = {
      ...(this.roomInfo || {}),
      roomName: payload.roomName,
      roomCode: payload.roomCode,
      roomPassword: payload.roomPassword
    };
    this.playerInfo = {
      ...(this.playerInfo || {}),
      playerName: payload.playerName,
      playerId: payload.playerId,
    };

    this.gameStateManager.setCurrentPlayerObjectId(payload.playerId);

    // Notify game info change
    if (this.options.onGameInfoChange) {
      this.options.onGameInfoChange({
        ...this.roomInfo,
        ...this.playerInfo,
      });
    }
  }

  handleGameStateUpdate(payload) {
    this.gameStateManager.handleGameStateUpdate(payload);
  }

  handleExitGameResponse(payload) {
    console.log("Exit game response:", payload);
  }

  handleErrorMessage(payload) {
    console.error('Server error:', payload);

    if (this.options.onError) {
      this.options.onError(payload.message || 'Server error');
    }
  }

  // Game actions
  createGame(roomName, playerName) {
    this.sendMessage('CreateGame', {
      roomName,
      playerName
    });
  }

  joinGame(roomCode, roomPassword, playerName) {
    this.sendMessage('JoinGame', {
      roomCode,
      roomPassword,
      playerName
    });
  }

  rejoinGame(roomId, playerId, playerToken) {
    // Store player and room info
    this.playerInfo = {
      ...this.playerInfo,
      playerId,
      playerToken
    };

    this.roomInfo = {
      ...this.roomInfo,
      roomId
    };

    // Send rejoin message
    this.sendMessage('RejoinGame', {
      roomId,
      playerId,
      playerToken
    });
  }

  // Input handling
  handleKeyDown(e) {
    if (['W', 'A', 'S', 'D'].includes(e.key.toUpperCase()) && !e.repeat) {
      this.keysPressed.add(e.key.toUpperCase());
      this.sendKeyStatus();
    }
  }

  handleKeyUp(e) {
    if (['W', 'A', 'S', 'D'].includes(e.key.toUpperCase())) {
      this.keysPressed.delete(e.key.toUpperCase());
      this.sendKeyStatus();
    }
  }

  handleCanvasMouseMove(e) {
    const relativeX = e.clientX - this.canvas.offsetLeft;
    const relativeY = e.clientY - this.canvas.offsetTop;
    this.gameStateManager.handleMouseMove(relativeX, relativeY);
  }

  handleCanvasMouseDown(e) {
    // Only handle left (0) and right (2) clicks
    // https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/button
    if (e.button !== 0 && e.button !== 2) return;
    
    const relativeX = e.clientX - this.canvas.offsetLeft;
    const relativeY = e.clientY - this.canvas.offsetTop;
    this.sendMessage('Click', { x: relativeX, y: relativeY, isDown: true, button: e.button });
  }

  handleCanvasMouseUp(e) {
    // Only handle left (0) and right (2) clicks
    if (e.button !== 0 && e.button !== 2) return;
    
    const relativeX = e.clientX - this.canvas.offsetLeft;
    const relativeY = e.clientY - this.canvas.offsetTop;
    this.sendMessage('Click', { x: relativeX, y: relativeY, isDown: false, button: e.button });
  }

  sendKeyStatus() {
    if (this.wsConnected && this.playerInfo && this.roomInfo) {
      this.sendMessage('Keys', {
        keysPressed: Array.from(this.keysPressed)
      });
    }
  }

  sendDirection() {
    if (this.wsConnected && this.playerInfo && this.roomInfo) {
      this.sendMessage('Dir', {
        dir: this.direction
      });
    }
  }

  // Rendering
  animate(timestamp) {
    if (!this.isRunning) return;

    this.gameStateManager.render(this.ctx, timestamp);
    this.animationFrame = requestAnimationFrame(this.animate);
  }
}

// Create a singleton instance
const gameInstance = new Game();
console.log("Game instance created");
gameInstance.initConnection();
window.gameInstance = gameInstance;

export default gameInstance;
