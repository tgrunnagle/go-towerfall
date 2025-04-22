# Multiplayer Game WebSocket Server

*Based on [go-ws-server](https://github.com/tgrunnagle/go-ws-server).*

A Go-based WebSocket server with a React frontend designed for multiplayer online games. This server maintains state about connected players and provides real-time game state updates.

## Features

- WebSocket-based communication using Gorilla WebSocket
- Player state management with position tracking
- Game room creation and management
- Player authentication with tokens
- Real-time game state broadcasting
- Position interpolation for smooth movement
- React-based frontend with canvas rendering
- Keyboard controls for player movement
- Shoot arrows by holding left-click to power up and releasing (right click to cancel)

## Project Structure

- `/backend` - Go WebSocket server
- `/frontend` - React frontend application
- `/test` - Selenium tests

## Running the Application

### Using Docker Compose (Recommended)

This will start both the backend server and frontend application:

```bash
docker compose up --build -d
```

The backend server will be available at `ws://localhost:4000/ws`
The frontend application will be available at `http://localhost:4001`

### Running Locally

#### Backend Server

```bash
cd backend
go run main.go
```

The server will be available at `ws://localhost:4000/ws`

#### Frontend Application

```bash
cd frontend
npm install
npm start
```

The frontend will be available at `http://localhost:4001`

### Manual test helper

The test helper is a Selenium script that can be used to manually test the application. It will start two browser instances and create a new game in one instance, then join the game in the other instance. Make sure the servers are running before running the test helper.

```bash
cd test/selenium
npm install
npm test
```

### Bot

The bot is a ML model that can play the game. It runs in python and is located in the `bot` directory. Initialize the virtual environment with:

```bash
cd bot
# on macOS
python3.11 -m venv env
source env/bin/activate
# on windows (in powershell, Set-ExecutionPolicy RemoteSigned)
py -3.11 -m venv env
.\env\Scripts\activate
# exit venv
deactivate
```

Run the bot with:

```bash
// TODO
```

## Game Controls

- Use W, A, S, D keys to move your player
- Click, hold, and release left mouse button to shoot arrows, right mouse button to cancel
- Create a new game room or join an existing one from the landing page
- View room information and other players in the game view
