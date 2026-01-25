# backend - Go Game Server

An authoritative game server for a Towerfall-inspired arena combat game. Handles real-time physics simulation, WebSocket communication, and HTTP APIs for game management and ML bot integration.

For detailed architecture information, see [docs/architecture/backend.md](../docs/architecture/backend.md).

## Quick Start

### Prerequisites

- Go 1.24+
- Task (task runner)
- golangci-lint (for linting)

Install golangci-lint:

```bash
# macOS
brew install golangci-lint

# Windows
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Linux
curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin
```

### Installation

```bash
task be:tidy
task be:build
```

### Running the Server

```bash
task be:run
```

The server starts on port 4000 with:
- WebSocket endpoint: `ws://localhost:4000/ws`
- HTTP API: `http://localhost:4000/api/`

### Running Tests

```bash
# Run all unit tests
task be:test

# Run tests in short mode (faster)
task be:test:short
```

### Running Checks

```bash
# Run all checks (build, vet, test)
task be:check

# Individual checks
task be:lint        # Linting with golangci-lint
task be:format      # Format code with go fmt
task be:vet         # Static analysis with go vet
```

## API Endpoints

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
