package types

import (
	"encoding/json"
	"go-ws-server/pkg/server/game_objects"
)

// Message represents a WebSocket message
type Message struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

// Request/Response structs
type CreateGameRequest struct {
	RoomName   string `json:"roomName"`
	PlayerName string `json:"playerName"`
}

type CreateGameResponse struct {
	Success      bool   `json:"success"`
	Error        string `json:"error"`
	RoomID       string `json:"roomId"`
	RoomName     string `json:"roomName"`
	RoomCode     string `json:"roomCode"`
	RoomPassword string `json:"roomPassword"`
	PlayerID     string `json:"playerId"`
	PlayerToken  string `json:"playerToken"`
}

type JoinGameRequest struct {
	RoomCode     string `json:"roomCode"`
	RoomPassword string `json:"roomPassword"`
	PlayerName   string `json:"playerName"`
}

type JoinGameResponse struct {
	Success     bool   `json:"success"`
	Error       string `json:"error"`
	RoomID      string `json:"roomId"`
	RoomName    string `json:"roomName"`
	RoomCode    string `json:"roomCode"`
	PlayerID    string `json:"playerId"`
	PlayerToken string `json:"playerToken"`
}

type RejoinGameRequest struct {
	RoomID      string `json:"roomId"`
	PlayerID    string `json:"playerId"`
	PlayerToken string `json:"playerToken"`
}

type RejoinGameResponse struct {
	Success      bool   `json:"success"`
	Error        string `json:"error"`
	RoomName     string `json:"roomName"`
	RoomCode     string `json:"roomCode"`
	RoomPassword string `json:"roomPassword"`
	PlayerName   string `json:"playerName"`
	PlayerID     string `json:"playerId"`
}

type KeyStatusRequest struct {
	Key    string `json:"key"`    // Key that changed (W, A, S, D)
	IsDown bool   `json:"isDown"` // True when pressed, false when released
}

type ClientStateRequest struct {
	Direction float64 `json:"dir"` // Direction in radians
}

type PlayerClickRequest struct {
	X      float64 `json:"x"`      // Click position X
	Y      float64 `json:"y"`      // Click position Y
	IsDown bool    `json:"isDown"` // True for mouse down, false for mouse up
	Button int     `json:"button"` // 0 for left click, 2 for right click
}

type GameUpdateEvent struct {
	Type game_objects.EventType `json:"type"`
	Data map[string]interface{} `json:"data"`
}

type GameUpdate struct {
	FullUpdate   bool                              `json:"fullUpdate"`
	ObjectStates map[string]map[string]interface{} `json:"objectStates"` // Map of ObjectID -> ObjectState
	Events       []GameUpdateEvent                 `json:"events"`       // List of events
	// Training mode state (only included when training mode is enabled)
	TrainingComplete bool              `json:"trainingComplete,omitempty"` // True when training completion conditions are met
	TrainingInfo     *TrainingStateInfo `json:"trainingInfo,omitempty"`     // Training metadata for spectators
}

// TrainingStateInfo contains training-specific information for spectators
type TrainingStateInfo struct {
	Episode        int     `json:"episode"`        // Current training episode
	TotalKills     int     `json:"totalKills"`     // Total kills this episode
	ElapsedTime    float64 `json:"elapsedTime"`    // Seconds since episode start
	TickMultiplier float64 `json:"tickMultiplier"` // Current tick speed multiplier
}

type SpectatorUpdate struct {
	Spectators []string `json:"spectators"`
}

// ExitGameRequest is sent when a player wants to exit a game
type ExitGameRequest struct {
}

type ExitGameResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error"`
}

// ErrorMessage is sent when an error occurs
type ErrorMessage struct {
	Message string `json:"message"`
}

// BotAction represents a single bot action (key, click, or direction)
type BotAction struct {
	Type      string  `json:"type"`                // "key", "click", or "direction"
	Key       string  `json:"key,omitempty"`       // For key actions: W/A/S/D
	IsDown    bool    `json:"isDown,omitempty"`    // For key/click actions
	X         float64 `json:"x,omitempty"`         // For click actions
	Y         float64 `json:"y,omitempty"`         // For click actions
	Button    int     `json:"button,omitempty"`    // For click actions: 0=left, 2=right
	Direction float64 `json:"direction,omitempty"` // For direction actions (radians)
}

// BotActionRequest represents a request to submit bot actions
type BotActionRequest struct {
	Actions []BotAction `json:"actions"`
}

// BotActionResponse represents the response to a bot action request
type BotActionResponse struct {
	Success          bool   `json:"success"`
	ActionsProcessed int    `json:"actionsProcessed,omitempty"`
	Timestamp        int64  `json:"timestamp,omitempty"`
	Error            string `json:"error,omitempty"`
}
