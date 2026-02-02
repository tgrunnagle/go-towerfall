package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/game_maps"
	"go-ws-server/pkg/server/game_objects"
	"go-ws-server/pkg/server/types"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// writeJSON writes a JSON response to the http.ResponseWriter and logs any encoding errors
func writeJSON(w http.ResponseWriter, v interface{}) {
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("Failed to encode JSON response: %v", err)
	}
}

// HandleGetMaps handles HTTP requests to get available maps
func (s *Server) HandleGetMaps(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow GET requests
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		writeJSON(w, map[string]interface{}{
			"error": "Method not allowed",
		})
		return
	}

	// Get all map metadata
	metadata, err := game_maps.GetAllMapsMetadata()
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		writeJSON(w, map[string]interface{}{
			"error": fmt.Sprintf("Failed to get maps: %v", err),
		})
		return
	}

	// Convert metadata to response format
	maps := make([]MapInfo, 0, len(metadata))
	for _, md := range metadata {
		// Get map type from metadata
		mapType := strings.TrimPrefix(string(md.MapType), "meta/")
		mapType = strings.TrimSuffix(mapType, filepath.Ext(mapType))
		canvasSizeX := int(float64(md.ViewSize.X) * float64(constants.BlockSizeUnitPixels))
		canvasSizeY := int(float64(md.ViewSize.Y) * float64(constants.BlockSizeUnitPixels))

		maps = append(maps, MapInfo{
			Type:        mapType,
			Name:        md.MapName,
			CanvasSizeX: canvasSizeX,
			CanvasSizeY: canvasSizeY,
		})
	}

	// Send response
	w.WriteHeader(http.StatusOK)
	writeJSON(w, GetMapsResponse{
		Maps: maps,
	})
}

// CreateGameRequest represents the request to create a new game
type CreateGameHTTPRequest struct {
	PlayerName string `json:"playerName"`
	RoomName   string `json:"roomName"`
	MapType    string `json:"mapType"`
	// Training mode options (all optional)
	TrainingMode        bool    `json:"trainingMode,omitempty"`
	TickMultiplier      float64 `json:"tickMultiplier,omitempty"`      // e.g., 10.0 for 10x speed
	MaxGameDurationSec  int     `json:"maxGameDurationSec,omitempty"`  // Auto-terminate after N seconds
	DisableRespawnTimer bool    `json:"disableRespawnTimer,omitempty"` // Instant respawn for faster episodes
	MaxKills            int     `json:"maxKills,omitempty"`            // End game after N total kills
}

// CreateGameResponse represents the response to a create game request
type CreateGameHTTPResponse struct {
	Success      bool   `json:"success"`
	PlayerID     string `json:"playerId,omitempty"`
	PlayerToken  string `json:"playerToken,omitempty"`
	RoomID       string `json:"roomId,omitempty"`
	RoomCode     string `json:"roomCode,omitempty"`
	RoomName     string `json:"roomName,omitempty"`
	RoomPassword string `json:"roomPassword,omitempty"`
	CanvasSizeX  int    `json:"canvasSizeX,omitempty"`
	CanvasSizeY  int    `json:"canvasSizeY,omitempty"`
	Error        string `json:"error,omitempty"`
	// Training mode settings (returned when training mode is enabled)
	TrainingMode        bool    `json:"trainingMode,omitempty"`
	TickMultiplier      float64 `json:"tickMultiplier,omitempty"`
	MaxGameDurationSec  int     `json:"maxGameDurationSec,omitempty"`
	DisableRespawnTimer bool    `json:"disableRespawnTimer,omitempty"`
	MaxKills            int     `json:"maxKills,omitempty"`
}

// JoinGameRequest represents the request to join an existing game
type JoinGameHTTPRequest struct {
	PlayerName   string `json:"playerName"`
	RoomCode     string `json:"roomCode"`
	RoomPassword string `json:"roomPassword"`
	IsSpectator  bool   `json:"isSpectator,omitempty"`
}

// JoinGameResponse represents the response to a join game request
type JoinGameHTTPResponse struct {
	Success     bool   `json:"success"`
	PlayerID    string `json:"playerId,omitempty"`
	PlayerToken string `json:"playerToken,omitempty"`
	RoomID      string `json:"roomId,omitempty"`
	RoomCode    string `json:"roomCode,omitempty"`
	IsSpectator bool   `json:"isSpectator"`
	CanvasSizeX int    `json:"canvasSizeX,omitempty"`
	CanvasSizeY int    `json:"canvasSizeY,omitempty"`
	Error       string `json:"error,omitempty"`
	// Training mode settings (returned when joining a training game as spectator)
	TrainingMode       bool    `json:"trainingMode,omitempty"`
	TickMultiplier     float64 `json:"tickMultiplier,omitempty"`
	MaxGameDurationSec int     `json:"maxGameDurationSec,omitempty"`
	MaxKills           int     `json:"maxKills,omitempty"`
}

// GetRoomStateResponse represents the response to a room state request
type GetRoomStateResponse struct {
	Success      bool                              `json:"success"`
	RoomID       string                            `json:"roomId,omitempty"`
	Timestamp    string                            `json:"timestamp,omitempty"`
	ObjectStates map[string]map[string]interface{} `json:"objectStates,omitempty"`
	Error        string                            `json:"error,omitempty"`
}

// ResetGameHTTPRequest represents the request to reset a game room
type ResetGameHTTPRequest struct {
	RoomPassword string `json:"roomPassword,omitempty"`
}

// ResetGameHTTPResponse represents the response to a reset game request
type ResetGameHTTPResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

// AddBotRequest represents the request to add a bot to a game room
type AddBotRequest struct {
	BotType    string `json:"botType"`              // "rule_based" or "neural_network"
	ModelID    string `json:"modelId,omitempty"`    // For neural_network bots
	Generation *int   `json:"generation,omitempty"` // Alternative to modelId
	PlayerName string `json:"playerName,omitempty"` // Bot's display name (default: "Bot")
}

// AddBotResponse represents the response when adding a bot
type AddBotResponse struct {
	Success bool   `json:"success"`
	BotID   string `json:"botId,omitempty"`
	Error   string `json:"error,omitempty"`
}

// RemoveBotResponse represents the response when removing a bot
type RemoveBotResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

// botServiceSpawnRequest is the internal request format for Bot Service
type botServiceSpawnRequest struct {
	RoomCode     string              `json:"room_code"`
	RoomPassword string              `json:"room_password"`
	BotConfig    botServiceBotConfig `json:"bot_config"`
}

type botServiceBotConfig struct {
	BotType    string `json:"bot_type"`
	ModelID    string `json:"model_id,omitempty"`
	Generation *int   `json:"generation,omitempty"`
	PlayerName string `json:"player_name"`
}

// botServiceSpawnResponse is the response format from Bot Service
type botServiceSpawnResponse struct {
	Success bool   `json:"success"`
	BotID   string `json:"bot_id,omitempty"`
	Error   string `json:"error,omitempty"`
}

// botServiceRemoveResponse is the response format from Bot Service for remove operations
type botServiceRemoveResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

// HandleCreateGame handles HTTP requests to create a new game
func (s *Server) HandleCreateGame(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow POST requests
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		writeJSON(w, CreateGameHTTPResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Parse request body
	var req CreateGameHTTPRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, CreateGameHTTPResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Validate request
	if req.PlayerName == "" || req.RoomName == "" {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, CreateGameHTTPResponse{
			Success: false,
			Error:   "PlayerName and RoomName are required",
		})
		return
	}

	// Build training options if training mode is enabled
	var trainingOptions *TrainingOptions
	if req.TrainingMode {
		// Validate training parameters
		// Note: Max multiplier is 20x because DefaultTickInterval (20ms) / MinTickInterval (1ms) = 20.
		// The issue spec mentions 1.0-100.0, but the server's MinTickInterval of 1ms limits practical max to 20x.
		if req.TickMultiplier != 0 && (req.TickMultiplier < 1.0 || req.TickMultiplier > 20.0) {
			w.WriteHeader(http.StatusBadRequest)
			writeJSON(w, CreateGameHTTPResponse{
				Success: false,
				Error:   "tickMultiplier must be between 1.0 and 20.0",
			})
			return
		}
		if req.MaxGameDurationSec < 0 || req.MaxGameDurationSec > 3600 {
			w.WriteHeader(http.StatusBadRequest)
			writeJSON(w, CreateGameHTTPResponse{
				Success: false,
				Error:   "maxGameDurationSec must be between 0 and 3600",
			})
			return
		}
		if req.MaxKills < 0 {
			w.WriteHeader(http.StatusBadRequest)
			writeJSON(w, CreateGameHTTPResponse{
				Success: false,
				Error:   "maxKills must be non-negative",
			})
			return
		}

		trainingOptions = &TrainingOptions{
			Enabled:             true,
			TickMultiplier:      req.TickMultiplier,
			MaxGameDurationSec:  req.MaxGameDurationSec,
			DisableRespawnTimer: req.DisableRespawnTimer,
			MaxKills:            req.MaxKills,
		}
	}

	// Validate map type before creating the game
	if !game_maps.IsValidMapType(req.MapType) {
		validTypes, _ := game_maps.GetValidMapTypes()
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, CreateGameHTTPResponse{
			Success: false,
			Error:   fmt.Sprintf("invalid mapType '%s', valid types are: %v", req.MapType, validTypes),
		})
		return
	}

	// Create game with specified map type and training options
	mapType := game_maps.MapType("meta/" + req.MapType + ".json")
	room, player, err := NewGameWithPlayerAndTrainingConfig(req.RoomName, req.PlayerName, mapType, nil, trainingOptions)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		writeJSON(w, CreateGameHTTPResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}
	s.AddGameRoomAndStartTick(room)

	s.serverLock.Lock()
	s.lastActivity[room.ID] = time.Now()
	s.serverLock.Unlock()

	// Build response
	canvasSizeX, canvasSizeY := room.ObjectManager.Map.GetCanvasSize()
	response := CreateGameHTTPResponse{
		Success:      true,
		RoomID:       room.ID,
		RoomCode:     room.RoomCode,
		RoomName:     room.Name,
		RoomPassword: room.Password,
		PlayerID:     player.ID,
		PlayerToken:  player.Token,
		CanvasSizeX:  canvasSizeX,
		CanvasSizeY:  canvasSizeY,
	}

	// Include training settings in response if training mode is enabled
	if room.TrainingOptions != nil && room.TrainingOptions.Enabled {
		response.TrainingMode = true
		// Return the actual applied tick multiplier (from room.TickMultiplier) rather than the
		// requested value, as the room may have applied defaults or constraints
		response.TickMultiplier = room.TickMultiplier
		response.MaxGameDurationSec = room.TrainingOptions.MaxGameDurationSec
		response.DisableRespawnTimer = room.TrainingOptions.DisableRespawnTimer
		response.MaxKills = room.TrainingOptions.MaxKills
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	writeJSON(w, response)

	log.Printf("Created game room %s with code %s via HTTP API", room.ID, room.RoomCode)
	log.Printf("Player %s joined game room %s via HTTP API", player.ID, room.ID)
}

// HandleJoinGame handles HTTP requests to join an existing game
func (s *Server) HandleJoinGame(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow POST requests
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		writeJSON(w, JoinGameHTTPResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Parse request body
	var req JoinGameHTTPRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, JoinGameHTTPResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Validate request
	if req.PlayerName == "" || req.RoomCode == "" || req.RoomPassword == "" {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, JoinGameHTTPResponse{
			Success: false,
			Error:   "PlayerName, RoomCode, and RoomPassword are required",
		})
		return
	}

	// Find room by code
	room, exists := s.roomManager.GetGameRoomByCode(req.RoomCode)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, JoinGameHTTPResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Add player to room
	isSpectator := req.IsSpectator || false
	player, err := AddPlayerToGame(room, req.PlayerName, req.RoomPassword, isSpectator)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		writeJSON(w, JoinGameHTTPResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	// Get canvas size
	canvasSizeX, canvasSizeY := room.ObjectManager.Map.GetCanvasSize()

	// update last activity for the room
	s.serverLock.Lock()
	s.lastActivity[room.ID] = time.Now()
	s.serverLock.Unlock()

	// Build response
	response := JoinGameHTTPResponse{
		Success:     true,
		PlayerID:    player.ID,
		PlayerToken: player.Token,
		RoomID:      room.ID,
		RoomCode:    room.RoomCode,
		IsSpectator: isSpectator,
		CanvasSizeX: canvasSizeX,
		CanvasSizeY: canvasSizeY,
	}

	// Include training settings in response if joining a training game
	if room.TrainingOptions != nil && room.TrainingOptions.Enabled {
		response.TrainingMode = true
		response.TickMultiplier = room.TickMultiplier
		response.MaxGameDurationSec = room.TrainingOptions.MaxGameDurationSec
		response.MaxKills = room.TrainingOptions.MaxKills
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	writeJSON(w, response)

	log.Printf("Player %s joined game room %s via HTTP API", player.ID, room.ID)
}

// HandleGetRoomState handles HTTP requests to get the current game state for a room
func (s *Server) HandleGetRoomState(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Player-Token")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow GET requests
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		writeJSON(w, GetRoomStateResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract roomId from URL path: /api/rooms/{roomId}/state
	path := r.URL.Path
	prefix := "/api/rooms/"
	suffix := "/state"
	if !strings.HasPrefix(path, prefix) || !strings.HasSuffix(path, suffix) {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, GetRoomStateResponse{
			Success: false,
			Error:   "Invalid URL format",
		})
		return
	}
	roomID := strings.TrimSuffix(strings.TrimPrefix(path, prefix), suffix)
	if roomID == "" {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, GetRoomStateResponse{
			Success: false,
			Error:   "Room ID is required",
		})
		return
	}

	// Get player token from query parameter or header
	playerToken := r.URL.Query().Get("playerToken")
	if playerToken == "" {
		playerToken = r.Header.Get("X-Player-Token")
	}
	if playerToken == "" {
		// Check Authorization header (Bearer token format)
		auth := r.Header.Get("Authorization")
		if strings.HasPrefix(auth, "Bearer ") {
			playerToken = strings.TrimPrefix(auth, "Bearer ")
		}
	}
	if playerToken == "" {
		w.WriteHeader(http.StatusUnauthorized)
		writeJSON(w, GetRoomStateResponse{
			Success: false,
			Error:   "Player token is required (provide via playerToken query param, X-Player-Token header, or Authorization: Bearer header)",
		})
		return
	}

	// Get room by ID
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, GetRoomStateResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Verify player token belongs to a player in the room (thread-safe)
	if !room.IsPlayerTokenValid(playerToken) {
		w.WriteHeader(http.StatusForbidden)
		writeJSON(w, GetRoomStateResponse{
			Success: false,
			Error:   "Player token is not authorized for this room",
		})
		return
	}

	// Get game state
	objectStates := room.GetAllGameObjectStates()

	// Return success response
	w.WriteHeader(http.StatusOK)
	writeJSON(w, GetRoomStateResponse{
		Success:      true,
		RoomID:       room.ID,
		Timestamp:    time.Now().UTC().Format(time.RFC3339),
		ObjectStates: objectStates,
	})
}

// HandleResetGame handles HTTP requests to reset a game room for a new training episode
func (s *Server) HandleResetGame(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Player-Token")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow POST requests
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		writeJSON(w, ResetGameHTTPResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract roomId from URL path: /api/rooms/{roomId}/reset
	path := r.URL.Path
	prefix := "/api/rooms/"
	suffix := "/reset"
	if !strings.HasPrefix(path, prefix) || !strings.HasSuffix(path, suffix) {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, ResetGameHTTPResponse{
			Success: false,
			Error:   "Invalid URL format",
		})
		return
	}
	roomID := strings.TrimSuffix(strings.TrimPrefix(path, prefix), suffix)
	if roomID == "" {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, ResetGameHTTPResponse{
			Success: false,
			Error:   "Room ID is required",
		})
		return
	}

	// Get player token from header for authentication
	playerToken := r.Header.Get("X-Player-Token")
	if playerToken == "" {
		// Check Authorization header (Bearer token format)
		auth := r.Header.Get("Authorization")
		if strings.HasPrefix(auth, "Bearer ") {
			playerToken = strings.TrimPrefix(auth, "Bearer ")
		}
	}
	if playerToken == "" {
		w.WriteHeader(http.StatusUnauthorized)
		writeJSON(w, ResetGameHTTPResponse{
			Success: false,
			Error:   "Player token is required (provide via X-Player-Token header or Authorization: Bearer header)",
		})
		return
	}

	// Get room by ID
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, ResetGameHTTPResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Verify player token belongs to a player in the room (thread-safe)
	if !room.IsPlayerTokenValid(playerToken) {
		w.WriteHeader(http.StatusForbidden)
		writeJSON(w, ResetGameHTTPResponse{
			Success: false,
			Error:   "Player token is not authorized for this room",
		})
		return
	}

	// Parse optional request body for room password validation
	var req ResetGameHTTPRequest
	if r.Body != nil && r.ContentLength > 0 {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			writeJSON(w, ResetGameHTTPResponse{
				Success: false,
				Error:   "Invalid request format",
			})
			return
		}

		// Validate room password if provided
		if req.RoomPassword != "" && room.Password != strings.ToUpper(req.RoomPassword) {
			w.WriteHeader(http.StatusUnauthorized)
			writeJSON(w, ResetGameHTTPResponse{
				Success: false,
				Error:   "Invalid room password",
			})
			return
		}
	}

	// Reset the game room
	room.Reset()

	// Update room activity
	s.serverLock.Lock()
	s.lastActivity[roomID] = time.Now()
	s.serverLock.Unlock()

	// Queue a full game state update to broadcast to all connected WebSocket clients
	s.gameStateUpdateQueue <- GameUpdateQueueItem{
		RoomID: roomID,
		Update: &types.GameUpdate{FullUpdate: true},
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	writeJSON(w, ResetGameHTTPResponse{
		Success: true,
	})

	log.Printf("Game room %s reset via HTTP API", roomID)
}

// HandleGetRoomStats handles HTTP requests to get player statistics for a room
func (s *Server) HandleGetRoomStats(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Player-Token")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow GET requests
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		writeJSON(w, GetRoomStatsHTTPResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract roomId from URL path: /api/rooms/{roomId}/stats
	path := r.URL.Path
	prefix := "/api/rooms/"
	suffix := "/stats"
	if !strings.HasPrefix(path, prefix) || !strings.HasSuffix(path, suffix) {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, GetRoomStatsHTTPResponse{
			Success: false,
			Error:   "Invalid URL format",
		})
		return
	}
	roomID := strings.TrimSuffix(strings.TrimPrefix(path, prefix), suffix)
	if roomID == "" {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, GetRoomStatsHTTPResponse{
			Success: false,
			Error:   "Room ID is required",
		})
		return
	}

	// Get player token from query parameter or header
	playerToken := r.URL.Query().Get("playerToken")
	if playerToken == "" {
		playerToken = r.Header.Get("X-Player-Token")
	}
	if playerToken == "" {
		// Check Authorization header (Bearer token format)
		auth := r.Header.Get("Authorization")
		if strings.HasPrefix(auth, "Bearer ") {
			playerToken = strings.TrimPrefix(auth, "Bearer ")
		}
	}
	if playerToken == "" {
		w.WriteHeader(http.StatusUnauthorized)
		writeJSON(w, GetRoomStatsHTTPResponse{
			Success: false,
			Error:   "Player token is required (provide via playerToken query param, X-Player-Token header, or Authorization: Bearer header)",
		})
		return
	}

	// Get room by ID
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, GetRoomStatsHTTPResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Verify player token belongs to a player in the room (thread-safe)
	if !room.IsPlayerTokenValid(playerToken) {
		w.WriteHeader(http.StatusForbidden)
		writeJSON(w, GetRoomStatsHTTPResponse{
			Success: false,
			Error:   "Player token is not authorized for this room",
		})
		return
	}

	// Get player stats with names atomically
	stats := room.GetAllPlayerStatsWithNames()
	playerStatsDTO := make(map[string]*PlayerStatsDTO, len(stats))
	for playerID, stat := range stats {
		playerStatsDTO[playerID] = &PlayerStatsDTO{
			PlayerID:   stat.PlayerID,
			PlayerName: stat.PlayerName,
			Kills:      stat.Kills,
			Deaths:     stat.Deaths,
		}
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	writeJSON(w, GetRoomStatsHTTPResponse{
		Success:     true,
		RoomID:      room.ID,
		PlayerStats: playerStatsDTO,
	})
}

// extractBotActionPathParams extracts roomId and playerId from the URL path
// Expected format: /api/rooms/{roomId}/players/{playerId}/action
func extractBotActionPathParams(path string) (roomID string, playerID string, ok bool) {
	parts := strings.Split(path, "/")
	// parts = ["", "api", "rooms", "{roomId}", "players", "{playerId}", "action"]
	if len(parts) == 7 && parts[1] == "api" && parts[2] == "rooms" &&
		parts[4] == "players" && parts[6] == "action" {
		return parts[3], parts[5], true
	}
	return "", "", false
}

// extractAddBotPathParams extracts roomId from /api/rooms/{roomId}/bots
func extractAddBotPathParams(path string) (roomID string, ok bool) {
	prefix := "/api/rooms/"
	suffix := "/bots"
	if !strings.HasPrefix(path, prefix) || !strings.HasSuffix(path, suffix) {
		return "", false
	}
	roomID = strings.TrimSuffix(strings.TrimPrefix(path, prefix), suffix)
	return roomID, roomID != ""
}

// extractRemoveBotPathParams extracts roomId and botId from /api/rooms/{roomId}/bots/{botId}
func extractRemoveBotPathParams(path string) (roomID string, botID string, ok bool) {
	parts := strings.Split(path, "/")
	// parts = ["", "api", "rooms", "{roomId}", "bots", "{botId}"]
	if len(parts) == 6 && parts[1] == "api" && parts[2] == "rooms" && parts[4] == "bots" {
		return parts[3], parts[5], parts[3] != "" && parts[5] != ""
	}
	return "", "", false
}

// HandleBotAction handles HTTP requests to submit bot actions
func (s *Server) HandleBotAction(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Player-Token")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow POST requests
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		writeJSON(w, types.BotActionResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract roomId and playerId from URL path
	roomID, playerID, ok := extractBotActionPathParams(r.URL.Path)
	if !ok {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, types.BotActionResponse{
			Success: false,
			Error:   "Invalid URL format. Expected: /api/rooms/{roomId}/players/{playerId}/action",
		})
		return
	}

	// Get player token from X-Player-Token header or Authorization header (Bearer token format)
	playerToken := r.Header.Get("X-Player-Token")
	if playerToken == "" {
		auth := r.Header.Get("Authorization")
		if strings.HasPrefix(auth, "Bearer ") {
			playerToken = strings.TrimPrefix(auth, "Bearer ")
		}
	}
	if playerToken == "" {
		log.Printf("HandleBotAction: Missing player token for room %s, player %s", roomID, playerID)
		w.WriteHeader(http.StatusUnauthorized)
		writeJSON(w, types.BotActionResponse{
			Success: false,
			Error:   "Player token is required (provide via X-Player-Token header or Authorization: Bearer header)",
		})
		return
	}

	// Get room by ID
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, types.BotActionResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Get player by ID
	player, exists := room.GetPlayer(playerID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, types.BotActionResponse{
			Success: false,
			Error:   "Player not found",
		})
		return
	}

	// Validate player token
	if player.Token != playerToken {
		log.Printf("HandleBotAction: Invalid player token for room %s, player %s", roomID, playerID)
		w.WriteHeader(http.StatusUnauthorized)
		writeJSON(w, types.BotActionResponse{
			Success: false,
			Error:   "Invalid player token",
		})
		return
	}

	// Parse request body (limit to 1MB to prevent abuse)
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	var req types.BotActionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, types.BotActionResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Process each action
	actionsProcessed := 0
	for _, action := range req.Actions {
		var event *game_objects.GameEvent

		switch action.Type {
		case "key":
			// Validate key value
			key := strings.ToUpper(action.Key)
			if key != "W" && key != "A" && key != "S" && key != "D" {
				w.WriteHeader(http.StatusBadRequest)
				writeJSON(w, types.BotActionResponse{
					Success: false,
					Error:   "Invalid key value. Must be W, A, S, or D",
				})
				return
			}
			eventData := map[string]interface{}{
				"playerId": playerID,
				"key":      key,
				"isDown":   action.IsDown,
			}
			event = game_objects.NewGameEvent(
				roomID,
				game_objects.EventPlayerKeyInput,
				eventData,
				1,
				nil,
			)

		case "click":
			eventData := map[string]interface{}{
				"playerId": playerID,
				"x":        action.X,
				"y":        action.Y,
				"isDown":   action.IsDown,
				"button":   action.Button,
			}
			event = game_objects.NewGameEvent(
				roomID,
				game_objects.EventPlayerClickInput,
				eventData,
				1,
				nil,
			)

		case "direction":
			eventData := map[string]interface{}{
				"playerId":  playerID,
				"direction": action.Direction,
			}
			event = game_objects.NewGameEvent(
				roomID,
				game_objects.EventPlayerDirection,
				eventData,
				1,
				nil,
			)

		default:
			w.WriteHeader(http.StatusBadRequest)
			writeJSON(w, types.BotActionResponse{
				Success: false,
				Error:   fmt.Sprintf("Invalid action type: %s. Must be key, click, or direction", action.Type),
			})
			return
		}

		// Process the event (reuses processEvent from message_handlers.go)
		s.processEvent(room, event)
		actionsProcessed++
	}

	// Update room activity
	s.serverLock.Lock()
	s.lastActivity[roomID] = time.Now()
	s.serverLock.Unlock()

	// Return success response
	w.WriteHeader(http.StatusOK)
	writeJSON(w, types.BotActionResponse{
		Success:          true,
		ActionsProcessed: actionsProcessed,
		Timestamp:        time.Now().UnixMilli(),
	})

	log.Printf("Bot action submitted for player %s in room %s: %d actions processed", playerID, roomID, actionsProcessed)
}

// HandleGetTrainingSessions handles HTTP requests to list active training sessions
func (s *Server) HandleGetTrainingSessions(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow GET requests
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		writeJSON(w, map[string]interface{}{
			"error": "Method not allowed",
		})
		return
	}

	// Get all training rooms
	trainingRooms := s.roomManager.GetTrainingRooms()

	// Build response
	sessions := make([]TrainingSessionInfo, 0, len(trainingRooms))
	for _, room := range trainingRooms {
		// Format duration as a human-readable string
		duration := room.GetRoomDuration()
		durationStr := duration.Round(time.Second).String()

		sessions = append(sessions, TrainingSessionInfo{
			RoomID:         room.ID,
			RoomCode:       room.RoomCode,
			TickMultiplier: room.TickMultiplier,
			PlayerCount:    room.GetPlayerCount(),
			SpectatorCount: room.GetSpectatorCount(),
			Duration:       durationStr,
		})
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	writeJSON(w, GetTrainingSessionsResponse{
		Sessions: sessions,
	})
}

// HandleHealth handles HTTP requests for server health checks
func (s *Server) HandleHealth(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow GET requests
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		writeJSON(w, map[string]interface{}{
			"error": "Method not allowed",
		})
		return
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	writeJSON(w, HealthCheckResponse{
		Status:    "ok",
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	})
}

// getBotServiceURL returns the Bot Service URL from environment or default
func getBotServiceURL() string {
	url := os.Getenv("BOT_SERVICE_URL")
	if url == "" {
		return "http://localhost:8080"
	}
	return url
}

// botServiceClient is a reusable HTTP client for Bot Service requests
// This enables connection pooling and HTTP keep-alive
var botServiceClient = &http.Client{Timeout: 10 * time.Second}

// HandleAddBot handles POST /api/rooms/{roomId}/bots
func (s *Server) HandleAddBot(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Player-Token")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow POST requests
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract roomId from URL path: /api/rooms/{roomId}/bots
	roomID, ok := extractAddBotPathParams(r.URL.Path)
	if !ok {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Invalid URL format. Expected: /api/rooms/{roomId}/bots",
		})
		return
	}

	if roomID == "" {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Room ID is required",
		})
		return
	}

	// Get player token from X-Player-Token header or Authorization header
	playerToken := r.Header.Get("X-Player-Token")
	if playerToken == "" {
		auth := r.Header.Get("Authorization")
		if strings.HasPrefix(auth, "Bearer ") {
			playerToken = strings.TrimPrefix(auth, "Bearer ")
		}
	}
	if playerToken == "" {
		w.WriteHeader(http.StatusUnauthorized)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Player token is required (provide via X-Player-Token header or Authorization: Bearer header)",
		})
		return
	}

	// Get room by ID
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Verify player token belongs to a player in the room
	if !room.IsPlayerTokenValid(playerToken) {
		w.WriteHeader(http.StatusForbidden)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Player token is not authorized for this room",
		})
		return
	}

	// Parse request body (limit to 1MB to prevent abuse)
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	var req AddBotRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Validate bot type
	if req.BotType != "rule_based" && req.BotType != "neural_network" {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Invalid botType. Must be 'rule_based' or 'neural_network'",
		})
		return
	}

	// Set default player name if not provided
	playerName := req.PlayerName
	if playerName == "" {
		playerName = "Bot"
	}

	// Construct Bot Service request
	botServiceReq := botServiceSpawnRequest{
		RoomCode:     room.RoomCode,
		RoomPassword: room.Password,
		BotConfig: botServiceBotConfig{
			BotType:    req.BotType,
			ModelID:    req.ModelID,
			Generation: req.Generation,
			PlayerName: playerName,
		},
	}

	// Marshal request body
	reqBody, err := json.Marshal(botServiceReq)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Failed to construct request",
		})
		return
	}

	// Forward request to Bot Service (using request context for cancellation propagation)
	botServiceURL := getBotServiceURL()
	proxyReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, botServiceURL+"/bots/spawn", bytes.NewReader(reqBody))
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Failed to construct request",
		})
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	resp, err := botServiceClient.Do(proxyReq)
	if err != nil {
		log.Printf("HandleAddBot: Failed to connect to Bot Service: %v", err)
		w.WriteHeader(http.StatusServiceUnavailable)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Bot Service unavailable",
		})
		return
	}
	defer resp.Body.Close()

	// Read response body (limit to 1MB to prevent memory exhaustion)
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Failed to read Bot Service response",
		})
		return
	}

	// Handle Bot Service errors
	if resp.StatusCode == http.StatusNotFound {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Room not found in Bot Service",
		})
		return
	}

	// Handle other non-success status codes from Bot Service
	if resp.StatusCode >= 400 {
		log.Printf("HandleAddBot: Bot Service returned status %d", resp.StatusCode)
		w.WriteHeader(resp.StatusCode)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   fmt.Sprintf("Bot Service error (status %d)", resp.StatusCode),
		})
		return
	}

	// Parse Bot Service response
	var botServiceResp botServiceSpawnResponse
	if err := json.Unmarshal(respBody, &botServiceResp); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		writeJSON(w, AddBotResponse{
			Success: false,
			Error:   "Failed to parse Bot Service response",
		})
		return
	}

	// Update room activity
	s.serverLock.Lock()
	s.lastActivity[roomID] = time.Now()
	s.serverLock.Unlock()

	// Return response
	w.WriteHeader(http.StatusOK)
	writeJSON(w, AddBotResponse(botServiceResp))

	if botServiceResp.Success {
		log.Printf("Bot %s added to room %s via HTTP API", botServiceResp.BotID, roomID)
	}
}

// HandleRemoveBot handles DELETE /api/rooms/{roomId}/bots/{botId}
func (s *Server) HandleRemoveBot(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Player-Token")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow DELETE requests
	if r.Method != "DELETE" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract roomId and botId from URL path: /api/rooms/{roomId}/bots/{botId}
	roomID, botID, ok := extractRemoveBotPathParams(r.URL.Path)
	if !ok {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Invalid URL format. Expected: /api/rooms/{roomId}/bots/{botId}",
		})
		return
	}

	if roomID == "" {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Room ID is required",
		})
		return
	}

	if botID == "" {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Bot ID is required",
		})
		return
	}

	// Get player token from X-Player-Token header or Authorization header
	playerToken := r.Header.Get("X-Player-Token")
	if playerToken == "" {
		auth := r.Header.Get("Authorization")
		if strings.HasPrefix(auth, "Bearer ") {
			playerToken = strings.TrimPrefix(auth, "Bearer ")
		}
	}
	if playerToken == "" {
		w.WriteHeader(http.StatusUnauthorized)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Player token is required (provide via X-Player-Token header or Authorization: Bearer header)",
		})
		return
	}

	// Get room by ID
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Verify player token belongs to a player in the room
	if !room.IsPlayerTokenValid(playerToken) {
		w.WriteHeader(http.StatusForbidden)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Player token is not authorized for this room",
		})
		return
	}

	// Forward DELETE request to Bot Service (using request context for cancellation propagation)
	botServiceURL := getBotServiceURL()
	proxyReq, err := http.NewRequestWithContext(r.Context(), http.MethodDelete, botServiceURL+"/bots/"+botID, nil)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Failed to construct request",
		})
		return
	}

	resp, err := botServiceClient.Do(proxyReq)
	if err != nil {
		log.Printf("HandleRemoveBot: Failed to connect to Bot Service: %v", err)
		w.WriteHeader(http.StatusServiceUnavailable)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Bot Service unavailable",
		})
		return
	}
	defer resp.Body.Close()

	// Read response body (limit to 1MB to prevent memory exhaustion)
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Failed to read Bot Service response",
		})
		return
	}

	// Handle Bot Service 404 (bot not found)
	if resp.StatusCode == http.StatusNotFound {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Bot not found",
		})
		return
	}

	// Handle other non-success status codes from Bot Service
	if resp.StatusCode >= 400 {
		log.Printf("HandleRemoveBot: Bot Service returned status %d", resp.StatusCode)
		w.WriteHeader(resp.StatusCode)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   fmt.Sprintf("Bot Service error (status %d)", resp.StatusCode),
		})
		return
	}

	// Parse Bot Service response
	var botServiceResp botServiceRemoveResponse
	if err := json.Unmarshal(respBody, &botServiceResp); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		writeJSON(w, RemoveBotResponse{
			Success: false,
			Error:   "Failed to parse Bot Service response",
		})
		return
	}

	// Update room activity
	s.serverLock.Lock()
	s.lastActivity[roomID] = time.Now()
	s.serverLock.Unlock()

	// Return response
	w.WriteHeader(http.StatusOK)
	writeJSON(w, RemoveBotResponse(botServiceResp))

	if botServiceResp.Success {
		log.Printf("Bot %s removed from room %s via HTTP API", botID, roomID)
	}
}
