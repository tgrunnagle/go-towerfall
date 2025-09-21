package server

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"time"

	"go-ws-server/pkg/server/game_maps"
)

// TrainingConfig represents training-specific room configuration
type TrainingConfig struct {
	SpeedMultiplier  float64 `json:"speedMultiplier"`
	HeadlessMode     bool    `json:"headlessMode"`
	TrainingMode     bool    `json:"trainingMode"`
	SessionID        string  `json:"sessionId,omitempty"`
	DirectStateAccess bool   `json:"directStateAccess,omitempty"`
}

// CreateTrainingRoomRequest represents the request to create a training room
type CreateTrainingRoomRequest struct {
	PlayerName     string         `json:"playerName"`
	RoomName       string         `json:"roomName"`
	MapType        string         `json:"mapType"`
	TrainingConfig TrainingConfig `json:"trainingConfig"`
}

// CreateTrainingRoomResponse represents the response to create training room
type CreateTrainingRoomResponse struct {
	Success        bool    `json:"success"`
	PlayerID       string  `json:"playerId,omitempty"`
	PlayerToken    string  `json:"playerToken,omitempty"`
	RoomID         string  `json:"roomId,omitempty"`
	RoomCode       string  `json:"roomCode,omitempty"`
	RoomName       string  `json:"roomName,omitempty"`
	CanvasSizeX    int     `json:"canvasSizeX,omitempty"`
	CanvasSizeY    int     `json:"canvasSizeY,omitempty"`
	SpeedMultiplier float64 `json:"speedMultiplier,omitempty"`
	HeadlessMode   bool    `json:"headlessMode,omitempty"`
	Error          string  `json:"error,omitempty"`
}

// JoinTrainingRoomRequest represents the request to join a training room
type JoinTrainingRoomRequest struct {
	PlayerName         string `json:"playerName"`
	RoomCode           string `json:"roomCode"`
	RoomPassword       string `json:"roomPassword"`
	EnableDirectAccess bool   `json:"enableDirectAccess,omitempty"`
}

// JoinTrainingRoomResponse represents the response to join training room
type JoinTrainingRoomResponse struct {
	Success          bool    `json:"success"`
	PlayerID         string  `json:"playerId,omitempty"`
	PlayerToken      string  `json:"playerToken,omitempty"`
	RoomID           string  `json:"roomId,omitempty"`
	RoomCode         string  `json:"roomCode,omitempty"`
	CanvasSizeX      int     `json:"canvasSizeX,omitempty"`
	CanvasSizeY      int     `json:"canvasSizeY,omitempty"`
	TrainingEnabled  bool    `json:"trainingEnabled"`
	SpeedMultiplier  float64 `json:"speedMultiplier,omitempty"`
	HeadlessMode     bool    `json:"headlessMode,omitempty"`
	Error            string  `json:"error,omitempty"`
}

// SetRoomSpeedRequest represents the request to set room speed
type SetRoomSpeedRequest struct {
	SpeedMultiplier float64 `json:"speedMultiplier"`
}

// SetRoomSpeedResponse represents the response to set room speed
type SetRoomSpeedResponse struct {
	Success         bool    `json:"success"`
	SpeedMultiplier float64 `json:"speedMultiplier,omitempty"`
	Error           string  `json:"error,omitempty"`
}

// GetRoomStateResponse represents the response for direct state access
type GetRoomStateResponse struct {
	Success   bool                   `json:"success"`
	State     map[string]interface{} `json:"state,omitempty"`
	Timestamp int64                  `json:"timestamp,omitempty"`
	Error     string                 `json:"error,omitempty"`
}

// ConfigureTrainingRoomRequest represents the request to configure training room
type ConfigureTrainingRoomRequest struct {
	TrainingMode      string  `json:"trainingMode"`
	SpeedMultiplier   float64 `json:"speedMultiplier"`
	DirectStateAccess bool    `json:"directStateAccess"`
}

// ConfigureTrainingRoomResponse represents the response to configure training room
type ConfigureTrainingRoomResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

// HandleCreateTrainingRoom handles HTTP requests to create a training room
func (s *Server) HandleCreateTrainingRoom(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow POST requests
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(CreateTrainingRoomResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Parse request body
	var req CreateTrainingRoomRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(CreateTrainingRoomResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Validate request
	if req.PlayerName == "" || req.RoomName == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(CreateTrainingRoomResponse{
			Success: false,
			Error:   "PlayerName and RoomName are required",
		})
		return
	}

	// Validate speed multiplier
	if req.TrainingConfig.SpeedMultiplier <= 0 || req.TrainingConfig.SpeedMultiplier > 100 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(CreateTrainingRoomResponse{
			Success: false,
			Error:   "SpeedMultiplier must be between 0.1 and 100",
		})
		return
	}

	// Create training room with specified configuration
	mapType := game_maps.MapType("meta/" + req.MapType + ".json")
	room, player, err := NewTrainingGameWithPlayer(
		req.RoomName, 
		req.PlayerName, 
		mapType, 
		req.TrainingConfig,
	)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(CreateTrainingRoomResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	s.roomManager.AddGameRoom(room)

	s.serverLock.Lock()
	s.lastActivity[room.ID] = time.Now()
	s.serverLock.Unlock()

	// Return success response
	w.WriteHeader(http.StatusOK)
	canvasSizeX, canvasSizeY := room.ObjectManager.Map.GetCanvasSize()
	json.NewEncoder(w).Encode(CreateTrainingRoomResponse{
		Success:         true,
		RoomID:          room.ID,
		RoomCode:        room.RoomCode,
		RoomName:        room.Name,
		PlayerID:        player.ID,
		PlayerToken:     player.Token,
		CanvasSizeX:     canvasSizeX,
		CanvasSizeY:     canvasSizeY,
		SpeedMultiplier: req.TrainingConfig.SpeedMultiplier,
		HeadlessMode:    req.TrainingConfig.HeadlessMode,
	})

	log.Printf("Created training room %s with code %s (Speed: %.1fx, Headless: %v)", 
		room.ID, room.RoomCode, req.TrainingConfig.SpeedMultiplier, req.TrainingConfig.HeadlessMode)
}

// HandleJoinTrainingRoom handles HTTP requests to join a training room
func (s *Server) HandleJoinTrainingRoom(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow POST requests
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(JoinTrainingRoomResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Parse request body
	var req JoinTrainingRoomRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(JoinTrainingRoomResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Validate request
	if req.PlayerName == "" || req.RoomCode == "" || req.RoomPassword == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(JoinTrainingRoomResponse{
			Success: false,
			Error:   "PlayerName, RoomCode, and RoomPassword are required",
		})
		return
	}

	// Find room by code
	room, exists := s.roomManager.GetGameRoomByCode(req.RoomCode)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(JoinTrainingRoomResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Add player to room
	player, err := AddPlayerToGame(room, req.PlayerName, req.RoomPassword, false)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(JoinTrainingRoomResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	// Get canvas size
	canvasSizeX, canvasSizeY := room.ObjectManager.Map.GetCanvasSize()

	// Update last activity for the room
	s.serverLock.Lock()
	s.lastActivity[room.ID] = time.Now()
	s.serverLock.Unlock()

	// Check if room has training capabilities
	trainingEnabled := room.IsTrainingRoom()
	var speedMultiplier float64 = 1.0
	var headlessMode bool = false
	
	if trainingEnabled {
		speedMultiplier = room.GetSpeedMultiplier()
		headlessMode = room.IsHeadlessMode()
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(JoinTrainingRoomResponse{
		Success:         true,
		PlayerID:        player.ID,
		PlayerToken:     player.Token,
		RoomID:          room.ID,
		RoomCode:        room.RoomCode,
		CanvasSizeX:     canvasSizeX,
		CanvasSizeY:     canvasSizeY,
		TrainingEnabled: trainingEnabled,
		SpeedMultiplier: speedMultiplier,
		HeadlessMode:    headlessMode,
	})

	log.Printf("Player %s joined training room %s", player.ID, room.ID)
}

// HandleSetRoomSpeed handles HTTP requests to set room speed
func (s *Server) HandleSetRoomSpeed(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow POST requests
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(SetRoomSpeedResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract room ID from URL path
	roomID := extractRoomIDFromPath(r.URL.Path, "/api/training/rooms/", "/speed")
	if roomID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(SetRoomSpeedResponse{
			Success: false,
			Error:   "Invalid room ID in URL",
		})
		return
	}

	// Validate authorization
	if !s.validatePlayerToken(r, roomID) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(SetRoomSpeedResponse{
			Success: false,
			Error:   "Unauthorized",
		})
		return
	}

	// Parse request body
	var req SetRoomSpeedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(SetRoomSpeedResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Validate speed multiplier
	if req.SpeedMultiplier <= 0 || req.SpeedMultiplier > 100 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(SetRoomSpeedResponse{
			Success: false,
			Error:   "SpeedMultiplier must be between 0.1 and 100",
		})
		return
	}

	// Find room
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(SetRoomSpeedResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Check if room supports training
	if !room.IsTrainingRoom() {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(SetRoomSpeedResponse{
			Success: false,
			Error:   "Room does not support speed control",
		})
		return
	}

	// Set room speed
	err := room.SetSpeedMultiplier(req.SpeedMultiplier)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(SetRoomSpeedResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(SetRoomSpeedResponse{
		Success:         true,
		SpeedMultiplier: req.SpeedMultiplier,
	})

	log.Printf("Set speed multiplier to %.1fx for room %s", req.SpeedMultiplier, roomID)
}

// HandleGetRoomState handles HTTP requests to get room state directly
func (s *Server) HandleGetRoomState(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow GET requests
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(GetRoomStateResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract room ID from URL path
	roomID := extractRoomIDFromPath(r.URL.Path, "/api/training/rooms/", "/state")
	if roomID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(GetRoomStateResponse{
			Success: false,
			Error:   "Invalid room ID in URL",
		})
		return
	}

	// Validate authorization
	if !s.validatePlayerToken(r, roomID) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(GetRoomStateResponse{
			Success: false,
			Error:   "Unauthorized",
		})
		return
	}

	// Find room
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(GetRoomStateResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Get current game state
	gameState := room.GetDirectGameState()
	
	// Return success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(GetRoomStateResponse{
		Success:   true,
		State:     gameState,
		Timestamp: time.Now().UnixMilli(),
	})
}

// HandleConfigureTrainingRoom handles HTTP requests to configure training room
func (s *Server) HandleConfigureTrainingRoom(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow POST requests
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(ConfigureTrainingRoomResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract room ID from URL path
	roomID := extractRoomIDFromPath(r.URL.Path, "/api/training/rooms/", "/configure")
	if roomID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ConfigureTrainingRoomResponse{
			Success: false,
			Error:   "Invalid room ID in URL",
		})
		return
	}

	// Validate authorization
	if !s.validatePlayerToken(r, roomID) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(ConfigureTrainingRoomResponse{
			Success: false,
			Error:   "Unauthorized",
		})
		return
	}

	// Parse request body
	var req ConfigureTrainingRoomRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ConfigureTrainingRoomResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Find room
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(ConfigureTrainingRoomResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Configure training room
	err := room.ConfigureTraining(req.TrainingMode, req.SpeedMultiplier, req.DirectStateAccess)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ConfigureTrainingRoomResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(ConfigureTrainingRoomResponse{
		Success: true,
	})

	log.Printf("Configured training room %s: mode=%s, speed=%.1fx, directAccess=%v", 
		roomID, req.TrainingMode, req.SpeedMultiplier, req.DirectStateAccess)
}

// Helper function to extract room ID from URL path
func extractRoomIDFromPath(path, prefix, suffix string) string {
	if !strings.HasPrefix(path, prefix) || !strings.HasSuffix(path, suffix) {
		return ""
	}
	
	start := len(prefix)
	end := len(path) - len(suffix)
	
	if start >= end {
		return ""
	}
	
	return path[start:end]
}

// HandleTrainingRoomRequests routes training room requests to appropriate handlers
func (s *Server) HandleTrainingRoomRequests(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	
	// Route based on path pattern
	if strings.Contains(path, "/speed") {
		s.HandleSetRoomSpeed(w, r)
	} else if strings.Contains(path, "/state") {
		s.HandleGetRoomState(w, r)
	} else if strings.Contains(path, "/configure") {
		s.HandleConfigureTrainingRoom(w, r)
	} else {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   "Endpoint not found",
		})
	}
}

// Helper function to validate player token
func (s *Server) validatePlayerToken(r *http.Request, roomID string) bool {
	authHeader := r.Header.Get("Authorization")
	if !strings.HasPrefix(authHeader, "Bearer ") {
		return false
	}
	
	token := strings.TrimPrefix(authHeader, "Bearer ")
	
	// Find room and validate token
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		return false
	}
	
	return room.ValidatePlayerToken(token)
}