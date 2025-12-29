package server

import (
	"encoding/json"
	"fmt"
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/game_maps"
	"log"
	"net/http"
	"path/filepath"
	"strings"
	"time"
)

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
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": "Method not allowed",
		})
		return
	}

	// Get all map metadata
	metadata, err := game_maps.GetAllMapsMetadata()
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]interface{}{
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
	json.NewEncoder(w).Encode(GetMapsResponse{
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
	Success     bool   `json:"success"`
	PlayerID    string `json:"playerId,omitempty"`
	PlayerToken string `json:"playerToken,omitempty"`
	RoomID      string `json:"roomId,omitempty"`
	RoomCode    string `json:"roomCode,omitempty"`
	RoomName    string `json:"roomName,omitempty"`
	CanvasSizeX int    `json:"canvasSizeX,omitempty"`
	CanvasSizeY int    `json:"canvasSizeY,omitempty"`
	Error       string `json:"error,omitempty"`
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
		json.NewEncoder(w).Encode(CreateGameHTTPResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Parse request body
	var req CreateGameHTTPRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(CreateGameHTTPResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Validate request
	if req.PlayerName == "" || req.RoomName == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(CreateGameHTTPResponse{
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
			json.NewEncoder(w).Encode(CreateGameHTTPResponse{
				Success: false,
				Error:   "tickMultiplier must be between 1.0 and 20.0",
			})
			return
		}
		if req.MaxGameDurationSec < 0 || req.MaxGameDurationSec > 3600 {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(CreateGameHTTPResponse{
				Success: false,
				Error:   "maxGameDurationSec must be between 0 and 3600",
			})
			return
		}
		if req.MaxKills < 0 {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(CreateGameHTTPResponse{
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

	// Create game with specified map type and training options
	mapType := game_maps.MapType("meta/" + req.MapType + ".json")
	room, player, err := NewGameWithPlayerAndTrainingConfig(req.RoomName, req.PlayerName, mapType, nil, trainingOptions)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(CreateGameHTTPResponse{
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
		Success:     true,
		RoomID:      room.ID,
		RoomCode:    room.RoomCode,
		RoomName:    room.Name,
		PlayerID:    player.ID,
		PlayerToken: player.Token,
		CanvasSizeX: canvasSizeX,
		CanvasSizeY: canvasSizeY,
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
	json.NewEncoder(w).Encode(response)

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
		json.NewEncoder(w).Encode(JoinGameHTTPResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Parse request body
	var req JoinGameHTTPRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(JoinGameHTTPResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Validate request
	if req.PlayerName == "" || req.RoomCode == "" || req.RoomPassword == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(JoinGameHTTPResponse{
			Success: false,
			Error:   "PlayerName, RoomCode, and RoomPassword are required",
		})
		return
	}

	// Find room by code
	room, exists := s.roomManager.GetGameRoomByCode(req.RoomCode)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(JoinGameHTTPResponse{
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
		json.NewEncoder(w).Encode(JoinGameHTTPResponse{
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

	// Return success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(JoinGameHTTPResponse{
		Success:     true,
		PlayerID:    player.ID,
		PlayerToken: player.Token,
		RoomID:      room.ID,
		RoomCode:    room.RoomCode,
		IsSpectator: isSpectator,
		CanvasSizeX: canvasSizeX,
		CanvasSizeY: canvasSizeY,
	})

	log.Printf("Player %s joined game room %s via HTTP API", player.ID, room.ID)
}
