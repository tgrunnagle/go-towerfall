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

	// Create game with specified map type
	mapType := game_maps.MapType("meta/" + req.MapType + ".json")
	room, player, err := NewGameWithPlayer(req.RoomName, req.PlayerName, mapType)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(CreateGameHTTPResponse{
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
	json.NewEncoder(w).Encode(CreateGameHTTPResponse{
		Success:     true,
		RoomID:      room.ID,
		RoomCode:    room.RoomCode,
		RoomName:    room.Name,
		PlayerID:    player.ID,
		PlayerToken: player.Token,
		CanvasSizeX: canvasSizeX,
		CanvasSizeY: canvasSizeY,
	})

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

// RoomDetailsResponse represents the response for room details
type RoomDetailsResponse struct {
	Success      bool   `json:"success"`
	RoomID       string `json:"roomId,omitempty"`
	RoomCode     string `json:"roomCode,omitempty"`
	RoomName     string `json:"roomName,omitempty"`
	RoomPassword string `json:"roomPassword,omitempty"`
	Error        string `json:"error,omitempty"`
}

// HandleGetRoomDetails handles HTTP requests to get room details by room ID
func (s *Server) HandleGetRoomDetails(w http.ResponseWriter, r *http.Request) {
	log.Printf("HandleGetRoomDetails called with path: %s", r.URL.Path)

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
		log.Printf("Method not allowed: %s", r.Method)
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(RoomDetailsResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract room ID from URL path
	roomID := extractRoomIDFromPath(r.URL.Path, "/api/rooms/", "/details")
	log.Printf("Extracted room ID: %s", roomID)

	if roomID == "" {
		log.Printf("Invalid room ID extracted from path: %s", r.URL.Path)
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(RoomDetailsResponse{
			Success: false,
			Error:   "Invalid room ID in URL",
		})
		return
	}

	// Find room
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		log.Printf("Room not found: %s", roomID)
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(RoomDetailsResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	log.Printf("Found room: %s, returning details", roomID)

	// Return room details
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(RoomDetailsResponse{
		Success:      true,
		RoomID:       room.ID,
		RoomCode:     room.RoomCode,
		RoomName:     room.Name,
		RoomPassword: room.Password,
	})
}

// HandleRoomDetails routes room detail requests to appropriate handlers
func (s *Server) HandleRoomDetails(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers for all room detail requests
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	path := r.URL.Path
	log.Printf("HandleRoomDetails called with path: %s, method: %s", path, r.Method)

	// Route based on path pattern
	if strings.Contains(path, "/details") && r.Method == "GET" {
		// GET /api/rooms/{roomId}/details - Get room details
		log.Printf("Routing to HandleGetRoomDetails")
		s.HandleGetRoomDetails(w, r)
	} else {
		log.Printf("No matching route found for path: %s", path)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   "Endpoint not found",
		})
	}
}
