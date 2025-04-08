package server

import (
	"encoding/json"
	"log"
	"net/http"
	"time"
)

// CreateGameRequest represents the request to create a new game
type CreateGameHTTPRequest struct {
	PlayerName string `json:"playerName"`
	RoomName   string `json:"roomName"`
}

// CreateGameResponse represents the response to a create game request
type CreateGameHTTPResponse struct {
	Success      bool   `json:"success"`
	PlayerID     string `json:"playerId,omitempty"`
	PlayerToken  string `json:"playerToken,omitempty"`
	RoomID       string `json:"roomId,omitempty"`
	RoomCode     string `json:"roomCode,omitempty"`
	RoomPassword string `json:"roomPassword,omitempty"`
	Error        string `json:"error,omitempty"`
}

// JoinGameRequest represents the request to join an existing game
type JoinGameHTTPRequest struct {
	PlayerName   string `json:"playerName"`
	RoomCode     string `json:"roomCode"`
	RoomPassword string `json:"roomPassword"`
}

// JoinGameResponse represents the response to a join game request
type JoinGameHTTPResponse struct {
	Success     bool   `json:"success"`
	PlayerID    string `json:"playerId,omitempty"`
	PlayerToken string `json:"playerToken,omitempty"`
	RoomID      string `json:"roomId,omitempty"`
	RoomCode    string `json:"roomCode,omitempty"`
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

	room, player, err := NewGameWithPlayer(req.RoomName, req.PlayerName)
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
	json.NewEncoder(w).Encode(CreateGameHTTPResponse{
		Success:      true,
		PlayerID:     player.ID,
		PlayerToken:  player.Token,
		RoomID:       room.ID,
		RoomCode:     room.RoomCode,
		RoomPassword: room.Password,
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
	player, err := AddPlayerToGame(room, req.PlayerName, req.RoomPassword)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(JoinGameHTTPResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

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
	})

	log.Printf("Player %s joined game room %s via HTTP API", player.ID, room.ID)
}
