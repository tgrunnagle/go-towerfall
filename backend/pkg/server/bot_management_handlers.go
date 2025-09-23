package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// Bot server integration configuration
const (
	BOT_SERVER_URL     = "http://localhost:8001"
	BOT_SERVER_TIMEOUT = 10 * time.Second
)

// Bot management API types

// GetAvailableBotsResponse represents the response for available bots
type GetAvailableBotsResponse struct {
	Success  bool               `json:"success"`
	BotTypes []AvailableBotType `json:"botTypes,omitempty"`
	Error    string             `json:"error,omitempty"`
}

// AvailableBotType represents a type of bot available for spawning
type AvailableBotType struct {
	Type                 string   `json:"type"`
	Name                 string   `json:"name"`
	Description          string   `json:"description"`
	Difficulties         []string `json:"difficulties,omitempty"`
	AvailableGenerations []int    `json:"availableGenerations,omitempty"`
	SupportsTrainingMode bool     `json:"supportsTrainingMode"`
}

// AddBotToRoomRequest represents the request to add a bot to a room
type AddBotToRoomRequest struct {
	BotType      string `json:"botType"`
	Difficulty   string `json:"difficulty"`
	BotName      string `json:"botName"`
	Generation   *int   `json:"generation,omitempty"`
	TrainingMode bool   `json:"trainingMode,omitempty"`
}

// AddBotToRoomResponse represents the response to add bot request
type AddBotToRoomResponse struct {
	Success bool   `json:"success"`
	BotID   string `json:"botId,omitempty"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
}

// RemoveBotFromRoomResponse represents the response to remove bot request
type RemoveBotFromRoomResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
}

// GetRoomBotsResponse represents the response for room bots
type GetRoomBotsResponse struct {
	Success bool      `json:"success"`
	Bots    []BotInfo `json:"bots,omitempty"`
	Error   string    `json:"error,omitempty"`
}

// BotInfo represents information about a bot in a room
type BotInfo struct {
	BotID       string                 `json:"botId"`
	BotType     string                 `json:"botType"`
	Difficulty  string                 `json:"difficulty"`
	Name        string                 `json:"name"`
	Generation  *int                   `json:"generation,omitempty"`
	Status      string                 `json:"status"`
	CreatedAt   string                 `json:"createdAt"`
	Performance map[string]interface{} `json:"performance,omitempty"`
}

// ConfigureBotDifficultyRequest represents the request to configure bot difficulty
type ConfigureBotDifficultyRequest struct {
	Difficulty string `json:"difficulty"`
}

// ConfigureBotDifficultyResponse represents the response to configure bot difficulty
type ConfigureBotDifficultyResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
}

// GetBotServerStatusResponse represents the response for bot server status
type GetBotServerStatusResponse struct {
	Success       bool                   `json:"success"`
	ServerRunning bool                   `json:"serverRunning"`
	TotalBots     int                    `json:"totalBots"`
	ActiveRooms   int                    `json:"activeRooms"`
	ServerStats   map[string]interface{} `json:"serverStats,omitempty"`
	Error         string                 `json:"error,omitempty"`
}

// HandleGetAvailableBots handles HTTP requests to get available bot types
func (s *Server) HandleGetAvailableBots(w http.ResponseWriter, r *http.Request) {
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
		json.NewEncoder(w).Encode(GetAvailableBotsResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Make request to Python bot server
	response, err := s.makeBotServerRequest("GET", "/api/bots/available", nil)
	if err != nil {
		// Fallback to static bot types if bot server is unavailable
		log.Printf("Bot server unavailable, returning static bot types: %v", err)
		botTypes := []AvailableBotType{
			{
				Type:                 "rules_based",
				Name:                 "Rules-Based Bot",
				Description:          "Traditional AI with configurable difficulty levels",
				Difficulties:         []string{"beginner", "intermediate", "advanced", "expert"},
				SupportsTrainingMode: true,
			},
		}

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(GetAvailableBotsResponse{
			Success:  true,
			BotTypes: botTypes,
		})
		return
	}

	// Parse response
	var botServerResponse struct {
		Success  bool `json:"success"`
		BotTypes []struct {
			Type                 string   `json:"type"`
			Name                 string   `json:"name"`
			Description          string   `json:"description"`
			Difficulties         []string `json:"difficulties"`
			AvailableGenerations []int    `json:"available_generations"`
			SupportsTrainingMode bool     `json:"supports_training_mode"`
		} `json:"bot_types"`
		Error string `json:"error"`
	}

	if err := json.Unmarshal(response, &botServerResponse); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(GetAvailableBotsResponse{
			Success: false,
			Error:   "Failed to parse bot server response",
		})
		return
	}

	if !botServerResponse.Success {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(GetAvailableBotsResponse{
			Success: false,
			Error:   botServerResponse.Error,
		})
		return
	}

	// Convert to our response format
	botTypes := make([]AvailableBotType, len(botServerResponse.BotTypes))
	for i, bt := range botServerResponse.BotTypes {
		botTypes[i] = AvailableBotType{
			Type:                 bt.Type,
			Name:                 bt.Name,
			Description:          bt.Description,
			Difficulties:         bt.Difficulties,
			AvailableGenerations: bt.AvailableGenerations,
			SupportsTrainingMode: bt.SupportsTrainingMode,
		}
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(GetAvailableBotsResponse{
		Success:  true,
		BotTypes: botTypes,
	})

	log.Printf("Returned %d available bot types from bot server", len(botTypes))
}

// HandleAddBotToRoom handles HTTP requests to add a bot to a room
func (s *Server) HandleAddBotToRoom(w http.ResponseWriter, r *http.Request) {
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
		json.NewEncoder(w).Encode(AddBotToRoomResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract room ID from URL path
	roomID := extractRoomIDFromPath(r.URL.Path, "/api/rooms/", "/bots")
	if roomID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(AddBotToRoomResponse{
			Success: false,
			Error:   "Invalid room ID in URL",
		})
		return
	}

	// Validate authorization
	if !s.validatePlayerToken(r, roomID) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(AddBotToRoomResponse{
			Success: false,
			Error:   "Unauthorized",
		})
		return
	}

	// Parse request body
	var req AddBotToRoomRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(AddBotToRoomResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Validate request
	if req.BotType == "" || req.Difficulty == "" || req.BotName == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(AddBotToRoomResponse{
			Success: false,
			Error:   "BotType, Difficulty, and BotName are required",
		})
		return
	}

	// Find room
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(AddBotToRoomResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Get room code for bot spawning
	roomCode := room.RoomCode
	roomPassword := room.Password

	// Spawn bot via Python bot server
	botID, err := s.spawnBotViaServer(roomCode, roomPassword, req)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(AddBotToRoomResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(AddBotToRoomResponse{
		Success: true,
		BotID:   botID,
		Message: "Bot added successfully",
	})

	log.Printf("Added bot %s (%s, %s) to room %s", botID, req.BotType, req.Difficulty, roomID)
}

// HandleRemoveBotFromRoom handles HTTP requests to remove a bot from a room
func (s *Server) HandleRemoveBotFromRoom(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow DELETE requests
	if r.Method != "DELETE" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(RemoveBotFromRoomResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract room ID and bot ID from URL path
	roomID := extractRoomIDFromPath(r.URL.Path, "/api/rooms/", "/bots/")
	botID := extractBotIDFromPath(r.URL.Path)

	if roomID == "" || botID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(RemoveBotFromRoomResponse{
			Success: false,
			Error:   "Invalid room ID or bot ID in URL",
		})
		return
	}

	// Validate authorization
	if !s.validatePlayerToken(r, roomID) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(RemoveBotFromRoomResponse{
			Success: false,
			Error:   "Unauthorized",
		})
		return
	}

	// Find room
	_, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(RemoveBotFromRoomResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Terminate bot via Python bot server
	err := s.terminateBotViaServer(botID)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(RemoveBotFromRoomResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(RemoveBotFromRoomResponse{
		Success: true,
		Message: "Bot removed successfully",
	})

	log.Printf("Removed bot %s from room %s", botID, roomID)
}

// HandleGetRoomBots handles HTTP requests to get bots in a room
func (s *Server) HandleGetRoomBots(w http.ResponseWriter, r *http.Request) {
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
		json.NewEncoder(w).Encode(GetRoomBotsResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract room ID from URL path
	roomID := extractRoomIDFromPath(r.URL.Path, "/api/rooms/", "/bots")
	if roomID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(GetRoomBotsResponse{
			Success: false,
			Error:   "Invalid room ID in URL",
		})
		return
	}

	// Validate authorization
	if !s.validatePlayerToken(r, roomID) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(GetRoomBotsResponse{
			Success: false,
			Error:   "Unauthorized",
		})
		return
	}

	// Find room
	_, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(GetRoomBotsResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Get bots from Python bot server
	bots, err := s.getRoomBotsViaServer(roomID)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(GetRoomBotsResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(GetRoomBotsResponse{
		Success: true,
		Bots:    bots,
	})

	log.Printf("Returned %d bots for room %s", len(bots), roomID)
}

// HandleConfigureBotDifficulty handles HTTP requests to configure bot difficulty
func (s *Server) HandleConfigureBotDifficulty(w http.ResponseWriter, r *http.Request) {
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "PUT, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Only allow PUT requests
	if r.Method != "PUT" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(ConfigureBotDifficultyResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Extract room ID and bot ID from URL path
	roomID := extractRoomIDFromPath(r.URL.Path, "/api/rooms/", "/bots/")
	botID := extractBotIDFromPath(r.URL.Path)

	if roomID == "" || botID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ConfigureBotDifficultyResponse{
			Success: false,
			Error:   "Invalid room ID or bot ID in URL",
		})
		return
	}

	// Validate authorization
	if !s.validatePlayerToken(r, roomID) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(ConfigureBotDifficultyResponse{
			Success: false,
			Error:   "Unauthorized",
		})
		return
	}

	// Parse request body
	var req ConfigureBotDifficultyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ConfigureBotDifficultyResponse{
			Success: false,
			Error:   "Invalid request format",
		})
		return
	}

	// Validate request
	if req.Difficulty == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ConfigureBotDifficultyResponse{
			Success: false,
			Error:   "Difficulty is required",
		})
		return
	}

	// Find room
	_, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(ConfigureBotDifficultyResponse{
			Success: false,
			Error:   "Room not found",
		})
		return
	}

	// Configure bot via Python bot server
	err := s.configureBotViaServer(botID, req.Difficulty)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ConfigureBotDifficultyResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(ConfigureBotDifficultyResponse{
		Success: true,
		Message: "Bot difficulty updated successfully",
	})

	log.Printf("Updated bot %s difficulty to %s in room %s", botID, req.Difficulty, roomID)
}

// HandleGetBotServerStatus handles HTTP requests to get bot server status
func (s *Server) HandleGetBotServerStatus(w http.ResponseWriter, r *http.Request) {
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
		json.NewEncoder(w).Encode(GetBotServerStatusResponse{
			Success: false,
			Error:   "Method not allowed",
		})
		return
	}

	// Get status from Python bot server
	status, err := s.getBotServerStatusViaServer()
	if err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(GetBotServerStatusResponse{
			Success:       false,
			ServerRunning: false,
			TotalBots:     0,
			ActiveRooms:   0,
			Error:         err.Error(),
		})
		return
	}

	// Return success response
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(status)

	log.Printf("Returned bot server status")
}

// Helper functions for bot management

// extractBotIDFromPath extracts bot ID from URL path like /api/rooms/{roomId}/bots/{botId}/difficulty
func extractBotIDFromPath(path string) string {
	parts := strings.Split(path, "/")
	for i, part := range parts {
		if part == "bots" && i+1 < len(parts) {
			botIDPart := parts[i+1]
			// Remove any suffix like "/difficulty"
			if idx := strings.Index(botIDPart, "/"); idx != -1 {
				return botIDPart[:idx]
			}
			return botIDPart
		}
	}
	return ""
}

// generateShortID generates a short unique ID for bots
func generateShortID() string {
	return time.Now().Format("20060102150405")
}

// Bot server integration functions

// spawnBotViaServer spawns a bot via the Python bot server
func (s *Server) spawnBotViaServer(roomCode, roomPassword string, req AddBotToRoomRequest) (string, error) {
	requestBody := map[string]interface{}{
		"bot_type":      req.BotType,
		"difficulty":    req.Difficulty,
		"bot_name":      req.BotName,
		"room_code":     roomCode,
		"room_password": roomPassword,
		"generation":    req.Generation,
		"training_mode": req.TrainingMode,
	}

	requestData, _ := json.Marshal(requestBody)
	response, err := s.makeBotServerRequest("POST", "/api/bots/spawn", requestData)
	if err != nil {
		return "", fmt.Errorf("failed to spawn bot: %v", err)
	}

	var spawnResponse struct {
		Success bool   `json:"success"`
		BotID   string `json:"bot_id"`
		Error   string `json:"error"`
	}

	if err := json.Unmarshal(response, &spawnResponse); err != nil {
		return "", fmt.Errorf("failed to parse spawn response: %v", err)
	}

	if !spawnResponse.Success {
		return "", fmt.Errorf("bot server error: %s", spawnResponse.Error)
	}

	return spawnResponse.BotID, nil
}

// terminateBotViaServer terminates a bot via the Python bot server
func (s *Server) terminateBotViaServer(botID string) error {
	_, err := s.makeBotServerRequest("POST", fmt.Sprintf("/api/bots/%s/terminate", botID), nil)
	if err != nil {
		return fmt.Errorf("failed to terminate bot: %v", err)
	}
	return nil
}

// getRoomBotsViaServer gets room bots via the Python bot server
func (s *Server) getRoomBotsViaServer(roomID string) ([]BotInfo, error) {
	response, err := s.makeBotServerRequest("GET", fmt.Sprintf("/api/rooms/%s/bots", roomID), nil)
	if err != nil {
		// Return empty list if bot server is unavailable
		log.Printf("Bot server unavailable for room bots: %v", err)
		return []BotInfo{}, nil
	}

	var botsResponse struct {
		Success bool `json:"success"`
		Bots    []struct {
			BotID       string                 `json:"bot_id"`
			BotType     string                 `json:"bot_type"`
			Difficulty  string                 `json:"difficulty"`
			Name        string                 `json:"name"`
			Generation  *int                   `json:"generation"`
			Status      string                 `json:"status"`
			CreatedAt   string                 `json:"created_at"`
			Performance map[string]interface{} `json:"performance"`
		} `json:"bots"`
		Error string `json:"error"`
	}

	if err := json.Unmarshal(response, &botsResponse); err != nil {
		return nil, fmt.Errorf("failed to parse bots response: %v", err)
	}

	if !botsResponse.Success {
		return nil, fmt.Errorf("bot server error: %s", botsResponse.Error)
	}

	// Convert to our format
	bots := make([]BotInfo, len(botsResponse.Bots))
	for i, bot := range botsResponse.Bots {
		bots[i] = BotInfo{
			BotID:       bot.BotID,
			BotType:     bot.BotType,
			Difficulty:  bot.Difficulty,
			Name:        bot.Name,
			Generation:  bot.Generation,
			Status:      bot.Status,
			CreatedAt:   bot.CreatedAt,
			Performance: bot.Performance,
		}
	}

	return bots, nil
}

// configureBotViaServer configures a bot via the Python bot server
func (s *Server) configureBotViaServer(botID, difficulty string) error {
	requestBody := map[string]string{
		"difficulty": difficulty,
	}

	requestData, _ := json.Marshal(requestBody)
	_, err := s.makeBotServerRequest("PUT", fmt.Sprintf("/api/bots/%s/configure", botID), requestData)
	if err != nil {
		return fmt.Errorf("failed to configure bot: %v", err)
	}

	return nil
}

// getBotServerStatusViaServer gets bot server status via the Python bot server
func (s *Server) getBotServerStatusViaServer() (*GetBotServerStatusResponse, error) {
	response, err := s.makeBotServerRequest("GET", "/api/bots/status", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get bot server status: %v", err)
	}

	var statusResponse struct {
		Success       bool                   `json:"success"`
		ServerRunning bool                   `json:"server_running"`
		TotalBots     int                    `json:"total_bots"`
		ActiveRooms   int                    `json:"active_rooms"`
		ServerStats   map[string]interface{} `json:"server_stats"`
		Error         string                 `json:"error"`
	}

	if err := json.Unmarshal(response, &statusResponse); err != nil {
		return nil, fmt.Errorf("failed to parse status response: %v", err)
	}

	if !statusResponse.Success {
		return nil, fmt.Errorf("bot server error: %s", statusResponse.Error)
	}

	return &GetBotServerStatusResponse{
		Success:       true,
		ServerRunning: statusResponse.ServerRunning,
		TotalBots:     statusResponse.TotalBots,
		ActiveRooms:   statusResponse.ActiveRooms,
		ServerStats:   statusResponse.ServerStats,
	}, nil
}

// makeBotServerRequest makes an HTTP request to the Python bot server
func (s *Server) makeBotServerRequest(method, endpoint string, body []byte) ([]byte, error) {
	client := &http.Client{
		Timeout: BOT_SERVER_TIMEOUT,
	}

	url := BOT_SERVER_URL + endpoint

	var req *http.Request
	var err error

	if body != nil {
		req, err = http.NewRequest(method, url, bytes.NewBuffer(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
	} else {
		req, err = http.NewRequest(method, url, nil)
		if err != nil {
			return nil, err
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("bot server returned status %d: %s", resp.StatusCode, string(responseBody))
	}

	return responseBody, nil
}

// HandleRoomBotRequests routes room bot requests to appropriate handlers
func (s *Server) HandleRoomBotRequests(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path

	// Route based on path pattern
	if strings.Contains(path, "/bots") && r.Method == "POST" {
		// POST /api/rooms/{roomId}/bots - Add bot to room
		s.HandleAddBotToRoom(w, r)
	} else if strings.Contains(path, "/bots") && r.Method == "GET" && !strings.Contains(path, "/bots/") {
		// GET /api/rooms/{roomId}/bots - Get room bots
		s.HandleGetRoomBots(w, r)
	} else if strings.Contains(path, "/bots/") && r.Method == "DELETE" {
		// DELETE /api/rooms/{roomId}/bots/{botId} - Remove bot from room
		s.HandleRemoveBotFromRoom(w, r)
	} else if strings.Contains(path, "/bots/") && strings.Contains(path, "/difficulty") && r.Method == "PUT" {
		// PUT /api/rooms/{roomId}/bots/{botId}/difficulty - Configure bot difficulty
		s.HandleConfigureBotDifficulty(w, r)
	} else {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   "Endpoint not found",
		})
	}
}
