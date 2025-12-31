package server

import (
	"bytes"
	"encoding/json"
	"go-ws-server/pkg/server/types"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandleCreateGame_TrainingModeValidation(t *testing.T) {
	server := NewServer()

	tests := []struct {
		name           string
		request        CreateGameHTTPRequest
		wantStatusCode int
		wantSuccess    bool
		wantError      string
	}{
		{
			name: "Valid training mode request",
			request: CreateGameHTTPRequest{
				PlayerName:          "TestPlayer",
				RoomName:            "TestRoom",
				MapType:             "default",
				TrainingMode:        true,
				TickMultiplier:      10.0,
				MaxGameDurationSec:  60,
				DisableRespawnTimer: true,
				MaxKills:            20,
			},
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
			wantError:      "",
		},
		{
			name: "Training mode with minimum tick multiplier",
			request: CreateGameHTTPRequest{
				PlayerName:     "TestPlayer",
				RoomName:       "TestRoom",
				MapType:        "default",
				TrainingMode:   true,
				TickMultiplier: 1.0,
			},
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
			wantError:      "",
		},
		{
			name: "Training mode with maximum tick multiplier",
			request: CreateGameHTTPRequest{
				PlayerName:     "TestPlayer",
				RoomName:       "TestRoom",
				MapType:        "default",
				TrainingMode:   true,
				TickMultiplier: 20.0, // Max is 20x due to MinTickInterval of 1ms
			},
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
			wantError:      "",
		},
		{
			name: "Tick multiplier below minimum",
			request: CreateGameHTTPRequest{
				PlayerName:     "TestPlayer",
				RoomName:       "TestRoom",
				MapType:        "default",
				TrainingMode:   true,
				TickMultiplier: 0.5,
			},
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "tickMultiplier must be between 1.0 and 20.0",
		},
		{
			name: "Tick multiplier above maximum",
			request: CreateGameHTTPRequest{
				PlayerName:     "TestPlayer",
				RoomName:       "TestRoom",
				MapType:        "default",
				TrainingMode:   true,
				TickMultiplier: 21.0,
			},
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "tickMultiplier must be between 1.0 and 20.0",
		},
		{
			name: "Negative max game duration",
			request: CreateGameHTTPRequest{
				PlayerName:         "TestPlayer",
				RoomName:           "TestRoom",
				MapType:            "default",
				TrainingMode:       true,
				MaxGameDurationSec: -1,
			},
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "maxGameDurationSec must be between 0 and 3600",
		},
		{
			name: "Max game duration above limit",
			request: CreateGameHTTPRequest{
				PlayerName:         "TestPlayer",
				RoomName:           "TestRoom",
				MapType:            "default",
				TrainingMode:       true,
				MaxGameDurationSec: 3601,
			},
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "maxGameDurationSec must be between 0 and 3600",
		},
		{
			name: "Negative max kills",
			request: CreateGameHTTPRequest{
				PlayerName:   "TestPlayer",
				RoomName:     "TestRoom",
				MapType:      "default",
				TrainingMode: true,
				MaxKills:     -1,
			},
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "maxKills must be non-negative",
		},
		{
			name: "Training options ignored when training mode is false",
			request: CreateGameHTTPRequest{
				PlayerName:     "TestPlayer",
				RoomName:       "TestRoom",
				MapType:        "default",
				TrainingMode:   false,
				TickMultiplier: 200.0, // Invalid but should be ignored
				MaxKills:       -5,    // Invalid but should be ignored
			},
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
			wantError:      "",
		},
		{
			name: "Valid training mode with zero tick multiplier (uses default)",
			request: CreateGameHTTPRequest{
				PlayerName:     "TestPlayer",
				RoomName:       "TestRoom",
				MapType:        "default",
				TrainingMode:   true,
				TickMultiplier: 0, // Should be allowed, uses default
			},
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
			wantError:      "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create request body
			body, err := json.Marshal(tt.request)
			if err != nil {
				t.Fatalf("Failed to marshal request: %v", err)
			}

			// Create HTTP request
			req := httptest.NewRequest(http.MethodPost, "/api/createGame", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			// Create response recorder
			rr := httptest.NewRecorder()

			// Call handler
			server.HandleCreateGame(rr, req)

			// Check status code
			if rr.Code != tt.wantStatusCode {
				t.Errorf("HandleCreateGame() status = %v, want %v", rr.Code, tt.wantStatusCode)
			}

			// Parse response
			var response CreateGameHTTPResponse
			if err := json.NewDecoder(rr.Body).Decode(&response); err != nil {
				t.Fatalf("Failed to decode response: %v", err)
			}

			// Check success field
			if response.Success != tt.wantSuccess {
				t.Errorf("HandleCreateGame() success = %v, want %v", response.Success, tt.wantSuccess)
			}

			// Check error message
			if tt.wantError != "" && response.Error != tt.wantError {
				t.Errorf("HandleCreateGame() error = %q, want %q", response.Error, tt.wantError)
			}

			// For successful training mode requests, verify response includes training settings
			if tt.wantSuccess && tt.request.TrainingMode {
				if !response.TrainingMode {
					t.Error("Response should have TrainingMode=true")
				}
				if tt.request.TickMultiplier > 0 && response.TickMultiplier != tt.request.TickMultiplier {
					t.Errorf("Response TickMultiplier = %v, want %v", response.TickMultiplier, tt.request.TickMultiplier)
				}
				if response.MaxGameDurationSec != tt.request.MaxGameDurationSec {
					t.Errorf("Response MaxGameDurationSec = %v, want %v", response.MaxGameDurationSec, tt.request.MaxGameDurationSec)
				}
				if response.DisableRespawnTimer != tt.request.DisableRespawnTimer {
					t.Errorf("Response DisableRespawnTimer = %v, want %v", response.DisableRespawnTimer, tt.request.DisableRespawnTimer)
				}
				if response.MaxKills != tt.request.MaxKills {
					t.Errorf("Response MaxKills = %v, want %v", response.MaxKills, tt.request.MaxKills)
				}
			}

			// For successful non-training mode requests, verify response does not include training settings
			if tt.wantSuccess && !tt.request.TrainingMode {
				if response.TrainingMode {
					t.Error("Response should have TrainingMode=false for non-training requests")
				}
			}
		})
	}
}

func TestHandleCreateGame_TrainingModeResponse(t *testing.T) {
	server := NewServer()

	// Create a training mode game and verify all fields are returned correctly
	request := CreateGameHTTPRequest{
		PlayerName:          "TestPlayer",
		RoomName:            "TestRoom",
		MapType:             "default",
		TrainingMode:        true,
		TickMultiplier:      5.0,
		MaxGameDurationSec:  120,
		DisableRespawnTimer: true,
		MaxKills:            50,
	}

	body, _ := json.Marshal(request)
	req := httptest.NewRequest(http.MethodPost, "/api/createGame", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	server.HandleCreateGame(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d", rr.Code)
	}

	var response CreateGameHTTPResponse
	json.NewDecoder(rr.Body).Decode(&response)

	// Verify all standard fields are present
	if response.RoomID == "" {
		t.Error("RoomID should not be empty")
	}
	if response.RoomCode == "" {
		t.Error("RoomCode should not be empty")
	}
	if response.PlayerID == "" {
		t.Error("PlayerID should not be empty")
	}
	if response.PlayerToken == "" {
		t.Error("PlayerToken should not be empty")
	}

	// Verify training settings are returned
	if !response.TrainingMode {
		t.Error("TrainingMode should be true")
	}
	if response.TickMultiplier != 5.0 {
		t.Errorf("TickMultiplier = %v, want 5.0", response.TickMultiplier)
	}
	if response.MaxGameDurationSec != 120 {
		t.Errorf("MaxGameDurationSec = %v, want 120", response.MaxGameDurationSec)
	}
	if !response.DisableRespawnTimer {
		t.Error("DisableRespawnTimer should be true")
	}
	if response.MaxKills != 50 {
		t.Errorf("MaxKills = %v, want 50", response.MaxKills)
	}
}

func TestHandleGetRoomState(t *testing.T) {
	server := NewServer()

	// First create a game to get a valid room and player token
	createReq := CreateGameHTTPRequest{
		PlayerName: "TestPlayer",
		RoomName:   "TestRoom",
		MapType:    "default",
	}
	createBody, _ := json.Marshal(createReq)
	createHttpReq := httptest.NewRequest(http.MethodPost, "/api/createGame", bytes.NewReader(createBody))
	createHttpReq.Header.Set("Content-Type", "application/json")
	createRR := httptest.NewRecorder()
	server.HandleCreateGame(createRR, createHttpReq)

	var createResp CreateGameHTTPResponse
	json.NewDecoder(createRR.Body).Decode(&createResp)
	if !createResp.Success {
		t.Fatalf("Failed to create test game: %s", createResp.Error)
	}

	roomID := createResp.RoomID
	playerToken := createResp.PlayerToken

	tests := []struct {
		name           string
		url            string
		token          string
		tokenLocation  string // "query", "header", "bearer"
		wantStatusCode int
		wantSuccess    bool
		wantError      string
	}{
		{
			name:           "Valid request with query param token",
			url:            "/api/rooms/" + roomID + "/state?playerToken=" + playerToken,
			tokenLocation:  "query",
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
		},
		{
			name:           "Valid request with X-Player-Token header",
			url:            "/api/rooms/" + roomID + "/state",
			token:          playerToken,
			tokenLocation:  "header",
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
		},
		{
			name:           "Valid request with Bearer token",
			url:            "/api/rooms/" + roomID + "/state",
			token:          playerToken,
			tokenLocation:  "bearer",
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
		},
		{
			name:           "Missing player token",
			url:            "/api/rooms/" + roomID + "/state",
			tokenLocation:  "",
			wantStatusCode: http.StatusUnauthorized,
			wantSuccess:    false,
			wantError:      "Player token is required",
		},
		{
			name:           "Invalid player token",
			url:            "/api/rooms/" + roomID + "/state",
			token:          "invalid-token",
			tokenLocation:  "header",
			wantStatusCode: http.StatusForbidden,
			wantSuccess:    false,
			wantError:      "Player token is not authorized for this room",
		},
		{
			name:           "Room not found",
			url:            "/api/rooms/nonexistent-room-id/state",
			token:          playerToken,
			tokenLocation:  "header",
			wantStatusCode: http.StatusNotFound,
			wantSuccess:    false,
			wantError:      "Room not found",
		},
		{
			name:           "Invalid URL format - missing /state suffix",
			url:            "/api/rooms/" + roomID,
			token:          playerToken,
			tokenLocation:  "header",
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "Invalid URL format",
		},
		{
			name:           "Empty room ID",
			url:            "/api/rooms//state",
			token:          playerToken,
			tokenLocation:  "header",
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "Room ID is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, tt.url, nil)

			// Set token based on location
			switch tt.tokenLocation {
			case "header":
				req.Header.Set("X-Player-Token", tt.token)
			case "bearer":
				req.Header.Set("Authorization", "Bearer "+tt.token)
			// "query" case is already in URL
			}

			rr := httptest.NewRecorder()
			server.HandleGetRoomState(rr, req)

			if rr.Code != tt.wantStatusCode {
				t.Errorf("HandleGetRoomState() status = %v, want %v", rr.Code, tt.wantStatusCode)
			}

			var response GetRoomStateResponse
			if err := json.NewDecoder(rr.Body).Decode(&response); err != nil {
				t.Fatalf("Failed to decode response: %v", err)
			}

			if response.Success != tt.wantSuccess {
				t.Errorf("HandleGetRoomState() success = %v, want %v", response.Success, tt.wantSuccess)
			}

			if tt.wantError != "" && response.Error == "" {
				t.Errorf("HandleGetRoomState() expected error containing %q, got empty", tt.wantError)
			} else if tt.wantError != "" && !strings.Contains(response.Error, tt.wantError) {
				t.Errorf("HandleGetRoomState() error = %q, want to contain %q", response.Error, tt.wantError)
			}

			// For successful requests, verify response structure
			if tt.wantSuccess {
				if response.RoomID != roomID {
					t.Errorf("HandleGetRoomState() roomId = %v, want %v", response.RoomID, roomID)
				}
				if response.Timestamp == "" {
					t.Error("HandleGetRoomState() timestamp should not be empty")
				}
				if response.ObjectStates == nil {
					t.Error("HandleGetRoomState() objectStates should not be nil")
				}
			}
		})
	}
}

func TestHandleGetRoomState_MethodNotAllowed(t *testing.T) {
	server := NewServer()

	// Test that non-GET methods are rejected (except OPTIONS for CORS)
	methods := []string{http.MethodPost, http.MethodPut, http.MethodDelete, http.MethodPatch}

	for _, method := range methods {
		t.Run(method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/api/rooms/test-room/state", nil)
			rr := httptest.NewRecorder()

			server.HandleGetRoomState(rr, req)

			if rr.Code != http.StatusMethodNotAllowed {
				t.Errorf("HandleGetRoomState() with %s status = %v, want %v", method, rr.Code, http.StatusMethodNotAllowed)
			}
		})
	}
}

func TestHandleGetRoomState_CORSPreflight(t *testing.T) {
	server := NewServer()

	req := httptest.NewRequest(http.MethodOptions, "/api/rooms/test-room/state", nil)
	rr := httptest.NewRecorder()

	server.HandleGetRoomState(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("HandleGetRoomState() OPTIONS status = %v, want %v", rr.Code, http.StatusOK)
	}

	// Verify CORS headers
	if rr.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("Missing or incorrect Access-Control-Allow-Origin header")
	}
	if rr.Header().Get("Access-Control-Allow-Methods") != "GET, OPTIONS" {
		t.Error("Missing or incorrect Access-Control-Allow-Methods header")
	}
}

func TestHandleGetRoomState_ResponseContainsPlayerState(t *testing.T) {
	server := NewServer()

	// Create a game
	createReq := CreateGameHTTPRequest{
		PlayerName: "TestPlayer",
		RoomName:   "TestRoom",
		MapType:    "default",
	}
	createBody, _ := json.Marshal(createReq)
	createHttpReq := httptest.NewRequest(http.MethodPost, "/api/createGame", bytes.NewReader(createBody))
	createHttpReq.Header.Set("Content-Type", "application/json")
	createRR := httptest.NewRecorder()
	server.HandleCreateGame(createRR, createHttpReq)

	var createResp CreateGameHTTPResponse
	json.NewDecoder(createRR.Body).Decode(&createResp)

	// Get room state
	req := httptest.NewRequest(http.MethodGet, "/api/rooms/"+createResp.RoomID+"/state", nil)
	req.Header.Set("X-Player-Token", createResp.PlayerToken)
	rr := httptest.NewRecorder()

	server.HandleGetRoomState(rr, req)

	var response GetRoomStateResponse
	json.NewDecoder(rr.Body).Decode(&response)

	// Verify player object exists in the state
	playerState, exists := response.ObjectStates[createResp.PlayerID]
	if !exists {
		t.Fatalf("Player state not found in objectStates")
	}

	// Verify player state has expected fields (using short names from constants)
	// id, name, x, y are the key fields we expect for any player object
	expectedFields := []string{"id", "name", "x", "y"}
	for _, field := range expectedFields {
		if _, ok := playerState[field]; !ok {
			t.Errorf("Player state missing expected field: %s", field)
		}
	}

	// Verify player name matches what we created
	if playerName, ok := playerState["name"].(string); !ok || playerName != "TestPlayer" {
		t.Errorf("Player name = %v, want 'TestPlayer'", playerState["name"])
	}
}

func TestExtractBotActionPathParams(t *testing.T) {
	tests := []struct {
		name         string
		path         string
		wantRoomID   string
		wantPlayerID string
		wantOk       bool
	}{
		{
			name:         "Valid path",
			path:         "/api/rooms/room123/players/player456/action",
			wantRoomID:   "room123",
			wantPlayerID: "player456",
			wantOk:       true,
		},
		{
			name:         "Valid path with UUIDs",
			path:         "/api/rooms/550e8400-e29b-41d4-a716-446655440000/players/6ba7b810-9dad-11d1-80b4-00c04fd430c8/action",
			wantRoomID:   "550e8400-e29b-41d4-a716-446655440000",
			wantPlayerID: "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
			wantOk:       true,
		},
		{
			name:   "Invalid path - missing action",
			path:   "/api/rooms/room123/players/player456",
			wantOk: false,
		},
		{
			name:   "Invalid path - wrong endpoint",
			path:   "/api/rooms/room123/players/player456/state",
			wantOk: false,
		},
		{
			name:   "Invalid path - missing players segment",
			path:   "/api/rooms/room123/action",
			wantOk: false,
		},
		{
			name:   "Invalid path - wrong prefix",
			path:   "/rooms/room123/players/player456/action",
			wantOk: false,
		},
		{
			name:   "Invalid path - extra segments",
			path:   "/api/rooms/room123/players/player456/action/extra",
			wantOk: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			roomID, playerID, ok := extractBotActionPathParams(tt.path)
			if ok != tt.wantOk {
				t.Errorf("extractBotActionPathParams() ok = %v, want %v", ok, tt.wantOk)
			}
			if ok && roomID != tt.wantRoomID {
				t.Errorf("extractBotActionPathParams() roomID = %v, want %v", roomID, tt.wantRoomID)
			}
			if ok && playerID != tt.wantPlayerID {
				t.Errorf("extractBotActionPathParams() playerID = %v, want %v", playerID, tt.wantPlayerID)
			}
		})
	}
}

func TestHandleBotAction(t *testing.T) {
	server := NewServer()

	// First create a game to get a valid room and player token
	createReq := CreateGameHTTPRequest{
		PlayerName: "TestBot",
		RoomName:   "TestRoom",
		MapType:    "default",
	}
	createBody, _ := json.Marshal(createReq)
	createHttpReq := httptest.NewRequest(http.MethodPost, "/api/createGame", bytes.NewReader(createBody))
	createHttpReq.Header.Set("Content-Type", "application/json")
	createRR := httptest.NewRecorder()
	server.HandleCreateGame(createRR, createHttpReq)

	var createResp CreateGameHTTPResponse
	json.NewDecoder(createRR.Body).Decode(&createResp)
	if !createResp.Success {
		t.Fatalf("Failed to create test game: %s", createResp.Error)
	}

	roomID := createResp.RoomID
	playerID := createResp.PlayerID
	playerToken := createResp.PlayerToken

	tests := []struct {
		name           string
		url            string
		token          string
		request        types.BotActionRequest
		wantStatusCode int
		wantSuccess    bool
		wantError      string
		wantProcessed  int
	}{
		{
			name:  "Valid key action",
			url:   "/api/rooms/" + roomID + "/players/" + playerID + "/action",
			token: playerToken,
			request: types.BotActionRequest{
				Actions: []types.BotAction{
					{Type: "key", Key: "d", IsDown: true},
				},
			},
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
			wantProcessed:  1,
		},
		{
			name:  "Valid click action",
			url:   "/api/rooms/" + roomID + "/players/" + playerID + "/action",
			token: playerToken,
			request: types.BotActionRequest{
				Actions: []types.BotAction{
					{Type: "click", X: 150.5, Y: 200.0, Button: 0, IsDown: true},
				},
			},
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
			wantProcessed:  1,
		},
		{
			name:  "Valid direction action",
			url:   "/api/rooms/" + roomID + "/players/" + playerID + "/action",
			token: playerToken,
			request: types.BotActionRequest{
				Actions: []types.BotAction{
					{Type: "direction", Direction: 1.57},
				},
			},
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
			wantProcessed:  1,
		},
		{
			name:  "Multiple actions",
			url:   "/api/rooms/" + roomID + "/players/" + playerID + "/action",
			token: playerToken,
			request: types.BotActionRequest{
				Actions: []types.BotAction{
					{Type: "key", Key: "w", IsDown: true},
					{Type: "click", X: 100.0, Y: 100.0, Button: 0, IsDown: true},
					{Type: "direction", Direction: 3.14},
				},
			},
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
			wantProcessed:  3,
		},
		{
			name:  "Empty actions array",
			url:   "/api/rooms/" + roomID + "/players/" + playerID + "/action",
			token: playerToken,
			request: types.BotActionRequest{
				Actions: []types.BotAction{},
			},
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
			wantProcessed:  0,
		},
		{
			name:           "Missing authorization header",
			url:            "/api/rooms/" + roomID + "/players/" + playerID + "/action",
			token:          "",
			request:        types.BotActionRequest{Actions: []types.BotAction{}},
			wantStatusCode: http.StatusUnauthorized,
			wantSuccess:    false,
			wantError:      "Authorization header with Bearer token is required",
		},
		{
			name:           "Invalid player token",
			url:            "/api/rooms/" + roomID + "/players/" + playerID + "/action",
			token:          "invalid-token",
			request:        types.BotActionRequest{Actions: []types.BotAction{}},
			wantStatusCode: http.StatusUnauthorized,
			wantSuccess:    false,
			wantError:      "Invalid player token",
		},
		{
			name:           "Room not found",
			url:            "/api/rooms/nonexistent-room/players/" + playerID + "/action",
			token:          playerToken,
			request:        types.BotActionRequest{Actions: []types.BotAction{}},
			wantStatusCode: http.StatusNotFound,
			wantSuccess:    false,
			wantError:      "Room not found",
		},
		{
			name:           "Player not found",
			url:            "/api/rooms/" + roomID + "/players/nonexistent-player/action",
			token:          playerToken,
			request:        types.BotActionRequest{Actions: []types.BotAction{}},
			wantStatusCode: http.StatusNotFound,
			wantSuccess:    false,
			wantError:      "Player not found",
		},
		{
			name:  "Invalid action type",
			url:   "/api/rooms/" + roomID + "/players/" + playerID + "/action",
			token: playerToken,
			request: types.BotActionRequest{
				Actions: []types.BotAction{
					{Type: "invalid"},
				},
			},
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "Invalid action type: invalid",
		},
		{
			name:  "Invalid key value",
			url:   "/api/rooms/" + roomID + "/players/" + playerID + "/action",
			token: playerToken,
			request: types.BotActionRequest{
				Actions: []types.BotAction{
					{Type: "key", Key: "x", IsDown: true},
				},
			},
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "Invalid key value",
		},
		{
			name:           "Invalid URL format",
			url:            "/api/rooms/" + roomID + "/players/" + playerID,
			token:          playerToken,
			request:        types.BotActionRequest{Actions: []types.BotAction{}},
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "Invalid URL format",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, _ := json.Marshal(tt.request)
			req := httptest.NewRequest(http.MethodPost, tt.url, bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			if tt.token != "" {
				req.Header.Set("Authorization", "Bearer "+tt.token)
			}

			rr := httptest.NewRecorder()
			server.HandleBotAction(rr, req)

			if rr.Code != tt.wantStatusCode {
				t.Errorf("HandleBotAction() status = %v, want %v", rr.Code, tt.wantStatusCode)
			}

			var response types.BotActionResponse
			if err := json.NewDecoder(rr.Body).Decode(&response); err != nil {
				t.Fatalf("Failed to decode response: %v", err)
			}

			if response.Success != tt.wantSuccess {
				t.Errorf("HandleBotAction() success = %v, want %v", response.Success, tt.wantSuccess)
			}

			if tt.wantError != "" && !strings.Contains(response.Error, tt.wantError) {
				t.Errorf("HandleBotAction() error = %q, want to contain %q", response.Error, tt.wantError)
			}

			if tt.wantSuccess && response.ActionsProcessed != tt.wantProcessed {
				t.Errorf("HandleBotAction() actionsProcessed = %v, want %v", response.ActionsProcessed, tt.wantProcessed)
			}

			if tt.wantSuccess && response.Timestamp == 0 {
				t.Error("HandleBotAction() timestamp should not be zero for successful requests")
			}
		})
	}
}

func TestHandleBotAction_MethodNotAllowed(t *testing.T) {
	server := NewServer()

	// Test that non-POST methods are rejected (except OPTIONS for CORS)
	methods := []string{http.MethodGet, http.MethodPut, http.MethodDelete, http.MethodPatch}

	for _, method := range methods {
		t.Run(method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/api/rooms/test-room/players/test-player/action", nil)
			rr := httptest.NewRecorder()

			server.HandleBotAction(rr, req)

			if rr.Code != http.StatusMethodNotAllowed {
				t.Errorf("HandleBotAction() with %s status = %v, want %v", method, rr.Code, http.StatusMethodNotAllowed)
			}
		})
	}
}

func TestHandleBotAction_CORSPreflight(t *testing.T) {
	server := NewServer()

	req := httptest.NewRequest(http.MethodOptions, "/api/rooms/test-room/players/test-player/action", nil)
	rr := httptest.NewRecorder()

	server.HandleBotAction(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("HandleBotAction() OPTIONS status = %v, want %v", rr.Code, http.StatusOK)
	}

	// Verify CORS headers
	if rr.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("Missing or incorrect Access-Control-Allow-Origin header")
	}
	if rr.Header().Get("Access-Control-Allow-Methods") != "POST, OPTIONS" {
		t.Error("Missing or incorrect Access-Control-Allow-Methods header")
	}
}

func TestHandleBotAction_KeyCaseInsensitive(t *testing.T) {
	server := NewServer()

	// Create a game
	createReq := CreateGameHTTPRequest{
		PlayerName: "TestBot",
		RoomName:   "TestRoom",
		MapType:    "default",
	}
	createBody, _ := json.Marshal(createReq)
	createHttpReq := httptest.NewRequest(http.MethodPost, "/api/createGame", bytes.NewReader(createBody))
	createHttpReq.Header.Set("Content-Type", "application/json")
	createRR := httptest.NewRecorder()
	server.HandleCreateGame(createRR, createHttpReq)

	var createResp CreateGameHTTPResponse
	json.NewDecoder(createRR.Body).Decode(&createResp)

	roomID := createResp.RoomID
	playerID := createResp.PlayerID
	playerToken := createResp.PlayerToken

	// Test lowercase keys work
	lowercaseKeys := []string{"w", "a", "s", "d"}
	for _, key := range lowercaseKeys {
		t.Run("lowercase_"+key, func(t *testing.T) {
			request := types.BotActionRequest{
				Actions: []types.BotAction{
					{Type: "key", Key: key, IsDown: true},
				},
			}
			body, _ := json.Marshal(request)
			req := httptest.NewRequest(http.MethodPost, "/api/rooms/"+roomID+"/players/"+playerID+"/action", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer "+playerToken)

			rr := httptest.NewRecorder()
			server.HandleBotAction(rr, req)

			if rr.Code != http.StatusOK {
				t.Errorf("HandleBotAction() with key %q status = %v, want %v", key, rr.Code, http.StatusOK)
			}
		})
	}

	// Test uppercase keys also work
	uppercaseKeys := []string{"W", "A", "S", "D"}
	for _, key := range uppercaseKeys {
		t.Run("uppercase_"+key, func(t *testing.T) {
			request := types.BotActionRequest{
				Actions: []types.BotAction{
					{Type: "key", Key: key, IsDown: true},
				},
			}
			body, _ := json.Marshal(request)
			req := httptest.NewRequest(http.MethodPost, "/api/rooms/"+roomID+"/players/"+playerID+"/action", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer "+playerToken)

			rr := httptest.NewRecorder()
			server.HandleBotAction(rr, req)

			if rr.Code != http.StatusOK {
				t.Errorf("HandleBotAction() with key %q status = %v, want %v", key, rr.Code, http.StatusOK)
			}
		})
	}
}

func TestHandleResetGame(t *testing.T) {
	server := NewServer()

	// First create a game to get a valid room and player token
	createReq := CreateGameHTTPRequest{
		PlayerName:   "TestPlayer",
		RoomName:     "TestRoom",
		MapType:      "default",
		TrainingMode: true,
	}
	createBody, _ := json.Marshal(createReq)
	createHttpReq := httptest.NewRequest(http.MethodPost, "/api/createGame", bytes.NewReader(createBody))
	createHttpReq.Header.Set("Content-Type", "application/json")
	createRR := httptest.NewRecorder()
	server.HandleCreateGame(createRR, createHttpReq)

	var createResp CreateGameHTTPResponse
	json.NewDecoder(createRR.Body).Decode(&createResp)
	if !createResp.Success {
		t.Fatalf("Failed to create test game: %s", createResp.Error)
	}

	roomID := createResp.RoomID
	playerToken := createResp.PlayerToken

	tests := []struct {
		name           string
		url            string
		token          string
		tokenLocation  string // "header", "bearer"
		body           *ResetGameHTTPRequest
		wantStatusCode int
		wantSuccess    bool
		wantError      string
	}{
		{
			name:           "Valid reset with X-Player-Token header",
			url:            "/api/rooms/" + roomID + "/reset",
			token:          playerToken,
			tokenLocation:  "header",
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
		},
		{
			name:           "Valid reset with Bearer token",
			url:            "/api/rooms/" + roomID + "/reset",
			token:          playerToken,
			tokenLocation:  "bearer",
			wantStatusCode: http.StatusOK,
			wantSuccess:    true,
		},
		{
			name:           "Missing player token",
			url:            "/api/rooms/" + roomID + "/reset",
			tokenLocation:  "",
			wantStatusCode: http.StatusUnauthorized,
			wantSuccess:    false,
			wantError:      "Player token is required",
		},
		{
			name:           "Invalid player token",
			url:            "/api/rooms/" + roomID + "/reset",
			token:          "invalid-token",
			tokenLocation:  "header",
			wantStatusCode: http.StatusForbidden,
			wantSuccess:    false,
			wantError:      "Player token is not authorized for this room",
		},
		{
			name:           "Room not found",
			url:            "/api/rooms/nonexistent-room-id/reset",
			token:          playerToken,
			tokenLocation:  "header",
			wantStatusCode: http.StatusNotFound,
			wantSuccess:    false,
			wantError:      "Room not found",
		},
		{
			name:           "Invalid URL format - missing /reset suffix",
			url:            "/api/rooms/" + roomID,
			token:          playerToken,
			tokenLocation:  "header",
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "Invalid URL format",
		},
		{
			name:           "Empty room ID",
			url:            "/api/rooms//reset",
			token:          playerToken,
			tokenLocation:  "header",
			wantStatusCode: http.StatusBadRequest,
			wantSuccess:    false,
			wantError:      "Room ID is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var bodyReader *bytes.Reader
			if tt.body != nil {
				bodyBytes, _ := json.Marshal(tt.body)
				bodyReader = bytes.NewReader(bodyBytes)
			} else {
				bodyReader = bytes.NewReader(nil)
			}

			req := httptest.NewRequest(http.MethodPost, tt.url, bodyReader)
			req.Header.Set("Content-Type", "application/json")

			// Set token based on location
			switch tt.tokenLocation {
			case "header":
				req.Header.Set("X-Player-Token", tt.token)
			case "bearer":
				req.Header.Set("Authorization", "Bearer "+tt.token)
			}

			rr := httptest.NewRecorder()
			server.HandleResetGame(rr, req)

			if rr.Code != tt.wantStatusCode {
				t.Errorf("HandleResetGame() status = %v, want %v", rr.Code, tt.wantStatusCode)
			}

			var response ResetGameHTTPResponse
			if err := json.NewDecoder(rr.Body).Decode(&response); err != nil {
				t.Fatalf("Failed to decode response: %v", err)
			}

			if response.Success != tt.wantSuccess {
				t.Errorf("HandleResetGame() success = %v, want %v", response.Success, tt.wantSuccess)
			}

			if tt.wantError != "" && response.Error == "" {
				t.Errorf("HandleResetGame() expected error containing %q, got empty", tt.wantError)
			} else if tt.wantError != "" && !strings.Contains(response.Error, tt.wantError) {
				t.Errorf("HandleResetGame() error = %q, want to contain %q", response.Error, tt.wantError)
			}
		})
	}
}

func TestHandleResetGame_MethodNotAllowed(t *testing.T) {
	server := NewServer()

	// Test that non-POST methods are rejected (except OPTIONS for CORS)
	methods := []string{http.MethodGet, http.MethodPut, http.MethodDelete, http.MethodPatch}

	for _, method := range methods {
		t.Run(method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/api/rooms/test-room/reset", nil)
			rr := httptest.NewRecorder()

			server.HandleResetGame(rr, req)

			if rr.Code != http.StatusMethodNotAllowed {
				t.Errorf("HandleResetGame() with %s status = %v, want %v", method, rr.Code, http.StatusMethodNotAllowed)
			}
		})
	}
}

func TestHandleResetGame_CORSPreflight(t *testing.T) {
	server := NewServer()

	req := httptest.NewRequest(http.MethodOptions, "/api/rooms/test-room/reset", nil)
	rr := httptest.NewRecorder()

	server.HandleResetGame(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("HandleResetGame() OPTIONS status = %v, want %v", rr.Code, http.StatusOK)
	}

	// Verify CORS headers
	if rr.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("Missing or incorrect Access-Control-Allow-Origin header")
	}
	if rr.Header().Get("Access-Control-Allow-Methods") != "POST, OPTIONS" {
		t.Error("Missing or incorrect Access-Control-Allow-Methods header")
	}
}

func TestHandleResetGame_ResetsPlayerState(t *testing.T) {
	server := NewServer()

	// Create a game
	createReq := CreateGameHTTPRequest{
		PlayerName:   "TestPlayer",
		RoomName:     "TestRoom",
		MapType:      "default",
		TrainingMode: true,
	}
	createBody, _ := json.Marshal(createReq)
	createHttpReq := httptest.NewRequest(http.MethodPost, "/api/createGame", bytes.NewReader(createBody))
	createHttpReq.Header.Set("Content-Type", "application/json")
	createRR := httptest.NewRecorder()
	server.HandleCreateGame(createRR, createHttpReq)

	var createResp CreateGameHTTPResponse
	json.NewDecoder(createRR.Body).Decode(&createResp)

	roomID := createResp.RoomID
	playerID := createResp.PlayerID
	playerToken := createResp.PlayerToken

	// Get initial state
	getStateReq := httptest.NewRequest(http.MethodGet, "/api/rooms/"+roomID+"/state", nil)
	getStateReq.Header.Set("X-Player-Token", playerToken)
	getStateRR := httptest.NewRecorder()
	server.HandleGetRoomState(getStateRR, getStateReq)

	var initialState GetRoomStateResponse
	json.NewDecoder(getStateRR.Body).Decode(&initialState)
	initialPlayerState := initialState.ObjectStates[playerID]
	initialX := initialPlayerState["x"].(float64)
	initialY := initialPlayerState["y"].(float64)

	// Modify player state by sending key input
	actionReq := types.BotActionRequest{
		Actions: []types.BotAction{
			{Type: "key", Key: "d", IsDown: true}, // Move right
		},
	}
	actionBody, _ := json.Marshal(actionReq)
	botActionReq := httptest.NewRequest(http.MethodPost, "/api/rooms/"+roomID+"/players/"+playerID+"/action", bytes.NewReader(actionBody))
	botActionReq.Header.Set("Content-Type", "application/json")
	botActionReq.Header.Set("Authorization", "Bearer "+playerToken)
	server.HandleBotAction(httptest.NewRecorder(), botActionReq)

	// Reset the game
	resetReq := httptest.NewRequest(http.MethodPost, "/api/rooms/"+roomID+"/reset", nil)
	resetReq.Header.Set("X-Player-Token", playerToken)
	resetRR := httptest.NewRecorder()
	server.HandleResetGame(resetRR, resetReq)

	if resetRR.Code != http.StatusOK {
		t.Fatalf("HandleResetGame() failed with status %v", resetRR.Code)
	}

	// Get state after reset
	getStateReq2 := httptest.NewRequest(http.MethodGet, "/api/rooms/"+roomID+"/state", nil)
	getStateReq2.Header.Set("X-Player-Token", playerToken)
	getStateRR2 := httptest.NewRecorder()
	server.HandleGetRoomState(getStateRR2, getStateReq2)

	var resetState GetRoomStateResponse
	json.NewDecoder(getStateRR2.Body).Decode(&resetState)
	resetPlayerState := resetState.ObjectStates[playerID]

	// Verify player is still in the game
	if resetPlayerState == nil {
		t.Fatal("Player should still exist after reset")
	}

	// Verify dx and dy are reset to 0
	if dx, ok := resetPlayerState["dx"].(float64); !ok || dx != 0.0 {
		t.Errorf("Player dx after reset = %v, want 0", resetPlayerState["dx"])
	}
	if dy, ok := resetPlayerState["dy"].(float64); !ok || dy != 0.0 {
		t.Errorf("Player dy after reset = %v, want 0", resetPlayerState["dy"])
	}

	// Verify player is not dead
	if dead, ok := resetPlayerState["dead"].(bool); !ok || dead {
		t.Errorf("Player dead after reset = %v, want false", resetPlayerState["dead"])
	}

	// Verify arrows are reset to starting count (4)
	// Note: The state key is "ac" for arrow count, and it comes as float64 from JSON
	if arrowCount, ok := resetPlayerState["ac"].(float64); !ok || arrowCount != 4.0 {
		t.Errorf("Player arrows after reset = %v, want 4", resetPlayerState["ac"])
	}

	// Note: Position may have changed to a new respawn location, so we just verify it's set
	if _, ok := resetPlayerState["x"].(float64); !ok {
		t.Error("Player x position should be set after reset")
	}
	if _, ok := resetPlayerState["y"].(float64); !ok {
		t.Error("Player y position should be set after reset")
	}

	// Log initial and reset positions for informational purposes
	t.Logf("Initial position: (%v, %v), Reset position: (%v, %v)", initialX, initialY, resetPlayerState["x"], resetPlayerState["y"])
}
