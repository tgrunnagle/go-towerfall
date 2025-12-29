package server

import (
	"bytes"
	"encoding/json"
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
