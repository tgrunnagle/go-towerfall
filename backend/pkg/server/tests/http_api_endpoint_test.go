package tests

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go-ws-server/pkg/server"
	"go-ws-server/pkg/server/game_maps"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/uuid"
)

// TestHandleGetMaps tests the /api/maps endpoint
func TestHandleGetMaps(t *testing.T) {
	s := server.NewServer()

	tests := []struct {
		name           string
		method         string
		expectedStatus int
		checkResponse  bool
	}{
		{
			name:           "GET maps - success",
			method:         "GET",
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:           "OPTIONS maps - preflight",
			method:         "OPTIONS",
			expectedStatus: http.StatusOK,
			checkResponse:  false,
		},
		{
			name:           "POST maps - method not allowed",
			method:         "POST",
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  true,
		},
		{
			name:           "PUT maps - method not allowed",
			method:         "PUT",
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  true,
		},
		{
			name:           "DELETE maps - method not allowed",
			method:         "DELETE",
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/api/maps", nil)
			w := httptest.NewRecorder()

			s.HandleGetMaps(w, req)

			// Check status code
			if w.Code != tt.expectedStatus {
				t.Errorf("Status code = %d, want %d", w.Code, tt.expectedStatus)
			}

			// Check CORS headers
			if w.Header().Get("Access-Control-Allow-Origin") != "*" {
				t.Errorf("Missing or incorrect CORS origin header")
			}
			if w.Header().Get("Access-Control-Allow-Methods") != "GET, OPTIONS" {
				t.Errorf("Missing or incorrect CORS methods header")
			}
			if w.Header().Get("Access-Control-Allow-Headers") != "Content-Type" {
				t.Errorf("Missing or incorrect CORS headers header")
			}

			// Check content type for non-OPTIONS requests
			if tt.method != "OPTIONS" && w.Header().Get("Content-Type") != "application/json" {
				t.Errorf("Missing or incorrect content type header")
			}

			// Check response body for successful GET requests
			if tt.checkResponse && tt.expectedStatus == http.StatusOK {
				var response server.GetMapsResponse
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Errorf("Failed to unmarshal response: %v", err)
				}

				if len(response.Maps) == 0 {
					t.Errorf("Expected at least one map in response")
				}

				// Verify map structure
				for _, mapInfo := range response.Maps {
					if mapInfo.Type == "" {
						t.Errorf("Map type should not be empty")
					}
					if mapInfo.Name == "" {
						t.Errorf("Map name should not be empty")
					}
					if mapInfo.CanvasSizeX <= 0 {
						t.Errorf("Canvas size X should be positive, got %d", mapInfo.CanvasSizeX)
					}
					if mapInfo.CanvasSizeY <= 0 {
						t.Errorf("Canvas size Y should be positive, got %d", mapInfo.CanvasSizeY)
					}
				}
			}

			// Check error response for method not allowed
			if tt.checkResponse && tt.expectedStatus == http.StatusMethodNotAllowed {
				var errorResponse map[string]interface{}
				if err := json.Unmarshal(w.Body.Bytes(), &errorResponse); err != nil {
					t.Errorf("Failed to unmarshal error response: %v", err)
				}
				if errorResponse["error"] != "Method not allowed" {
					t.Errorf("Expected 'Method not allowed' error message")
				}
			}
		})
	}
}

// TestHandleCreateGame tests the /api/createGame endpoint
func TestHandleCreateGame(t *testing.T) {
	s := server.NewServer()

	tests := []struct {
		name           string
		method         string
		requestBody    interface{}
		expectedStatus int
		checkResponse  bool
	}{
		{
			name:   "Create game - success",
			method: "POST",
			requestBody: server.CreateGameHTTPRequest{
				PlayerName: "Test Player",
				RoomName:   "Test Room",
				MapType:    "default",
			},
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:           "OPTIONS create game - preflight",
			method:         "OPTIONS",
			requestBody:    nil,
			expectedStatus: http.StatusOK,
			checkResponse:  false,
		},
		{
			name:           "GET create game - method not allowed",
			method:         "GET",
			requestBody:    nil,
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  true,
		},
		{
			name:   "Create game - missing player name",
			method: "POST",
			requestBody: server.CreateGameHTTPRequest{
				PlayerName: "",
				RoomName:   "Test Room",
				MapType:    "default",
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:   "Create game - missing room name",
			method: "POST",
			requestBody: server.CreateGameHTTPRequest{
				PlayerName: "Test Player",
				RoomName:   "",
				MapType:    "default",
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:           "Create game - invalid JSON",
			method:         "POST",
			requestBody:    "invalid json",
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req *http.Request
			if tt.requestBody != nil {
				var body []byte
				var err error
				if str, ok := tt.requestBody.(string); ok {
					body = []byte(str)
				} else {
					body, err = json.Marshal(tt.requestBody)
					if err != nil {
						t.Fatalf("Failed to marshal request body: %v", err)
					}
				}
				req = httptest.NewRequest(tt.method, "/api/createGame", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
			} else {
				req = httptest.NewRequest(tt.method, "/api/createGame", nil)
			}

			w := httptest.NewRecorder()
			s.HandleCreateGame(w, req)

			// Check status code
			if w.Code != tt.expectedStatus {
				t.Errorf("Status code = %d, want %d", w.Code, tt.expectedStatus)
			}

			// Check CORS headers
			if w.Header().Get("Access-Control-Allow-Origin") != "*" {
				t.Errorf("Missing or incorrect CORS origin header")
			}
			if w.Header().Get("Access-Control-Allow-Methods") != "POST, OPTIONS" {
				t.Errorf("Missing or incorrect CORS methods header")
			}
			if w.Header().Get("Access-Control-Allow-Headers") != "Content-Type" {
				t.Errorf("Missing or incorrect CORS headers header")
			}

			// Check content type for non-OPTIONS requests
			if tt.method != "OPTIONS" && w.Header().Get("Content-Type") != "application/json" {
				t.Errorf("Missing or incorrect content type header")
			}

			// Check response body
			if tt.checkResponse {
				var response server.CreateGameHTTPResponse
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Errorf("Failed to unmarshal response: %v", err)
				}

				if tt.expectedStatus == http.StatusOK {
					if !response.Success {
						t.Errorf("Expected success = true, got false")
					}
					if response.PlayerID == "" {
						t.Errorf("Player ID should not be empty")
					}
					if response.PlayerToken == "" {
						t.Errorf("Player token should not be empty")
					}
					if response.RoomID == "" {
						t.Errorf("Room ID should not be empty")
					}
					if response.RoomCode == "" {
						t.Errorf("Room code should not be empty")
					}
					if response.RoomName == "" {
						t.Errorf("Room name should not be empty")
					}
					if response.CanvasSizeX <= 0 {
						t.Errorf("Canvas size X should be positive, got %d", response.CanvasSizeX)
					}
					if response.CanvasSizeY <= 0 {
						t.Errorf("Canvas size Y should be positive, got %d", response.CanvasSizeY)
					}
				} else {
					if response.Success {
						t.Errorf("Expected success = false, got true")
					}
					if response.Error == "" {
						t.Errorf("Error message should not be empty for failed requests")
					}
				}
			}
		})
	}
}

// TestHandleJoinGame tests the /api/joinGame endpoint
func TestHandleJoinGame(t *testing.T) {
	s := server.NewServer()

	// Create a room first
	room, _, err := server.NewGameWithPlayer("Test Room", "Host Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create test room: %v", err)
	}
	s.GetRoomManager().AddGameRoom(room)

	tests := []struct {
		name           string
		method         string
		requestBody    interface{}
		expectedStatus int
		checkResponse  bool
	}{
		{
			name:   "Join game - success",
			method: "POST",
			requestBody: server.JoinGameHTTPRequest{
				PlayerName:   "Joining Player",
				RoomCode:     room.RoomCode,
				RoomPassword: room.Password,
				IsSpectator:  false,
			},
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:   "Join game as spectator - success",
			method: "POST",
			requestBody: server.JoinGameHTTPRequest{
				PlayerName:   "Spectator Player",
				RoomCode:     room.RoomCode,
				RoomPassword: room.Password,
				IsSpectator:  true,
			},
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:           "OPTIONS join game - preflight",
			method:         "OPTIONS",
			requestBody:    nil,
			expectedStatus: http.StatusOK,
			checkResponse:  false,
		},
		{
			name:           "GET join game - method not allowed",
			method:         "GET",
			requestBody:    nil,
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  true,
		},
		{
			name:   "Join game - missing player name",
			method: "POST",
			requestBody: server.JoinGameHTTPRequest{
				PlayerName:   "",
				RoomCode:     room.RoomCode,
				RoomPassword: room.Password,
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:   "Join game - missing room code",
			method: "POST",
			requestBody: server.JoinGameHTTPRequest{
				PlayerName:   "Test Player",
				RoomCode:     "",
				RoomPassword: room.Password,
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:   "Join game - missing password",
			method: "POST",
			requestBody: server.JoinGameHTTPRequest{
				PlayerName:   "Test Player",
				RoomCode:     room.RoomCode,
				RoomPassword: "",
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:   "Join game - wrong password",
			method: "POST",
			requestBody: server.JoinGameHTTPRequest{
				PlayerName:   "Test Player",
				RoomCode:     room.RoomCode,
				RoomPassword: "WRONG",
			},
			expectedStatus: http.StatusInternalServerError,
			checkResponse:  true,
		},
		{
			name:   "Join game - room not found",
			method: "POST",
			requestBody: server.JoinGameHTTPRequest{
				PlayerName:   "Test Player",
				RoomCode:     "NOTFOUND",
				RoomPassword: "password",
			},
			expectedStatus: http.StatusNotFound,
			checkResponse:  true,
		},
		{
			name:           "Join game - invalid JSON",
			method:         "POST",
			requestBody:    "invalid json",
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req *http.Request
			if tt.requestBody != nil {
				var body []byte
				var err error
				if str, ok := tt.requestBody.(string); ok {
					body = []byte(str)
				} else {
					body, err = json.Marshal(tt.requestBody)
					if err != nil {
						t.Fatalf("Failed to marshal request body: %v", err)
					}
				}
				req = httptest.NewRequest(tt.method, "/api/joinGame", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
			} else {
				req = httptest.NewRequest(tt.method, "/api/joinGame", nil)
			}

			w := httptest.NewRecorder()
			s.HandleJoinGame(w, req)

			// Check status code
			if w.Code != tt.expectedStatus {
				t.Errorf("Status code = %d, want %d", w.Code, tt.expectedStatus)
			}

			// Check CORS headers
			if w.Header().Get("Access-Control-Allow-Origin") != "*" {
				t.Errorf("Missing or incorrect CORS origin header")
			}
			if w.Header().Get("Access-Control-Allow-Methods") != "POST, OPTIONS" {
				t.Errorf("Missing or incorrect CORS methods header")
			}
			if w.Header().Get("Access-Control-Allow-Headers") != "Content-Type" {
				t.Errorf("Missing or incorrect CORS headers header")
			}

			// Check content type for non-OPTIONS requests
			if tt.method != "OPTIONS" && w.Header().Get("Content-Type") != "application/json" {
				t.Errorf("Missing or incorrect content type header")
			}

			// Check response body
			if tt.checkResponse {
				var response server.JoinGameHTTPResponse
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Errorf("Failed to unmarshal response: %v", err)
				}

				if tt.expectedStatus == http.StatusOK {
					if !response.Success {
						t.Errorf("Expected success = true, got false")
					}
					if response.PlayerID == "" {
						t.Errorf("Player ID should not be empty")
					}
					if response.PlayerToken == "" {
						t.Errorf("Player token should not be empty")
					}
					if response.RoomID == "" {
						t.Errorf("Room ID should not be empty")
					}
					if response.RoomCode == "" {
						t.Errorf("Room code should not be empty")
					}
					if response.CanvasSizeX <= 0 {
						t.Errorf("Canvas size X should be positive, got %d", response.CanvasSizeX)
					}
					if response.CanvasSizeY <= 0 {
						t.Errorf("Canvas size Y should be positive, got %d", response.CanvasSizeY)
					}

					// Check spectator flag matches request
					if reqBody, ok := tt.requestBody.(server.JoinGameHTTPRequest); ok {
						if response.IsSpectator != reqBody.IsSpectator {
							t.Errorf("Spectator flag = %v, want %v", response.IsSpectator, reqBody.IsSpectator)
						}
					}
				} else {
					if response.Success {
						t.Errorf("Expected success = false, got true")
					}
					if response.Error == "" {
						t.Errorf("Error message should not be empty for failed requests")
					}
				}
			}
		})
	}
}

// TestHandleCreateTrainingRoom tests the /api/training/createRoom endpoint
func TestHandleCreateTrainingRoom(t *testing.T) {
	s := server.NewServer()

	tests := []struct {
		name           string
		method         string
		requestBody    interface{}
		expectedStatus int
		checkResponse  bool
	}{
		{
			name:   "Create training room - success",
			method: "POST",
			requestBody: server.CreateTrainingRoomRequest{
				PlayerName: "Training Player",
				RoomName:   "Training Room",
				MapType:    "default",
				TrainingConfig: server.TrainingConfig{
					SpeedMultiplier:   5.0,
					HeadlessMode:      false,
					TrainingMode:      true,
					SessionID:         "test-session-1",
					DirectStateAccess: true,
				},
			},
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:   "Create training room - headless mode",
			method: "POST",
			requestBody: server.CreateTrainingRoomRequest{
				PlayerName: "Training Player",
				RoomName:   "Headless Training Room",
				MapType:    "default",
				TrainingConfig: server.TrainingConfig{
					SpeedMultiplier:   50.0,
					HeadlessMode:      true,
					TrainingMode:      true,
					SessionID:         "test-session-2",
					DirectStateAccess: true,
				},
			},
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:           "OPTIONS create training room - preflight",
			method:         "OPTIONS",
			requestBody:    nil,
			expectedStatus: http.StatusOK,
			checkResponse:  false,
		},
		{
			name:           "GET create training room - method not allowed",
			method:         "GET",
			requestBody:    nil,
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  true,
		},
		{
			name:   "Create training room - missing player name",
			method: "POST",
			requestBody: server.CreateTrainingRoomRequest{
				PlayerName: "",
				RoomName:   "Training Room",
				MapType:    "default",
				TrainingConfig: server.TrainingConfig{
					SpeedMultiplier: 5.0,
					TrainingMode:    true,
				},
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:   "Create training room - missing room name",
			method: "POST",
			requestBody: server.CreateTrainingRoomRequest{
				PlayerName: "Training Player",
				RoomName:   "",
				MapType:    "default",
				TrainingConfig: server.TrainingConfig{
					SpeedMultiplier: 5.0,
					TrainingMode:    true,
				},
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:   "Create training room - invalid speed multiplier (too high)",
			method: "POST",
			requestBody: server.CreateTrainingRoomRequest{
				PlayerName: "Training Player",
				RoomName:   "Training Room",
				MapType:    "default",
				TrainingConfig: server.TrainingConfig{
					SpeedMultiplier: 150.0,
					TrainingMode:    true,
				},
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:   "Create training room - invalid speed multiplier (zero)",
			method: "POST",
			requestBody: server.CreateTrainingRoomRequest{
				PlayerName: "Training Player",
				RoomName:   "Training Room",
				MapType:    "default",
				TrainingConfig: server.TrainingConfig{
					SpeedMultiplier: 0.0,
					TrainingMode:    true,
				},
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:   "Create training room - invalid speed multiplier (negative)",
			method: "POST",
			requestBody: server.CreateTrainingRoomRequest{
				PlayerName: "Training Player",
				RoomName:   "Training Room",
				MapType:    "default",
				TrainingConfig: server.TrainingConfig{
					SpeedMultiplier: -1.0,
					TrainingMode:    true,
				},
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:           "Create training room - invalid JSON",
			method:         "POST",
			requestBody:    "invalid json",
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req *http.Request
			if tt.requestBody != nil {
				var body []byte
				var err error
				if str, ok := tt.requestBody.(string); ok {
					body = []byte(str)
				} else {
					body, err = json.Marshal(tt.requestBody)
					if err != nil {
						t.Fatalf("Failed to marshal request body: %v", err)
					}
				}
				req = httptest.NewRequest(tt.method, "/api/training/createRoom", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
			} else {
				req = httptest.NewRequest(tt.method, "/api/training/createRoom", nil)
			}

			w := httptest.NewRecorder()
			s.HandleCreateTrainingRoom(w, req)

			// Check status code
			if w.Code != tt.expectedStatus {
				t.Errorf("Status code = %d, want %d", w.Code, tt.expectedStatus)
			}

			// Check CORS headers
			if w.Header().Get("Access-Control-Allow-Origin") != "*" {
				t.Errorf("Missing or incorrect CORS origin header")
			}
			if w.Header().Get("Access-Control-Allow-Methods") != "POST, OPTIONS" {
				t.Errorf("Missing or incorrect CORS methods header")
			}
			if w.Header().Get("Access-Control-Allow-Headers") != "Content-Type, Authorization" {
				t.Errorf("Missing or incorrect CORS headers header")
			}

			// Check content type for non-OPTIONS requests
			if tt.method != "OPTIONS" && w.Header().Get("Content-Type") != "application/json" {
				t.Errorf("Missing or incorrect content type header")
			}

			// Check response body
			if tt.checkResponse {
				var response server.CreateTrainingRoomResponse
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Errorf("Failed to unmarshal response: %v", err)
				}

				if tt.expectedStatus == http.StatusOK {
					if !response.Success {
						t.Errorf("Expected success = true, got false")
					}
					if response.PlayerID == "" {
						t.Errorf("Player ID should not be empty")
					}
					if response.PlayerToken == "" {
						t.Errorf("Player token should not be empty")
					}
					if response.RoomID == "" {
						t.Errorf("Room ID should not be empty")
					}
					if response.RoomCode == "" {
						t.Errorf("Room code should not be empty")
					}
					if response.RoomName == "" {
						t.Errorf("Room name should not be empty")
					}
					if response.CanvasSizeX <= 0 {
						t.Errorf("Canvas size X should be positive, got %d", response.CanvasSizeX)
					}
					if response.CanvasSizeY <= 0 {
						t.Errorf("Canvas size Y should be positive, got %d", response.CanvasSizeY)
					}

					// Check training-specific fields
					if reqBody, ok := tt.requestBody.(server.CreateTrainingRoomRequest); ok {
						if response.SpeedMultiplier != reqBody.TrainingConfig.SpeedMultiplier {
							t.Errorf("Speed multiplier = %v, want %v", response.SpeedMultiplier, reqBody.TrainingConfig.SpeedMultiplier)
						}
						if response.HeadlessMode != reqBody.TrainingConfig.HeadlessMode {
							t.Errorf("Headless mode = %v, want %v", response.HeadlessMode, reqBody.TrainingConfig.HeadlessMode)
						}
					}
				} else {
					if response.Success {
						t.Errorf("Expected success = false, got true")
					}
					if response.Error == "" {
						t.Errorf("Error message should not be empty for failed requests")
					}
				}
			}
		})
	}
}

// TestHandleJoinTrainingRoom tests the /api/training/joinRoom endpoint
func TestHandleJoinTrainingRoom(t *testing.T) {
	s := server.NewServer()

	// Create a training room first
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   10.0,
		HeadlessMode:      false,
		TrainingMode:      true,
		SessionID:         "test-session",
		DirectStateAccess: true,
	}
	room, _, err := server.NewTrainingGameWithPlayer("Training Room", "Host Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}
	s.GetRoomManager().AddGameRoom(room)

	tests := []struct {
		name           string
		method         string
		requestBody    interface{}
		expectedStatus int
		checkResponse  bool
	}{
		{
			name:   "Join training room - success",
			method: "POST",
			requestBody: server.JoinTrainingRoomRequest{
				PlayerName:         "Joining Player",
				RoomCode:           room.RoomCode,
				RoomPassword:       room.Password,
				EnableDirectAccess: true,
			},
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:           "OPTIONS join training room - preflight",
			method:         "OPTIONS",
			requestBody:    nil,
			expectedStatus: http.StatusOK,
			checkResponse:  false,
		},
		{
			name:           "GET join training room - method not allowed",
			method:         "GET",
			requestBody:    nil,
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  true,
		},
		{
			name:   "Join training room - missing player name",
			method: "POST",
			requestBody: server.JoinTrainingRoomRequest{
				PlayerName:   "",
				RoomCode:     room.RoomCode,
				RoomPassword: room.Password,
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:   "Join training room - missing room code",
			method: "POST",
			requestBody: server.JoinTrainingRoomRequest{
				PlayerName:   "Test Player",
				RoomCode:     "",
				RoomPassword: room.Password,
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:   "Join training room - missing password",
			method: "POST",
			requestBody: server.JoinTrainingRoomRequest{
				PlayerName:   "Test Player",
				RoomCode:     room.RoomCode,
				RoomPassword: "",
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:   "Join training room - wrong password",
			method: "POST",
			requestBody: server.JoinTrainingRoomRequest{
				PlayerName:   "Test Player",
				RoomCode:     room.RoomCode,
				RoomPassword: "WRONG",
			},
			expectedStatus: http.StatusInternalServerError,
			checkResponse:  true,
		},
		{
			name:   "Join training room - room not found",
			method: "POST",
			requestBody: server.JoinTrainingRoomRequest{
				PlayerName:   "Test Player",
				RoomCode:     "NOTFOUND",
				RoomPassword: "password",
			},
			expectedStatus: http.StatusNotFound,
			checkResponse:  true,
		},
		{
			name:           "Join training room - invalid JSON",
			method:         "POST",
			requestBody:    "invalid json",
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req *http.Request
			if tt.requestBody != nil {
				var body []byte
				var err error
				if str, ok := tt.requestBody.(string); ok {
					body = []byte(str)
				} else {
					body, err = json.Marshal(tt.requestBody)
					if err != nil {
						t.Fatalf("Failed to marshal request body: %v", err)
					}
				}
				req = httptest.NewRequest(tt.method, "/api/training/joinRoom", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
			} else {
				req = httptest.NewRequest(tt.method, "/api/training/joinRoom", nil)
			}

			w := httptest.NewRecorder()
			s.HandleJoinTrainingRoom(w, req)

			// Check status code
			if w.Code != tt.expectedStatus {
				t.Errorf("Status code = %d, want %d", w.Code, tt.expectedStatus)
			}

			// Check CORS headers
			if w.Header().Get("Access-Control-Allow-Origin") != "*" {
				t.Errorf("Missing or incorrect CORS origin header")
			}
			if w.Header().Get("Access-Control-Allow-Methods") != "POST, OPTIONS" {
				t.Errorf("Missing or incorrect CORS methods header")
			}
			if w.Header().Get("Access-Control-Allow-Headers") != "Content-Type, Authorization" {
				t.Errorf("Missing or incorrect CORS headers header")
			}

			// Check content type for non-OPTIONS requests
			if tt.method != "OPTIONS" && w.Header().Get("Content-Type") != "application/json" {
				t.Errorf("Missing or incorrect content type header")
			}

			// Check response body
			if tt.checkResponse {
				var response server.JoinTrainingRoomResponse
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Errorf("Failed to unmarshal response: %v", err)
				}

				if tt.expectedStatus == http.StatusOK {
					if !response.Success {
						t.Errorf("Expected success = true, got false")
					}
					if response.PlayerID == "" {
						t.Errorf("Player ID should not be empty")
					}
					if response.PlayerToken == "" {
						t.Errorf("Player token should not be empty")
					}
					if response.RoomID == "" {
						t.Errorf("Room ID should not be empty")
					}
					if response.RoomCode == "" {
						t.Errorf("Room code should not be empty")
					}
					if response.CanvasSizeX <= 0 {
						t.Errorf("Canvas size X should be positive, got %d", response.CanvasSizeX)
					}
					if response.CanvasSizeY <= 0 {
						t.Errorf("Canvas size Y should be positive, got %d", response.CanvasSizeY)
					}

					// Check training-specific fields
					if !response.TrainingEnabled {
						t.Errorf("Training enabled should be true for training room")
					}
					if response.SpeedMultiplier != 10.0 {
						t.Errorf("Speed multiplier = %v, want 10.0", response.SpeedMultiplier)
					}
					if response.HeadlessMode != false {
						t.Errorf("Headless mode = %v, want false", response.HeadlessMode)
					}
				} else {
					if response.Success {
						t.Errorf("Expected success = false, got true")
					}
					if response.Error == "" {
						t.Errorf("Error message should not be empty for failed requests")
					}
				}
			}
		})
	}
}

// TestHandleSetRoomSpeed tests the /api/training/rooms/{roomId}/speed endpoint
func TestHandleSetRoomSpeed(t *testing.T) {
	s := server.NewServer()

	// Create a training room first
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   5.0,
		HeadlessMode:      false,
		TrainingMode:      true,
		SessionID:         "speed-test-session",
		DirectStateAccess: true,
	}
	room, player, err := server.NewTrainingGameWithPlayer("Speed Test Room", "Host Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}
	s.GetRoomManager().AddGameRoom(room)

	// Create a normal room for testing error cases
	normalRoom, normalPlayer, err := server.NewGameWithPlayer("Normal Room", "Normal Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create normal room: %v", err)
	}
	s.GetRoomManager().AddGameRoom(normalRoom)

	tests := []struct {
		name           string
		method         string
		roomID         string
		playerToken    string
		requestBody    interface{}
		expectedStatus int
		checkResponse  bool
	}{
		{
			name:        "Set room speed - success",
			method:      "POST",
			roomID:      room.ID,
			playerToken: player.Token,
			requestBody: server.SetRoomSpeedRequest{
				SpeedMultiplier: 20.0,
			},
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:           "OPTIONS set room speed - preflight",
			method:         "OPTIONS",
			roomID:         room.ID,
			playerToken:    "",
			requestBody:    nil,
			expectedStatus: http.StatusOK,
			checkResponse:  false,
		},
		{
			name:           "GET set room speed - method not allowed",
			method:         "GET",
			roomID:         room.ID,
			playerToken:    "",
			requestBody:    nil,
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  true,
		},
		{
			name:        "Set room speed - invalid room ID",
			method:      "POST",
			roomID:      "",
			playerToken: player.Token,
			requestBody: server.SetRoomSpeedRequest{
				SpeedMultiplier: 10.0,
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:        "Set room speed - unauthorized (no token)",
			method:      "POST",
			roomID:      room.ID,
			playerToken: "",
			requestBody: server.SetRoomSpeedRequest{
				SpeedMultiplier: 10.0,
			},
			expectedStatus: http.StatusUnauthorized,
			checkResponse:  true,
		},
		{
			name:        "Set room speed - unauthorized (invalid token)",
			method:      "POST",
			roomID:      room.ID,
			playerToken: "invalid-token",
			requestBody: server.SetRoomSpeedRequest{
				SpeedMultiplier: 10.0,
			},
			expectedStatus: http.StatusUnauthorized,
			checkResponse:  true,
		},
		{
			name:        "Set room speed - room not found",
			method:      "POST",
			roomID:      uuid.New().String(),
			playerToken: player.Token,
			requestBody: server.SetRoomSpeedRequest{
				SpeedMultiplier: 10.0,
			},
			expectedStatus: http.StatusUnauthorized, // Auth check happens before room existence check
			checkResponse:  true,
		},
		{
			name:        "Set room speed - non-training room",
			method:      "POST",
			roomID:      normalRoom.ID,
			playerToken: normalPlayer.Token,
			requestBody: server.SetRoomSpeedRequest{
				SpeedMultiplier: 10.0,
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:        "Set room speed - invalid speed (too high)",
			method:      "POST",
			roomID:      room.ID,
			playerToken: player.Token,
			requestBody: server.SetRoomSpeedRequest{
				SpeedMultiplier: 150.0,
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:        "Set room speed - invalid speed (zero)",
			method:      "POST",
			roomID:      room.ID,
			playerToken: player.Token,
			requestBody: server.SetRoomSpeedRequest{
				SpeedMultiplier: 0.0,
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:           "Set room speed - invalid JSON",
			method:         "POST",
			roomID:         room.ID,
			playerToken:    player.Token,
			requestBody:    "invalid json",
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req *http.Request
			url := fmt.Sprintf("/api/training/rooms/%s/speed", tt.roomID)

			if tt.requestBody != nil {
				var body []byte
				var err error
				if str, ok := tt.requestBody.(string); ok {
					body = []byte(str)
				} else {
					body, err = json.Marshal(tt.requestBody)
					if err != nil {
						t.Fatalf("Failed to marshal request body: %v", err)
					}
				}
				req = httptest.NewRequest(tt.method, url, bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
			} else {
				req = httptest.NewRequest(tt.method, url, nil)
			}

			if tt.playerToken != "" {
				req.Header.Set("Authorization", "Bearer "+tt.playerToken)
			}

			w := httptest.NewRecorder()
			s.HandleSetRoomSpeed(w, req)

			// Check status code
			if w.Code != tt.expectedStatus {
				t.Errorf("Status code = %d, want %d", w.Code, tt.expectedStatus)
			}

			// Check CORS headers
			if w.Header().Get("Access-Control-Allow-Origin") != "*" {
				t.Errorf("Missing or incorrect CORS origin header")
			}
			if w.Header().Get("Access-Control-Allow-Methods") != "POST, OPTIONS" {
				t.Errorf("Missing or incorrect CORS methods header")
			}
			if w.Header().Get("Access-Control-Allow-Headers") != "Content-Type, Authorization" {
				t.Errorf("Missing or incorrect CORS headers header")
			}

			// Check content type for non-OPTIONS requests
			if tt.method != "OPTIONS" && w.Header().Get("Content-Type") != "application/json" {
				t.Errorf("Missing or incorrect content type header")
			}

			// Check response body
			if tt.checkResponse {
				var response server.SetRoomSpeedResponse
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Errorf("Failed to unmarshal response: %v", err)
				}

				if tt.expectedStatus == http.StatusOK {
					if !response.Success {
						t.Errorf("Expected success = true, got false")
					}
					if reqBody, ok := tt.requestBody.(server.SetRoomSpeedRequest); ok {
						if response.SpeedMultiplier != reqBody.SpeedMultiplier {
							t.Errorf("Speed multiplier = %v, want %v", response.SpeedMultiplier, reqBody.SpeedMultiplier)
						}
					}
				} else {
					if response.Success {
						t.Errorf("Expected success = false, got true")
					}
					if response.Error == "" {
						t.Errorf("Error message should not be empty for failed requests")
					}
				}
			}
		})
	}
}

// TestHandleGetRoomState tests the /api/training/rooms/{roomId}/state endpoint
func TestHandleGetRoomState(t *testing.T) {
	s := server.NewServer()

	// Create a training room first
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   5.0,
		HeadlessMode:      false,
		TrainingMode:      true,
		SessionID:         "state-test-session",
		DirectStateAccess: true,
	}
	room, player, err := server.NewTrainingGameWithPlayer("State Test Room", "Host Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}
	s.GetRoomManager().AddGameRoom(room)

	tests := []struct {
		name           string
		method         string
		roomID         string
		playerToken    string
		expectedStatus int
		checkResponse  bool
	}{
		{
			name:           "Get room state - success",
			method:         "GET",
			roomID:         room.ID,
			playerToken:    player.Token,
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:           "OPTIONS get room state - preflight",
			method:         "OPTIONS",
			roomID:         room.ID,
			playerToken:    "",
			expectedStatus: http.StatusOK,
			checkResponse:  false,
		},
		{
			name:           "POST get room state - method not allowed",
			method:         "POST",
			roomID:         room.ID,
			playerToken:    "",
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  true,
		},
		{
			name:           "Get room state - invalid room ID",
			method:         "GET",
			roomID:         "",
			playerToken:    player.Token,
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:           "Get room state - unauthorized (no token)",
			method:         "GET",
			roomID:         room.ID,
			playerToken:    "",
			expectedStatus: http.StatusUnauthorized,
			checkResponse:  true,
		},
		{
			name:           "Get room state - unauthorized (invalid token)",
			method:         "GET",
			roomID:         room.ID,
			playerToken:    "invalid-token",
			expectedStatus: http.StatusUnauthorized,
			checkResponse:  true,
		},
		{
			name:           "Get room state - room not found",
			method:         "GET",
			roomID:         uuid.New().String(),
			playerToken:    player.Token,
			expectedStatus: http.StatusUnauthorized, // Auth check happens before room existence check
			checkResponse:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url := fmt.Sprintf("/api/training/rooms/%s/state", tt.roomID)
			req := httptest.NewRequest(tt.method, url, nil)

			if tt.playerToken != "" {
				req.Header.Set("Authorization", "Bearer "+tt.playerToken)
			}

			w := httptest.NewRecorder()
			s.HandleGetRoomState(w, req)

			// Check status code
			if w.Code != tt.expectedStatus {
				t.Errorf("Status code = %d, want %d", w.Code, tt.expectedStatus)
			}

			// Check CORS headers
			if w.Header().Get("Access-Control-Allow-Origin") != "*" {
				t.Errorf("Missing or incorrect CORS origin header")
			}
			if w.Header().Get("Access-Control-Allow-Methods") != "GET, OPTIONS" {
				t.Errorf("Missing or incorrect CORS methods header")
			}
			if w.Header().Get("Access-Control-Allow-Headers") != "Content-Type, Authorization" {
				t.Errorf("Missing or incorrect CORS headers header")
			}

			// Check content type for non-OPTIONS requests
			if tt.method != "OPTIONS" && w.Header().Get("Content-Type") != "application/json" {
				t.Errorf("Missing or incorrect content type header")
			}

			// Check response body
			if tt.checkResponse {
				var response server.GetRoomStateResponse
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Errorf("Failed to unmarshal response: %v", err)
				}

				if tt.expectedStatus == http.StatusOK {
					if !response.Success {
						t.Errorf("Expected success = true, got false")
					}
					if response.State == nil {
						t.Errorf("State should not be nil")
					}
					if response.Timestamp <= 0 {
						t.Errorf("Timestamp should be positive, got %d", response.Timestamp)
					}

					// Verify state structure
					if response.State["objects"] == nil {
						t.Errorf("State should contain objects")
					}
					if response.State["room"] == nil {
						t.Errorf("State should contain room information")
					}
				} else {
					if response.Success {
						t.Errorf("Expected success = false, got true")
					}
					if response.Error == "" {
						t.Errorf("Error message should not be empty for failed requests")
					}
				}
			}
		})
	}
}

// TestHandleConfigureTrainingRoom tests the /api/training/rooms/{roomId}/configure endpoint
func TestHandleConfigureTrainingRoom(t *testing.T) {
	s := server.NewServer()

	// Create a training room first
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   5.0,
		HeadlessMode:      false,
		TrainingMode:      true,
		SessionID:         "config-test-session",
		DirectStateAccess: false,
	}
	room, player, err := server.NewTrainingGameWithPlayer("Config Test Room", "Host Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}
	s.GetRoomManager().AddGameRoom(room)

	// Create a normal room for testing error cases
	normalRoom, normalPlayer, err := server.NewGameWithPlayer("Normal Room", "Normal Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create normal room: %v", err)
	}
	s.GetRoomManager().AddGameRoom(normalRoom)

	tests := []struct {
		name           string
		method         string
		roomID         string
		playerToken    string
		requestBody    interface{}
		expectedStatus int
		checkResponse  bool
	}{
		{
			name:        "Configure training room - enable headless",
			method:      "POST",
			roomID:      room.ID,
			playerToken: player.Token,
			requestBody: server.ConfigureTrainingRoomRequest{
				TrainingMode:      "headless",
				SpeedMultiplier:   10.0,
				DirectStateAccess: true,
			},
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:        "Configure training room - disable headless",
			method:      "POST",
			roomID:      room.ID,
			playerToken: player.Token,
			requestBody: server.ConfigureTrainingRoomRequest{
				TrainingMode:      "normal",
				SpeedMultiplier:   5.0,
				DirectStateAccess: false,
			},
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:           "OPTIONS configure training room - preflight",
			method:         "OPTIONS",
			roomID:         room.ID,
			playerToken:    "",
			requestBody:    nil,
			expectedStatus: http.StatusOK,
			checkResponse:  false,
		},
		{
			name:           "GET configure training room - method not allowed",
			method:         "GET",
			roomID:         room.ID,
			playerToken:    "",
			requestBody:    nil,
			expectedStatus: http.StatusMethodNotAllowed,
			checkResponse:  true,
		},
		{
			name:        "Configure training room - invalid room ID",
			method:      "POST",
			roomID:      "",
			playerToken: player.Token,
			requestBody: server.ConfigureTrainingRoomRequest{
				TrainingMode:    "headless",
				SpeedMultiplier: 10.0,
			},
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
		{
			name:        "Configure training room - unauthorized (no token)",
			method:      "POST",
			roomID:      room.ID,
			playerToken: "",
			requestBody: server.ConfigureTrainingRoomRequest{
				TrainingMode:    "headless",
				SpeedMultiplier: 10.0,
			},
			expectedStatus: http.StatusUnauthorized,
			checkResponse:  true,
		},
		{
			name:        "Configure training room - unauthorized (invalid token)",
			method:      "POST",
			roomID:      room.ID,
			playerToken: "invalid-token",
			requestBody: server.ConfigureTrainingRoomRequest{
				TrainingMode:    "headless",
				SpeedMultiplier: 10.0,
			},
			expectedStatus: http.StatusUnauthorized,
			checkResponse:  true,
		},
		{
			name:        "Configure training room - room not found",
			method:      "POST",
			roomID:      uuid.New().String(),
			playerToken: player.Token,
			requestBody: server.ConfigureTrainingRoomRequest{
				TrainingMode:    "headless",
				SpeedMultiplier: 10.0,
			},
			expectedStatus: http.StatusUnauthorized, // Auth check happens before room existence check
			checkResponse:  true,
		},
		{
			name:        "Configure training room - non-training room",
			method:      "POST",
			roomID:      normalRoom.ID,
			playerToken: normalPlayer.Token,
			requestBody: server.ConfigureTrainingRoomRequest{
				TrainingMode:    "headless",
				SpeedMultiplier: 10.0,
			},
			expectedStatus: http.StatusInternalServerError,
			checkResponse:  true,
		},
		{
			name:           "Configure training room - invalid JSON",
			method:         "POST",
			roomID:         room.ID,
			playerToken:    player.Token,
			requestBody:    "invalid json",
			expectedStatus: http.StatusBadRequest,
			checkResponse:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req *http.Request
			url := fmt.Sprintf("/api/training/rooms/%s/configure", tt.roomID)

			if tt.requestBody != nil {
				var body []byte
				var err error
				if str, ok := tt.requestBody.(string); ok {
					body = []byte(str)
				} else {
					body, err = json.Marshal(tt.requestBody)
					if err != nil {
						t.Fatalf("Failed to marshal request body: %v", err)
					}
				}
				req = httptest.NewRequest(tt.method, url, bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
			} else {
				req = httptest.NewRequest(tt.method, url, nil)
			}

			if tt.playerToken != "" {
				req.Header.Set("Authorization", "Bearer "+tt.playerToken)
			}

			w := httptest.NewRecorder()
			s.HandleConfigureTrainingRoom(w, req)

			// Check status code
			if w.Code != tt.expectedStatus {
				t.Errorf("Status code = %d, want %d", w.Code, tt.expectedStatus)
			}

			// Check CORS headers
			if w.Header().Get("Access-Control-Allow-Origin") != "*" {
				t.Errorf("Missing or incorrect CORS origin header")
			}
			if w.Header().Get("Access-Control-Allow-Methods") != "POST, OPTIONS" {
				t.Errorf("Missing or incorrect CORS methods header")
			}
			if w.Header().Get("Access-Control-Allow-Headers") != "Content-Type, Authorization" {
				t.Errorf("Missing or incorrect CORS headers header")
			}

			// Check content type for non-OPTIONS requests
			if tt.method != "OPTIONS" && w.Header().Get("Content-Type") != "application/json" {
				t.Errorf("Missing or incorrect content type header")
			}

			// Check response body
			if tt.checkResponse {
				var response server.ConfigureTrainingRoomResponse
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Errorf("Failed to unmarshal response: %v", err)
				}

				if tt.expectedStatus == http.StatusOK {
					if !response.Success {
						t.Errorf("Expected success = true, got false")
					}
				} else {
					if response.Success {
						t.Errorf("Expected success = false, got true")
					}
					if response.Error == "" {
						t.Errorf("Error message should not be empty for failed requests")
					}
				}
			}
		})
	}
}

// TestHandleTrainingRoomRequests tests the routing functionality for training room endpoints
func TestHandleTrainingRoomRequests(t *testing.T) {
	s := server.NewServer()

	// Create a training room first
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   5.0,
		HeadlessMode:      false,
		TrainingMode:      true,
		SessionID:         "routing-test-session",
		DirectStateAccess: true,
	}
	room, player, err := server.NewTrainingGameWithPlayer("Routing Test Room", "Host Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}
	s.GetRoomManager().AddGameRoom(room)

	tests := []struct {
		name           string
		method         string
		url            string
		playerToken    string
		requestBody    interface{}
		expectedStatus int
	}{
		{
			name:           "Route to speed endpoint",
			method:         "POST",
			url:            fmt.Sprintf("/api/training/rooms/%s/speed", room.ID),
			playerToken:    player.Token,
			requestBody:    server.SetRoomSpeedRequest{SpeedMultiplier: 10.0},
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Route to state endpoint",
			method:         "GET",
			url:            fmt.Sprintf("/api/training/rooms/%s/state", room.ID),
			playerToken:    player.Token,
			requestBody:    nil,
			expectedStatus: http.StatusOK,
		},
		{
			name:        "Route to configure endpoint",
			method:      "POST",
			url:         fmt.Sprintf("/api/training/rooms/%s/configure", room.ID),
			playerToken: player.Token,
			requestBody: server.ConfigureTrainingRoomRequest{
				TrainingMode:    "headless",
				SpeedMultiplier: 15.0,
			},
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Route to unknown endpoint",
			method:         "POST",
			url:            fmt.Sprintf("/api/training/rooms/%s/unknown", room.ID),
			playerToken:    player.Token,
			requestBody:    nil,
			expectedStatus: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req *http.Request

			if tt.requestBody != nil {
				body, err := json.Marshal(tt.requestBody)
				if err != nil {
					t.Fatalf("Failed to marshal request body: %v", err)
				}
				req = httptest.NewRequest(tt.method, tt.url, bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
			} else {
				req = httptest.NewRequest(tt.method, tt.url, nil)
			}

			if tt.playerToken != "" {
				req.Header.Set("Authorization", "Bearer "+tt.playerToken)
			}

			w := httptest.NewRecorder()
			s.HandleTrainingRoomRequests(w, req)

			// Check status code
			if w.Code != tt.expectedStatus {
				t.Errorf("Status code = %d, want %d", w.Code, tt.expectedStatus)
			}

			// Check that response is valid JSON
			if w.Body.Len() > 0 {
				var response map[string]interface{}
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Errorf("Failed to unmarshal response: %v", err)
				}
			}
		})
	}
}

// TestErrorHandlingAndValidation tests comprehensive error handling across all endpoints
func TestErrorHandlingAndValidation(t *testing.T) {
	s := server.NewServer()

	tests := []struct {
		name           string
		endpoint       string
		method         string
		contentType    string
		body           string
		expectedStatus int
		expectedError  string
	}{
		{
			name:           "Invalid content type - maps",
			endpoint:       "/api/maps",
			method:         "GET",
			contentType:    "text/plain",
			body:           "",
			expectedStatus: http.StatusOK, // Maps endpoint doesn't check content type for GET
			expectedError:  "",
		},

		{
			name:           "Empty body - create game",
			endpoint:       "/api/createGame",
			method:         "POST",
			contentType:    "application/json",
			body:           "",
			expectedStatus: http.StatusBadRequest,
			expectedError:  "Invalid request format",
		},
		{
			name:           "Malformed JSON - join game",
			endpoint:       "/api/joinGame",
			method:         "POST",
			contentType:    "application/json",
			body:           `{"playerName":"Test","roomCode":}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "Invalid request format",
		},
		{
			name:           "Missing required fields - create training room",
			endpoint:       "/api/training/createRoom",
			method:         "POST",
			contentType:    "application/json",
			body:           `{"trainingConfig":{"speedMultiplier":5.0}}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "PlayerName and RoomName are required",
		},
		{
			name:           "Invalid speed multiplier - create training room",
			endpoint:       "/api/training/createRoom",
			method:         "POST",
			contentType:    "application/json",
			body:           `{"playerName":"Test","roomName":"Test","mapType":"default","trainingConfig":{"speedMultiplier":200.0}}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "SpeedMultiplier must be between 0.1 and 100",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.endpoint, strings.NewReader(tt.body))
			if tt.contentType != "" {
				req.Header.Set("Content-Type", tt.contentType)
			}

			w := httptest.NewRecorder()

			// Route to appropriate handler
			switch {
			case strings.Contains(tt.endpoint, "/api/maps"):
				s.HandleGetMaps(w, req)
			case strings.Contains(tt.endpoint, "/api/createGame"):
				s.HandleCreateGame(w, req)
			case strings.Contains(tt.endpoint, "/api/joinGame"):
				s.HandleJoinGame(w, req)
			case strings.Contains(tt.endpoint, "/api/training/createRoom"):
				s.HandleCreateTrainingRoom(w, req)
			case strings.Contains(tt.endpoint, "/api/training/joinRoom"):
				s.HandleJoinTrainingRoom(w, req)
			}

			// Check status code
			if w.Code != tt.expectedStatus {
				t.Errorf("Status code = %d, want %d", w.Code, tt.expectedStatus)
			}

			// Check error message if expected
			if tt.expectedError != "" {
				var response map[string]interface{}
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Errorf("Failed to unmarshal error response: %v", err)
				}
				if errorMsg, ok := response["error"].(string); ok {
					if !strings.Contains(errorMsg, tt.expectedError) {
						t.Errorf("Error message = %v, want to contain %v", errorMsg, tt.expectedError)
					}
				} else {
					t.Errorf("Expected error message in response")
				}
			}
		})
	}
}

// TestCORSConfigurationAndSecurityHeaders tests CORS and security headers across all endpoints
func TestCORSConfigurationAndSecurityHeaders(t *testing.T) {
	s := server.NewServer()

	endpoints := []struct {
		name    string
		path    string
		handler func(http.ResponseWriter, *http.Request)
		methods string
		headers string
	}{
		{
			name:    "Maps endpoint",
			path:    "/api/maps",
			handler: s.HandleGetMaps,
			methods: "GET, OPTIONS",
			headers: "Content-Type",
		},
		{
			name:    "Create game endpoint",
			path:    "/api/createGame",
			handler: s.HandleCreateGame,
			methods: "POST, OPTIONS",
			headers: "Content-Type",
		},
		{
			name:    "Join game endpoint",
			path:    "/api/joinGame",
			handler: s.HandleJoinGame,
			methods: "POST, OPTIONS",
			headers: "Content-Type",
		},
		{
			name:    "Create training room endpoint",
			path:    "/api/training/createRoom",
			handler: s.HandleCreateTrainingRoom,
			methods: "POST, OPTIONS",
			headers: "Content-Type, Authorization",
		},
		{
			name:    "Join training room endpoint",
			path:    "/api/training/joinRoom",
			handler: s.HandleJoinTrainingRoom,
			methods: "POST, OPTIONS",
			headers: "Content-Type, Authorization",
		},
	}

	for _, endpoint := range endpoints {
		t.Run(endpoint.name, func(t *testing.T) {
			// Test OPTIONS preflight request
			req := httptest.NewRequest("OPTIONS", endpoint.path, nil)
			req.Header.Set("Origin", "https://example.com")
			req.Header.Set("Access-Control-Request-Method", "POST")
			req.Header.Set("Access-Control-Request-Headers", "Content-Type")

			w := httptest.NewRecorder()
			endpoint.handler(w, req)

			// Check CORS headers
			if w.Header().Get("Access-Control-Allow-Origin") != "*" {
				t.Errorf("Access-Control-Allow-Origin = %v, want *", w.Header().Get("Access-Control-Allow-Origin"))
			}
			if w.Header().Get("Access-Control-Allow-Methods") != endpoint.methods {
				t.Errorf("Access-Control-Allow-Methods = %v, want %v", w.Header().Get("Access-Control-Allow-Methods"), endpoint.methods)
			}
			if w.Header().Get("Access-Control-Allow-Headers") != endpoint.headers {
				t.Errorf("Access-Control-Allow-Headers = %v, want %v", w.Header().Get("Access-Control-Allow-Headers"), endpoint.headers)
			}

			// Check status code for OPTIONS
			if w.Code != http.StatusOK {
				t.Errorf("OPTIONS status code = %d, want %d", w.Code, http.StatusOK)
			}
		})
	}
}
