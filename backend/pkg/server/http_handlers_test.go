package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
