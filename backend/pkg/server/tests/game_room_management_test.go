package tests

import (
	"go-ws-server/pkg/server"
	"go-ws-server/pkg/server/game_maps"
	"go-ws-server/pkg/util"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
)

// TestRoomCreationNormal tests creating normal game rooms
func TestRoomCreationNormal(t *testing.T) {
	tests := []struct {
		name     string
		roomName string
		mapType  game_maps.MapType
		wantErr  bool
	}{
		{
			name:     "Create normal room with default map",
			roomName: "Test Room",
			mapType:  game_maps.MapDefault,
			wantErr:  false,
		},
		{
			name:     "Create room with empty name",
			roomName: "",
			mapType:  game_maps.MapDefault,
			wantErr:  false, // Empty name should be allowed
		},
		{
			name:     "Create room with long name",
			roomName: "This is a very long room name that should still work fine",
			mapType:  game_maps.MapDefault,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			roomID := uuid.New().String()
			password := util.GeneratePassword()
			roomCode := util.GenerateRoomCode()

			room, err := server.NewGameRoom(roomID, tt.roomName, password, roomCode, tt.mapType)

			if (err != nil) != tt.wantErr {
				t.Errorf("NewGameRoom() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				// Verify room properties
				if room.ID != roomID {
					t.Errorf("Room ID = %v, want %v", room.ID, roomID)
				}
				if room.Name != tt.roomName {
					t.Errorf("Room Name = %v, want %v", room.Name, tt.roomName)
				}
				if room.Password != password {
					t.Errorf("Room Password = %v, want %v", room.Password, password)
				}
				if room.RoomCode != roomCode {
					t.Errorf("Room Code = %v, want %v", room.RoomCode, roomCode)
				}
				if room.IsTraining {
					t.Errorf("Normal room should not be training room")
				}
				if room.SpeedMultiplier != 0 {
					t.Errorf("Normal room speed multiplier = %v, want 0", room.SpeedMultiplier)
				}
				if room.HeadlessMode {
					t.Errorf("Normal room should not be headless")
				}
				if room.DirectStateAccess {
					t.Errorf("Normal room should not have direct state access")
				}
				if room.Players == nil {
					t.Errorf("Room players map should be initialized")
				}
				if room.EventManager == nil {
					t.Errorf("Room event manager should be initialized")
				}
				if room.ObjectManager == nil {
					t.Errorf("Room object manager should be initialized")
				}
			}
		})
	}
}

// TestRoomCreationTraining tests creating training game rooms
func TestRoomCreationTraining(t *testing.T) {
	tests := []struct {
		name           string
		roomName       string
		mapType        game_maps.MapType
		trainingConfig server.TrainingConfig
		wantErr        bool
	}{
		{
			name:     "Create training room with normal speed",
			roomName: "Training Room",
			mapType:  game_maps.MapDefault,
			trainingConfig: server.TrainingConfig{
				SpeedMultiplier:   1.0,
				HeadlessMode:      false,
				TrainingMode:      true,
				SessionID:         "session-123",
				DirectStateAccess: true,
			},
			wantErr: false,
		},
		{
			name:     "Create training room with high speed",
			roomName: "Fast Training",
			mapType:  game_maps.MapDefault,
			trainingConfig: server.TrainingConfig{
				SpeedMultiplier:   10.0,
				HeadlessMode:      true,
				TrainingMode:      true,
				SessionID:         "session-456",
				DirectStateAccess: true,
			},
			wantErr: false,
		},
		{
			name:     "Create training room with maximum speed",
			roomName: "Max Speed Training",
			mapType:  game_maps.MapDefault,
			trainingConfig: server.TrainingConfig{
				SpeedMultiplier:   100.0,
				HeadlessMode:      true,
				TrainingMode:      true,
				SessionID:         "session-789",
				DirectStateAccess: true,
			},
			wantErr: false,
		},
		{
			name:     "Create training room with minimum speed",
			roomName: "Slow Training",
			mapType:  game_maps.MapDefault,
			trainingConfig: server.TrainingConfig{
				SpeedMultiplier:   0.1,
				HeadlessMode:      false,
				TrainingMode:      true,
				SessionID:         "session-slow",
				DirectStateAccess: false,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			roomID := uuid.New().String()
			password := util.GeneratePassword()
			roomCode := util.GenerateRoomCode()

			room, err := server.NewTrainingGameRoom(roomID, tt.roomName, password, roomCode, tt.mapType, tt.trainingConfig)

			if (err != nil) != tt.wantErr {
				t.Errorf("NewTrainingGameRoom() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				// Verify basic room properties
				if room.ID != roomID {
					t.Errorf("Room ID = %v, want %v", room.ID, roomID)
				}
				if room.Name != tt.roomName {
					t.Errorf("Room Name = %v, want %v", room.Name, tt.roomName)
				}

				// Verify training-specific properties
				if !room.IsTraining {
					t.Errorf("Training room should have IsTraining = true")
				}
				if room.SpeedMultiplier != tt.trainingConfig.SpeedMultiplier {
					t.Errorf("Speed multiplier = %v, want %v", room.SpeedMultiplier, tt.trainingConfig.SpeedMultiplier)
				}
				if room.HeadlessMode != tt.trainingConfig.HeadlessMode {
					t.Errorf("Headless mode = %v, want %v", room.HeadlessMode, tt.trainingConfig.HeadlessMode)
				}
				if room.DirectStateAccess != tt.trainingConfig.DirectStateAccess {
					t.Errorf("Direct state access = %v, want %v", room.DirectStateAccess, tt.trainingConfig.DirectStateAccess)
				}
				if room.TrainingSessionID != tt.trainingConfig.SessionID {
					t.Errorf("Training session ID = %v, want %v", room.TrainingSessionID, tt.trainingConfig.SessionID)
				}

				// Verify custom tick rate is set correctly
				expectedTickRate := time.Duration(float64(20*time.Millisecond) / tt.trainingConfig.SpeedMultiplier)
				if room.CustomTickRate != expectedTickRate {
					t.Errorf("Custom tick rate = %v, want %v", room.CustomTickRate, expectedTickRate)
				}
			}
		})
	}
}

// TestRoomCreationHeadless tests creating headless training rooms
func TestRoomCreationHeadless(t *testing.T) {
	roomID := uuid.New().String()
	password := util.GeneratePassword()
	roomCode := util.GenerateRoomCode()

	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   50.0,
		HeadlessMode:      true,
		TrainingMode:      true,
		SessionID:         "headless-session",
		DirectStateAccess: true,
	}

	room, err := server.NewTrainingGameRoom(roomID, "Headless Room", password, roomCode, game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create headless training room: %v", err)
	}

	// Verify headless mode is enabled
	if !room.IsHeadlessMode() {
		t.Errorf("Headless room should have headless mode enabled")
	}

	// Verify training room methods work
	if !room.IsTrainingRoom() {
		t.Errorf("Headless room should be a training room")
	}

	// Verify speed multiplier
	if room.GetSpeedMultiplier() != 50.0 {
		t.Errorf("Speed multiplier = %v, want 50.0", room.GetSpeedMultiplier())
	}
}

// TestRoomCleanupAndResourceManagement tests room cleanup functionality
func TestRoomCleanupAndResourceManagement(t *testing.T) {
	// Create a room with players
	room, player, err := server.NewGameWithPlayer("Test Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room with player: %v", err)
	}

	// Verify player was added
	if room.GetNumberOfConnectedPlayers() != 1 {
		t.Errorf("Expected 1 player, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Add another player
	player2, err := server.AddPlayerToGame(room, "Player 2", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add second player: %v", err)
	}

	if room.GetNumberOfConnectedPlayers() != 2 {
		t.Errorf("Expected 2 players, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Add a spectator
	spectator, err := server.AddPlayerToGame(room, "Spectator", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator: %v", err)
	}

	if room.GetNumberOfConnectedPlayers() != 3 {
		t.Errorf("Expected 3 players (including spectator), got %d", room.GetNumberOfConnectedPlayers())
	}

	// Verify spectator list
	spectators := room.GetSpectators()
	if len(spectators) != 1 {
		t.Errorf("Expected 1 spectator, got %d", len(spectators))
	}
	if spectators[0] != "Spectator" {
		t.Errorf("Expected spectator name 'Spectator', got '%s'", spectators[0])
	}

	// Remove players one by one
	room.RemovePlayer(player.ID)
	if room.GetNumberOfConnectedPlayers() != 2 {
		t.Errorf("Expected 2 players after removal, got %d", room.GetNumberOfConnectedPlayers())
	}

	room.RemovePlayer(player2.ID)
	if room.GetNumberOfConnectedPlayers() != 1 {
		t.Errorf("Expected 1 player after second removal, got %d", room.GetNumberOfConnectedPlayers())
	}

	room.RemovePlayer(spectator.ID)
	if room.GetNumberOfConnectedPlayers() != 0 {
		t.Errorf("Expected 0 players after all removals, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Verify spectator list is empty
	spectators = room.GetSpectators()
	if len(spectators) != 0 {
		t.Errorf("Expected 0 spectators after removal, got %d", len(spectators))
	}
}

// TestRoomStatePersistenceAndRetrieval tests room state management
func TestRoomStatePersistenceAndRetrieval(t *testing.T) {
	// Create training room
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   5.0,
		HeadlessMode:      false,
		TrainingMode:      true,
		SessionID:         "state-test-session",
		DirectStateAccess: true,
	}

	room, player, err := server.NewTrainingGameWithPlayer("State Test Room", "Test Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}

	// Test direct state access
	gameState := room.GetDirectGameState()
	if gameState == nil {
		t.Fatalf("Game state should not be nil")
	}

	// Verify state structure
	if gameState["objects"] == nil {
		t.Errorf("Game state should contain objects")
	}

	roomInfo, ok := gameState["room"].(map[string]interface{})
	if !ok {
		t.Fatalf("Game state should contain room information")
	}

	if roomInfo["id"] != room.ID {
		t.Errorf("Room ID in state = %v, want %v", roomInfo["id"], room.ID)
	}
	if roomInfo["isTraining"] != true {
		t.Errorf("Training flag in state = %v, want true", roomInfo["isTraining"])
	}
	if roomInfo["speedMultiplier"] != 5.0 {
		t.Errorf("Speed multiplier in state = %v, want 5.0", roomInfo["speedMultiplier"])
	}

	// Verify player state
	players, ok := gameState["players"].([]map[string]interface{})
	if !ok {
		t.Fatalf("Game state should contain players array")
	}
	if len(players) != 1 {
		t.Errorf("Expected 1 player in state, got %d", len(players))
	}
	if players[0]["id"] != player.ID {
		t.Errorf("Player ID in state = %v, want %v", players[0]["id"], player.ID)
	}

	// Test regular state retrieval
	allStates := room.GetAllGameObjectStates()
	if allStates == nil {
		t.Errorf("All game object states should not be nil")
	}
}

// TestRoomPasswordValidationAndAccessControl tests password validation
func TestRoomPasswordValidationAndAccessControl(t *testing.T) {
	room, _, err := server.NewGameWithPlayer("Password Test Room", "Host Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	tests := []struct {
		name         string
		password     string
		playerName   string
		isSpectator  bool
		shouldSucceed bool
	}{
		{
			name:         "Correct password - player",
			password:     room.Password,
			playerName:   "Valid Player",
			isSpectator:  false,
			shouldSucceed: true,
		},
		{
			name:         "Correct password - spectator",
			password:     room.Password,
			playerName:   "Valid Spectator",
			isSpectator:  true,
			shouldSucceed: true,
		},
		{
			name:         "Incorrect password",
			password:     "WRONG",
			playerName:   "Invalid Player",
			isSpectator:  false,
			shouldSucceed: false,
		},
		{
			name:         "Empty password",
			password:     "",
			playerName:   "Empty Password Player",
			isSpectator:  false,
			shouldSucceed: false,
		},
		{
			name:         "Lowercase password (should succeed - case insensitive)",
			password:     strings.ToLower(room.Password),
			playerName:   "Lowercase Player",
			isSpectator:  false,
			shouldSucceed: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			player, err := server.AddPlayerToGame(room, tt.playerName, tt.password, tt.isSpectator)

			if tt.shouldSucceed {
				if err != nil {
					t.Errorf("Expected success but got error: %v", err)
				}
				if player == nil {
					t.Errorf("Expected player to be created")
				}
				if player != nil && player.IsSpectator != tt.isSpectator {
					t.Errorf("Player spectator status = %v, want %v", player.IsSpectator, tt.isSpectator)
				}
			} else {
				if err == nil {
					t.Errorf("Expected error but got success")
				}
				if player != nil {
					t.Errorf("Expected no player to be created")
				}
			}
		})
	}
}

// TestTrainingRoomSpeedMultiplierFunctionality tests speed control
func TestTrainingRoomSpeedMultiplierFunctionality(t *testing.T) {
	// Create training room
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   1.0,
		HeadlessMode:      false,
		TrainingMode:      true,
		SessionID:         "speed-test-session",
		DirectStateAccess: true,
	}

	room, _, err := server.NewTrainingGameWithPlayer("Speed Test Room", "Test Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}

	tests := []struct {
		name           string
		speedMultiplier float64
		shouldSucceed   bool
	}{
		{
			name:           "Valid speed - 1x",
			speedMultiplier: 1.0,
			shouldSucceed:   true,
		},
		{
			name:           "Valid speed - 10x",
			speedMultiplier: 10.0,
			shouldSucceed:   true,
		},
		{
			name:           "Valid speed - 100x (maximum)",
			speedMultiplier: 100.0,
			shouldSucceed:   true,
		},
		{
			name:           "Valid speed - 0.1x (minimum)",
			speedMultiplier: 0.1,
			shouldSucceed:   true,
		},
		{
			name:           "Invalid speed - 0x",
			speedMultiplier: 0.0,
			shouldSucceed:   false,
		},
		{
			name:           "Invalid speed - negative",
			speedMultiplier: -1.0,
			shouldSucceed:   false,
		},
		{
			name:           "Invalid speed - too high",
			speedMultiplier: 101.0,
			shouldSucceed:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := room.SetSpeedMultiplier(tt.speedMultiplier)

			if tt.shouldSucceed {
				if err != nil {
					t.Errorf("Expected success but got error: %v", err)
				}
				if room.GetSpeedMultiplier() != tt.speedMultiplier {
					t.Errorf("Speed multiplier = %v, want %v", room.GetSpeedMultiplier(), tt.speedMultiplier)
				}
				// Verify tick rate was updated
				expectedTickRate := time.Duration(float64(20*time.Millisecond) / tt.speedMultiplier)
				if room.GetCustomTickRate() != expectedTickRate {
					t.Errorf("Custom tick rate = %v, want %v", room.GetCustomTickRate(), expectedTickRate)
				}
			} else {
				if err == nil {
					t.Errorf("Expected error but got success")
				}
			}
		})
	}

	// Test speed control on non-training room
	normalRoom, _, err := server.NewGameWithPlayer("Normal Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create normal room: %v", err)
	}

	err = normalRoom.SetSpeedMultiplier(10.0)
	if err == nil {
		t.Errorf("Expected error when setting speed on non-training room")
	}
}

// TestPlayerTokenValidation tests player token validation
func TestPlayerTokenValidation(t *testing.T) {
	room, player, err := server.NewGameWithPlayer("Token Test Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Add another player
	player2, err := server.AddPlayerToGame(room, "Player 2", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add second player: %v", err)
	}

	tests := []struct {
		name        string
		token       string
		shouldValid bool
	}{
		{
			name:        "Valid token - player 1",
			token:       player.Token,
			shouldValid: true,
		},
		{
			name:        "Valid token - player 2",
			token:       player2.Token,
			shouldValid: true,
		},
		{
			name:        "Invalid token",
			token:       "invalid-token",
			shouldValid: false,
		},
		{
			name:        "Empty token",
			token:       "",
			shouldValid: false,
		},
		{
			name:        "Random UUID token",
			token:       uuid.New().String(),
			shouldValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isValid := room.ValidatePlayerToken(tt.token)
			if isValid != tt.shouldValid {
				t.Errorf("Token validation = %v, want %v", isValid, tt.shouldValid)
			}
		})
	}

	// Test token validation after player removal
	room.RemovePlayer(player.ID)
	if room.ValidatePlayerToken(player.Token) {
		t.Errorf("Token should be invalid after player removal")
	}

	// Player 2 token should still be valid
	if !room.ValidatePlayerToken(player2.Token) {
		t.Errorf("Player 2 token should still be valid")
	}
}

// TestTrainingRoomConfiguration tests training room configuration changes
func TestTrainingRoomConfiguration(t *testing.T) {
	// Create training room
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   1.0,
		HeadlessMode:      false,
		TrainingMode:      true,
		SessionID:         "config-test-session",
		DirectStateAccess: false,
	}

	room, _, err := server.NewTrainingGameWithPlayer("Config Test Room", "Test Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}

	// Test configuration changes
	tests := []struct {
		name              string
		trainingMode      string
		speedMultiplier   float64
		directStateAccess bool
		shouldSucceed     bool
	}{
		{
			name:              "Enable headless mode",
			trainingMode:      "headless",
			speedMultiplier:   10.0,
			directStateAccess: true,
			shouldSucceed:     true,
		},
		{
			name:              "Disable headless mode",
			trainingMode:      "normal",
			speedMultiplier:   5.0,
			directStateAccess: false,
			shouldSucceed:     true,
		},
		{
			name:              "Invalid speed multiplier",
			trainingMode:      "normal",
			speedMultiplier:   150.0,
			directStateAccess: true,
			shouldSucceed:     false,
		},
		{
			name:              "Zero speed multiplier (should use existing)",
			trainingMode:      "headless",
			speedMultiplier:   0.0,
			directStateAccess: true,
			shouldSucceed:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalSpeed := room.GetSpeedMultiplier()
			
			err := room.ConfigureTraining(tt.trainingMode, tt.speedMultiplier, tt.directStateAccess)

			if tt.shouldSucceed {
				if err != nil {
					t.Errorf("Expected success but got error: %v", err)
				}

				// Verify headless mode
				expectedHeadless := (tt.trainingMode == "headless")
				if room.IsHeadlessMode() != expectedHeadless {
					t.Errorf("Headless mode = %v, want %v", room.IsHeadlessMode(), expectedHeadless)
				}

				// Verify direct state access
				if room.DirectStateAccess != tt.directStateAccess {
					t.Errorf("Direct state access = %v, want %v", room.DirectStateAccess, tt.directStateAccess)
				}

				// Verify speed multiplier
				if tt.speedMultiplier > 0 {
					if room.GetSpeedMultiplier() != tt.speedMultiplier {
						t.Errorf("Speed multiplier = %v, want %v", room.GetSpeedMultiplier(), tt.speedMultiplier)
					}
				} else {
					// Should keep original speed if 0 was provided
					if room.GetSpeedMultiplier() != originalSpeed {
						t.Errorf("Speed multiplier should remain %v when 0 provided, got %v", originalSpeed, room.GetSpeedMultiplier())
					}
				}
			} else {
				if err == nil {
					t.Errorf("Expected error but got success")
				}
			}
		})
	}

	// Test configuration on non-training room
	normalRoom, _, err := server.NewGameWithPlayer("Normal Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create normal room: %v", err)
	}

	err = normalRoom.ConfigureTraining("headless", 10.0, true)
	if err == nil {
		t.Errorf("Expected error when configuring non-training room")
	}
}

// TestRoomManagerIntegration tests room management with RoomManager
func TestRoomManagerIntegration(t *testing.T) {
	manager := server.NewRoomManager()

	// Create multiple rooms
	room1, _, err := server.NewGameWithPlayer("Room 1", "Player 1", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room 1: %v", err)
	}

	room2, _, err := server.NewTrainingGameWithPlayer("Training Room", "Player 2", game_maps.MapDefault, server.TrainingConfig{
		SpeedMultiplier:   5.0,
		HeadlessMode:      true,
		TrainingMode:      true,
		SessionID:         "manager-test-session",
		DirectStateAccess: true,
	})
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}

	// Add rooms to manager
	manager.AddGameRoom(room1)
	manager.AddGameRoom(room2)

	// Test room retrieval by ID
	retrievedRoom1, exists := manager.GetGameRoom(room1.ID)
	if !exists {
		t.Errorf("Room 1 should exist in manager")
	}
	if retrievedRoom1.ID != room1.ID {
		t.Errorf("Retrieved room ID = %v, want %v", retrievedRoom1.ID, room1.ID)
	}

	// Test room retrieval by code
	retrievedRoom2, exists := manager.GetGameRoomByCode(room2.RoomCode)
	if !exists {
		t.Errorf("Training room should exist in manager")
	}
	if retrievedRoom2.ID != room2.ID {
		t.Errorf("Retrieved room ID = %v, want %v", retrievedRoom2.ID, room2.ID)
	}

	// Test case-insensitive room code lookup
	lowerCaseCode := strings.ToLower(room1.RoomCode)
	retrievedRoom3, exists := manager.GetGameRoomByCode(lowerCaseCode)
	if !exists {
		t.Errorf("Room should be found with lowercase code")
	}
	if retrievedRoom3.ID != room1.ID {
		t.Errorf("Retrieved room ID = %v, want %v", retrievedRoom3.ID, room1.ID)
	}

	// Test player count retrieval
	count1, exists := manager.GetNumberOfConnectedPlayers(room1.ID)
	if !exists {
		t.Errorf("Should be able to get player count for room 1")
	}
	if count1 != 1 {
		t.Errorf("Room 1 player count = %d, want 1", count1)
	}

	// Test room removal
	manager.RemoveGameRoom(room1.ID)
	_, exists = manager.GetGameRoom(room1.ID)
	if exists {
		t.Errorf("Room 1 should not exist after removal")
	}

	// Room 2 should still exist
	_, exists = manager.GetGameRoom(room2.ID)
	if !exists {
		t.Errorf("Room 2 should still exist after room 1 removal")
	}

	// Test getting all room IDs
	roomIDs := manager.GetGameRoomIDs()
	if len(roomIDs) != 1 {
		t.Errorf("Expected 1 room ID, got %d", len(roomIDs))
	}
	if roomIDs[0] != room2.ID {
		t.Errorf("Room ID = %v, want %v", roomIDs[0], room2.ID)
	}
}