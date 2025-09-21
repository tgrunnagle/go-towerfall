package tests

import (
	"fmt"
	"go-ws-server/pkg/server"
	"go-ws-server/pkg/server/game_maps"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
)

// TestAddMultiplePlayersToRoom tests adding multiple players and spectators to rooms
func TestAddMultiplePlayersToRoom(t *testing.T) {
	// Create a room
	room, hostPlayer, err := server.NewGameWithPlayer("Multi-Player Test Room", "Host Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Verify initial state
	if room.GetNumberOfConnectedPlayers() != 1 {
		t.Errorf("Expected 1 player initially, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Test adding regular players
	players := make([]*server.ConnectedPlayer, 0)
	for i := 2; i <= 5; i++ {
		playerName := fmt.Sprintf("Player %d", i)
		player, err := server.AddPlayerToGame(room, playerName, room.Password, false)
		if err != nil {
			t.Fatalf("Failed to add player %s: %v", playerName, err)
		}
		players = append(players, player)

		// Verify player was added
		if room.GetNumberOfConnectedPlayers() != i {
			t.Errorf("Expected %d players after adding %s, got %d", i, playerName, room.GetNumberOfConnectedPlayers())
		}

		// Verify player properties
		if player.Name != playerName {
			t.Errorf("Player name = %s, want %s", player.Name, playerName)
		}
		if player.IsSpectator {
			t.Errorf("Regular player should not be spectator")
		}
		if player.Token == "" {
			t.Errorf("Player token should not be empty")
		}

		// Verify player can be retrieved
		retrievedPlayer, exists := room.GetPlayer(player.ID)
		if !exists {
			t.Errorf("Player %s should exist in room", player.ID)
		}
		if retrievedPlayer.ID != player.ID {
			t.Errorf("Retrieved player ID = %s, want %s", retrievedPlayer.ID, player.ID)
		}
	}

	// Test adding spectators
	spectators := make([]*server.ConnectedPlayer, 0)
	for i := 1; i <= 3; i++ {
		spectatorName := fmt.Sprintf("Spectator %d", i)
		spectator, err := server.AddPlayerToGame(room, spectatorName, room.Password, true)
		if err != nil {
			t.Fatalf("Failed to add spectator %s: %v", spectatorName, err)
		}
		spectators = append(spectators, spectator)

		expectedTotal := 5 + i // 5 players + i spectators
		if room.GetNumberOfConnectedPlayers() != expectedTotal {
			t.Errorf("Expected %d total connections after adding spectator %s, got %d", 
				expectedTotal, spectatorName, room.GetNumberOfConnectedPlayers())
		}

		// Verify spectator properties
		if spectator.Name != spectatorName {
			t.Errorf("Spectator name = %s, want %s", spectator.Name, spectatorName)
		}
		if !spectator.IsSpectator {
			t.Errorf("Spectator should have IsSpectator = true")
		}
		if spectator.Token == "" {
			t.Errorf("Spectator token should not be empty")
		}
	}

	// Verify spectator list
	spectatorNames := room.GetSpectators()
	if len(spectatorNames) != 3 {
		t.Errorf("Expected 3 spectators, got %d", len(spectatorNames))
	}

	// Check that all expected spectators are present (order doesn't matter)
	expectedSpectatorNames := []string{"Spectator 1", "Spectator 2", "Spectator 3"}
	spectatorNameMap := make(map[string]bool)
	for _, name := range spectatorNames {
		spectatorNameMap[name] = true
	}

	for _, expectedName := range expectedSpectatorNames {
		if !spectatorNameMap[expectedName] {
			t.Errorf("Expected spectator %s not found in spectator list", expectedName)
		}
	}

	// Test adding duplicate player (should fail)
	duplicatePlayer, err := server.AddPlayerToGame(room, "Duplicate Player", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to create duplicate player: %v", err)
	}

	// Try to add the same player again (should fail)
	success := room.AddPlayer(duplicatePlayer.ID, duplicatePlayer)
	if success {
		t.Errorf("Adding duplicate player should fail")
	}

	// Verify total count hasn't changed
	if room.GetNumberOfConnectedPlayers() != 9 { // 5 players + 3 spectators + 1 duplicate = 9
		t.Errorf("Expected 9 total connections, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Test token uniqueness
	allTokens := make(map[string]bool)
	allTokens[hostPlayer.Token] = true
	
	for _, player := range players {
		if allTokens[player.Token] {
			t.Errorf("Duplicate token found: %s", player.Token)
		}
		allTokens[player.Token] = true
	}
	
	for _, spectator := range spectators {
		if allTokens[spectator.Token] {
			t.Errorf("Duplicate token found: %s", spectator.Token)
		}
		allTokens[spectator.Token] = true
	}
}

// TestPlayerRemovalAndDisconnectionHandling tests player removal scenarios
func TestPlayerRemovalAndDisconnectionHandling(t *testing.T) {
	// Create room with multiple players and spectators
	room, hostPlayer, err := server.NewGameWithPlayer("Removal Test Room", "Host Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Add regular players
	player1, err := server.AddPlayerToGame(room, "Player 1", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player 1: %v", err)
	}

	player2, err := server.AddPlayerToGame(room, "Player 2", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player 2: %v", err)
	}

	// Add spectators
	spectator1, err := server.AddPlayerToGame(room, "Spectator 1", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator 1: %v", err)
	}

	spectator2, err := server.AddPlayerToGame(room, "Spectator 2", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator 2: %v", err)
	}

	// Verify initial state
	if room.GetNumberOfConnectedPlayers() != 5 {
		t.Errorf("Expected 5 total connections initially, got %d", room.GetNumberOfConnectedPlayers())
	}

	spectatorNames := room.GetSpectators()
	if len(spectatorNames) != 2 {
		t.Errorf("Expected 2 spectators initially, got %d", len(spectatorNames))
	}

	// Test removing regular player
	room.RemovePlayer(player1.ID)
	if room.GetNumberOfConnectedPlayers() != 4 {
		t.Errorf("Expected 4 connections after removing player 1, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Verify player is no longer retrievable
	_, exists := room.GetPlayer(player1.ID)
	if exists {
		t.Errorf("Player 1 should not exist after removal")
	}

	// Verify token is no longer valid
	if room.ValidatePlayerToken(player1.Token) {
		t.Errorf("Player 1 token should be invalid after removal")
	}

	// Test removing spectator
	room.RemovePlayer(spectator1.ID)
	if room.GetNumberOfConnectedPlayers() != 3 {
		t.Errorf("Expected 3 connections after removing spectator 1, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Verify spectator list is updated
	spectatorNames = room.GetSpectators()
	if len(spectatorNames) != 1 {
		t.Errorf("Expected 1 spectator after removal, got %d", len(spectatorNames))
	}
	if spectatorNames[0] != "Spectator 2" {
		t.Errorf("Remaining spectator name = %s, want 'Spectator 2'", spectatorNames[0])
	}

	// Test removing host player
	room.RemovePlayer(hostPlayer.ID)
	if room.GetNumberOfConnectedPlayers() != 2 {
		t.Errorf("Expected 2 connections after removing host, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Verify remaining players are still valid
	_, exists = room.GetPlayer(player2.ID)
	if !exists {
		t.Errorf("Player 2 should still exist")
	}

	_, exists = room.GetPlayer(spectator2.ID)
	if !exists {
		t.Errorf("Spectator 2 should still exist")
	}

	// Test removing non-existent player (should not crash)
	nonExistentID := uuid.New().String()
	room.RemovePlayer(nonExistentID) // Should not panic or error

	// Verify count unchanged
	if room.GetNumberOfConnectedPlayers() != 2 {
		t.Errorf("Expected 2 connections after removing non-existent player, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Test removing all remaining players
	room.RemovePlayer(player2.ID)
	room.RemovePlayer(spectator2.ID)

	if room.GetNumberOfConnectedPlayers() != 0 {
		t.Errorf("Expected 0 connections after removing all players, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Verify spectator list is empty
	spectatorNames = room.GetSpectators()
	if len(spectatorNames) != 0 {
		t.Errorf("Expected 0 spectators after removing all, got %d", len(spectatorNames))
	}
}

// TestPlayerTokenValidationAndAuthentication tests token validation scenarios
func TestPlayerTokenValidationAndAuthentication(t *testing.T) {
	// Create room with players
	room, hostPlayer, err := server.NewGameWithPlayer("Token Test Room", "Host Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Add more players
	player1, err := server.AddPlayerToGame(room, "Player 1", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player 1: %v", err)
	}

	spectator1, err := server.AddPlayerToGame(room, "Spectator 1", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator 1: %v", err)
	}

	// Test valid token validation
	validTokens := []struct {
		name   string
		token  string
		player *server.ConnectedPlayer
	}{
		{"Host player token", hostPlayer.Token, hostPlayer},
		{"Regular player token", player1.Token, player1},
		{"Spectator token", spectator1.Token, spectator1},
	}

	for _, tt := range validTokens {
		t.Run(tt.name, func(t *testing.T) {
			if !room.ValidatePlayerToken(tt.token) {
				t.Errorf("Token %s should be valid", tt.token)
			}
		})
	}

	// Test invalid token validation
	invalidTokens := []struct {
		name  string
		token string
	}{
		{"Empty token", ""},
		{"Random UUID", uuid.New().String()},
		{"Invalid string", "invalid-token-123"},
		{"Partial token", hostPlayer.Token[:10]},
		{"Modified token", hostPlayer.Token + "x"},
	}

	for _, tt := range invalidTokens {
		t.Run(tt.name, func(t *testing.T) {
			if room.ValidatePlayerToken(tt.token) {
				t.Errorf("Token %s should be invalid", tt.token)
			}
		})
	}

	// Test token validation after player removal
	originalToken := player1.Token
	room.RemovePlayer(player1.ID)

	if room.ValidatePlayerToken(originalToken) {
		t.Errorf("Token should be invalid after player removal")
	}

	// Test that other tokens are still valid
	if !room.ValidatePlayerToken(hostPlayer.Token) {
		t.Errorf("Host token should still be valid after other player removal")
	}

	if !room.ValidatePlayerToken(spectator1.Token) {
		t.Errorf("Spectator token should still be valid after other player removal")
	}

	// Test token uniqueness across multiple rooms
	room2, hostPlayer2, err := server.NewGameWithPlayer("Token Test Room 2", "Host Player 2", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create second room: %v", err)
	}

	// Tokens from different rooms should not be valid in other rooms
	if room.ValidatePlayerToken(hostPlayer2.Token) {
		t.Errorf("Token from room2 should not be valid in room1")
	}

	if room2.ValidatePlayerToken(hostPlayer.Token) {
		t.Errorf("Token from room1 should not be valid in room2")
	}

	// Test case sensitivity of tokens (tokens should be case sensitive)
	upperToken := strings.ToUpper(hostPlayer.Token)
	if upperToken != hostPlayer.Token && room.ValidatePlayerToken(upperToken) {
		t.Errorf("Token validation should be case sensitive")
	}
}

// TestPlayerStateSynchronization tests player state consistency
func TestPlayerStateSynchronization(t *testing.T) {
	// Create training room for direct state access
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   1.0,
		HeadlessMode:      false,
		TrainingMode:      true,
		SessionID:         "sync-test-session",
		DirectStateAccess: true,
	}

	room, hostPlayer, err := server.NewTrainingGameWithPlayer("Sync Test Room", "Host Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}

	// Add players and spectators
	player1, err := server.AddPlayerToGame(room, "Player 1", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player 1: %v", err)
	}

	spectator1, err := server.AddPlayerToGame(room, "Spectator 1", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator 1: %v", err)
	}

	// Test direct game state access
	gameState := room.GetDirectGameState()
	if gameState == nil {
		t.Fatalf("Game state should not be nil")
	}

	// Verify player information in game state
	players, ok := gameState["players"].([]map[string]interface{})
	if !ok {
		t.Fatalf("Game state should contain players array")
	}

	if len(players) != 3 {
		t.Errorf("Expected 3 players in game state, got %d", len(players))
	}

	// Create map of players by ID for easier verification
	playerMap := make(map[string]map[string]interface{})
	for _, p := range players {
		id, ok := p["id"].(string)
		if !ok {
			t.Errorf("Player should have string ID")
			continue
		}
		playerMap[id] = p
	}

	// Verify each player's state
	expectedPlayers := []*server.ConnectedPlayer{hostPlayer, player1, spectator1}
	for _, expectedPlayer := range expectedPlayers {
		playerState, exists := playerMap[expectedPlayer.ID]
		if !exists {
			t.Errorf("Player %s should exist in game state", expectedPlayer.ID)
			continue
		}

		// Verify player properties
		if playerState["name"] != expectedPlayer.Name {
			t.Errorf("Player %s name in state = %v, want %s", 
				expectedPlayer.ID, playerState["name"], expectedPlayer.Name)
		}

		if playerState["isSpectator"] != expectedPlayer.IsSpectator {
			t.Errorf("Player %s spectator status in state = %v, want %v", 
				expectedPlayer.ID, playerState["isSpectator"], expectedPlayer.IsSpectator)
		}

		// Token should not be exposed in game state for security
		if _, hasToken := playerState["token"]; hasToken {
			t.Errorf("Player token should not be exposed in game state")
		}
	}

	// Test state consistency after player changes
	_, err = server.AddPlayerToGame(room, "Player 2", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player 2: %v", err)
	}

	// Get updated state
	updatedGameState := room.GetDirectGameState()
	updatedPlayers, ok := updatedGameState["players"].([]map[string]interface{})
	if !ok {
		t.Fatalf("Updated game state should contain players array")
	}

	if len(updatedPlayers) != 4 {
		t.Errorf("Expected 4 players in updated game state, got %d", len(updatedPlayers))
	}

	// Remove a player and verify state update
	room.RemovePlayer(player1.ID)

	finalGameState := room.GetDirectGameState()
	finalPlayers, ok := finalGameState["players"].([]map[string]interface{})
	if !ok {
		t.Fatalf("Final game state should contain players array")
	}

	if len(finalPlayers) != 3 {
		t.Errorf("Expected 3 players in final game state, got %d", len(finalPlayers))
	}

	// Verify removed player is not in state
	for _, p := range finalPlayers {
		if p["id"] == player1.ID {
			t.Errorf("Removed player should not be in game state")
		}
	}

	// Test regular game object states
	allStates := room.GetAllGameObjectStates()
	if allStates == nil {
		t.Errorf("All game object states should not be nil")
	}

	// Verify room information in game state
	roomInfo, ok := finalGameState["room"].(map[string]interface{})
	if !ok {
		t.Fatalf("Game state should contain room information")
	}

	expectedRoomFields := map[string]interface{}{
		"id":              room.ID,
		"name":            room.Name,
		"roomCode":        room.RoomCode,
		"isTraining":      true,
		"speedMultiplier": 1.0,
		"headlessMode":    false,
	}

	for field, expectedValue := range expectedRoomFields {
		if roomInfo[field] != expectedValue {
			t.Errorf("Room %s = %v, want %v", field, roomInfo[field], expectedValue)
		}
	}
}

// TestSpectatorModeFunctionality tests spectator-specific features
func TestSpectatorModeFunctionality(t *testing.T) {
	// Create room
	room, hostPlayer, err := server.NewGameWithPlayer("Spectator Test Room", "Host Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Add regular players
	player1, err := server.AddPlayerToGame(room, "Player 1", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player 1: %v", err)
	}

	player2, err := server.AddPlayerToGame(room, "Player 2", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player 2: %v", err)
	}

	// Add spectators
	spectator1, err := server.AddPlayerToGame(room, "Spectator 1", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator 1: %v", err)
	}

	spectator2, err := server.AddPlayerToGame(room, "Spectator 2", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator 2: %v", err)
	}

	spectator3, err := server.AddPlayerToGame(room, "Spectator 3", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator 3: %v", err)
	}

	// Test spectator list functionality
	spectators := room.GetSpectators()
	expectedSpectators := []string{"Spectator 1", "Spectator 2", "Spectator 3"}

	if len(spectators) != len(expectedSpectators) {
		t.Errorf("Expected %d spectators, got %d", len(expectedSpectators), len(spectators))
	}

	// Check that all expected spectators are present (order doesn't matter)
	spectatorMap := make(map[string]bool)
	for _, name := range spectators {
		spectatorMap[name] = true
	}

	for _, expectedName := range expectedSpectators {
		if !spectatorMap[expectedName] {
			t.Errorf("Expected spectator %s not found in spectator list", expectedName)
		}
	}

	// Test spectator properties
	spectatorPlayers := []*server.ConnectedPlayer{spectator1, spectator2, spectator3}
	for _, spectator := range spectatorPlayers {
		// Verify spectator flag
		if !spectator.IsSpectator {
			t.Errorf("Spectator %s should have IsSpectator = true", spectator.Name)
		}

		// Verify spectator can be retrieved
		retrievedSpectator, exists := room.GetPlayer(spectator.ID)
		if !exists {
			t.Errorf("Spectator %s should be retrievable", spectator.ID)
		}

		if !retrievedSpectator.IsSpectator {
			t.Errorf("Retrieved spectator %s should have IsSpectator = true", spectator.ID)
		}

		// Verify spectator token validation
		if !room.ValidatePlayerToken(spectator.Token) {
			t.Errorf("Spectator token should be valid")
		}
	}

	// Test removing spectators
	room.RemovePlayer(spectator2.ID)

	updatedSpectators := room.GetSpectators()
	expectedAfterRemoval := []string{"Spectator 1", "Spectator 3"}

	if len(updatedSpectators) != len(expectedAfterRemoval) {
		t.Errorf("Expected %d spectators after removal, got %d", len(expectedAfterRemoval), len(updatedSpectators))
	}

	// Check that expected spectators are present after removal (order doesn't matter)
	updatedSpectatorMap := make(map[string]bool)
	for _, name := range updatedSpectators {
		updatedSpectatorMap[name] = true
	}

	for _, expectedName := range expectedAfterRemoval {
		if !updatedSpectatorMap[expectedName] {
			t.Errorf("Expected spectator %s not found after removal", expectedName)
		}
	}

	// Verify removed spectator is not present
	if updatedSpectatorMap["Spectator 2"] {
		t.Errorf("Removed spectator 'Spectator 2' should not be in list")
	}

	// Test that regular players are not in spectator list
	allSpectators := room.GetSpectators()
	regularPlayerNames := []string{hostPlayer.Name, player1.Name, player2.Name}

	for _, playerName := range regularPlayerNames {
		for _, spectatorName := range allSpectators {
			if playerName == spectatorName {
				t.Errorf("Regular player %s should not be in spectator list", playerName)
			}
		}
	}

	// Test mixed removal (players and spectators)
	room.RemovePlayer(player1.ID)
	room.RemovePlayer(spectator1.ID)

	finalSpectators := room.GetSpectators()
	if len(finalSpectators) != 1 {
		t.Errorf("Expected 1 spectator after mixed removal, got %d", len(finalSpectators))
	}

	if finalSpectators[0] != "Spectator 3" {
		t.Errorf("Remaining spectator = %s, want 'Spectator 3'", finalSpectators[0])
	}

	// Verify total player count
	if room.GetNumberOfConnectedPlayers() != 3 { // host + player2 + spectator3
		t.Errorf("Expected 3 total connections after mixed removal, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Test adding spectator with same name as regular player (should be allowed)
	duplicateNameSpectator, err := server.AddPlayerToGame(room, hostPlayer.Name, room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator with duplicate name: %v", err)
	}

	if !duplicateNameSpectator.IsSpectator {
		t.Errorf("Duplicate name player should be spectator")
	}

	// Verify both players exist
	_, hostExists := room.GetPlayer(hostPlayer.ID)
	_, spectatorExists := room.GetPlayer(duplicateNameSpectator.ID)

	if !hostExists {
		t.Errorf("Original host player should still exist")
	}

	if !spectatorExists {
		t.Errorf("Duplicate name spectator should exist")
	}

	// Test spectator list includes duplicate name
	finalSpectatorList := room.GetSpectators()
	duplicateNameFound := false
	for _, name := range finalSpectatorList {
		if name == hostPlayer.Name {
			duplicateNameFound = true
			break
		}
	}

	if !duplicateNameFound {
		t.Errorf("Spectator list should include duplicate name spectator")
	}
}

// TestPlayerManagementInTrainingRooms tests player management in training rooms
func TestPlayerManagementInTrainingRooms(t *testing.T) {
	// Create training room
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   5.0,
		HeadlessMode:      true,
		TrainingMode:      true,
		SessionID:         "training-player-test",
		DirectStateAccess: true,
	}

	room, hostPlayer, err := server.NewTrainingGameWithPlayer("Training Player Test", "Host Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}

	// Test adding players to training room
	player1, err := server.AddPlayerToGame(room, "Training Player 1", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player to training room: %v", err)
	}

	spectator1, err := server.AddPlayerToGame(room, "Training Spectator 1", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator to training room: %v", err)
	}

	// Verify training room properties are maintained
	if !room.IsTrainingRoom() {
		t.Errorf("Room should remain a training room after adding players")
	}

	if room.GetSpeedMultiplier() != 5.0 {
		t.Errorf("Speed multiplier should remain 5.0, got %f", room.GetSpeedMultiplier())
	}

	if !room.IsHeadlessMode() {
		t.Errorf("Headless mode should remain enabled")
	}

	// Test direct state access with players
	gameState := room.GetDirectGameState()
	if gameState == nil {
		t.Fatalf("Training room should provide direct game state")
	}

	// Verify training-specific state information
	roomInfo, ok := gameState["room"].(map[string]interface{})
	if !ok {
		t.Fatalf("Game state should contain room information")
	}

	if roomInfo["isTraining"] != true {
		t.Errorf("Room state should indicate training mode")
	}

	if roomInfo["speedMultiplier"] != 5.0 {
		t.Errorf("Room state speed multiplier = %v, want 5.0", roomInfo["speedMultiplier"])
	}

	if roomInfo["headlessMode"] != true {
		t.Errorf("Room state should indicate headless mode")
	}

	// Test player token validation in training room
	if !room.ValidatePlayerToken(hostPlayer.Token) {
		t.Errorf("Host token should be valid in training room")
	}

	if !room.ValidatePlayerToken(player1.Token) {
		t.Errorf("Player token should be valid in training room")
	}

	if !room.ValidatePlayerToken(spectator1.Token) {
		t.Errorf("Spectator token should be valid in training room")
	}

	// Test player removal in training room
	room.RemovePlayer(player1.ID)

	if room.GetNumberOfConnectedPlayers() != 2 {
		t.Errorf("Expected 2 players after removal in training room, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Verify training properties are still intact
	if !room.IsTrainingRoom() {
		t.Errorf("Room should remain training room after player removal")
	}

	// Test spectator functionality in training room
	spectators := room.GetSpectators()
	if len(spectators) != 1 {
		t.Errorf("Expected 1 spectator in training room, got %d", len(spectators))
	}

	if spectators[0] != "Training Spectator 1" {
		t.Errorf("Spectator name = %s, want 'Training Spectator 1'", spectators[0])
	}
}

// TestConcurrentPlayerOperations tests thread safety of player operations
func TestConcurrentPlayerOperations(t *testing.T) {
	// Create room
	room, _, err := server.NewGameWithPlayer("Concurrent Test Room", "Host Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Test concurrent player additions
	const numConcurrentPlayers = 10
	playerChan := make(chan *server.ConnectedPlayer, numConcurrentPlayers)
	errorChan := make(chan error, numConcurrentPlayers)

	// Add players concurrently
	for i := 0; i < numConcurrentPlayers; i++ {
		go func(index int) {
			playerName := fmt.Sprintf("Concurrent Player %d", index)
			player, err := server.AddPlayerToGame(room, playerName, room.Password, index%2 == 0) // Alternate between players and spectators
			if err != nil {
				errorChan <- err
				return
			}
			playerChan <- player
		}(i)
	}

	// Collect results
	players := make([]*server.ConnectedPlayer, 0, numConcurrentPlayers)
	for i := 0; i < numConcurrentPlayers; i++ {
		select {
		case player := <-playerChan:
			players = append(players, player)
		case err := <-errorChan:
			t.Errorf("Concurrent player addition failed: %v", err)
		case <-time.After(5 * time.Second):
			t.Fatalf("Timeout waiting for concurrent player addition")
		}
	}

	// Verify all players were added
	expectedTotal := numConcurrentPlayers + 1 // +1 for host player
	if room.GetNumberOfConnectedPlayers() != expectedTotal {
		t.Errorf("Expected %d total players after concurrent addition, got %d", 
			expectedTotal, room.GetNumberOfConnectedPlayers())
	}

	// Test concurrent token validation
	validationChan := make(chan bool, len(players))
	for _, player := range players {
		go func(p *server.ConnectedPlayer) {
			validationChan <- room.ValidatePlayerToken(p.Token)
		}(player)
	}

	// Verify all tokens are valid
	for i := 0; i < len(players); i++ {
		select {
		case isValid := <-validationChan:
			if !isValid {
				t.Errorf("Player token should be valid in concurrent test")
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("Timeout waiting for concurrent token validation")
		}
	}

	// Test concurrent player removal
	removalChan := make(chan string, len(players))
	for _, player := range players {
		go func(p *server.ConnectedPlayer) {
			room.RemovePlayer(p.ID)
			removalChan <- p.ID
		}(player)
	}

	// Wait for all removals to complete
	for i := 0; i < len(players); i++ {
		select {
		case <-removalChan:
			// Player removed successfully
		case <-time.After(5 * time.Second):
			t.Fatalf("Timeout waiting for concurrent player removal")
		}
	}

	// Verify only host player remains
	if room.GetNumberOfConnectedPlayers() != 1 {
		t.Errorf("Expected 1 player after concurrent removal, got %d", room.GetNumberOfConnectedPlayers())
	}

	// Verify spectator list is empty
	spectators := room.GetSpectators()
	if len(spectators) != 0 {
		t.Errorf("Expected 0 spectators after concurrent removal, got %d", len(spectators))
	}
}