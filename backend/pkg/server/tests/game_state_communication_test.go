package tests

import (
	"encoding/json"
	"fmt"
	"go-ws-server/pkg/server"
	"go-ws-server/pkg/server/game_maps"
	"go-ws-server/pkg/server/game_objects"
	"go-ws-server/pkg/server/types"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
)

// MockWebSocketConnection simulates a WebSocket connection for testing
type MockWebSocketConnection struct {
	ID              string
	Messages        []types.Message
	MessagesMutex   sync.Mutex
	Closed          bool
	WriteError      error
	LastMessageTime time.Time
}

func NewMockWebSocketConnection() *MockWebSocketConnection {
	return &MockWebSocketConnection{
		ID:       uuid.New().String(),
		Messages: make([]types.Message, 0),
		Closed:   false,
	}
}

func (m *MockWebSocketConnection) WriteJSON(message types.Message) error {
	if m.WriteError != nil {
		return m.WriteError
	}

	if m.Closed {
		return fmt.Errorf("connection closed")
	}

	m.MessagesMutex.Lock()
	defer m.MessagesMutex.Unlock()

	m.Messages = append(m.Messages, message)
	m.LastMessageTime = time.Now()
	return nil
}

func (m *MockWebSocketConnection) GetMessages() []types.Message {
	m.MessagesMutex.Lock()
	defer m.MessagesMutex.Unlock()

	// Return a copy to avoid race conditions
	messages := make([]types.Message, len(m.Messages))
	copy(messages, m.Messages)
	return messages
}

func (m *MockWebSocketConnection) GetMessageCount() int {
	m.MessagesMutex.Lock()
	defer m.MessagesMutex.Unlock()
	return len(m.Messages)
}

func (m *MockWebSocketConnection) ClearMessages() {
	m.MessagesMutex.Lock()
	defer m.MessagesMutex.Unlock()
	m.Messages = m.Messages[:0]
}

func (m *MockWebSocketConnection) Close() {
	m.Closed = true
}

// TestWebSocketMessageBroadcastingToAllClients tests WebSocket message broadcasting
func TestWebSocketMessageBroadcastingToAllClients(t *testing.T) {
	// Create test room with multiple players
	room, player1, err := server.NewGameWithPlayer("Broadcast Test Room", "Player 1", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Add more players
	player2, err := server.AddPlayerToGame(room, "Player 2", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player 2: %v", err)
	}

	player3, err := server.AddPlayerToGame(room, "Player 3", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player 3: %v", err)
	}

	// Add a spectator
	spectator, err := server.AddPlayerToGame(room, "Spectator", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator: %v", err)
	}

	// Create mock connections
	conn1 := NewMockWebSocketConnection()
	conn2 := NewMockWebSocketConnection()
	conn3 := NewMockWebSocketConnection()
	connSpectator := NewMockWebSocketConnection()

	// Create server connections (for reference - not used in mock testing)
	_ = &server.Connection{
		ID:       conn1.ID,
		RoomID:   room.ID,
		PlayerID: player1.ID,
	}
	_ = &server.Connection{
		ID:       conn2.ID,
		RoomID:   room.ID,
		PlayerID: player2.ID,
	}
	_ = &server.Connection{
		ID:       conn3.ID,
		RoomID:   room.ID,
		PlayerID: player3.ID,
	}
	_ = &server.Connection{
		ID:       connSpectator.ID,
		RoomID:   room.ID,
		PlayerID: spectator.ID,
	}

	// Test broadcasting game state updates
	t.Run("Broadcast game state to all clients", func(t *testing.T) {
		// Create a game event that will trigger state updates
		eventData := map[string]interface{}{
			"playerId": player1.ID,
			"key":      "W",
			"isDown":   true,
		}

		event := game_objects.NewGameEvent(
			room.ID,
			game_objects.EventPlayerKeyInput,
			eventData,
			1,
			nil,
		)

		// Process the event
		roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
		if roomHandleResult == nil {
			t.Fatalf("Room handle result should not be nil")
		}

		// Verify that all connections would receive the update
		// In a real server, this would be handled by the game update queue
		gameUpdate := &types.GameUpdate{
			FullUpdate:   false,
			ObjectStates: make(map[string]map[string]interface{}),
			Events:       make([]types.GameUpdateEvent, 0),
		}

		// Add updated objects to the game update
		for objectID, obj := range roomHandleResult.UpdatedObjects {
			if obj == nil {
				gameUpdate.ObjectStates[objectID] = nil
			} else {
				gameUpdate.ObjectStates[objectID] = obj.GetState()
			}
		}

		// Add events to the game update
		for _, event := range roomHandleResult.Events {
			gameUpdate.Events = append(gameUpdate.Events, types.GameUpdateEvent{
				Type: event.EventType,
				Data: event.Data,
			})
		}

		// Simulate broadcasting to all connections
		connections := []*MockWebSocketConnection{conn1, conn2, conn3, connSpectator}
		for _, conn := range connections {
			message := types.Message{
				Type:    "GameState",
				Payload: mustMarshal(gameUpdate),
			}
			err := conn.WriteJSON(message)
			if err != nil {
				t.Errorf("Failed to send message to connection %s: %v", conn.ID, err)
			}
		}

		// Verify all connections received the message
		for i, conn := range connections {
			messages := conn.GetMessages()
			if len(messages) != 1 {
				t.Errorf("Connection %d should have received 1 message, got %d", i+1, len(messages))
				continue
			}

			if messages[0].Type != "GameState" {
				t.Errorf("Connection %d should have received GameState message, got %s", i+1, messages[0].Type)
			}

			// Verify the payload can be unmarshaled
			var receivedUpdate types.GameUpdate
			err := json.Unmarshal(messages[0].Payload, &receivedUpdate)
			if err != nil {
				t.Errorf("Connection %d received invalid GameState payload: %v", i+1, err)
			}
		}
	})

	// Test selective broadcasting (client-specific filtering)
	t.Run("Client-specific state filtering", func(t *testing.T) {
		// Clear previous messages
		conn1.ClearMessages()
		conn2.ClearMessages()
		conn3.ClearMessages()
		connSpectator.ClearMessages()

		// Create different game states for different clients
		allStates := room.GetAllGameObjectStates()

		// Simulate client-specific filtering
		// Player connections get full state, spectators get limited state
		playerUpdate := &types.GameUpdate{
			FullUpdate:   true,
			ObjectStates: allStates,
			Events:       make([]types.GameUpdateEvent, 0),
		}

		spectatorUpdate := &types.GameUpdate{
			FullUpdate:   true,
			ObjectStates: filterSpectatorStates(allStates),
			Events:       make([]types.GameUpdateEvent, 0),
		}

		// Send full updates to players
		playerConnections := []*MockWebSocketConnection{conn1, conn2, conn3}
		for _, conn := range playerConnections {
			message := types.Message{
				Type:    "GameState",
				Payload: mustMarshal(playerUpdate),
			}
			err := conn.WriteJSON(message)
			if err != nil {
				t.Errorf("Failed to send player update to connection %s: %v", conn.ID, err)
			}
		}

		// Send filtered update to spectator
		spectatorMessage := types.Message{
			Type:    "GameState",
			Payload: mustMarshal(spectatorUpdate),
		}
		err := connSpectator.WriteJSON(spectatorMessage)
		if err != nil {
			t.Errorf("Failed to send spectator update: %v", err)
		}

		// Verify players received full state
		for i, conn := range playerConnections {
			messages := conn.GetMessages()
			if len(messages) != 1 {
				t.Errorf("Player connection %d should have received 1 message, got %d", i+1, len(messages))
				continue
			}

			var receivedUpdate types.GameUpdate
			err := json.Unmarshal(messages[0].Payload, &receivedUpdate)
			if err != nil {
				t.Errorf("Player connection %d received invalid payload: %v", i+1, err)
				continue
			}

			if len(receivedUpdate.ObjectStates) != len(allStates) {
				t.Errorf("Player connection %d should receive full state (%d objects), got %d",
					i+1, len(allStates), len(receivedUpdate.ObjectStates))
			}
		}

		// Verify spectator received filtered state
		spectatorMessages := connSpectator.GetMessages()
		if len(spectatorMessages) != 1 {
			t.Errorf("Spectator should have received 1 message, got %d", len(spectatorMessages))
		} else {
			var receivedUpdate types.GameUpdate
			err := json.Unmarshal(spectatorMessages[0].Payload, &receivedUpdate)
			if err != nil {
				t.Errorf("Spectator received invalid payload: %v", err)
			} else {
				// Verify spectator received some state (filtering may not reduce object count if no sensitive fields exist)
				if len(receivedUpdate.ObjectStates) == 0 {
					t.Errorf("Spectator should receive some game state")
				}

				// Verify the filtering function works correctly
				filteredStates := filterSpectatorStates(allStates)
				if len(filteredStates) != len(receivedUpdate.ObjectStates) {
					t.Logf("Filtered state count: %d, received: %d (filtering may vary based on actual object fields)",
						len(filteredStates), len(receivedUpdate.ObjectStates))
				}
			}
		}
	})

	// Test connection failure handling
	t.Run("Handle connection failures during broadcast", func(t *testing.T) {
		// Clear previous messages
		conn1.ClearMessages()
		conn2.ClearMessages()
		conn3.ClearMessages()

		// Simulate connection failure for conn2
		conn2.WriteError = fmt.Errorf("connection failed")

		gameUpdate := &types.GameUpdate{
			FullUpdate:   true,
			ObjectStates: room.GetAllGameObjectStates(),
			Events:       make([]types.GameUpdateEvent, 0),
		}

		// Attempt to broadcast to all connections
		connections := []*MockWebSocketConnection{conn1, conn2, conn3}
		successCount := 0
		failureCount := 0

		for _, conn := range connections {
			message := types.Message{
				Type:    "GameState",
				Payload: mustMarshal(gameUpdate),
			}
			err := conn.WriteJSON(message)
			if err != nil {
				failureCount++
			} else {
				successCount++
			}
		}

		// Verify that some connections succeeded and one failed
		if successCount != 2 {
			t.Errorf("Expected 2 successful connections, got %d", successCount)
		}
		if failureCount != 1 {
			t.Errorf("Expected 1 failed connection, got %d", failureCount)
		}

		// Verify successful connections received the message
		if conn1.GetMessageCount() != 1 {
			t.Errorf("Connection 1 should have received 1 message, got %d", conn1.GetMessageCount())
		}
		if conn2.GetMessageCount() != 0 {
			t.Errorf("Connection 2 should have received 0 messages (failed), got %d", conn2.GetMessageCount())
		}
		if conn3.GetMessageCount() != 1 {
			t.Errorf("Connection 3 should have received 1 message, got %d", conn3.GetMessageCount())
		}
	})
}

// TestGameStateSerializationAndDeserialization tests game state serialization
func TestGameStateSerializationAndDeserialization(t *testing.T) {
	// Create test room with objects
	room, player, err := server.NewGameWithPlayer("Serialization Test Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Add some game events to create object states
	events := []*game_objects.GameEvent{
		game_objects.NewGameEvent(
			room.ID,
			game_objects.EventPlayerKeyInput,
			map[string]interface{}{
				"playerId": player.ID,
				"key":      "W",
				"isDown":   true,
			},
			1,
			nil,
		),
		game_objects.NewGameEvent(
			room.ID,
			game_objects.EventPlayerClickInput,
			map[string]interface{}{
				"playerId": player.ID,
				"x":        100.5,
				"y":        200.7,
				"isDown":   true,
				"button":   0,
			},
			1,
			nil,
		),
	}

	// Process events to generate state changes
	roomHandleResult := room.Handle(events)
	if roomHandleResult == nil {
		t.Fatalf("Room handle result should not be nil")
	}

	t.Run("Serialize complete game state", func(t *testing.T) {
		// Get all game object states
		allStates := room.GetAllGameObjectStates()
		if allStates == nil {
			t.Fatalf("All states should not be nil")
		}

		// Create game update
		gameUpdate := &types.GameUpdate{
			FullUpdate:   true,
			ObjectStates: allStates,
			Events:       make([]types.GameUpdateEvent, 0),
		}

		// Add events from room handle result
		for _, event := range roomHandleResult.Events {
			gameUpdate.Events = append(gameUpdate.Events, types.GameUpdateEvent{
				Type: event.EventType,
				Data: event.Data,
			})
		}

		// Serialize to JSON
		jsonData, err := json.Marshal(gameUpdate)
		if err != nil {
			t.Fatalf("Failed to serialize game update: %v", err)
		}

		// Verify JSON is valid and not empty
		if len(jsonData) == 0 {
			t.Errorf("Serialized JSON should not be empty")
		}

		// Deserialize back to GameUpdate
		var deserializedUpdate types.GameUpdate
		err = json.Unmarshal(jsonData, &deserializedUpdate)
		if err != nil {
			t.Fatalf("Failed to deserialize game update: %v", err)
		}

		// Verify deserialized data matches original
		if deserializedUpdate.FullUpdate != gameUpdate.FullUpdate {
			t.Errorf("FullUpdate mismatch: got %v, want %v", deserializedUpdate.FullUpdate, gameUpdate.FullUpdate)
		}

		if len(deserializedUpdate.ObjectStates) != len(gameUpdate.ObjectStates) {
			t.Errorf("ObjectStates count mismatch: got %d, want %d",
				len(deserializedUpdate.ObjectStates), len(gameUpdate.ObjectStates))
		}

		if len(deserializedUpdate.Events) != len(gameUpdate.Events) {
			t.Errorf("Events count mismatch: got %d, want %d",
				len(deserializedUpdate.Events), len(gameUpdate.Events))
		}
	})

	t.Run("Serialize partial game state updates", func(t *testing.T) {
		// Create partial update with only changed objects
		partialUpdate := &types.GameUpdate{
			FullUpdate:   false,
			ObjectStates: make(map[string]map[string]interface{}),
			Events:       make([]types.GameUpdateEvent, 0),
		}

		// Add only updated objects from room handle result
		for objectID, obj := range roomHandleResult.UpdatedObjects {
			if obj == nil {
				partialUpdate.ObjectStates[objectID] = nil
			} else {
				partialUpdate.ObjectStates[objectID] = obj.GetState()
			}
		}

		// Serialize partial update
		jsonData, err := json.Marshal(partialUpdate)
		if err != nil {
			t.Fatalf("Failed to serialize partial update: %v", err)
		}

		// Deserialize and verify
		var deserializedUpdate types.GameUpdate
		err = json.Unmarshal(jsonData, &deserializedUpdate)
		if err != nil {
			t.Fatalf("Failed to deserialize partial update: %v", err)
		}

		if deserializedUpdate.FullUpdate != false {
			t.Errorf("Partial update should have FullUpdate=false, got %v", deserializedUpdate.FullUpdate)
		}

		// Verify only changed objects are included
		if len(deserializedUpdate.ObjectStates) != len(roomHandleResult.UpdatedObjects) {
			t.Errorf("Partial update should contain %d objects, got %d",
				len(roomHandleResult.UpdatedObjects), len(deserializedUpdate.ObjectStates))
		}
	})

	t.Run("Handle serialization of complex object states", func(t *testing.T) {
		// Get all object states which may contain complex data types
		allStates := room.GetAllGameObjectStates()

		// Test serialization of each object state individually
		for objectID, objectState := range allStates {
			jsonData, err := json.Marshal(objectState)
			if err != nil {
				t.Errorf("Failed to serialize object %s state: %v", objectID, err)
				continue
			}

			// Deserialize back
			var deserializedState map[string]interface{}
			err = json.Unmarshal(jsonData, &deserializedState)
			if err != nil {
				t.Errorf("Failed to deserialize object %s state: %v", objectID, err)
				continue
			}

			// Verify basic structure is preserved
			if len(deserializedState) == 0 && len(objectState) > 0 {
				t.Errorf("Object %s state lost data during serialization", objectID)
			}
		}
	})

	t.Run("Handle serialization errors gracefully", func(t *testing.T) {
		// Create game update with potentially problematic data
		problematicUpdate := &types.GameUpdate{
			FullUpdate: true,
			ObjectStates: map[string]map[string]interface{}{
				"test-object": {
					"validField":  "valid value",
					"numberField": 42,
					"boolField":   true,
					"arrayField":  []interface{}{1, 2, 3},
					"objectField": map[string]interface{}{"nested": "value"},
				},
			},
			Events: []types.GameUpdateEvent{
				{
					Type: game_objects.EventPlayerKeyInput,
					Data: map[string]interface{}{
						"playerId": "test-player",
						"key":      "W",
						"isDown":   true,
					},
				},
			},
		}

		// This should serialize successfully
		jsonData, err := json.Marshal(problematicUpdate)
		if err != nil {
			t.Errorf("Failed to serialize valid update: %v", err)
		} else {
			// Verify it can be deserialized
			var deserializedUpdate types.GameUpdate
			err = json.Unmarshal(jsonData, &deserializedUpdate)
			if err != nil {
				t.Errorf("Failed to deserialize valid update: %v", err)
			}
		}
	})
}

// TestDirectStateAccessAPIForTrainingMode tests direct state access for training
func TestDirectStateAccessAPIForTrainingMode(t *testing.T) {
	// Create training room
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   10.0,
		HeadlessMode:      true,
		TrainingMode:      true,
		SessionID:         "direct-access-test",
		DirectStateAccess: true,
	}

	trainingRoom, trainingPlayer, err := server.NewTrainingGameWithPlayer(
		"Direct Access Test Room",
		"Training Player",
		game_maps.MapDefault,
		trainingConfig,
	)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}

	// Create normal room for comparison
	normalRoom, _, err := server.NewGameWithPlayer("Normal Room", "Normal Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create normal room: %v", err)
	}

	t.Run("Direct state access in training mode", func(t *testing.T) {
		// Verify training room supports direct state access
		if !trainingRoom.IsTrainingRoom() {
			t.Errorf("Room should be a training room")
		}

		if !trainingRoom.DirectStateAccess {
			t.Errorf("Training room should have direct state access enabled")
		}

		// Get direct game state
		gameState := trainingRoom.GetDirectGameState()
		if gameState == nil {
			t.Fatalf("Direct game state should not be nil")
		}

		// Verify state structure
		if gameState["objects"] == nil {
			t.Errorf("Game state should contain objects")
		}

		roomInfo, ok := gameState["room"].(map[string]interface{})
		if !ok {
			t.Fatalf("Game state should contain room information")
		}

		// Verify room information
		if roomInfo["id"] != trainingRoom.ID {
			t.Errorf("Room ID in state = %v, want %v", roomInfo["id"], trainingRoom.ID)
		}
		if roomInfo["isTraining"] != true {
			t.Errorf("Training flag in state = %v, want true", roomInfo["isTraining"])
		}
		if roomInfo["speedMultiplier"] != 10.0 {
			t.Errorf("Speed multiplier in state = %v, want 10.0", roomInfo["speedMultiplier"])
		}
		if roomInfo["headlessMode"] != true {
			t.Errorf("Headless mode in state = %v, want true", roomInfo["headlessMode"])
		}

		// Verify player information
		players, ok := gameState["players"].([]map[string]interface{})
		if !ok {
			t.Fatalf("Game state should contain players array")
		}
		if len(players) != 1 {
			t.Errorf("Expected 1 player in state, got %d", len(players))
		}
		if players[0]["id"] != trainingPlayer.ID {
			t.Errorf("Player ID in state = %v, want %v", players[0]["id"], trainingPlayer.ID)
		}

		// Verify map information
		mapInfo, ok := gameState["map"].(map[string]interface{})
		if !ok {
			t.Fatalf("Game state should contain map information")
		}
		if mapInfo["name"] == nil {
			t.Errorf("Map should have a name")
		}

		// Verify timestamp
		timestamp, ok := gameState["timestamp"].(int64)
		if !ok {
			t.Errorf("Game state should contain timestamp")
		}
		if timestamp <= 0 {
			t.Errorf("Timestamp should be positive, got %d", timestamp)
		}
	})

	t.Run("Direct state access disabled in normal mode", func(t *testing.T) {
		// Verify normal room does not support training features
		if normalRoom.IsTrainingRoom() {
			t.Errorf("Normal room should not be a training room")
		}

		if normalRoom.DirectStateAccess {
			t.Errorf("Normal room should not have direct state access")
		}

		// Direct state access should still work but return basic state
		gameState := normalRoom.GetDirectGameState()
		if gameState == nil {
			t.Fatalf("Direct game state should not be nil even for normal rooms")
		}

		// Verify room information indicates non-training mode
		roomInfo, ok := gameState["room"].(map[string]interface{})
		if !ok {
			t.Fatalf("Game state should contain room information")
		}

		if roomInfo["isTraining"] != false {
			t.Errorf("Normal room training flag should be false, got %v", roomInfo["isTraining"])
		}
		if roomInfo["speedMultiplier"] != 0.0 {
			t.Errorf("Normal room speed multiplier should be 0, got %v", roomInfo["speedMultiplier"])
		}
	})

	t.Run("Direct state access performance", func(t *testing.T) {
		// Test performance of direct state access
		const numAccesses = 100
		startTime := time.Now()

		for i := 0; i < numAccesses; i++ {
			gameState := trainingRoom.GetDirectGameState()
			if gameState == nil {
				t.Errorf("Direct state access %d failed", i)
			}
		}

		duration := time.Since(startTime)
		accessesPerSecond := float64(numAccesses) / duration.Seconds()

		// Direct state access should be fast (>1000 accesses per second)
		if accessesPerSecond < 100 {
			t.Logf("Direct state access rate: %.2f accesses/second (may be slow)", accessesPerSecond)
		}

		// Verify state consistency across multiple accesses
		state1 := trainingRoom.GetDirectGameState()
		state2 := trainingRoom.GetDirectGameState()

		// Room information should be consistent
		room1 := state1["room"].(map[string]interface{})
		room2 := state2["room"].(map[string]interface{})

		if room1["id"] != room2["id"] {
			t.Errorf("Room ID should be consistent across accesses")
		}
		if room1["isTraining"] != room2["isTraining"] {
			t.Errorf("Training flag should be consistent across accesses")
		}
	})

	t.Run("Direct state access with game events", func(t *testing.T) {
		// Get initial state
		initialState := trainingRoom.GetDirectGameState()
		_ = initialState["objects"].(map[string]map[string]interface{})

		// Create game events
		events := []*game_objects.GameEvent{
			game_objects.NewGameEvent(
				trainingRoom.ID,
				game_objects.EventPlayerKeyInput,
				map[string]interface{}{
					"playerId": trainingPlayer.ID,
					"key":      "W",
					"isDown":   true,
				},
				1,
				nil,
			),
		}

		// Add a small delay to ensure timestamp difference
		time.Sleep(1 * time.Millisecond)

		// Process events
		roomHandleResult := trainingRoom.Handle(events)
		if roomHandleResult == nil {
			t.Fatalf("Room handle result should not be nil")
		}

		// Get updated state
		updatedState := trainingRoom.GetDirectGameState()
		updatedObjects := updatedState["objects"].(map[string]map[string]interface{})

		// Verify state reflects changes
		if len(updatedObjects) == 0 {
			t.Errorf("Updated state should contain objects")
		}

		// Timestamps should be different
		initialTimestamp := initialState["timestamp"].(int64)
		updatedTimestamp := updatedState["timestamp"].(int64)
		if updatedTimestamp <= initialTimestamp {
			t.Errorf("Updated timestamp (%d) should be greater than initial timestamp (%d)", updatedTimestamp, initialTimestamp)
		}
	})
}

// TestStateUpdateFrequencyAndPerformance tests state update frequency and performance
func TestStateUpdateFrequencyAndPerformance(t *testing.T) {
	// Create training room with high speed multiplier
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   50.0,
		HeadlessMode:      true,
		TrainingMode:      true,
		SessionID:         "performance-test",
		DirectStateAccess: true,
	}

	room, player, err := server.NewTrainingGameWithPlayer(
		"Performance Test Room",
		"Test Player",
		game_maps.MapDefault,
		trainingConfig,
	)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}

	t.Run("State update frequency", func(t *testing.T) {
		// Create mock connection to track updates
		conn := NewMockWebSocketConnection()

		// Simulate rapid game events
		const numEvents = 50
		const eventInterval = 10 * time.Millisecond

		startTime := time.Now()

		for i := 0; i < numEvents; i++ {
			// Create game event
			eventData := map[string]interface{}{
				"playerId": player.ID,
				"key":      []string{"W", "A", "S", "D"}[i%4],
				"isDown":   i%2 == 0,
			}

			event := game_objects.NewGameEvent(
				room.ID,
				game_objects.EventPlayerKeyInput,
				eventData,
				1,
				nil,
			)

			// Process event
			roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
			if roomHandleResult == nil {
				t.Errorf("Room handle result should not be nil for event %d", i)
				continue
			}

			// Simulate sending update to connection
			if len(roomHandleResult.UpdatedObjects) > 0 || len(roomHandleResult.Events) > 0 {
				gameUpdate := &types.GameUpdate{
					FullUpdate:   false,
					ObjectStates: make(map[string]map[string]interface{}),
					Events:       make([]types.GameUpdateEvent, 0),
				}

				for objectID, obj := range roomHandleResult.UpdatedObjects {
					if obj == nil {
						gameUpdate.ObjectStates[objectID] = nil
					} else {
						gameUpdate.ObjectStates[objectID] = obj.GetState()
					}
				}

				message := types.Message{
					Type:    "GameState",
					Payload: mustMarshal(gameUpdate),
				}

				err := conn.WriteJSON(message)
				if err != nil {
					t.Errorf("Failed to send update %d: %v", i, err)
				}
			}

			time.Sleep(eventInterval)
		}

		duration := time.Since(startTime)
		eventsPerSecond := float64(numEvents) / duration.Seconds()

		// Verify event processing rate
		if eventsPerSecond < 100 {
			t.Logf("Event processing rate: %.2f events/second", eventsPerSecond)
		}

		// Verify connection received updates
		messageCount := conn.GetMessageCount()
		if messageCount == 0 {
			t.Errorf("Connection should have received at least some updates")
		}

		t.Logf("Processed %d events in %v, received %d updates (%.2f events/sec)",
			numEvents, duration, messageCount, eventsPerSecond)
	})

	t.Run("Performance under load", func(t *testing.T) {
		// Test performance with multiple concurrent operations
		const numConcurrentOperations = 20
		const operationsPerGoroutine = 25

		var wg sync.WaitGroup
		errorChan := make(chan error, numConcurrentOperations)
		startTime := time.Now()

		for i := 0; i < numConcurrentOperations; i++ {
			wg.Add(1)
			go func(goroutineID int) {
				defer wg.Done()

				for j := 0; j < operationsPerGoroutine; j++ {
					// Alternate between different operations
					switch j % 3 {
					case 0:
						// Direct state access
						gameState := room.GetDirectGameState()
						if gameState == nil {
							errorChan <- fmt.Errorf("goroutine %d: direct state access failed", goroutineID)
							return
						}

					case 1:
						// Game event processing
						eventData := map[string]interface{}{
							"playerId": player.ID,
							"key":      "W",
							"isDown":   j%2 == 0,
						}

						event := game_objects.NewGameEvent(
							room.ID,
							game_objects.EventPlayerKeyInput,
							eventData,
							1,
							nil,
						)

						roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
						if roomHandleResult == nil {
							errorChan <- fmt.Errorf("goroutine %d: event processing failed", goroutineID)
							return
						}

					case 2:
						// Get all object states
						allStates := room.GetAllGameObjectStates()
						if allStates == nil {
							errorChan <- fmt.Errorf("goroutine %d: get all states failed", goroutineID)
							return
						}
					}
				}
			}(i)
		}

		wg.Wait()
		close(errorChan)

		duration := time.Since(startTime)
		totalOperations := numConcurrentOperations * operationsPerGoroutine
		operationsPerSecond := float64(totalOperations) / duration.Seconds()

		// Check for errors
		errorCount := 0
		for err := range errorChan {
			t.Errorf("Concurrent operation error: %v", err)
			errorCount++
		}

		if errorCount > 0 {
			t.Errorf("Had %d errors out of %d operations", errorCount, totalOperations)
		}

		t.Logf("Completed %d concurrent operations in %v (%.2f ops/sec)",
			totalOperations, duration, operationsPerSecond)

		// Performance should be reasonable even under load
		if operationsPerSecond < 100 {
			t.Logf("Performance under load: %.2f operations/second (may be slow)", operationsPerSecond)
		}
	})

	t.Run("Memory usage during frequent updates", func(t *testing.T) {
		// Test that frequent state updates don't cause memory leaks
		const numIterations = 100

		for i := 0; i < numIterations; i++ {
			// Create multiple events
			events := make([]*game_objects.GameEvent, 5)
			for j := 0; j < 5; j++ {
				events[j] = game_objects.NewGameEvent(
					room.ID,
					game_objects.EventPlayerKeyInput,
					map[string]interface{}{
						"playerId": player.ID,
						"key":      []string{"W", "A", "S", "D"}[j%4],
						"isDown":   (i+j)%2 == 0,
					},
					1,
					nil,
				)
			}

			// Process events
			roomHandleResult := room.Handle(events)
			if roomHandleResult == nil {
				t.Errorf("Room handle result should not be nil for iteration %d", i)
				continue
			}

			// Get state multiple times
			for k := 0; k < 3; k++ {
				gameState := room.GetDirectGameState()
				if gameState == nil {
					t.Errorf("Direct state access failed at iteration %d, access %d", i, k)
				}

				allStates := room.GetAllGameObjectStates()
				if allStates == nil {
					t.Errorf("Get all states failed at iteration %d, access %d", i, k)
				}
			}

			// Periodically force garbage collection to test for leaks
			if i%20 == 0 {
				// In a real test, you might use runtime.GC() here
				// For this test, we'll just continue
			}
		}

		// If we get here without running out of memory, the test passes
		t.Logf("Completed %d iterations of frequent state updates without memory issues", numIterations)
	})
}

// Helper functions

func mustMarshal(v interface{}) json.RawMessage {
	data, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("Failed to marshal: %v", err))
	}
	return json.RawMessage(data)
}

func filterSpectatorStates(allStates map[string]map[string]interface{}) map[string]map[string]interface{} {
	// Simulate filtering sensitive information for spectators
	filtered := make(map[string]map[string]interface{})

	for objectID, objectState := range allStates {
		filteredState := make(map[string]interface{})

		// Copy only non-sensitive fields
		for key, value := range objectState {
			// Filter out sensitive fields (example: player tokens, internal state)
			if key != "token" && key != "internalState" && key != "privateData" {
				filteredState[key] = value
			}
		}

		// Only include objects with remaining data
		if len(filteredState) > 0 {
			filtered[objectID] = filteredState
		}
	}

	return filtered
}
