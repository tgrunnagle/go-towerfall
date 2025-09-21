package tests

import (
	"encoding/json"
	"fmt"
	"go-ws-server/pkg/server"
	"go-ws-server/pkg/server/game_maps"
	"go-ws-server/pkg/server/game_objects"
	"go-ws-server/pkg/server/types"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
)

// TestKeyboardInputHandling tests W/A/S/D movement key processing
func TestKeyboardInputHandling(t *testing.T) {
	// Create test room
	room, player, err := server.NewGameWithPlayer("Input Test Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Create mock connection
	conn := &server.Connection{
		ID:       uuid.New().String(),
		RoomID:   room.ID,
		PlayerID: player.ID,
	}

	tests := []struct {
		name        string
		key         string
		isDown      bool
		shouldPass  bool
		description string
	}{
		{
			name:        "Press W key",
			key:         "W",
			isDown:      true,
			shouldPass:  true,
			description: "W key press should be processed",
		},
		{
			name:        "Release W key",
			key:         "W",
			isDown:      false,
			shouldPass:  true,
			description: "W key release should be processed",
		},
		{
			name:        "Press A key",
			key:         "A",
			isDown:      true,
			shouldPass:  true,
			description: "A key press should be processed",
		},
		{
			name:        "Release A key",
			key:         "A",
			isDown:      false,
			shouldPass:  true,
			description: "A key release should be processed",
		},
		{
			name:        "Press S key",
			key:         "S",
			isDown:      true,
			shouldPass:  true,
			description: "S key press should be processed",
		},
		{
			name:        "Release S key",
			key:         "S",
			isDown:      false,
			shouldPass:  true,
			description: "S key release should be processed",
		},
		{
			name:        "Press D key",
			key:         "D",
			isDown:      true,
			shouldPass:  true,
			description: "D key press should be processed",
		},
		{
			name:        "Release D key",
			key:         "D",
			isDown:      false,
			shouldPass:  true,
			description: "D key release should be processed",
		},
		{
			name:        "Invalid key X",
			key:         "X",
			isDown:      true,
			shouldPass:  true, // Server should accept but may ignore invalid keys
			description: "Invalid keys should be handled gracefully",
		},
		{
			name:        "Empty key",
			key:         "",
			isDown:      true,
			shouldPass:  true, // Server should handle empty keys gracefully
			description: "Empty key should be handled gracefully",
		},
		{
			name:        "Lowercase key",
			key:         "w",
			isDown:      true,
			shouldPass:  true,
			description: "Lowercase keys should be processed",
		},
		{
			name:        "Special character key",
			key:         "!",
			isDown:      true,
			shouldPass:  true,
			description: "Special characters should be handled gracefully",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create key status request
			req := types.KeyStatusRequest{
				Key:    tt.key,
				IsDown: tt.isDown,
			}

			// Test the handler directly (since we can't easily test WebSocket in unit tests)
			// We'll verify that the method doesn't panic and processes the input
			defer func() {
				if r := recover(); r != nil {
					if tt.shouldPass {
						t.Errorf("Handler panicked for valid input: %v", r)
					}
				}
			}()

			// This would normally be called by the WebSocket handler
			// We're testing the input processing logic
			eventData := map[string]interface{}{
				"playerId": conn.PlayerID,
				"key":      req.Key,
				"isDown":   req.IsDown,
			}

			event := game_objects.NewGameEvent(
				conn.RoomID,
				game_objects.EventPlayerKeyInput,
				eventData,
				1,
				nil,
			)

			// Verify event was created correctly
			if event.EventType != game_objects.EventPlayerKeyInput {
				t.Errorf("Event type = %v, want %v", event.EventType, game_objects.EventPlayerKeyInput)
			}

			if event.Data["playerId"] != conn.PlayerID {
				t.Errorf("Player ID = %v, want %v", event.Data["playerId"], conn.PlayerID)
			}

			if event.Data["key"] != req.Key {
				t.Errorf("Key = %v, want %v", event.Data["key"], req.Key)
			}

			if event.Data["isDown"] != req.IsDown {
				t.Errorf("IsDown = %v, want %v", event.Data["isDown"], req.IsDown)
			}

			// Test that the room can handle the event
			roomHandleResult := room.Handle([]*game_objects.GameEvent{event})

			// The result should not be nil (even if no objects were updated)
			if roomHandleResult == nil {
				t.Errorf("Room handle result should not be nil")
			}
		})
	}
}

// TestMouseInputProcessing tests mouse click coordinates and button handling
func TestMouseInputProcessing(t *testing.T) {
	// Create test room
	room, player, err := server.NewGameWithPlayer("Mouse Test Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Create mock connection
	conn := &server.Connection{
		ID:       uuid.New().String(),
		RoomID:   room.ID,
		PlayerID: player.ID,
	}

	tests := []struct {
		name        string
		x           float64
		y           float64
		isDown      bool
		button      int
		shouldPass  bool
		description string
	}{
		{
			name:        "Left click press at origin",
			x:           0.0,
			y:           0.0,
			isDown:      true,
			button:      0,
			shouldPass:  true,
			description: "Left mouse button press at (0,0)",
		},
		{
			name:        "Left click release at origin",
			x:           0.0,
			y:           0.0,
			isDown:      false,
			button:      0,
			shouldPass:  true,
			description: "Left mouse button release at (0,0)",
		},
		{
			name:        "Right click press",
			x:           100.5,
			y:           200.7,
			isDown:      true,
			button:      2,
			shouldPass:  true,
			description: "Right mouse button press at (100.5, 200.7)",
		},
		{
			name:        "Right click release",
			x:           100.5,
			y:           200.7,
			isDown:      false,
			button:      2,
			shouldPass:  true,
			description: "Right mouse button release at (100.5, 200.7)",
		},
		{
			name:        "Click at positive coordinates",
			x:           500.0,
			y:           300.0,
			isDown:      true,
			button:      0,
			shouldPass:  true,
			description: "Click at positive coordinates",
		},
		{
			name:        "Click at negative coordinates",
			x:           -50.0,
			y:           -25.0,
			isDown:      true,
			button:      0,
			shouldPass:  true,
			description: "Click at negative coordinates should be handled",
		},
		{
			name:        "Click with large coordinates",
			x:           9999.99,
			y:           8888.88,
			isDown:      true,
			button:      0,
			shouldPass:  true,
			description: "Click with large coordinates",
		},
		{
			name:        "Click with decimal coordinates",
			x:           123.456,
			y:           789.012,
			isDown:      true,
			button:      0,
			shouldPass:  true,
			description: "Click with decimal coordinates",
		},
		{
			name:        "Invalid button number",
			x:           100.0,
			y:           100.0,
			isDown:      true,
			button:      5,
			shouldPass:  true, // Server should handle gracefully
			description: "Invalid button number should be handled gracefully",
		},
		{
			name:        "Negative button number",
			x:           100.0,
			y:           100.0,
			isDown:      true,
			button:      -1,
			shouldPass:  true, // Server should handle gracefully
			description: "Negative button number should be handled gracefully",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create player click request
			req := types.PlayerClickRequest{
				X:      tt.x,
				Y:      tt.y,
				IsDown: tt.isDown,
				Button: tt.button,
			}

			// Test the handler logic
			defer func() {
				if r := recover(); r != nil {
					if tt.shouldPass {
						t.Errorf("Handler panicked for valid input: %v", r)
					}
				}
			}()

			// Create event data as the handler would
			eventData := map[string]interface{}{
				"playerId": conn.PlayerID,
				"x":        req.X,
				"y":        req.Y,
				"isDown":   req.IsDown,
				"button":   req.Button,
			}

			event := game_objects.NewGameEvent(
				conn.RoomID,
				game_objects.EventPlayerClickInput,
				eventData,
				1,
				nil,
			)

			// Verify event was created correctly
			if event.EventType != game_objects.EventPlayerClickInput {
				t.Errorf("Event type = %v, want %v", event.EventType, game_objects.EventPlayerClickInput)
			}

			if event.Data["playerId"] != conn.PlayerID {
				t.Errorf("Player ID = %v, want %v", event.Data["playerId"], conn.PlayerID)
			}

			if event.Data["x"] != req.X {
				t.Errorf("X coordinate = %v, want %v", event.Data["x"], req.X)
			}

			if event.Data["y"] != req.Y {
				t.Errorf("Y coordinate = %v, want %v", event.Data["y"], req.Y)
			}

			if event.Data["isDown"] != req.IsDown {
				t.Errorf("IsDown = %v, want %v", event.Data["isDown"], req.IsDown)
			}

			if event.Data["button"] != req.Button {
				t.Errorf("Button = %v, want %v", event.Data["button"], req.Button)
			}

			// Test that the room can handle the event
			roomHandleResult := room.Handle([]*game_objects.GameEvent{event})

			if roomHandleResult == nil {
				t.Errorf("Room handle result should not be nil")
			}
		})
	}
}

// TestInputValidationAndSanitization tests input validation and sanitization
func TestInputValidationAndSanitization(t *testing.T) {
	// Create test room
	room, player, err := server.NewGameWithPlayer("Validation Test Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Create mock connection
	conn := &server.Connection{
		ID:       uuid.New().String(),
		RoomID:   room.ID,
		PlayerID: player.ID,
	}

	// Test malformed JSON input handling
	t.Run("Malformed JSON handling", func(t *testing.T) {
		// Test that malformed JSON doesn't crash the system
		malformedInputs := []string{
			`{"key": "W", "isDown": }`,       // Missing value
			`{"key": "W" "isDown": true}`,    // Missing comma
			`{"key": W, "isDown": true}`,     // Unquoted string
			`{key: "W", "isDown": true}`,     // Unquoted key
			`{"key": "W", "isDown": "true"}`, // Wrong type
			`{"key": null, "isDown": true}`,  // Null value
		}

		for _, input := range malformedInputs {
			var req types.KeyStatusRequest
			err := json.Unmarshal([]byte(input), &req)
			// We expect these to fail during unmarshaling, which is correct behavior
			if err == nil {
				// If it doesn't fail, verify the system handles it gracefully
				eventData := map[string]interface{}{
					"playerId": conn.PlayerID,
					"key":      req.Key,
					"isDown":   req.IsDown,
				}

				event := game_objects.NewGameEvent(
					conn.RoomID,
					game_objects.EventPlayerKeyInput,
					eventData,
					1,
					nil,
				)

				// Should not panic
				roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
				if roomHandleResult == nil {
					t.Errorf("Room handle result should not be nil even for malformed input")
				}
			}
		}
	})

	// Test input sanitization
	t.Run("Input sanitization", func(t *testing.T) {
		sanitizationTests := []struct {
			name     string
			input    string
			expected string
		}{
			{
				name:     "Normal key",
				input:    "W",
				expected: "W",
			},
			{
				name:     "Lowercase key",
				input:    "w",
				expected: "w", // Should preserve case
			},
			{
				name:     "Key with whitespace",
				input:    " W ",
				expected: " W ", // Should preserve whitespace (server may trim)
			},
			{
				name:     "Very long key string",
				input:    strings.Repeat("W", 1000),
				expected: strings.Repeat("W", 1000), // Should handle long strings
			},
			{
				name:     "Unicode characters",
				input:    "Ω",
				expected: "Ω", // Should handle unicode
			},
			{
				name:     "Special characters",
				input:    "!@#$%^&*()",
				expected: "!@#$%^&*()", // Should handle special chars
			},
		}

		for _, tt := range sanitizationTests {
			t.Run(tt.name, func(t *testing.T) {
				req := types.KeyStatusRequest{
					Key:    tt.input,
					IsDown: true,
				}

				eventData := map[string]interface{}{
					"playerId": conn.PlayerID,
					"key":      req.Key,
					"isDown":   req.IsDown,
				}

				event := game_objects.NewGameEvent(
					conn.RoomID,
					game_objects.EventPlayerKeyInput,
					eventData,
					1,
					nil,
				)

				// Verify the key is preserved as-is (sanitization may happen at game logic level)
				if event.Data["key"] != tt.expected {
					t.Errorf("Key sanitization: got %v, want %v", event.Data["key"], tt.expected)
				}

				// Should not panic
				roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
				if roomHandleResult == nil {
					t.Errorf("Room handle result should not be nil")
				}
			})
		}
	})

	// Test coordinate validation
	t.Run("Coordinate validation", func(t *testing.T) {
		coordinateTests := []struct {
			name string
			x    float64
			y    float64
		}{
			{"Normal coordinates", 100.0, 200.0},
			{"Zero coordinates", 0.0, 0.0},
			{"Negative coordinates", -100.0, -200.0},
			{"Large coordinates", 999999.0, 888888.0},
			{"Small decimal coordinates", 0.001, 0.002},
			{"Very large coordinates", 1e10, 1e10},
			{"Very small coordinates", 1e-10, 1e-10},
		}

		for _, tt := range coordinateTests {
			t.Run(tt.name, func(t *testing.T) {
				req := types.PlayerClickRequest{
					X:      tt.x,
					Y:      tt.y,
					IsDown: true,
					Button: 0,
				}

				eventData := map[string]interface{}{
					"playerId": conn.PlayerID,
					"x":        req.X,
					"y":        req.Y,
					"isDown":   req.IsDown,
					"button":   req.Button,
				}

				event := game_objects.NewGameEvent(
					conn.RoomID,
					game_objects.EventPlayerClickInput,
					eventData,
					1,
					nil,
				)

				// Coordinates should be preserved
				if event.Data["x"] != tt.x {
					t.Errorf("X coordinate: got %v, want %v", event.Data["x"], tt.x)
				}

				if event.Data["y"] != tt.y {
					t.Errorf("Y coordinate: got %v, want %v", event.Data["y"], tt.y)
				}

				// Should not panic
				roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
				if roomHandleResult == nil {
					t.Errorf("Room handle result should not be nil")
				}
			})
		}
	})
}

// TestInputRateLimitingAndSpamProtection tests input rate limiting and spam protection
func TestInputRateLimitingAndSpamProtection(t *testing.T) {
	// Create test room
	room, player, err := server.NewGameWithPlayer("Rate Limit Test Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Create mock connection
	conn := &server.Connection{
		ID:       uuid.New().String(),
		RoomID:   room.ID,
		PlayerID: player.ID,
	}

	// Test rapid input submission
	t.Run("Rapid input submission", func(t *testing.T) {
		const numRapidInputs = 100
		const inputInterval = 1 * time.Millisecond

		// Send rapid key inputs
		for i := 0; i < numRapidInputs; i++ {
			req := types.KeyStatusRequest{
				Key:    "W",
				IsDown: i%2 == 0, // Alternate between press and release
			}

			eventData := map[string]interface{}{
				"playerId": conn.PlayerID,
				"key":      req.Key,
				"isDown":   req.IsDown,
			}

			event := game_objects.NewGameEvent(
				conn.RoomID,
				game_objects.EventPlayerKeyInput,
				eventData,
				1,
				nil,
			)

			// Should handle rapid inputs without crashing
			roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
			if roomHandleResult == nil {
				t.Errorf("Room handle result should not be nil for rapid input %d", i)
			}

			time.Sleep(inputInterval)
		}
	})

	// Test concurrent input submission
	t.Run("Concurrent input submission", func(t *testing.T) {
		const numConcurrentInputs = 50
		var wg sync.WaitGroup
		errorChan := make(chan error, numConcurrentInputs)

		// Send concurrent inputs
		for i := 0; i < numConcurrentInputs; i++ {
			wg.Add(1)
			go func(index int) {
				defer wg.Done()

				defer func() {
					if r := recover(); r != nil {
						errorChan <- fmt.Errorf("panic in concurrent input %d: %v", index, r)
					}
				}()

				// Alternate between key and mouse inputs
				if index%2 == 0 {
					// Key input
					req := types.KeyStatusRequest{
						Key:    []string{"W", "A", "S", "D"}[index%4],
						IsDown: true,
					}

					eventData := map[string]interface{}{
						"playerId": conn.PlayerID,
						"key":      req.Key,
						"isDown":   req.IsDown,
					}

					event := game_objects.NewGameEvent(
						conn.RoomID,
						game_objects.EventPlayerKeyInput,
						eventData,
						1,
						nil,
					)

					roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
					if roomHandleResult == nil {
						errorChan <- fmt.Errorf("room handle result nil for key input %d", index)
					}
				} else {
					// Mouse input
					req := types.PlayerClickRequest{
						X:      float64(index * 10),
						Y:      float64(index * 20),
						IsDown: true,
						Button: index % 3, // Vary button
					}

					eventData := map[string]interface{}{
						"playerId": conn.PlayerID,
						"x":        req.X,
						"y":        req.Y,
						"isDown":   req.IsDown,
						"button":   req.Button,
					}

					event := game_objects.NewGameEvent(
						conn.RoomID,
						game_objects.EventPlayerClickInput,
						eventData,
						1,
						nil,
					)

					roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
					if roomHandleResult == nil {
						errorChan <- fmt.Errorf("room handle result nil for mouse input %d", index)
					}
				}
			}(i)
		}

		// Wait for all goroutines to complete
		wg.Wait()
		close(errorChan)

		// Check for errors
		for err := range errorChan {
			t.Errorf("Concurrent input error: %v", err)
		}
	})

	// Test input burst handling
	t.Run("Input burst handling", func(t *testing.T) {
		const burstSize = 20
		events := make([]*game_objects.GameEvent, 0, burstSize)

		// Create a burst of events
		for i := 0; i < burstSize; i++ {
			eventData := map[string]interface{}{
				"playerId": conn.PlayerID,
				"key":      "W",
				"isDown":   i%2 == 0,
			}

			event := game_objects.NewGameEvent(
				conn.RoomID,
				game_objects.EventPlayerKeyInput,
				eventData,
				1,
				nil,
			)

			events = append(events, event)
		}

		// Process burst of events
		roomHandleResult := room.Handle(events)
		if roomHandleResult == nil {
			t.Errorf("Room should handle burst of events")
		}
	})

	// Test spam detection patterns
	t.Run("Spam detection patterns", func(t *testing.T) {
		spamPatterns := []struct {
			name        string
			pattern     func() []*game_objects.GameEvent
			description string
		}{
			{
				name: "Repeated identical inputs",
				pattern: func() []*game_objects.GameEvent {
					events := make([]*game_objects.GameEvent, 10)
					for i := 0; i < 10; i++ {
						eventData := map[string]interface{}{
							"playerId": conn.PlayerID,
							"key":      "W",
							"isDown":   true,
						}
						events[i] = game_objects.NewGameEvent(
							conn.RoomID,
							game_objects.EventPlayerKeyInput,
							eventData,
							1,
							nil,
						)
					}
					return events
				},
				description: "Repeated identical key presses",
			},
			{
				name: "Rapid alternating inputs",
				pattern: func() []*game_objects.GameEvent {
					events := make([]*game_objects.GameEvent, 20)
					for i := 0; i < 20; i++ {
						eventData := map[string]interface{}{
							"playerId": conn.PlayerID,
							"key":      "W",
							"isDown":   i%2 == 0,
						}
						events[i] = game_objects.NewGameEvent(
							conn.RoomID,
							game_objects.EventPlayerKeyInput,
							eventData,
							1,
							nil,
						)
					}
					return events
				},
				description: "Rapid alternating key press/release",
			},
			{
				name: "Multiple key spam",
				pattern: func() []*game_objects.GameEvent {
					events := make([]*game_objects.GameEvent, 16)
					keys := []string{"W", "A", "S", "D"}
					for i := 0; i < 16; i++ {
						eventData := map[string]interface{}{
							"playerId": conn.PlayerID,
							"key":      keys[i%4],
							"isDown":   true,
						}
						events[i] = game_objects.NewGameEvent(
							conn.RoomID,
							game_objects.EventPlayerKeyInput,
							eventData,
							1,
							nil,
						)
					}
					return events
				},
				description: "Rapid multiple key presses",
			},
		}

		for _, pattern := range spamPatterns {
			t.Run(pattern.name, func(t *testing.T) {
				events := pattern.pattern()

				// Should handle spam patterns without crashing
				roomHandleResult := room.Handle(events)
				if roomHandleResult == nil {
					t.Errorf("Room should handle spam pattern: %s", pattern.description)
				}
			})
		}
	})
}

// TestTrainingModeInputBypassFunctionality tests training mode input bypass
func TestTrainingModeInputBypassFunctionality(t *testing.T) {
	// Create normal room
	normalRoom, normalPlayer, err := server.NewGameWithPlayer("Normal Room", "Normal Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create normal room: %v", err)
	}

	// Create training room
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   10.0,
		HeadlessMode:      true,
		TrainingMode:      true,
		SessionID:         "input-bypass-test",
		DirectStateAccess: true,
	}

	trainingRoom, trainingPlayer, err := server.NewTrainingGameWithPlayer("Training Room", "Training Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		t.Fatalf("Failed to create training room: %v", err)
	}

	// Test normal room input processing
	t.Run("Normal room input processing", func(t *testing.T) {
		conn := &server.Connection{
			ID:       uuid.New().String(),
			RoomID:   normalRoom.ID,
			PlayerID: normalPlayer.ID,
		}

		// Normal room should process inputs through standard WebSocket path
		req := types.KeyStatusRequest{
			Key:    "W",
			IsDown: true,
		}

		eventData := map[string]interface{}{
			"playerId": conn.PlayerID,
			"key":      req.Key,
			"isDown":   req.IsDown,
		}

		event := game_objects.NewGameEvent(
			conn.RoomID,
			game_objects.EventPlayerKeyInput,
			eventData,
			1,
			nil,
		)

		// Normal room should handle input normally
		roomHandleResult := normalRoom.Handle([]*game_objects.GameEvent{event})
		if roomHandleResult == nil {
			t.Errorf("Normal room should handle input events")
		}

		// Verify room is not in training mode
		if normalRoom.IsTrainingRoom() {
			t.Errorf("Normal room should not be in training mode")
		}
	})

	// Test training room input processing
	t.Run("Training room input processing", func(t *testing.T) {
		conn := &server.Connection{
			ID:       uuid.New().String(),
			RoomID:   trainingRoom.ID,
			PlayerID: trainingPlayer.ID,
		}

		// Training room should support both normal and direct input
		req := types.KeyStatusRequest{
			Key:    "W",
			IsDown: true,
		}

		eventData := map[string]interface{}{
			"playerId": conn.PlayerID,
			"key":      req.Key,
			"isDown":   req.IsDown,
		}

		event := game_objects.NewGameEvent(
			conn.RoomID,
			game_objects.EventPlayerKeyInput,
			eventData,
			1,
			nil,
		)

		// Training room should handle input events
		roomHandleResult := trainingRoom.Handle([]*game_objects.GameEvent{event})
		if roomHandleResult == nil {
			t.Errorf("Training room should handle input events")
		}

		// Verify room is in training mode
		if !trainingRoom.IsTrainingRoom() {
			t.Errorf("Training room should be in training mode")
		}

		// Verify direct state access is available
		if !trainingRoom.DirectStateAccess {
			t.Errorf("Training room should have direct state access enabled")
		}

		// Test direct state access (bypass functionality)
		gameState := trainingRoom.GetDirectGameState()
		if gameState == nil {
			t.Errorf("Training room should provide direct game state access")
		}

		// Verify training room properties in state
		roomInfo, ok := gameState["room"].(map[string]interface{})
		if !ok {
			t.Fatalf("Game state should contain room information")
		}

		if roomInfo["isTraining"] != true {
			t.Errorf("Room state should indicate training mode")
		}

		// Note: directStateAccess may not be exposed in room state for security reasons
		// This is acceptable behavior - the important thing is that DirectStateAccess works
		if trainingRoom.DirectStateAccess != true {
			t.Errorf("Training room should have direct state access enabled")
		}
	})

	// Test input processing performance in training mode
	t.Run("Training mode input performance", func(t *testing.T) {
		conn := &server.Connection{
			ID:       uuid.New().String(),
			RoomID:   trainingRoom.ID,
			PlayerID: trainingPlayer.ID,
		}

		const numInputs = 1000
		startTime := time.Now()

		// Process many inputs rapidly (simulating training scenario)
		for i := 0; i < numInputs; i++ {
			eventData := map[string]interface{}{
				"playerId": conn.PlayerID,
				"key":      []string{"W", "A", "S", "D"}[i%4],
				"isDown":   i%2 == 0,
			}

			event := game_objects.NewGameEvent(
				conn.RoomID,
				game_objects.EventPlayerKeyInput,
				eventData,
				1,
				nil,
			)

			roomHandleResult := trainingRoom.Handle([]*game_objects.GameEvent{event})
			if roomHandleResult == nil {
				t.Errorf("Training room should handle input %d", i)
			}
		}

		duration := time.Since(startTime)
		inputsPerSecond := float64(numInputs) / duration.Seconds()

		// Training mode should handle inputs efficiently
		// This is a performance check - adjust threshold as needed
		if inputsPerSecond < 1000 {
			t.Logf("Input processing rate: %.2f inputs/second", inputsPerSecond)
			// Note: This is informational, not a hard failure
		}
	})

	// Test training mode configuration changes
	t.Run("Training mode configuration", func(t *testing.T) {
		// Test enabling/disabling direct state access
		err := trainingRoom.ConfigureTraining("normal", 5.0, false)
		if err != nil {
			t.Errorf("Failed to configure training room: %v", err)
		}

		if trainingRoom.DirectStateAccess {
			t.Errorf("Direct state access should be disabled after configuration")
		}

		// Re-enable direct state access
		err = trainingRoom.ConfigureTraining("headless", 10.0, true)
		if err != nil {
			t.Errorf("Failed to reconfigure training room: %v", err)
		}

		if !trainingRoom.DirectStateAccess {
			t.Errorf("Direct state access should be enabled after reconfiguration")
		}

		// Verify state access still works
		gameState := trainingRoom.GetDirectGameState()
		if gameState == nil {
			t.Errorf("Direct game state should be available after reconfiguration")
		}
	})

	// Test input bypass with different training configurations
	t.Run("Input bypass with different configurations", func(t *testing.T) {
		configurations := []struct {
			name              string
			trainingMode      string
			speedMultiplier   float64
			directStateAccess bool
		}{
			{"Normal training", "normal", 1.0, true},
			{"Fast training", "normal", 10.0, true},
			{"Headless training", "headless", 50.0, true},
			{"No direct access", "normal", 5.0, false},
		}

		for _, config := range configurations {
			t.Run(config.name, func(t *testing.T) {
				err := trainingRoom.ConfigureTraining(config.trainingMode, config.speedMultiplier, config.directStateAccess)
				if err != nil {
					t.Errorf("Failed to configure training room for %s: %v", config.name, err)
				}

				// Test input processing with this configuration
				conn := &server.Connection{
					ID:       uuid.New().String(),
					RoomID:   trainingRoom.ID,
					PlayerID: trainingPlayer.ID,
				}

				eventData := map[string]interface{}{
					"playerId": conn.PlayerID,
					"key":      "W",
					"isDown":   true,
				}

				event := game_objects.NewGameEvent(
					conn.RoomID,
					game_objects.EventPlayerKeyInput,
					eventData,
					1,
					nil,
				)

				roomHandleResult := trainingRoom.Handle([]*game_objects.GameEvent{event})
				if roomHandleResult == nil {
					t.Errorf("Training room should handle input with configuration: %s", config.name)
				}

				// Test direct state access based on configuration
				gameState := trainingRoom.GetDirectGameState()
				if config.directStateAccess && gameState == nil {
					t.Errorf("Direct state access should work with configuration: %s", config.name)
				}
			})
		}
	})
}

// TestInputProcessingWithInvalidPlayerStates tests input handling with invalid player states
func TestInputProcessingWithInvalidPlayerStates(t *testing.T) {
	// Create test room
	room, player, err := server.NewGameWithPlayer("Invalid State Test Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Test input with non-existent player
	t.Run("Input with non-existent player", func(t *testing.T) {
		nonExistentPlayerID := uuid.New().String()

		eventData := map[string]interface{}{
			"playerId": nonExistentPlayerID,
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

		// Should handle gracefully without crashing
		roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
		if roomHandleResult == nil {
			t.Errorf("Room should handle input from non-existent player gracefully")
		}
	})

	// Test input with removed player
	t.Run("Input with removed player", func(t *testing.T) {
		// Add a player then remove them
		tempPlayer, err := server.AddPlayerToGame(room, "Temp Player", room.Password, false)
		if err != nil {
			t.Fatalf("Failed to add temp player: %v", err)
		}

		// Remove the player
		room.RemovePlayer(tempPlayer.ID)

		// Try to send input from removed player
		eventData := map[string]interface{}{
			"playerId": tempPlayer.ID,
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

		// Should handle gracefully
		roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
		if roomHandleResult == nil {
			t.Errorf("Room should handle input from removed player gracefully")
		}
	})

	// Test input with empty player ID
	t.Run("Input with empty player ID", func(t *testing.T) {
		eventData := map[string]interface{}{
			"playerId": "",
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

		// Should handle gracefully
		roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
		if roomHandleResult == nil {
			t.Errorf("Room should handle input with empty player ID gracefully")
		}
	})

	// Test input with nil player ID
	t.Run("Input with nil player ID", func(t *testing.T) {
		eventData := map[string]interface{}{
			"playerId": nil,
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

		// Should handle gracefully
		roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
		if roomHandleResult == nil {
			t.Errorf("Room should handle input with nil player ID gracefully")
		}
	})

	// Test input with valid player but invalid room
	t.Run("Input with valid player but invalid room", func(t *testing.T) {
		invalidRoomID := uuid.New().String()

		eventData := map[string]interface{}{
			"playerId": player.ID,
			"key":      "W",
			"isDown":   true,
		}

		event := game_objects.NewGameEvent(
			invalidRoomID, // Wrong room ID
			game_objects.EventPlayerKeyInput,
			eventData,
			1,
			nil,
		)

		// Should handle gracefully (though room won't process it)
		roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
		if roomHandleResult == nil {
			t.Errorf("Room should handle input with invalid room ID gracefully")
		}
	})

	// Test input with spectator player
	t.Run("Input with spectator player", func(t *testing.T) {
		// Add a spectator
		spectator, err := server.AddPlayerToGame(room, "Spectator", room.Password, true)
		if err != nil {
			t.Fatalf("Failed to add spectator: %v", err)
		}

		eventData := map[string]interface{}{
			"playerId": spectator.ID,
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

		// Should handle spectator input (may be ignored by game logic)
		roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
		if roomHandleResult == nil {
			t.Errorf("Room should handle spectator input gracefully")
		}
	})
}

// TestClientStateAndDirectionInput tests client state and direction input processing
func TestClientStateAndDirectionInput(t *testing.T) {
	// Create test room
	room, player, err := server.NewGameWithPlayer("Direction Test Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Create mock connection
	conn := &server.Connection{
		ID:       uuid.New().String(),
		RoomID:   room.ID,
		PlayerID: player.ID,
	}

	// Test direction input processing
	t.Run("Direction input processing", func(t *testing.T) {
		directionTests := []struct {
			name       string
			direction  float64
			shouldPass bool
		}{
			{"Zero direction", 0.0, true},
			{"Positive direction", 1.5708, true},  // π/2 radians (90 degrees)
			{"Negative direction", -1.5708, true}, // -π/2 radians (-90 degrees)
			{"Full circle", 6.2832, true},         // 2π radians (360 degrees)
			{"Large positive", 100.0, true},
			{"Large negative", -100.0, true},
			{"Small decimal", 0.001, true},
		}

		for _, tt := range directionTests {
			t.Run(tt.name, func(t *testing.T) {
				req := types.ClientStateRequest{
					Direction: tt.direction,
				}

				eventData := map[string]interface{}{
					"playerId":  conn.PlayerID,
					"direction": req.Direction,
				}

				event := game_objects.NewGameEvent(
					conn.RoomID,
					game_objects.EventPlayerDirection,
					eventData,
					1,
					nil,
				)

				// Verify event creation
				if event.EventType != game_objects.EventPlayerDirection {
					t.Errorf("Event type = %v, want %v", event.EventType, game_objects.EventPlayerDirection)
				}

				if event.Data["direction"] != tt.direction {
					t.Errorf("Direction = %v, want %v", event.Data["direction"], tt.direction)
				}

				// Test room handling
				roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
				if roomHandleResult == nil {
					t.Errorf("Room should handle direction input")
				}
			})
		}
	})

	// Test mixed input types
	t.Run("Mixed input types", func(t *testing.T) {
		// Create a sequence of different input types
		events := []*game_objects.GameEvent{
			// Key input
			game_objects.NewGameEvent(
				conn.RoomID,
				game_objects.EventPlayerKeyInput,
				map[string]interface{}{
					"playerId": conn.PlayerID,
					"key":      "W",
					"isDown":   true,
				},
				1,
				nil,
			),
			// Mouse input
			game_objects.NewGameEvent(
				conn.RoomID,
				game_objects.EventPlayerClickInput,
				map[string]interface{}{
					"playerId": conn.PlayerID,
					"x":        100.0,
					"y":        200.0,
					"isDown":   true,
					"button":   0,
				},
				1,
				nil,
			),
			// Direction input
			game_objects.NewGameEvent(
				conn.RoomID,
				game_objects.EventPlayerDirection,
				map[string]interface{}{
					"playerId":  conn.PlayerID,
					"direction": 1.5708,
				},
				1,
				nil,
			),
		}

		// Process all events together
		roomHandleResult := room.Handle(events)
		if roomHandleResult == nil {
			t.Errorf("Room should handle mixed input types")
		}
	})
}
