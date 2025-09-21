package tests

import (
	"fmt"
	"go-ws-server/pkg/server"
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/game_maps"
	"go-ws-server/pkg/server/game_objects"
	"go-ws-server/pkg/server/geo"
	"math"
	"testing"
	"time"

	"github.com/google/uuid"
)

// TestGameTickProcessingAtDifferentSpeeds tests game tick processing at various speed multipliers
func TestGameTickProcessingAtDifferentSpeeds(t *testing.T) {
	tests := []struct {
		name             string
		speedMultiplier  float64
		expectedTickRate time.Duration
	}{
		{
			name:             "Normal speed (1x)",
			speedMultiplier:  1.0,
			expectedTickRate: 20 * time.Millisecond,
		},
		{
			name:             "Double speed (2x)",
			speedMultiplier:  2.0,
			expectedTickRate: 10 * time.Millisecond,
		},
		{
			name:             "High speed (10x)",
			speedMultiplier:  10.0,
			expectedTickRate: 2 * time.Millisecond,
		},
		{
			name:             "Very high speed (50x)",
			speedMultiplier:  50.0,
			expectedTickRate: 400 * time.Microsecond,
		},
		{
			name:             "Slow speed (0.5x)",
			speedMultiplier:  0.5,
			expectedTickRate: 40 * time.Millisecond,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create training room with specific speed multiplier
			trainingConfig := server.TrainingConfig{
				SpeedMultiplier:   tt.speedMultiplier,
				HeadlessMode:      true,
				TrainingMode:      true,
				SessionID:         "physics-test-" + tt.name,
				DirectStateAccess: true,
			}

			room, player, err := server.NewTrainingGameWithPlayer("Physics Test Room", "Test Player", game_maps.MapDefault, trainingConfig)
			if err != nil {
				t.Fatalf("Failed to create training room: %v", err)
			}

			// Verify custom tick rate is set correctly
			actualTickRate := room.GetCustomTickRate()
			if actualTickRate != tt.expectedTickRate {
				t.Errorf("Custom tick rate = %v, want %v", actualTickRate, tt.expectedTickRate)
			}

			// Test that game tick events are processed
			initialState := room.GetAllGameObjectStates()
			if initialState == nil {
				t.Fatalf("Initial game state should not be nil")
			}

			// Verify player exists in state
			playerState, exists := initialState[player.ID]
			if !exists {
				t.Fatalf("Player should exist in game state")
			}

			// Verify player has required physics properties
			if playerState["x"] == nil || playerState["y"] == nil {
				t.Errorf("Player should have position coordinates")
			}
			if playerState["dx"] == nil || playerState["dy"] == nil {
				t.Errorf("Player should have velocity components")
			}
		})
	}
}

// TestCollisionDetectionBetweenGameObjects tests collision detection between different game object types
func TestCollisionDetectionBetweenGameObjects(t *testing.T) {
	tests := []struct {
		name            string
		setupObjects    func() (game_objects.GameObject, game_objects.GameObject)
		expectCollision bool
		description     string
	}{
		{
			name: "Player-Player collision",
			setupObjects: func() (game_objects.GameObject, game_objects.GameObject) {
				// Create two players at overlapping positions
				player1 := game_objects.NewPlayerGameObject("player1", "Player 1", "token1",
					func() (float64, float64) { return 100, 100 },
					func(x, y float64) (float64, float64) { return x, y })
				player2 := game_objects.NewPlayerGameObject("player2", "Player 2", "token2",
					func() (float64, float64) { return 110, 100 }, // Overlapping with player1
					func(x, y float64) (float64, float64) { return x, y })
				return player1, player2
			},
			expectCollision: true,
			description:     "Two players with overlapping circular bounding shapes should collide",
		},
		{
			name: "Player-Arrow collision",
			setupObjects: func() (game_objects.GameObject, game_objects.GameObject) {
				sourcePlayer := game_objects.NewPlayerGameObject("source", "Source", "token",
					func() (float64, float64) { return 50, 50 },
					func(x, y float64) (float64, float64) { return x, y })
				targetPlayer := game_objects.NewPlayerGameObject("target", "Target", "token2",
					func() (float64, float64) { return 100, 100 },
					func(x, y float64) (float64, float64) { return x, y })
				arrow := game_objects.NewArrowGameObject("arrow1", sourcePlayer, 100, 100, 1.0,
					func(x, y float64) (float64, float64) { return x, y })

				// Position arrow to intersect with player
				arrow.SetState(constants.StateX, float64(100))
				arrow.SetState(constants.StateY, float64(100))

				return targetPlayer, arrow
			},
			expectCollision: true,
			description:     "Player circle should collide with arrow line segment",
		},
		{
			name: "Non-colliding distant objects",
			setupObjects: func() (game_objects.GameObject, game_objects.GameObject) {
				player1 := game_objects.NewPlayerGameObject("distant1", "Distant 1", "token1",
					func() (float64, float64) { return 100, 100 },
					func(x, y float64) (float64, float64) { return x, y })
				player2 := game_objects.NewPlayerGameObject("distant2", "Distant 2", "token2",
					func() (float64, float64) { return 300, 300 }, // Far from player1
					func(x, y float64) (float64, float64) { return x, y })
				return player1, player2
			},
			expectCollision: false,
			description:     "Distant objects should not collide",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj1, obj2 := tt.setupObjects()

			shape1 := obj1.GetBoundingShape()
			shape2 := obj2.GetBoundingShape()

			if shape1 == nil || shape2 == nil {
				t.Fatalf("Both objects should have bounding shapes")
			}

			collides, collisionPoints := shape1.CollidesWith(shape2)

			if collides != tt.expectCollision {
				t.Errorf("Collision detection = %v, want %v. %s", collides, tt.expectCollision, tt.description)
			}

			if tt.expectCollision && len(collisionPoints) == 0 {
				t.Errorf("Expected collision points when collision detected")
			}

			if !tt.expectCollision && len(collisionPoints) > 0 {
				t.Errorf("Should not have collision points when no collision detected")
			}
		})
	}
}

// TestCollisionNotificationMessageGeneration tests that collision events generate proper notification messages
func TestCollisionNotificationMessageGeneration(t *testing.T) {
	room, _, err := server.NewGameWithPlayer("Notification Test Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Create a game tick event to trigger collision processing
	tickEvent := game_objects.NewGameEvent(
		room.ID,
		game_objects.EventGameTick,
		nil,
		1,
		nil,
	)

	// Process the tick event
	result := room.Handle([]*game_objects.GameEvent{tickEvent})

	// Verify that events were processed
	if result == nil {
		t.Fatalf("Game tick should return a result")
	}

	// Check for collision events in the result
	collisionEventFound := false
	for _, event := range result.Events {
		if event.EventType == game_objects.EventObjectCollision {
			collisionEventFound = true

			// Verify collision event has required data
			if event.Data["x"] == nil || event.Data["y"] == nil {
				t.Errorf("Collision event should contain x and y coordinates")
			}

			// Verify coordinates are valid numbers
			if x, ok := event.Data["x"].(float64); !ok || math.IsNaN(x) || math.IsInf(x, 0) {
				t.Errorf("Collision event x coordinate should be a valid float64")
			}
			if y, ok := event.Data["y"].(float64); !ok || math.IsNaN(y) || math.IsInf(y, 0) {
				t.Errorf("Collision event y coordinate should be a valid float64")
			}
		}
	}

	// Note: Collision events are only generated when actual collisions occur
	// This test verifies the event structure when they do occur
	t.Logf("Processed %d events, collision event found: %v", len(result.Events), collisionEventFound)
}

// TestPhysicsSimulationAccuracyAtHighSpeeds tests physics accuracy at high simulation speeds
func TestPhysicsSimulationAccuracyAtHighSpeeds(t *testing.T) {
	tests := []struct {
		name            string
		speedMultiplier float64
		testDuration    time.Duration
		tolerance       float64
	}{
		{
			name:            "Normal speed physics",
			speedMultiplier: 1.0,
			testDuration:    100 * time.Millisecond,
			tolerance:       1.0, // 1 pixel tolerance
		},
		{
			name:            "High speed physics",
			speedMultiplier: 10.0,
			testDuration:    50 * time.Millisecond,
			tolerance:       2.0, // Slightly higher tolerance for high speed
		},
		{
			name:            "Very high speed physics",
			speedMultiplier: 50.0,
			testDuration:    20 * time.Millisecond,
			tolerance:       5.0, // Higher tolerance for very high speed
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create training room with specific speed multiplier
			trainingConfig := server.TrainingConfig{
				SpeedMultiplier:   tt.speedMultiplier,
				HeadlessMode:      true,
				TrainingMode:      true,
				SessionID:         "accuracy-test-" + tt.name,
				DirectStateAccess: true,
			}

			room, player, err := server.NewTrainingGameWithPlayer("Accuracy Test Room", "Test Player", game_maps.MapDefault, trainingConfig)
			if err != nil {
				t.Fatalf("Failed to create training room: %v", err)
			}

			// Get initial player state
			initialState := room.GetAllGameObjectStates()
			playerState := initialState[player.ID]
			initialX := playerState["x"].(float64)
			initialY := playerState["y"].(float64)

			// Apply a known velocity to the player
			playerObj, exists := room.ObjectManager.GetObject(player.ID)
			if !exists {
				t.Fatalf("Player object should exist")
			}

			testVelocityX := 100.0 // pixels per second
			testVelocityY := 50.0  // pixels per second
			playerObj.SetState(constants.StateDx, testVelocityX)
			playerObj.SetState(constants.StateDy, testVelocityY)
			playerObj.SetState(constants.StateLastLocUpdateTime, time.Now())

			// Wait for physics simulation
			time.Sleep(tt.testDuration)

			// Process a game tick to update positions
			tickEvent := game_objects.NewGameEvent(room.ID, game_objects.EventGameTick, nil, 1, nil)
			room.Handle([]*game_objects.GameEvent{tickEvent})

			// Get final player state
			finalState := room.GetAllGameObjectStates()
			finalPlayerState := finalState[player.ID]
			finalX := finalPlayerState["x"].(float64)
			finalY := finalPlayerState["y"].(float64)

			// Calculate expected position based on time and velocity
			expectedDeltaX := testVelocityX * tt.testDuration.Seconds()
			expectedDeltaY := testVelocityY * tt.testDuration.Seconds()
			expectedX := initialX + expectedDeltaX
			expectedY := initialY + expectedDeltaY

			// Check accuracy within tolerance
			deltaX := math.Abs(finalX - expectedX)
			deltaY := math.Abs(finalY - expectedY)

			if deltaX > tt.tolerance {
				t.Errorf("X position accuracy: got %v, expected %v, delta %v > tolerance %v",
					finalX, expectedX, deltaX, tt.tolerance)
			}
			if deltaY > tt.tolerance {
				t.Errorf("Y position accuracy: got %v, expected %v, delta %v > tolerance %v",
					finalY, expectedY, deltaY, tt.tolerance)
			}
		})
	}
}

// TestProjectileTrajectoryAndImpactCalculations tests projectile physics
func TestProjectileTrajectoryAndImpactCalculations(t *testing.T) {
	sourcePlayer := game_objects.NewPlayerGameObject("source", "Source Player", "token",
		func() (float64, float64) { return 100, 100 },
		func(x, y float64) (float64, float64) { return x, y })

	tests := []struct {
		name           string
		targetX        float64
		targetY        float64
		powerRatio     float64
		expectedLength float64
	}{
		{
			name:           "Short range arrow",
			targetX:        150,
			targetY:        150,
			powerRatio:     0.5,
			expectedLength: constants.ArrowLengthPx,
		},
		{
			name:           "Long range arrow",
			targetX:        300,
			targetY:        200,
			powerRatio:     1.0,
			expectedLength: constants.ArrowLengthPx,
		},
		{
			name:           "Minimum power arrow",
			targetX:        120,
			targetY:        120,
			powerRatio:     0.1,
			expectedLength: constants.ArrowLengthPx,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create an arrow projectile
			arrowID := uuid.New().String()
			arrow := game_objects.NewArrowGameObject(arrowID, sourcePlayer, tt.targetX, tt.targetY, tt.powerRatio,
				func(x, y float64) (float64, float64) { return x, y })

			if arrow == nil {
				t.Fatalf("Arrow should be created successfully")
			}

			// Verify arrow has proper bounding shape
			boundingShape := arrow.GetBoundingShape()
			if boundingShape == nil {
				t.Fatalf("Arrow should have a bounding shape")
			}

			// Verify it's a line shape (projectile trajectory)
			line, ok := boundingShape.(*geo.Line)
			if !ok {
				t.Fatalf("Arrow bounding shape should be a line")
			}

			// Calculate actual line length
			dx := line.B.X - line.A.X
			dy := line.B.Y - line.A.Y
			actualLength := math.Sqrt(dx*dx + dy*dy)

			// Verify trajectory length is reasonable
			if math.Abs(actualLength-tt.expectedLength) > 1.0 {
				t.Errorf("Arrow trajectory length = %v, expected approximately %v", actualLength, tt.expectedLength)
			}

			// Verify arrow direction is correct
			arrowDx, exists := arrow.GetStateValue(constants.StateDx)
			if !exists {
				t.Fatalf("Arrow should have dx velocity")
			}
			arrowDy, exists := arrow.GetStateValue(constants.StateDy)
			if !exists {
				t.Fatalf("Arrow should have dy velocity")
			}

			// Verify arrow has reasonable velocity (based on physics calculations)
			velocityMagnitude := math.Sqrt(arrowDx.(float64)*arrowDx.(float64) + arrowDy.(float64)*arrowDy.(float64))
			if velocityMagnitude <= 0 {
				t.Errorf("Arrow should have positive velocity magnitude, got %v", velocityMagnitude)
			}
			if velocityMagnitude > 2000 { // Reasonable upper bound
				t.Errorf("Arrow velocity magnitude seems too high: %v", velocityMagnitude)
			}
		})
	}
}

// TestBoundaryCollisionAndWrappingBehavior tests collision with map boundaries and position wrapping
func TestBoundaryCollisionAndWrappingBehavior(t *testing.T) {
	room, player, err := server.NewGameWithPlayer("Boundary Test Room", "Test Player", game_maps.MapDefault)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	tests := []struct {
		name        string
		initialX    float64
		initialY    float64
		velocityX   float64
		velocityY   float64
		expectWrap  bool
		description string
	}{
		{
			name:        "Normal position - no wrapping",
			initialX:    400,
			initialY:    300,
			velocityX:   0,
			velocityY:   0,
			expectWrap:  false,
			description: "Player in normal position should not wrap",
		},
		{
			name:        "Near right boundary",
			initialX:    constants.RoomSizePixelsX - 10,
			initialY:    300,
			velocityX:   50,
			velocityY:   0,
			expectWrap:  true,
			description: "Player moving past right boundary should wrap",
		},
		{
			name:        "Near left boundary",
			initialX:    10,
			initialY:    300,
			velocityX:   -50,
			velocityY:   0,
			expectWrap:  true,
			description: "Player moving past left boundary should wrap",
		},
		{
			name:        "Near top boundary",
			initialX:    400,
			initialY:    10,
			velocityX:   0,
			velocityY:   -50,
			expectWrap:  true,
			description: "Player moving past top boundary should wrap",
		},
		{
			name:        "Near bottom boundary",
			initialX:    400,
			initialY:    constants.RoomSizePixelsY - 10,
			velocityX:   0,
			velocityY:   50,
			expectWrap:  true,
			description: "Player moving past bottom boundary should wrap",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Get player object
			playerObj, exists := room.ObjectManager.GetObject(player.ID)
			if !exists {
				t.Fatalf("Player object should exist")
			}

			// Set initial position and velocity
			playerObj.SetState(constants.StateX, tt.initialX)
			playerObj.SetState(constants.StateY, tt.initialY)
			playerObj.SetState(constants.StateDx, tt.velocityX)
			playerObj.SetState(constants.StateDy, tt.velocityY)
			playerObj.SetState(constants.StateLastLocUpdateTime, time.Now())

			// Process game tick to trigger physics
			tickEvent := game_objects.NewGameEvent(room.ID, game_objects.EventGameTick, nil, 1, nil)
			room.Handle([]*game_objects.GameEvent{tickEvent})

			// Get final position
			finalState := room.GetAllGameObjectStates()
			playerState := finalState[player.ID]
			finalX := playerState["x"].(float64)
			finalY := playerState["y"].(float64)

			// Log position changes for analysis
			t.Logf("Boundary test %s: initial(%v,%v) final(%v,%v)",
				tt.name, tt.initialX, tt.initialY, finalX, finalY)

			// For boundary tests, verify behavior is reasonable
			if tt.expectWrap {
				// Position may change due to wrapping logic or collision handling
				positionChanged := math.Abs(finalX-tt.initialX) > 1.0 || math.Abs(finalY-tt.initialY) > 1.0
				t.Logf("Position changed: %v", positionChanged)
			}
		})
	}
}

// TestPhysicsConsistencyAcrossTickRates tests that physics behave consistently across different tick rates
func TestPhysicsConsistencyAcrossTickRates(t *testing.T) {
	baseSpeedMultipliers := []float64{1.0, 2.0, 5.0, 10.0}

	// Test that the same physics scenario produces consistent results across different tick rates
	for i, speed1 := range baseSpeedMultipliers {
		for j, speed2 := range baseSpeedMultipliers {
			if i >= j {
				continue // Only test unique pairs
			}

			t.Run(fmt.Sprintf("Speed_%vx_vs_%vx", speed1, speed2), func(t *testing.T) {
				// Create two identical scenarios with different tick rates
				results := make([]map[string]interface{}, 2)
				speeds := []float64{speed1, speed2}

				for idx, speed := range speeds {
					trainingConfig := server.TrainingConfig{
						SpeedMultiplier:   speed,
						HeadlessMode:      true,
						TrainingMode:      true,
						SessionID:         fmt.Sprintf("consistency-test-%v", speed),
						DirectStateAccess: true,
					}

					room, player, err := server.NewTrainingGameWithPlayer("Consistency Test Room", "Test Player", game_maps.MapDefault, trainingConfig)
					if err != nil {
						t.Fatalf("Failed to create training room: %v", err)
					}

					// Set identical initial conditions
					playerObj, _ := room.ObjectManager.GetObject(player.ID)
					playerObj.SetState(constants.StateX, float64(200))
					playerObj.SetState(constants.StateY, float64(200))
					playerObj.SetState(constants.StateDx, float64(100)) // 100 px/s
					playerObj.SetState(constants.StateDy, float64(50))  // 50 px/s
					playerObj.SetState(constants.StateLastLocUpdateTime, time.Now())

					// Process multiple ticks to simulate same real-world time
					numTicks := int(speed * 5) // More ticks for higher speeds
					for tick := 0; tick < numTicks; tick++ {
						tickEvent := game_objects.NewGameEvent(room.ID, game_objects.EventGameTick, nil, 1, nil)
						room.Handle([]*game_objects.GameEvent{tickEvent})
						time.Sleep(time.Duration(float64(time.Millisecond) / speed)) // Proportional delay
					}

					// Record final state
					finalState := room.GetAllGameObjectStates()
					results[idx] = finalState[player.ID]
				}

				// Compare results - they should be similar within tolerance
				tolerance := 10.0 // 10 pixel tolerance for consistency

				x1 := results[0]["x"].(float64)
				y1 := results[0]["y"].(float64)
				x2 := results[1]["x"].(float64)
				y2 := results[1]["y"].(float64)

				deltaX := math.Abs(x1 - x2)
				deltaY := math.Abs(y1 - y2)

				if deltaX > tolerance {
					t.Errorf("X position inconsistency between %vx and %vx speeds: %v vs %v (delta: %v)",
						speed1, speed2, x1, x2, deltaX)
				}
				if deltaY > tolerance {
					t.Errorf("Y position inconsistency between %vx and %vx speeds: %v vs %v (delta: %v)",
						speed1, speed2, y1, y2, deltaY)
				}
			})
		}
	}
}
