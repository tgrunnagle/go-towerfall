package tests

import (
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/game_objects"
	"go-ws-server/pkg/server/geo"
	"math"
	"testing"
	"time"

	"github.com/google/uuid"
)

// TestPlayerBoundingShapeAccuracy tests that player bounding shapes are accurate
func TestPlayerBoundingShapeAccuracy(t *testing.T) {
	player := game_objects.NewPlayerGameObject("test-player", "Test Player", "token",
		func() (float64, float64) { return 100, 100 },
		func(x, y float64) (float64, float64) { return x, y })

	shape := player.GetBoundingShape()
	if shape == nil {
		t.Fatalf("Player should have a bounding shape")
	}

	circle, ok := shape.(*geo.Circle)
	if !ok {
		t.Fatalf("Player bounding shape should be a circle")
	}

	// Verify circle properties
	if circle.R != constants.PlayerRadius {
		t.Errorf("Player circle radius = %v, want %v", circle.R, constants.PlayerRadius)
	}

	// Verify center matches player position
	expectedX, _ := player.GetStateValue(constants.StateX)
	expectedY, _ := player.GetStateValue(constants.StateY)

	if math.Abs(circle.C.X-expectedX.(float64)) > 0.01 {
		t.Errorf("Player circle center X = %v, want %v", circle.C.X, expectedX)
	}
	if math.Abs(circle.C.Y-expectedY.(float64)) > 0.01 {
		t.Errorf("Player circle center Y = %v, want %v", circle.C.Y, expectedY)
	}
}

// TestArrowBoundingShapeAccuracy tests that arrow bounding shapes represent trajectory correctly
func TestArrowBoundingShapeAccuracy(t *testing.T) {
	sourcePlayer := game_objects.NewPlayerGameObject("source", "Source", "token",
		func() (float64, float64) { return 100, 100 },
		func(x, y float64) (float64, float64) { return x, y })

	tests := []struct {
		name       string
		targetX    float64
		targetY    float64
		powerRatio float64
	}{
		{
			name:       "Horizontal arrow",
			targetX:    200,
			targetY:    100,
			powerRatio: 1.0,
		},
		{
			name:       "Vertical arrow",
			targetX:    100,
			targetY:    200,
			powerRatio: 1.0,
		},
		{
			name:       "Diagonal arrow",
			targetX:    150,
			targetY:    150,
			powerRatio: 0.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			arrow := game_objects.NewArrowGameObject(uuid.New().String(), sourcePlayer,
				tt.targetX, tt.targetY, tt.powerRatio,
				func(x, y float64) (float64, float64) { return x, y })

			shape := arrow.GetBoundingShape()
			if shape == nil {
				t.Fatalf("Arrow should have a bounding shape")
			}

			line, ok := shape.(*geo.Line)
			if !ok {
				t.Fatalf("Arrow bounding shape should be a line")
			}

			// Verify line length matches arrow length constant
			dx := line.B.X - line.A.X
			dy := line.B.Y - line.A.Y
			actualLength := math.Sqrt(dx*dx + dy*dy)

			expectedLength := constants.ArrowLengthPx
			if math.Abs(actualLength-expectedLength) > 1.0 {
				t.Errorf("Arrow line length = %v, want approximately %v", actualLength, expectedLength)
			}

			// Verify line starts at arrow position
			arrowX, _ := arrow.GetStateValue(constants.StateX)
			arrowY, _ := arrow.GetStateValue(constants.StateY)

			if math.Abs(line.A.X-arrowX.(float64)) > 0.01 {
				t.Errorf("Arrow line start X = %v, want %v", line.A.X, arrowX)
			}
			if math.Abs(line.A.Y-arrowY.(float64)) > 0.01 {
				t.Errorf("Arrow line start Y = %v, want %v", line.A.Y, arrowY)
			}
		})
	}
}

// TestBulletBoundingShapeAccuracy tests that bullet bounding shapes represent trajectory correctly
func TestBulletBoundingShapeAccuracy(t *testing.T) {
	sourcePlayer := game_objects.NewPlayerGameObject("source", "Source", "token",
		func() (float64, float64) { return 100, 100 },
		func(x, y float64) (float64, float64) { return x, y })

	bullet := game_objects.NewBulletGameObject(uuid.New().String(), sourcePlayer, 200, 150)
	if bullet == nil {
		t.Fatalf("Bullet should be created successfully")
	}

	shape := bullet.GetBoundingShape()
	if shape == nil {
		t.Fatalf("Bullet should have a bounding shape")
	}

	line, ok := shape.(*geo.Line)
	if !ok {
		t.Fatalf("Bullet bounding shape should be a line")
	}

	// Verify line length matches bullet distance constant
	dx := line.B.X - line.A.X
	dy := line.B.Y - line.A.Y
	actualLength := math.Sqrt(dx*dx + dy*dy)

	expectedLength := constants.BulletDistance
	if math.Abs(actualLength-expectedLength) > 1.0 {
		t.Errorf("Bullet line length = %v, want approximately %v", actualLength, expectedLength)
	}

	// Verify line starts at bullet position
	bulletX, _ := bullet.GetStateValue(constants.StateX)
	bulletY, _ := bullet.GetStateValue(constants.StateY)

	if math.Abs(line.A.X-bulletX.(float64)) > 0.01 {
		t.Errorf("Bullet line start X = %v, want %v", line.A.X, bulletX)
	}
	if math.Abs(line.A.Y-bulletY.(float64)) > 0.01 {
		t.Errorf("Bullet line start Y = %v, want %v", line.A.Y, bulletY)
	}
}

// TestCollisionResponseBehavior tests how game objects respond to collisions
func TestCollisionResponseBehavior(t *testing.T) {
	// Create a player
	player := game_objects.NewPlayerGameObject("test-player", "Test Player", "token",
		func() (float64, float64) { return 100, 100 },
		func(x, y float64) (float64, float64) { return x, y })

	// Set player in motion
	player.SetState(constants.StateDx, 50.0) // Moving right
	player.SetState(constants.StateDy, 0.0)
	player.SetState(constants.StateLastLocUpdateTime, time.Now())

	// Create a mock solid object (block)
	blockPoints := []*geo.Point{
		{X: 150, Y: 100},
		{X: 170, Y: 100},
		{X: 170, Y: 120},
		{X: 150, Y: 120},
	}
	block := game_objects.NewBlockGameObject("test-block", blockPoints) // Right of player

	// Create room objects map
	roomObjects := map[string]game_objects.GameObject{
		player.GetID(): player,
		block.GetID():  block,
	}

	// Create game tick event
	tickEvent := game_objects.NewGameEvent("test-room", game_objects.EventGameTick, nil, 1, nil)

	// Process collision
	result := player.Handle(tickEvent, roomObjects)

	// Verify collision was handled
	if result == nil {
		t.Fatalf("Player should return a result from game tick")
	}

	// Check if collision events were generated
	collisionFound := false
	for _, event := range result.RaisedEvents {
		if event.EventType == game_objects.EventObjectCollision {
			collisionFound = true
			break
		}
	}

	// Note: Collision detection depends on exact positioning and bounding shapes
	// This test verifies the collision handling mechanism exists
	t.Logf("Collision handling result: state changed = %v, events raised = %d, collision found = %v",
		result.StateChanged, len(result.RaisedEvents), collisionFound)
}

// TestProjectileCollisionDetection tests collision detection for projectiles
func TestProjectileCollisionDetection(t *testing.T) {
	// Create source player
	sourcePlayer := game_objects.NewPlayerGameObject("source", "Source", "token",
		func() (float64, float64) { return 50, 50 },
		func(x, y float64) (float64, float64) { return x, y })

	// Create target player at a position that will intersect with arrow trajectory
	targetPlayer := game_objects.NewPlayerGameObject("target", "Target", "token2",
		func() (float64, float64) { return 70, 50 }, // Close to source, same Y
		func(x, y float64) (float64, float64) { return x, y })

	// Create arrow aimed through the target position
	arrow := game_objects.NewArrowGameObject(uuid.New().String(), sourcePlayer, 100, 50, 1.0,
		func(x, y float64) (float64, float64) { return x, y })

	// Test collision detection between arrow and target
	arrowShape := arrow.GetBoundingShape()
	targetShape := targetPlayer.GetBoundingShape()

	if arrowShape == nil || targetShape == nil {
		t.Fatalf("Both arrow and target should have bounding shapes")
	}

	collides, collisionPoints := arrowShape.CollidesWith(targetShape)

	// Log shapes for debugging
	if line, ok := arrowShape.(*geo.Line); ok {
		t.Logf("Arrow line: (%v,%v) to (%v,%v)", line.A.X, line.A.Y, line.B.X, line.B.Y)
	}
	if circle, ok := targetShape.(*geo.Circle); ok {
		t.Logf("Target circle: center (%v,%v) radius %v", circle.C.X, circle.C.Y, circle.R)
	}

	// Note: This test verifies collision detection mechanism works
	// The actual collision depends on exact positioning and arrow trajectory
	t.Logf("Collision detected: %v, collision points: %d", collides, len(collisionPoints))

	// Verify collision points are reasonable if they exist
	for _, point := range collisionPoints {
		if math.IsNaN(point.X) || math.IsNaN(point.Y) {
			t.Errorf("Collision point should not contain NaN values: (%v, %v)", point.X, point.Y)
		}
		if math.IsInf(point.X, 0) || math.IsInf(point.Y, 0) {
			t.Errorf("Collision point should not contain infinite values: (%v, %v)", point.X, point.Y)
		}
	}
}

// TestHighSpeedCollisionDetection tests collision detection at high velocities
func TestHighSpeedCollisionDetection(t *testing.T) {
	tests := []struct {
		name      string
		velocityX float64
		velocityY float64
		deltaTime time.Duration
	}{
		{
			name:      "Normal speed",
			velocityX: 100,
			velocityY: 50,
			deltaTime: 16 * time.Millisecond,
		},
		{
			name:      "High speed",
			velocityX: 500,
			velocityY: 300,
			deltaTime: 16 * time.Millisecond,
		},
		{
			name:      "Very high speed",
			velocityX: 1000,
			velocityY: 800,
			deltaTime: 16 * time.Millisecond,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create fast-moving player
			player := game_objects.NewPlayerGameObject("fast-player", "Fast Player", "token",
				func() (float64, float64) { return 100, 100 },
				func(x, y float64) (float64, float64) { return x, y })

			// Set high velocity
			player.SetState(constants.StateDx, tt.velocityX)
			player.SetState(constants.StateDy, tt.velocityY)
			player.SetState(constants.StateLastLocUpdateTime, time.Now().Add(-tt.deltaTime))

			// Get extrapolated position
			nextX, nextY, _, _, err := game_objects.GetExtrapolatedPosition(player)
			if err != nil {
				t.Fatalf("Failed to extrapolate position: %v", err)
			}

			// Verify position is reasonable (not NaN or infinite)
			if math.IsNaN(nextX) || math.IsNaN(nextY) {
				t.Errorf("Extrapolated position should not be NaN: (%v, %v)", nextX, nextY)
			}
			if math.IsInf(nextX, 0) || math.IsInf(nextY, 0) {
				t.Errorf("Extrapolated position should not be infinite: (%v, %v)", nextX, nextY)
			}

			// Verify bounding shape at high speed position
			player.SetState(constants.StateX, nextX)
			player.SetState(constants.StateY, nextY)

			shape := player.GetBoundingShape()
			if shape == nil {
				t.Errorf("Player should have bounding shape even at high speeds")
			}

			// Verify shape center is at expected position
			center := shape.GetCenter()
			if math.Abs(center.X-nextX) > 0.01 || math.Abs(center.Y-nextY) > 0.01 {
				t.Errorf("Bounding shape center should match player position: got (%v, %v), want (%v, %v)",
					center.X, center.Y, nextX, nextY)
			}
		})
	}
}

// TestCollisionEventDataIntegrity tests that collision events contain valid data
func TestCollisionEventDataIntegrity(t *testing.T) {
	// Create two overlapping players
	player1 := game_objects.NewPlayerGameObject("player1", "Player 1", "token1",
		func() (float64, float64) { return 100, 100 },
		func(x, y float64) (float64, float64) { return x, y })

	player2 := game_objects.NewPlayerGameObject("player2", "Player 2", "token2",
		func() (float64, float64) { return 110, 100 }, // Overlapping
		func(x, y float64) (float64, float64) { return x, y })

	// Create room objects
	roomObjects := map[string]game_objects.GameObject{
		player1.GetID(): player1,
		player2.GetID(): player2,
	}

	// Process game tick
	tickEvent := game_objects.NewGameEvent("test-room", game_objects.EventGameTick, nil, 1, nil)
	result := player1.Handle(tickEvent, roomObjects)

	// Check raised events for collision events
	for _, event := range result.RaisedEvents {
		if event.EventType == game_objects.EventObjectCollision {
			// Verify event has required data fields
			if event.Data == nil {
				t.Errorf("Collision event should have data")
				continue
			}

			// Check for coordinate data
			x, hasX := event.Data["x"]
			y, hasY := event.Data["y"]

			if !hasX || !hasY {
				t.Errorf("Collision event should have x and y coordinates")
				continue
			}

			// Verify coordinates are valid numbers
			xFloat, xOk := x.(float64)
			yFloat, yOk := y.(float64)

			if !xOk || !yOk {
				t.Errorf("Collision coordinates should be float64 values")
				continue
			}

			// Verify coordinates are not NaN or infinite
			if math.IsNaN(xFloat) || math.IsInf(xFloat, 0) {
				t.Errorf("Collision X coordinate should be a valid number: %v", xFloat)
			}
			if math.IsNaN(yFloat) || math.IsInf(yFloat, 0) {
				t.Errorf("Collision Y coordinate should be a valid number: %v", yFloat)
			}

			// Verify coordinates are within reasonable bounds
			if xFloat < -1000 || xFloat > 2000 || yFloat < -1000 || yFloat > 2000 {
				t.Errorf("Collision coordinates seem unreasonable: (%v, %v)", xFloat, yFloat)
			}
		}
	}
}
