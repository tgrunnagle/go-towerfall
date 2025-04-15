package game_objects

import (
	"log"
	"math"
	"time"

	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/geo"

	"github.com/google/uuid"
)

// PlayerGameObject represents a player in the game
type PlayerGameObject struct {
	*BaseGameObject
	Name         string
	Token        string
	respawnTimer *time.Timer
	respawning   bool
}

// NewPlayerGameObject creates a new PlayerGameObject
func NewPlayerGameObject(id string, name string, token string) *PlayerGameObject {
	base := NewBaseGameObject(id, constants.ObjectTypePlayer)
	player := &PlayerGameObject{
		BaseGameObject: base,
		Name:           name,
		Token:          token,
		respawnTimer:   nil,
	}

	// Initialize player state
	player.SetState(constants.StateID, id)
	player.SetState(constants.StateName, name)
	player.SetState(constants.StateX, constants.PlayerStartingX) // Starting X position
	player.SetState(constants.StateY, constants.PlayerStartingY) // Starting Y position
	player.SetState(constants.StateDx, 0.0)                      // X velocity
	player.SetState(constants.StateDy, 0.0)                      // Y velocity
	player.SetState(constants.StateDir, math.Pi*3/2)             // Point downward
	player.SetState(constants.StateLastLocUpdateTime, time.Now())
	player.SetState(constants.StateRadius, constants.PlayerRadius)         // Player radius
	player.SetState(constants.StateHealth, constants.PlayerStartingHealth) // Player health
	player.SetState(constants.StateDead, false)                            // Player is not dead
	player.SetState(constants.StateShooting, false)                        // Player is not shooting
	player.SetState(constants.StateJumpCount, 0)

	player.respawning = false
	return player
}

// Handle processes events for the player
func (p *PlayerGameObject) Handle(event *GameEvent, roomObjects map[string]GameObject) *GameObjectHandleEventResult {
	if p.respawning {
		// If the player is respawning, send an update to the client
		p.respawning = false
		return NewGameObjectHandleEventResult(true, nil)
	}

	// Skip event handling if player is dead
	dead, exists := p.GetStateValue(constants.StateDead)
	if exists && dead.(bool) {
		return NewGameObjectHandleEventResult(false, nil)
	}

	stateChanged := false
	var raisedEvents []*GameEvent = nil
	switch event.EventType {
	case EventPlayerKeyInput:
		stateChanged, raisedEvents = p.handlePlayerKeyInput(event)
	case EventPlayerClickInput:
		stateChanged, raisedEvents = p.handlePlayerClickInput(event)
	case EventPlayerDirection:
		stateChanged, raisedEvents = p.handlePlayerDirection(event)
	case EventGameTick:
		stateChanged, raisedEvents = p.handleGameTick(event, roomObjects)
	}

	return NewGameObjectHandleEventResult(stateChanged, raisedEvents)
}

// GetState returns the current state of the game object
func (p *PlayerGameObject) GetState() map[string]interface{} {
	result := p.BaseGameObject.GetState() // Extrapolate position without updating state
	nextX, nextY, nextDx, nextDy, err := GetExtrapolatedPosition(p)
	if err != nil {
		log.Printf("Failed to extrapolate player position: %v", err)
		return result
	}
	result[constants.StateX] = nextX
	result[constants.StateY] = nextY
	result[constants.StateDx] = nextDx
	result[constants.StateDy] = nextDy
	return result
}

// handlePlayerKeyInput processes player key input events
func (p *PlayerGameObject) handlePlayerKeyInput(event *GameEvent) (bool, []*GameEvent) {
	// Check if this event is for this player
	playerID, ok := event.Data["playerId"].(string)
	if !ok || playerID != p.GetID() {
		return false, nil
	}

	key, ok := event.Data["key"].(string)
	if !ok {
		log.Printf("Invalid key type: %T", event.Data["key"])
		return false, nil
	}

	isDown, ok := event.Data["isDown"].(bool)
	if !ok {
		log.Printf("Invalid isDown type: %T", event.Data["isDown"])
		return false, nil
	}

	// extrapolate position from last update
	nextX, nextY, nextDx, nextDy, err := GetExtrapolatedPosition(p)
	if err != nil {
		log.Printf("Failed to extrapolate player position: %v", err)
		return false, nil
	}

	// Update velocity based on key event
	dx := nextDx
	dy := nextDy

	if isDown {
		switch key {
		case "W": // Jump
			// Only allow jumping if we haven't exceeded max jumps
			jumpCount, exists := p.GetStateValue(constants.StateJumpCount)
			if !exists || jumpCount.(int) < constants.PlayerMaxJumps {
				log.Printf("Player %s jumped", p.GetID())
				dy = -1.0 * constants.PlayerJumpSpeedMetersPerSec * constants.PxPerMeter
				if exists {
					p.SetState(constants.StateJumpCount, jumpCount.(int)+1)
				} else {
					p.SetState(constants.StateJumpCount, 1)
				}
			} else {
				log.Printf("Player %s cannot jump - max jumps reached", p.GetID())
			}
		case "S": // Dive
			// Only allow diving if we haven't exceeded max jumps
			jumpCount, exists := p.GetStateValue(constants.StateJumpCount)
			if !exists || jumpCount.(int) < constants.PlayerMaxJumps {
				log.Printf("Player %s dived", p.GetID())
				dy = 1.0 * constants.PlayerJumpSpeedMetersPerSec * constants.PxPerMeter
				if exists {
					p.SetState(constants.StateJumpCount, jumpCount.(int)+1)
				} else {
					p.SetState(constants.StateJumpCount, 1)
				}
			} else {
				log.Printf("Player %s cannot dive - max jumps reached", p.GetID())
			}
		case "A": // Left
			dx = -1.0 * constants.PlayerSpeedXMetersPerSec * constants.PxPerMeter
		case "D": // Right
			dx = 1.0 * constants.PlayerSpeedXMetersPerSec * constants.PxPerMeter
		}
	} else {
		// On key release, stop movement in that direction if we were moving that way
		switch key {
		case "A":
			if dx < 0 {
				dx = 0
			}
		case "D":
			if dx > 0 {
				dx = 0
			}
		}
	}

	p.SetState(constants.StateX, nextX)
	p.SetState(constants.StateY, nextY)
	p.SetState(constants.StateDx, dx)
	p.SetState(constants.StateDy, dy)
	p.SetState(constants.StateLastLocUpdateTime, time.Now())

	return true, nil
}

func (p *PlayerGameObject) handlePlayerClickInput(event *GameEvent) (bool, []*GameEvent) {
	// Check if this event is for this player
	playerID, ok := event.Data["playerId"].(string)
	if !ok || playerID != p.GetID() {
		return false, nil
	}

	// left click
	if event.Data["isDown"].(bool) && event.Data["button"].(int) == 0 {
		// if the x is within the room bounds, start shooting
		x := event.Data["x"].(float64)
		y := event.Data["y"].(float64)
		if x < 0 || x > constants.RoomSizePixelsX || y < 0 || y > constants.RoomSizePixelsY {
			return false, nil
		}
		p.SetState(constants.StateShooting, true)
		p.SetState(constants.StateShootingStartTime, time.Now())
		return true, nil
	}

	// left release
	if !event.Data["isDown"].(bool) && event.Data["button"].(int) == 0 {
		// if shooting, fire an arrow
		if shooting, exists := p.GetStateValue(constants.StateShooting); exists && shooting.(bool) {

			// stop shooting
			p.SetState(constants.StateShooting, false)

			// create an arrow
			startTime, _ := p.GetStateValue(constants.StateShootingStartTime)
			powerRatio := math.Min(time.Since(startTime.(time.Time)).Seconds()/constants.ArrowMaxPowerTimeSec, 1.0)
			xClick := event.Data["x"].(float64)
			yClick := event.Data["y"].(float64)
			arrow := NewArrowGameObject(uuid.New().String(), p, xClick, yClick, powerRatio)
			return true, []*GameEvent{NewGameEvent(
				"",
				EventObjectCreated,
				map[string]interface{}{
					"type":   constants.ObjectTypeArrow,
					"object": arrow,
				},
				1,
				p,
			)}
		}
		return false, nil
	}

	// right click
	if event.Data["button"].(int) == 2 {
		// if right click, stop shooting
		if shooting, exists := p.GetStateValue(constants.StateShooting); exists && shooting.(bool) {
			p.SetState(constants.StateShooting, false)
			return true, nil
		}
		return false, nil
	}
	return false, nil
}

func (p *PlayerGameObject) GetBoundingShape() geo.Shape {
	x, exists := p.GetStateValue(constants.StateX)
	if !exists {
		log.Printf("Player %s has no x state", p.GetID())
		return nil
	}
	y, exists := p.GetStateValue(constants.StateY)
	if !exists {
		log.Printf("Player %s has no y state", p.GetID())
		return nil
	}
	return p.getBoundingShapeFor(x.(float64), y.(float64))
}

func (p *PlayerGameObject) getBoundingShapeFor(x float64, y float64) geo.Shape {
	point := geo.NewPoint(x, y)
	return geo.NewCircle(point, constants.PlayerRadius)
}

func (p *PlayerGameObject) handlePlayerDirection(event *GameEvent) (bool, []*GameEvent) {
	// Check if this event is for this player
	playerID, ok := event.Data["playerId"].(string)
	if !ok || playerID != p.GetID() {
		return false, nil
	}

	direction, ok := event.Data["direction"].(float64)
	if !ok {
		log.Printf("Invalid direction type: %T", event.Data["direction"])
		return false, nil
	}

	p.SetState(constants.StateDir, direction)
	return true, nil
}

func (p *PlayerGameObject) handleGameTick(event *GameEvent, roomObjects map[string]GameObject) (bool, []*GameEvent) {
	// Check collisions with next location
	nextX, nextY, nextDx, nextDy, err := GetExtrapolatedPosition(p)
	if err != nil {
		log.Printf("Failed to extrapolate player position: %v", err)
		return false, nil
	}
	shape := p.getBoundingShapeFor(nextX, nextY)

	events := []*GameEvent{}
	isOnGround := false

	// Check for collisions with solid objects and enemy bullets
	for _, object := range roomObjects {
		if object == p {
			continue
		}
		iIsSolid, exists := object.GetProperty(GameObjectPropertyIsSolid)
		isSolid := exists && iIsSolid.(bool)
		isEnemyBullet := object.GetObjectType() == constants.ObjectTypeBullet && object.(*BulletGameObject).SourcePlayer != p
		if !isSolid && !isEnemyBullet {
			continue
		}

		otherShape := object.GetBoundingShape()
		if otherShape == nil {
			continue
		}
		collides, collisionPoints := shape.CollidesWith(otherShape)
		if collides {
			if isSolid {
				// Collision detected, stop moving
				// TODO only stop moving in the direction of the collision
				// Get the average of the angle of the collision point to the player x, y (center of the bounding shape)
				var totalAngle float64
				var count int
				for _, point := range collisionPoints {
					angle := math.Atan2(point.Y-nextY, point.X-nextX)
					totalAngle += angle
					count++
				}
				if count > 0 {
					averageAngle := totalAngle / float64(count)
					if math.Abs(math.Cos(averageAngle)) > 0.1 {
						nextDx = 0.0
					}
					if math.Abs(math.Sin(averageAngle)) > 0.1 {
						nextDy = 0.0
						isOnGround = true
					}
					nextX, nextY, _ = GetExtrapolatedPositionForDxDy(p, nextDx, nextDy)
				}
			}
			if isEnemyBullet {
				// Report collision to bullet
				if len(collisionPoints) > 0 {
					x := collisionPoints[0].X
					y := collisionPoints[0].Y
					object.(*BulletGameObject).ReportCollision(p, x, y)
				} else {
					log.Printf("Player %s collided with bullet %s but no collision point found", p.GetID(), object.GetID())
				}

				currentHealth, exists := p.GetStateValue(constants.StateHealth)
				if !exists {
					log.Printf("Player %s has no health state", p.GetID())
					return false, nil
				}
				newHealth := currentHealth.(float64) - 10.0
				p.SetState(constants.StateHealth, newHealth)

				if newHealth <= 0.0 {
					p.handleDeath()

					events = append(events, NewGameEvent(
						event.RoomID,
						EventPlayerDied,
						map[string]interface{}{
							"objectID": p.GetID(),
							"x":        shape.GetCenter().X,
							"y":        shape.GetCenter().Y,
						},
						10,
						p,
					))

					// Stop processing further collisions if the player dies
					return true, events
				}
			}
		}
	}

	// Reset jump count when on ground
	if isOnGround {
		p.SetState(constants.StateJumpCount, 0)
	}

	p.SetState(constants.StateX, nextX)
	p.SetState(constants.StateY, nextY)
	p.SetState(constants.StateDx, nextDx)
	p.SetState(constants.StateDy, nextDy)
	p.SetState(constants.StateLastLocUpdateTime, time.Now())

	return true, events
}

func (p *PlayerGameObject) handleDeath() {
	p.SetState(constants.StateDead, true)
	p.SetState(constants.StateDx, 0.0)
	p.SetState(constants.StateDy, 0.0)
	p.SetState(constants.StateLastLocUpdateTime, time.Now())
	p.respawnTimer = time.AfterFunc(constants.PlayerRespawnTimeSec*time.Second, func() {
		p.SetState(constants.StateDead, false)
		p.SetState(constants.StateHealth, constants.PlayerStartingHealth)
		p.SetState(constants.StateX, constants.PlayerStartingX)
		p.SetState(constants.StateY, constants.PlayerStartingY)
		p.SetState(constants.StateDx, 0.0)
		p.SetState(constants.StateDy, 0.0)
		p.SetState(constants.StateDir, math.Pi*3/2)
		p.SetState(constants.StateLastLocUpdateTime, time.Now())
		p.respawnTimer = nil
		log.Printf("Player %s respawned", p.GetID())
		p.respawning = true
	})
}

// GetProperty returns the game object's properties
func (p *PlayerGameObject) GetProperty(key GameObjectProperty) (interface{}, bool) {
	switch key {
	case GameObjectPropertyMassKg:
		return constants.PlayerMassKg, true
	default:
		return nil, false
	}
}

// GetEventTypes returns the event types this player is interested in
func (p *PlayerGameObject) GetEventTypes() []EventType {
	return []EventType{
		EventPlayerKeyInput,
		EventPlayerClickInput,
		EventPlayerDirection,
		EventGameTick,
	}
}
