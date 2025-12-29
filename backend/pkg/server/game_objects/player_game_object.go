package game_objects

import (
	"log"
	"math"
	"strings"
	"sync"
	"time"

	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/geo"

	"github.com/google/uuid"
)

// PlayerGameObject represents a player in the game
type PlayerGameObject struct {
	*BaseGameObject
	Name               string
	Token              string
	respawnTimer       *time.Timer
	respawning         bool
	respawnMutex       sync.Mutex // Protects respawnTimer and respawning fields
	getRespawnLocation func() (float64, float64)
	wrapPosition       func(float64, float64) (float64, float64)
	respawnTimeSec     float64
}

// NewPlayerGameObject creates a new PlayerGameObject with default respawn time
func NewPlayerGameObject(
	id string,
	name string,
	token string,
	getRespawnLocation func() (float64, float64),
	wrapPosition func(float64, float64) (float64, float64),
) *PlayerGameObject {
	return NewPlayerGameObjectWithRespawnTime(id, name, token, getRespawnLocation, wrapPosition, constants.PlayerRespawnTimeSec)
}

// NewPlayerGameObjectWithRespawnTime creates a new PlayerGameObject with configurable respawn time
func NewPlayerGameObjectWithRespawnTime(
	id string,
	name string,
	token string,
	getRespawnLocation func() (float64, float64),
	wrapPosition func(float64, float64) (float64, float64),
	respawnTimeSec float64,
) *PlayerGameObject {
	base := NewBaseGameObject(id, constants.ObjectTypePlayer)
	player := &PlayerGameObject{
		BaseGameObject:     base,
		Name:               name,
		Token:              token,
		respawnTimer:       nil,
		getRespawnLocation: getRespawnLocation,
		wrapPosition:       wrapPosition,
		respawnTimeSec:     respawnTimeSec,
	}
	player.SetState(constants.StateID, id)
	player.SetState(constants.StateName, name)
	respawnX, respawnY := getRespawnLocation()
	player.SetState(constants.StateX, respawnX)      // Starting X position
	player.SetState(constants.StateY, respawnY)      // Starting Y position
	player.SetState(constants.StateDx, 0.0)          // X velocity
	player.SetState(constants.StateDy, 0.0)          // Y velocity
	player.SetState(constants.StateDir, math.Pi*3/2) // Point downward
	player.SetState(constants.StateLastLocUpdateTime, time.Now())
	player.SetState(constants.StateRadius, constants.PlayerRadius)         // Player radius
	player.SetState(constants.StateHealth, constants.PlayerStartingHealth) // Player health
	player.SetState(constants.StateDead, false)                            // Player is not dead
	player.SetState(constants.StateShooting, false)                        // Player is not shooting
	player.SetState(constants.StateJumpCount, 0)
	player.SetState(constants.StateArrowCount, constants.PlayerStartingArrows) // Starting arrows

	player.respawning = false
	return player
}

// Handle processes events for the player
func (p *PlayerGameObject) Handle(event *GameEvent, roomObjects map[string]GameObject) *GameObjectHandleEventResult {
	// Check if player is respawning (protected by mutex since doRespawn runs in timer goroutine)
	p.respawnMutex.Lock()
	isRespawning := p.respawning
	if isRespawning {
		p.respawning = false
	}
	p.respawnMutex.Unlock()

	if isRespawning {
		// If the player is respawning, send an update to the client
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

	key = strings.ToUpper(key)
	if isDown {
		switch key {
		case "W": // Jump
			// Only allow jumping if we haven't exceeded max jumps
			jumpCount, exists := p.GetStateValue(constants.StateJumpCount)
			if !exists || jumpCount.(int) < constants.PlayerMaxJumps {
				dy = -1.0 * constants.PlayerJumpSpeedMetersPerSec * constants.PxPerMeter
				if exists {
					p.SetState(constants.StateJumpCount, jumpCount.(int)+1)
				} else {
					p.SetState(constants.StateJumpCount, 1)
				}
			}
		case "S": // Dive
			// Only allow diving if we haven't exceeded max jumps
			jumpCount, exists := p.GetStateValue(constants.StateJumpCount)
			if !exists || jumpCount.(int) < constants.PlayerMaxJumps {
				dy = 1.0 * constants.PlayerJumpSpeedMetersPerSec * constants.PxPerMeter
				if exists {
					p.SetState(constants.StateJumpCount, jumpCount.(int)+1)
				} else {
					p.SetState(constants.StateJumpCount, 1)
				}
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
		// Check if already shooting
		shooting, exists := p.GetStateValue(constants.StateShooting)
		if exists && shooting.(bool) {
			return false, nil
		}

		// Check if we have arrows
		arrowCount, exists := p.GetStateValue(constants.StateArrowCount)
		if !exists || arrowCount.(int) <= 0 {
			return false, nil
		}

		// if the x is within the room bounds, start shooting
		x := event.Data["x"].(float64)
		y := event.Data["y"].(float64)
		if x < 0 || x > constants.RoomSizePixelsX || y < 0 || y > constants.RoomSizePixelsY {
			return false, nil
		}

		// Start shooting
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
			// Double heck if we have arrows
			arrowCount, exists := p.GetStateValue(constants.StateArrowCount)
			if !exists {
				log.Printf("Player %s has no arrow count state", p.GetID())
				return false, nil
			}
			if arrowCount.(int) <= 0 {
				return false, nil
			}

			p.SetState(constants.StateArrowCount, arrowCount.(int)-1) // Decrement arrow count

			// create an arrow
			startTime, _ := p.GetStateValue(constants.StateShootingStartTime)
			powerRatio := math.Min(time.Since(startTime.(time.Time)).Seconds()/constants.ArrowMaxPowerTimeSec, 1.0)
			xClick := event.Data["x"].(float64)
			yClick := event.Data["y"].(float64)
			arrow := NewArrowGameObject(uuid.New().String(), p, xClick, yClick, powerRatio, p.wrapPosition)
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
	stateChanged := false
	var raisedEvents []*GameEvent

	nextX, nextY, nextDx, nextDy, err := GetExtrapolatedPosition(p)
	if err != nil {
		log.Printf("Failed to extrapolate player position: %v", err)
		return false, nil
	}

	playerShape := p.getBoundingShapeFor(nextX, nextY)
	if playerShape == nil {
		log.Printf("Player %s has no bounding shape", p.GetID())
		return false, nil
	}

	x, exists := p.GetStateValue(constants.StateX)
	if !exists {
		log.Printf("Player %s has no x state", p.GetID())
		return false, nil
	}
	y, exists := p.GetStateValue(constants.StateY)
	if !exists {
		log.Printf("Player %s has no y state", p.GetID())
		return false, nil
	}
	dx, exists := p.GetStateValue(constants.StateDx)
	if !exists {
		log.Printf("Player %s has no dx state", p.GetID())
		return false, nil
	}
	dy, exists := p.GetStateValue(constants.StateDy)
	if !exists {
		log.Printf("Player %s has no dy state", p.GetID())
		return false, nil
	}

	isOnGround := false
	died := false

	// Check for collisions with other objects
	for _, obj := range roomObjects {
		if obj.GetID() == p.GetID() {
			continue
		}

		otherShape := obj.GetBoundingShape()
		if otherShape == nil {
			continue
		}

		collides, collisionPoints := otherShape.CollidesWith(playerShape)
		if !collides {
			continue
		}

		// Handle collisions with solid objects
		if isSolid, exists := obj.GetProperty(GameObjectPropertyIsSolid); exists && isSolid.(bool) {
			// Adjust movement based on average angle of collision points
			var avgAngle float64
			for _, point := range collisionPoints {
				avgAngle += math.Atan2(point.Y-nextY, point.X-nextX)
				raisedEvents = append(raisedEvents, NewGameEvent(
					event.RoomID,
					EventObjectCollision,
					map[string]interface{}{
						"x": point.X,
						"y": point.Y,
					},
					1,
					p,
				))
			}
			avgAngle /= float64(len(collisionPoints))

			// Normalize angle to be between -π and π
			normalizedAngle := math.Mod(avgAngle+math.Pi, 2*math.Pi) - math.Pi

			// Check for horizontal collisions (left/right)
			// Right collision (angle close to 0)
			if math.Abs(normalizedAngle) < constants.CollisionAngleThreshold {
				if dx.(float64) > 0 { // Only stop if moving right
					stateChanged = true
					nextX = x.(float64)
					nextDx = 0.0
				}
			}
			// Left collision (angle close to π or -π)
			if math.Abs(normalizedAngle-math.Pi) < constants.CollisionAngleThreshold {
				if dx.(float64) < 0 { // Only stop if moving left
					stateChanged = true
					nextX = x.(float64)
					nextDx = 0.0
				}
			}

			// Check for vertical collisions (up/down)
			// Note: angle is reflected over 0 due to y axis being inverted
			// Up collision (angle close to -π/2)
			if math.Abs(normalizedAngle+math.Pi/2) < constants.CollisionAngleThreshold {
				if dy.(float64) < 0 { // Only stop if moving up
					stateChanged = true
					nextY = y.(float64)
					nextDy = 0.0
				}
			}
			// Down collision (angle close to π/2)
			if math.Abs(normalizedAngle-math.Pi/2) < constants.CollisionAngleThreshold {
				isOnGround = true
				if dy.(float64) > 0 { // Only stop if moving down
					stateChanged = true
					nextY = y.(float64)
					nextDy = 0.0
				}
			}
		}

		// Handle collisions with arrows
		objType := obj.GetObjectType()
		switch objType {
		case constants.ObjectTypeArrow:
			// Check if arrow is grounded
			if grounded, exists := obj.GetStateValue(constants.StateArrowGrounded); exists && grounded.(bool) {
				// Pick up arrow if we have room
				if arrowCount, exists := p.GetStateValue(constants.StateArrowCount); exists && arrowCount.(int) < constants.PlayerMaxArrows {
					p.SetState(constants.StateArrowCount, arrowCount.(int)+1)
					// Mark arrow as destroyed
					obj.SetState(constants.StateDestroyedAtX, collisionPoints[0].X)
					obj.SetState(constants.StateDestroyedAtY, collisionPoints[0].Y)
					obj.SetState(constants.StateDestroyed, true)
					stateChanged = true
				}
			} else {
				// Check if the arrow was shot by another player
				if obj.(*ArrowGameObject).SourcePlayer != p {
					// Handle regular arrow collision (damage)
					p.handleDeath()
					died = true

					// Create a grounded arrow if the player has any
					if arrowCount, exists := p.GetStateValue(constants.StateArrowCount); exists && arrowCount.(int) > 0 {
						arrow := NewArrowGameObject(uuid.New().String(), p, playerShape.GetCenter().X, playerShape.GetCenter().Y+(playerShape.(*geo.Circle).R), 0.0, p.wrapPosition)
						arrow.SetState(constants.StateArrowGrounded, true)
						raisedEvents = append(raisedEvents, NewGameEvent(
							event.RoomID,
							EventObjectCreated,
							map[string]interface{}{
								"type":   constants.ObjectTypeArrow,
								"object": arrow,
							},
							1,
							p,
						))
					}

					// Mark arrow as destroyed
					obj.SetState(constants.StateDestroyedAtX, collisionPoints[0].X)
					obj.SetState(constants.StateDestroyedAtY, collisionPoints[0].Y)
					obj.SetState(constants.StateDestroyed, true)

					raisedEvents = append(raisedEvents, NewGameEvent(
						event.RoomID,
						EventPlayerDied,
						map[string]interface{}{
							"objectID": p.GetID(),
							"x":        playerShape.GetCenter().X,
							"y":        playerShape.GetCenter().Y,
						},
						10,
						p,
					))
					stateChanged = true
				}
			}
		}
	}

	if x.(float64)-nextX != 0 || y.(float64)-nextY != 0 || nextDx != 0 || nextDy != 0 {
		stateChanged = true
	}

	if isOnGround {
		jumpCount, exists := p.GetStateValue(constants.StateJumpCount)
		if exists && jumpCount.(int) != 0 {
			stateChanged = true
			p.SetState(constants.StateJumpCount, 0)
		}
	}

	// Update location state
	if !died {
		nextX, nextY = p.wrapPosition(nextX, nextY)

		p.SetState(constants.StateX, nextX)
		p.SetState(constants.StateY, nextY)
		p.SetState(constants.StateDx, nextDx)
		p.SetState(constants.StateDy, nextDy)
		p.SetState(constants.StateLastLocUpdateTime, time.Now())
	}

	return stateChanged, raisedEvents
}

func (p *PlayerGameObject) handleDeath() {
	p.SetState(constants.StateDead, true)
	p.SetState(constants.StateDx, 0.0)
	p.SetState(constants.StateDy, 0.0)
	p.SetState(constants.StateLastLocUpdateTime, time.Now())

	// Use configured respawn time (0 = instant respawn)
	respawnTime := p.respawnTimeSec
	if respawnTime <= 0 {
		// Instant respawn
		p.doRespawn()
	} else {
		p.respawnMutex.Lock()
		p.respawnTimer = time.AfterFunc(time.Duration(respawnTime*float64(time.Second)), func() {
			p.doRespawn()
		})
		p.respawnMutex.Unlock()
	}
}

func (p *PlayerGameObject) doRespawn() {
	p.SetState(constants.StateDead, false)
	p.SetState(constants.StateHealth, constants.PlayerStartingHealth)
	respawnX, respawnY := p.getRespawnLocation()
	p.SetState(constants.StateX, respawnX)
	p.SetState(constants.StateY, respawnY)
	p.SetState(constants.StateDx, 0.0)
	p.SetState(constants.StateDy, 0.0)
	p.SetState(constants.StateDir, math.Pi*3/2)
	p.SetState(constants.StateLastLocUpdateTime, time.Now())

	p.respawnMutex.Lock()
	p.respawnTimer = nil
	p.respawning = true
	p.respawnMutex.Unlock()
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
