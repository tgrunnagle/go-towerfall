package game_objects

import (
	"log"
	"math"
	"time"

	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/geo"
)

// ArrowGameObject represents an arrow in the game
type ArrowGameObject struct {
	*BaseGameObject
	SourcePlayer GameObject
	CreatedAt    time.Time
}

func NewArrowGameObject(id string, source GameObject, direction float64, powerRatio float64) *ArrowGameObject {
	base := NewBaseGameObject(id, constants.ObjectTypeArrow)
	x, exists := source.GetStateValue(constants.StateX)
	if !exists {
		log.Printf("Failed to get x position for arrow from source: %v", source.GetState())
		return nil
	}
	y, exists := source.GetStateValue(constants.StateY)
	if !exists {
		log.Printf("Failed to get y position for arrow from source: %v", source.GetState())
		return nil
	}

	base.SetState(constants.StateX, x)
	base.SetState(constants.StateY, y)

	initialVelocityPxPerSec := math.Sqrt(2*powerRatio*constants.ArrowMaxPowerNewton/constants.ArrowMassKg) * constants.PxPerMeter
	dx := math.Cos(direction) * initialVelocityPxPerSec
	dy := math.Sin(direction) * initialVelocityPxPerSec
	base.SetState(constants.StateDx, dx)
	base.SetState(constants.StateDy, dy)
	base.SetState(constants.StateLastLocUpdateTime, time.Now())
	base.SetState(constants.StateArrowGrounded, false)
	return &ArrowGameObject{
		BaseGameObject: base,
		SourcePlayer:   source,
		CreatedAt:      time.Now(),
	}
}

func (a *ArrowGameObject) Handle(event *GameEvent, roomObjects map[string]GameObject) *GameObjectHandleEventResult {
	switch event.EventType {
	case EventGameTick:
		return a.handleGameTick(event, roomObjects)
	}
	// Base implementation does nothing
	return NewGameObjectHandleEventResult(false, nil)
}

func (a *ArrowGameObject) handleGameTick(event *GameEvent, roomObjects map[string]GameObject) *GameObjectHandleEventResult {
	// Check collisions with next location
	nextX, nextY, _, _, err := GetExtrapolatedPosition(a)
	if err != nil {
		log.Printf("Failed to extrapolate arrow position: %v", err)
		return NewGameObjectHandleEventResult(false, nil)
	}

	// Check if arrow is destroyed (e.g. from hitting a player)
	destroyed, exists := a.GetStateValue(constants.StateDestroyed)
	if exists && destroyed.(bool) {
		x, exists := a.GetStateValue(constants.StateDestroyedAtX)
		if !exists {
			x, _ = a.GetStateValue(constants.StateX)
		}
		y, exists := a.GetStateValue(constants.StateDestroyedAtY)
		if !exists {
			y, _ = a.GetStateValue(constants.StateY)
		}
		events := []*GameEvent{NewGameEvent(
			event.RoomID,
			EventObjectDestroyed,
			map[string]interface{}{
				"objectID": a.GetID(),
				"x":        x.(float64),
				"y":        y.(float64),
			},
			10,
			a,
		)}
		return NewGameObjectHandleEventResult(true, events)
	}

	// Check if arrow is out of bounds
	if nextX < -1*constants.ArrowDestroyDistancePx ||
		nextX > constants.RoomSizePixelsX+constants.ArrowDestroyDistancePx ||
		nextY < -1*constants.ArrowDestroyDistancePx ||
		nextY > constants.RoomSizePixelsY+constants.ArrowDestroyDistancePx {

		a.SetState(constants.StateDestroyed, true)
		a.SetState(constants.StateDestroyedAtX, nextX)
		a.SetState(constants.StateDestroyedAtY, nextY)
		events := []*GameEvent{NewGameEvent(
			event.RoomID,
			EventObjectDestroyed,
			map[string]interface{}{
				"objectID": a.GetID(),
				"x":        nextX,
				"y":        nextY,
			},
			10,
			a,
		)}
		return NewGameObjectHandleEventResult(true, events)
	}

	// Check for collisions with other objects
	dir, exists := a.GetStateValue(constants.StateDir)
	if !exists {
		log.Printf("GetBoundingShape: Failed to get direction for arrow: %v", a.GetState())
		return NewGameObjectHandleEventResult(false, nil)
	}

	shape := a.getBoundingShapeFor(nextX, nextY, dir.(float64))
	grounded := false
	for _, object := range roomObjects {
		if object == a {
			continue
		}
		isSolid, exists := object.GetProperty(GameObjectPropertyIsSolid)
		if !exists || !isSolid.(bool) {
			continue
		}

		otherShape := object.GetBoundingShape()
		if otherShape == nil {
			continue
		}
		collides, _ := shape.CollidesWith(otherShape)
		if collides {
			grounded = true
			// TODO: set nextX, nextY to the collision point closest to the current x, y
			break

		}
	}

	if grounded {
		a.SetState(constants.StateArrowGrounded, true)
		a.SetState(constants.StateDx, 0.0)
		a.SetState(constants.StateDy, 0.0)
	}

	a.SetState(constants.StateX, nextX)
	a.SetState(constants.StateY, nextY)
	a.SetState(constants.StateLastLocUpdateTime, time.Now())

	return NewGameObjectHandleEventResult(true, nil)
}

func (a *ArrowGameObject) ReportCollision(source GameObject, collidedAtX float64, collidedAtY float64) {
	// a.Mutex.Lock() // TODO will this deadlock?
	// defer a.Mutex.Unlock()
	a.SetState(constants.StateDestroyed, true)
	a.SetState(constants.StateDestroyedAtX, collidedAtX)
	a.SetState(constants.StateDestroyedAtY, collidedAtY)
}

func (a *ArrowGameObject) GetBoundingShape() geo.Shape {
	x, exists := a.GetStateValue(constants.StateX)
	if !exists {
		log.Printf("GetBoundingShape: Failed to get x position for arrow: %v", a.GetState())
		return nil
	}
	y, exists := a.GetStateValue(constants.StateY)
	if !exists {
		log.Printf("GetBoundingShape: Failed to get y position for arrow: %v", a.GetState())
		return nil
	}
	dir, exists := a.GetStateValue(constants.StateDir)
	if !exists {
		log.Printf("GetBoundingShape: Failed to get direction for arrow: %v", a.GetState())
		return nil
	}

	return a.getBoundingShapeFor(x.(float64), y.(float64), dir.(float64))
}

func (a *ArrowGameObject) getBoundingShapeFor(x float64, y float64, dir float64) geo.Shape {

	if grounded, exists := a.GetStateValue(constants.StateArrowGrounded); exists && grounded.(bool) {
		return geo.NewCircle(geo.NewPoint(x, y), constants.ArrowGroundedRadiusPx)
	}

	p0 := geo.NewPoint(x, y)
	p1 := geo.NewPoint(
		x+constants.ArrowLengthMeters*math.Cos(dir),
		y+constants.ArrowLengthMeters*math.Sin(dir),
	)
	return geo.NewLine(p0, p1)
}

// GetProperty returns the game object's properties
func (a *ArrowGameObject) GetProperty(key GameObjectProperty) (interface{}, bool) {
	switch key {
	case GameObjectPropertyMassKg:
		if grounded, exists := a.GetStateValue(constants.StateArrowGrounded); exists && grounded.(bool) {
			return nil, false
		}
		return constants.ArrowMassKg, true
	default:
		return nil, false
	}
}
