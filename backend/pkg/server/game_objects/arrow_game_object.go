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

func NewArrowGameObject(id string, source GameObject, toX float64, toY float64, powerRatio float64) *ArrowGameObject {
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

	// calculate initial velocity based on power ratio, E = 0.5*m*v^2
	initialVelocityPxPerSec := math.Sqrt(2*powerRatio*constants.ArrowMaxPowerNewton/constants.ArrowMassKg) * constants.PxPerMeter
	distanceX := toX - x.(float64)
	distanceY := toY - y.(float64)
	dxNorm := distanceX / math.Sqrt(distanceX*distanceX+distanceY*distanceY)
	dyNorm := distanceY / math.Sqrt(distanceX*distanceX+distanceY*distanceY)
	dx := dxNorm * initialVelocityPxPerSec
	dy := dyNorm * initialVelocityPxPerSec

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

func (a *ArrowGameObject) GetState() map[string]interface{} {
	state := a.BaseGameObject.GetState()

	// supplement with 'dir' based on dx, dy
	dx := state[constants.StateDx].(float64)
	dy := state[constants.StateDy].(float64)
	normDx := dx / math.Sqrt(dx*dx+dy*dy)
	normDy := dy / math.Sqrt(dx*dx+dy*dy)
	dir := math.Atan2(normDy, normDx)
	state[constants.StateDir] = dir
	return state
}

func (a *ArrowGameObject) Handle(event *GameEvent, roomObjects map[string]GameObject) *GameObjectHandleEventResult {
	switch event.EventType {
	case EventGameTick:
		return a.handleGameTick(event, roomObjects)
	}
	log.Printf("ArrowGameObject: Unexpected event: %v", event.EventType)
	// Base implementation does nothing
	return NewGameObjectHandleEventResult(false, nil)
}

func (a *ArrowGameObject) handleGameTick(event *GameEvent, roomObjects map[string]GameObject) *GameObjectHandleEventResult {
	// Check collisions with next location
	nextX, nextY, nextDx, nextDy, err := GetExtrapolatedPosition(a)
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
	// calculate direction based on nextDx, nextDy
	shape := a.getBoundingShapeFor(nextX, nextY, nextDx, nextDy)
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
		log.Printf("Arrow %s grounded", a.GetID())
		a.SetState(constants.StateArrowGrounded, true)
		nextDx = 0.0
		nextDy = 0.0
	}

	a.SetState(constants.StateDx, nextDx)
	a.SetState(constants.StateDy, nextDy)
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
		log.Printf("ArrowGameObject.GetBoundingShape: Failed to get x position for arrow: %v", a.GetState())
		return nil
	}
	y, exists := a.GetStateValue(constants.StateY)
	if !exists {
		log.Printf("ArrowGameObject.GetBoundingShape: Failed to get y position for arrow: %v", a.GetState())
		return nil
	}
	dx, exists := a.GetStateValue(constants.StateDx)
	if !exists {
		log.Printf("ArrowGameObject.GetBoundingShape: Failed to get dx for arrow: %v", a.GetState())
		return nil
	}
	dy, exists := a.GetStateValue(constants.StateDy)
	if !exists {
		log.Printf("ArrowGameObject.GetBoundingShape: Failed to get dy for arrow: %v", a.GetState())
		return nil
	}

	return a.getBoundingShapeFor(x.(float64), y.(float64), dx.(float64), dy.(float64))
}

func (a *ArrowGameObject) getBoundingShapeFor(x float64, y float64, dX float64, dY float64) geo.Shape {

	if grounded, exists := a.GetStateValue(constants.StateArrowGrounded); exists && grounded.(bool) {
		return geo.NewCircle(geo.NewPoint(x, y), constants.ArrowGroundedRadiusPx)
	}

	dXNorm := dX / math.Sqrt(dX*dX+dY*dY)
	dYNorm := dY / math.Sqrt(dX*dX+dY*dY)
	p0 := geo.NewPoint(x, y)
	p1 := geo.NewPoint(
		x+dXNorm*constants.ArrowLengthPx,
		y+dYNorm*constants.ArrowLengthPx,
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

func (a *ArrowGameObject) GetEventTypes() []EventType {
	return []EventType{
		EventGameTick,
	}
}
