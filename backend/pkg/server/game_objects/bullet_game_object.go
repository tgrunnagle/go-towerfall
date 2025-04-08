package game_objects

import (
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/geo"
	"log"
	"math"
	"time"
)

// BulletGameObject represents a bullet in the game

type BulletGameObject struct {
	*BaseGameObject
	SourcePlayer GameObject
	CreatedAt    time.Time
}

// NewBulletGameObject creates a new BulletGameObject
func NewBulletGameObject(id string, source GameObject, targetX float64, targetY float64) *BulletGameObject {
	base := NewBaseGameObject(id, constants.ObjectTypeBullet)
	x, exists := source.GetStateValue(constants.StateX)
	if !exists {
		log.Printf("Failed to get x position for bullet from source: %v", source.GetState())
		return nil
	}
	y, exists := source.GetStateValue(constants.StateY)
	if !exists {
		log.Printf("Failed to get y position for bullet from source: %v", source.GetState())
		return nil
	}

	base.SetState(constants.StateX, x)
	base.SetState(constants.StateY, y)
	base.SetState(constants.StateLastLocUpdateTime, time.Now())

	dx := targetX - x.(float64)
	dy := targetY - y.(float64)
	// normalize
	length := math.Sqrt(dx*dx + dy*dy)
	if length > 0 {
		dx /= length
		dy /= length
	}
	base.SetState(constants.StateDx, dx)
	base.SetState(constants.StateDy, dy)
	return &BulletGameObject{
		BaseGameObject: base,
		SourcePlayer:   source,
		CreatedAt:      time.Now(),
	}
}

// Handle processes events for the bullet
func (b *BulletGameObject) Handle(event *GameEvent, roomObjects map[string]GameObject) *GameObjectHandleEventResult {
	switch event.EventType {
	case EventGameTick:
		return b.handleGameTick(event)
	}
	// Base implementation does nothing
	return NewGameObjectHandleEventResult(false, nil)
}

// GetState returns the current state of the game object
func (b *BulletGameObject) GetState() map[string]interface{} {
	result := b.BaseGameObject.GetState()
	// Extrapolate position without updating state
	nextX, nextY, err := GetExtrapolatedPosition(b)
	if err != nil {
		log.Printf("Failed to extrapolate bullet position: %v", err)
		return result
	}
	result[constants.StateX] = nextX
	result[constants.StateY] = nextY
	return result
}

// handleGameTick processes game tick events for the bullet
func (b *BulletGameObject) handleGameTick(event *GameEvent) *GameObjectHandleEventResult {

	// Destroy the bullet after a tick. Specify where if it collided with something.
	events := make([]*GameEvent, 0)
	x, _ := b.GetStateValue(constants.StateDestroyedAtX)
	y, _ := b.GetStateValue(constants.StateDestroyedAtY)
	if x != nil && y != nil {
		events = append(events, NewGameEvent(
			event.RoomID,
			EventObjectDestroyed,
			map[string]interface{}{
				"objectID": b.GetID(),
				"x":        x.(float64),
				"y":        y.(float64),
			},
			10,
			b,
		))
	} else {
		events = append(events, NewGameEvent(
			event.RoomID,
			EventObjectDestroyed,
			map[string]interface{}{
				"objectID": b.GetID(),
			},
			10,
			b,
		))
	}

	return NewGameObjectHandleEventResult(true, events)
}

func (b *BulletGameObject) ReportCollision(source GameObject, collidedAtX float64, collidedAtY float64) {
	// b.Mutex.Lock()
	b.SetState(constants.StateDestroyedAtX, collidedAtX)
	b.SetState(constants.StateDestroyedAtY, collidedAtY)
	// b.Mutex.Unlock()
}

// GetEventTypes returns the event types this bullet is interested in
func (b *BulletGameObject) GetEventTypes() []EventType {
	return []EventType{
		EventGameTick,
	}
}

func (b *BulletGameObject) GetBoundingShape() geo.Shape {
	x0, exists := b.GetStateValue(constants.StateX)
	if !exists {
		log.Printf("GetBoundingShape: Failed to get x position for bullet: %v", b.GetState())
		return nil
	}
	y0, exists := b.GetStateValue(constants.StateY)
	if !exists {
		log.Printf("GetBoundingShape: Failed to get y position for bullet: %v", b.GetState())
		return nil
	}
	dx, exists := b.GetStateValue(constants.StateDx)
	if !exists {
		log.Printf("GetBoundingShape: Failed to get dx for bullet: %v", b.GetState())
		return nil
	}
	dy, exists := b.GetStateValue(constants.StateDy)
	if !exists {
		log.Printf("GetBoundingShape: Failed to get dy for bullet: %v", b.GetState())
		return nil
	}
	p0 := geo.NewPoint(x0.(float64), y0.(float64))
	p1 := geo.NewPoint(
		x0.(float64)+constants.BulletDistance*dx.(float64),
		y0.(float64)+constants.BulletDistance*dy.(float64),
	)
	return geo.NewLine(p0, p1)
}

// GetProperty returns the game object's properties
func (b *BulletGameObject) GetProperty(key GameObjectProperty) (interface{}, bool) {
	return nil, false
}
