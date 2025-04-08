package game_objects

import (
	"go-ws-server/pkg/server/geo"
	"sync"
)

// GameObjectHandleEventResult represents the result of a GameObject handling an event
type GameObjectHandleEventResult struct {
	StateChanged bool
	RaisedEvents []*GameEvent
}

// NewGameObjectHandleEventResult creates a new GameObjectHandleEventResult
func NewGameObjectHandleEventResult(stateChanged bool, raisedEvents []*GameEvent) *GameObjectHandleEventResult {
	return &GameObjectHandleEventResult{
		StateChanged: stateChanged,
		RaisedEvents: raisedEvents,
	}
}

type GameObjectProperty int

const (
	// true / false, indicates the object should stop when colliding with another solid object
	GameObjectPropertyIsSolid GameObjectProperty = iota
	// mass of the object in kilograms
	GameObjectPropertyMassKg
)

// GameObject is an interface that all game objects must implement
type GameObject interface {
	// GetID returns the unique identifier of the game object
	GetID() string

	// GetObjectType returns the type of the game object
	GetObjectType() string

	// Handle processes an event and returns the result
	Handle(event *GameEvent, roomObjects map[string]GameObject) *GameObjectHandleEventResult

	// GetState returns the current state of the game object
	GetState() map[string]interface{}

	// SetState sets a state value for the game object
	SetState(key string, value interface{})

	// GetStateValue gets a state value from the game object
	GetStateValue(key string) (interface{}, bool)

	// GetEventTypes returns the event types this object is interested in
	GetEventTypes() []EventType

	// GetBoundingShape returns the bounding shape of the game object
	GetBoundingShape() geo.Shape

	// GetProperty returns the game object's properties
	GetProperty(key GameObjectProperty) (interface{}, bool)
}

// BaseGameObject provides a base implementation of the GameObject interface
type BaseGameObject struct {
	ID         string
	ObjectType string
	State      map[string]interface{}
	Mutex      sync.RWMutex
}

// NewBaseGameObject creates a new BaseGameObject
func NewBaseGameObject(id string, objectType string) *BaseGameObject {
	return &BaseGameObject{
		ID:         id,
		ObjectType: objectType,
		State:      make(map[string]interface{}),
	}
}

// GetID returns the unique identifier of the game object
func (g *BaseGameObject) GetID() string {
	return g.ID
}

// GetObjectType returns the type of the game object
func (g *BaseGameObject) GetObjectType() string {
	return g.ObjectType
}

// Handle processes an event and returns the result
// This is a base implementation that should be overridden by specific game objects
func (g *BaseGameObject) Handle(event *GameEvent, roomObjects map[string]GameObject) *GameObjectHandleEventResult {
	// Base implementation does nothing
	return NewGameObjectHandleEventResult(false, nil)
}

// GetState returns the current state of the game object
// TODO support partial state updates
func (g *BaseGameObject) GetState() map[string]interface{} {
	g.Mutex.RLock()
	defer g.Mutex.RUnlock()

	// Create a copy of the state to avoid concurrent modification
	stateCopy := make(map[string]interface{})
	for k, v := range g.State {
		stateCopy[k] = v
	}
	// Add object type to state for the client
	stateCopy["objectType"] = g.ObjectType

	return stateCopy
}

// SetState sets a state value for the game object
func (g *BaseGameObject) SetState(key string, value interface{}) {
	g.Mutex.Lock()
	defer g.Mutex.Unlock()

	g.State[key] = value
}

// GetStateValue gets a state value from the game object
func (g *BaseGameObject) GetStateValue(key string) (interface{}, bool) {
	g.Mutex.RLock()
	defer g.Mutex.RUnlock()

	value, exists := g.State[key]
	return value, exists
}

// GetEventTypes returns the event types this object is interested in
func (g *BaseGameObject) GetEventTypes() []EventType {
	// Base implementation returns an empty list
	return []EventType{}
}

// GetBoundingShape returns the bounding shape of the game object
func (g *BaseGameObject) GetBoundingShape() geo.Shape {
	return nil
}

// GetProperty returns the game object's properties
func (g *BaseGameObject) GetProperty(key GameObjectProperty) (interface{}, bool) {
	return nil, false
}
