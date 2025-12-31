package server

import (
	"errors"
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/game_maps"
	"go-ws-server/pkg/server/game_objects"
	"log"
	"maps"
	"sync"
)

type GameObjectManagerHandleEventResult struct {
	AddedObjects   map[string]game_objects.GameObject
	RemovedObjects map[string]game_objects.GameObject
}

// GameObjectManager manages a collection of game objects
type GameObjectManager struct {
	Objects map[string]game_objects.GameObject
	Mutex   sync.RWMutex
	Map     game_maps.Map
}

// NewGameObjectManager creates a new GameObjectManager
func NewGameObjectManager(gameMap game_maps.Map) *GameObjectManager {
	return &GameObjectManager{
		Objects: make(map[string]game_objects.GameObject),
		Map:     gameMap,
	}
}

// AddObject adds a game object to the manager
func (m *GameObjectManager) AddObject(obj game_objects.GameObject) {
	m.Mutex.Lock()
	defer m.Mutex.Unlock()

	m.Objects[obj.GetID()] = obj
}

// RemoveObject removes a game object from the manager
func (m *GameObjectManager) RemoveObject(id string) {
	m.Mutex.Lock()
	defer m.Mutex.Unlock()

	// TODO figure out how to clean up Objects without dereference errors elsewhere
	m.Objects[id] = nil
}

// GetObject gets a game object by ID
func (m *GameObjectManager) GetObject(id string) (game_objects.GameObject, bool) {
	m.Mutex.RLock()
	defer m.Mutex.RUnlock()

	// Check dynamic objects first
	if obj, exists := m.Objects[id]; exists {
		return obj, true
	}

	// Check map objects
	if m.Map != nil {
		if obj, exists := m.Map.GetObjects()[id]; exists {
			return obj, true
		}
	}

	return nil, false
}

func (m *GameObjectManager) GetObjectIDs() []string {
	m.Mutex.RLock()
	defer m.Mutex.RUnlock()

	var ids []string
	// Add dynamic objects
	for id := range m.Objects {
		ids = append(ids, id)
	}
	// Add map objects
	if m.Map != nil {
		for id := range m.Map.GetObjects() {
			ids = append(ids, id)
		}
	}
	return ids
}

// GetAllObjects returns all game objects including map objects
func (m *GameObjectManager) GetAllObjects() map[string]game_objects.GameObject {
	m.Mutex.RLock()
	defer m.Mutex.RUnlock()
	result := make(map[string]game_objects.GameObject)

	// Add dynamic objects
	for id, obj := range m.Objects {
		if obj == nil {
			continue
		}
		result[id] = obj
	}

	// Add map objects
	if m.Map != nil {
		maps.Copy(result, m.Map.GetObjects())
	}
	return result
}

// GetAllStates returns the state of all game objects including map objects
func (m *GameObjectManager) GetAllStates() map[string]map[string]interface{} {
	m.Mutex.RLock()
	defer m.Mutex.RUnlock()

	allStates := make(map[string]map[string]interface{})

	// Add dynamic objects
	for id, obj := range m.Objects {
		if obj == nil {
			continue
		}
		allStates[id] = obj.GetState()
	}

	// Add map objects
	if m.Map != nil {
		for id, obj := range m.Map.GetObjects() {
			allStates[id] = obj.GetState()
		}
	}
	return allStates
}

// ClearNonPlayerObjects removes all non-player objects (arrows, bullets, etc.) from the manager
// This is used during game reset to clear the arena while keeping players
func (m *GameObjectManager) ClearNonPlayerObjects() {
	m.Mutex.Lock()
	defer m.Mutex.Unlock()

	for id, obj := range m.Objects {
		if obj == nil {
			continue
		}
		// Keep player objects, remove everything else (arrows, bullets, etc.)
		if obj.GetObjectType() != constants.ObjectTypePlayer {
			m.Objects[id] = nil
		}
	}
}

func (m *GameObjectManager) HandleEvent(e *game_objects.GameEvent) (*GameObjectManagerHandleEventResult, error) {
	if e == nil {
		return &GameObjectManagerHandleEventResult{}, nil
	}
	switch e.EventType {
	case game_objects.EventObjectCreated:
		gameObject, ok := e.Data["object"].(game_objects.GameObject)
		if !ok {
			log.Printf("Invalid event data: %v", e)
			return nil, errors.New("invalid event data")
		}

		_, exists := m.Objects[gameObject.GetID()]
		if exists {
			log.Printf("Object %s already exists", gameObject.GetID())
			return &GameObjectManagerHandleEventResult{}, nil
		}
		m.AddObject(gameObject)

		// TODO better way to clean this up so it doesn't end up sent to the client
		delete(e.Data, "object")
		e.Data["objectID"] = gameObject.GetID()

		return &GameObjectManagerHandleEventResult{
			AddedObjects: map[string]game_objects.GameObject{
				gameObject.GetID(): gameObject,
			},
		}, nil
	case game_objects.EventObjectDestroyed:
		id, exists := e.Data["objectID"].(string)
		if !exists {
			log.Printf("Invalid event data: %v", e)
			return nil, errors.New("invalid event data")
		}

		m.RemoveObject(id)
		return &GameObjectManagerHandleEventResult{
			RemovedObjects: map[string]game_objects.GameObject{
				id: nil,
			},
		}, nil
	default:
		return &GameObjectManagerHandleEventResult{}, nil
	}
}
