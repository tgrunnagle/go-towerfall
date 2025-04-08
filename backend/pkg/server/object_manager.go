package server

import (
	"errors"
	"go-ws-server/pkg/server/game_objects"
	"log"
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
}

// NewGameObjectManager creates a new GameObjectManager
func NewGameObjectManager() *GameObjectManager {
	return &GameObjectManager{
		Objects: make(map[string]game_objects.GameObject),
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

	obj, exists := m.Objects[id]
	return obj, exists
}

func (m *GameObjectManager) GetObjectIDs() []string {
	m.Mutex.RLock()
	defer m.Mutex.RUnlock()

	var ids []string
	for id := range m.Objects {
		ids = append(ids, id)
	}
	return ids
}

// GetAllObjects returns all game objects
func (m *GameObjectManager) GetAllObjects() map[string]game_objects.GameObject {
	m.Mutex.RLock()
	defer m.Mutex.RUnlock()
	result := make(map[string]game_objects.GameObject)
	for id, obj := range m.Objects {
		if obj == nil {
			continue
		}
		result[id] = obj
	}
	return result
}

// GetAllStates returns the state of all game objects
func (m *GameObjectManager) GetAllStates() map[string]map[string]interface{} {
	m.Mutex.RLock()
	defer m.Mutex.RUnlock()

	allStates := make(map[string]map[string]interface{})

	for id, obj := range m.Objects {
		if obj == nil {
			continue
		}
		allStates[id] = obj.GetState()
	}

	return allStates
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
