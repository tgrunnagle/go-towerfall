package server

import (
	"go-ws-server/pkg/server/game_objects"
	"sort"
	"sync"
)

// HandleEventResult represents the result of handling a set of events
type HandleEventResult struct {
	UpdatedObjects map[string]game_objects.GameObject
	Events         []*game_objects.GameEvent
}

// GameEventManager manages game events and subscriptions within a game room
// It is used to handle events and notify subscribed game objects
type GameEventManager struct {
	// Map of EventType -> Map of GameObject ID -> GameObject pointer
	subscribedObjects map[game_objects.EventType]map[string]game_objects.GameObject
	mutex             sync.RWMutex
}

// NewGameEventManager creates a new GameEventManager
func NewGameEventManager() *GameEventManager {
	return &GameEventManager{
		subscribedObjects: make(map[game_objects.EventType]map[string]game_objects.GameObject),
	}
}

// Subscribe adds a GameObject to the subscription list for a specific event type
func (m *GameEventManager) Subscribe(gameObject game_objects.GameObject, eventType game_objects.EventType) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Initialize map if it doesn't exist
	if _, exists := m.subscribedObjects[eventType]; !exists {
		m.subscribedObjects[eventType] = make(map[string]game_objects.GameObject)
	}

	// Add the GameObject to the subscription list
	m.subscribedObjects[eventType][gameObject.GetID()] = gameObject
}

// SubscribeToAll subscribes a GameObject to all event types it's interested in
func (m *GameEventManager) SubscribeToAll(gameObject game_objects.GameObject) {
	eventTypes := gameObject.GetEventTypes()
	for _, eventType := range eventTypes {
		m.Subscribe(gameObject, eventType)
	}
}

// Unsubscribe removes a GameObject from the subscription list for a specific event type
func (m *GameEventManager) Unsubscribe(gameObject game_objects.GameObject, eventType game_objects.EventType) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if the map exists
	if _, exists := m.subscribedObjects[eventType]; !exists {
		return
	}

	// Remove the GameObject from the subscription list
	delete(m.subscribedObjects[eventType], gameObject.GetID())

	// Clean up empty map
	if len(m.subscribedObjects[eventType]) == 0 {
		delete(m.subscribedObjects, eventType)
	}
}

// UnsubscribeAll removes a GameObject from all event subscriptions
func (m *GameEventManager) UnsubscribeAll(gameObjectID string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Remove the GameObject from all event types
	for eventType, objects := range m.subscribedObjects {
		delete(objects, gameObjectID)

		// Clean up empty maps
		if len(objects) == 0 {
			delete(m.subscribedObjects, eventType)
		}
	}
}

// Handle processes an event and returns the updated game objects
func (m *GameEventManager) Handle(
	events []*game_objects.GameEvent,
	roomObjects map[string]game_objects.GameObject,
) *HandleEventResult {

	// Map to track updated and created game objects: ObjectID -> GameObject
	result := &HandleEventResult{
		UpdatedObjects: make(map[string]game_objects.GameObject),
		Events:         make([]*game_objects.GameEvent, 0),
	}

	// Current queue of events to process
	eventQueue := events
	if len(eventQueue) > 1 {
		sort.Slice(eventQueue, func(i, j int) bool {
			return eventQueue[i].Priority >= eventQueue[j].Priority
		})
	}

	// Next set of events raised by processing the current set
	nextEventQueue := []*game_objects.GameEvent{}

	// Process events until the queue is empty
	for len(eventQueue) > 0 {
		// Get the next event
		currentEvent := eventQueue[0]
		eventQueue = eventQueue[1:]

		// Get the subscribed objects for this event
		m.mutex.RLock()
		subscribedObjects, eventExists := m.subscribedObjects[currentEvent.EventType]
		m.mutex.RUnlock()

		if !eventExists {
			continue
		}

		// Process the event for each subscribed object
		for _, gameObject := range subscribedObjects {
			// Handle the event
			gameObjectResult := gameObject.Handle(currentEvent, roomObjects)

			// If the state changed, add to updated objects
			if gameObjectResult.StateChanged {
				result.UpdatedObjects[gameObject.GetID()] = gameObject
			}

			// Add raised events to the next queue
			if len(gameObjectResult.RaisedEvents) > 0 {
				nextEventQueue = append(nextEventQueue, gameObjectResult.RaisedEvents...)
				for _, event := range gameObjectResult.RaisedEvents {
					if m.isExternalEvent(event) {
						result.Events = append(result.Events, event)
					}
				}
			}
		}

		// If the current event queue is empty, process the next queue
		if len(eventQueue) == 0 {
			// Sort nextEventQueue by priority in descending order
			sort.Slice(nextEventQueue, func(i, j int) bool {
				return nextEventQueue[i].Priority >= nextEventQueue[j].Priority
			})
			eventQueue = nextEventQueue
			nextEventQueue = []*game_objects.GameEvent{}
		}
	}

	return result
}

func (m *GameEventManager) isExternalEvent(event *game_objects.GameEvent) bool {
	return event.EventType == game_objects.EventObjectCreated ||
		event.EventType == game_objects.EventObjectDestroyed ||
		event.EventType == game_objects.EventPlayerDied
}
