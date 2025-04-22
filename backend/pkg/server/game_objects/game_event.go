package game_objects

// EventType represents the type of game event
type EventType string

// Event type constants
const (
	EventPlayerKeyInput   EventType = "player_key_input"
	EventPlayerClickInput EventType = "player_click_input"
	EventPlayerDirection  EventType = "player_direction"
	EventPlayerDied       EventType = "player_died"
	EventPlayerJoin       EventType = "player_join"
	EventPlayerLeave      EventType = "player_leave"
	EventGameStart        EventType = "game_start"
	EventGameEnd          EventType = "game_end"
	EventGameTick         EventType = "game_tick"
	EventObjectCreated    EventType = "object_created"
	EventObjectDestroyed  EventType = "object_destroyed"
	EventObjectCollision  EventType = "collision"
)

// GameEvent represents an event in the game
type GameEvent struct {
	RoomID       string
	EventType    EventType
	Data         map[string]interface{}
	Priority     int
	SourceObject GameObject
}

// NewGameEvent creates a new GameEvent
func NewGameEvent(roomID string, eventType EventType, data map[string]interface{}, priority int, sourceObject GameObject) *GameEvent {
	return &GameEvent{
		RoomID:       roomID,
		EventType:    eventType,
		Data:         data,
		Priority:     priority,
		SourceObject: sourceObject,
	}
}
