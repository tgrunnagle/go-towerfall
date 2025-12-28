package server

import (
	"errors"
	"fmt"
	"go-ws-server/pkg/server/game_maps"
	"go-ws-server/pkg/server/game_objects"
	"go-ws-server/pkg/util"
	"strings"

	"log"
	"maps"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ConnectedPlayer represents a connected player in the game
type ConnectedPlayer struct {
	ID          string
	Name        string
	Token       string
	IsSpectator bool
}

// Default tick interval constants
const (
	DefaultTickInterval = 20 * time.Millisecond
	MinTickInterval     = 1 * time.Millisecond
	MaxTickInterval     = 1000 * time.Millisecond
)

// TickConfig holds configuration for room tick rate
type TickConfig struct {
	// TickInterval is the duration between game ticks. Default is 20ms.
	// Takes precedence over TickMultiplier if both are set.
	TickInterval time.Duration
	// TickMultiplier is a convenience multiplier for tick speed.
	// 1.0 = normal speed (20ms), 10.0 = 10x faster (2ms), 0.5 = half speed (40ms)
	TickMultiplier float64
}

// GameRoom represents an instance of the game being played
type GameRoom struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Password string `json:"password"`
	RoomCode string `json:"roomCode"`

	LockObject sync.Mutex
	// Event manager for this room
	EventManager *GameEventManager
	// Object manager for this room
	ObjectManager *GameObjectManager
	// Last update time
	LastUpdateTime time.Time
	// Map of player ID -> player
	Players map[string]*ConnectedPlayer
	Map     game_maps.Map

	// Tick configuration
	TickInterval   time.Duration
	TickMultiplier float64

	// Tick goroutine management
	tickStopChan chan struct{}
	tickWg       sync.WaitGroup
	tickCallback func(*GameRoom, *game_objects.GameEvent)
}

// NewGameRoom creates a new game room with default tick configuration
func NewGameRoom(id string, name string, password string, roomCode string, mapType game_maps.MapType) (*GameRoom, error) {
	return NewGameRoomWithTickConfig(id, name, password, roomCode, mapType, nil)
}

// NewGameRoomWithTickConfig creates a new game room with custom tick configuration
func NewGameRoomWithTickConfig(id string, name string, password string, roomCode string, mapType game_maps.MapType, tickConfig *TickConfig) (*GameRoom, error) {
	// Calculate tick interval from config
	tickInterval, tickMultiplier, err := calculateTickInterval(tickConfig)
	if err != nil {
		return nil, fmt.Errorf("invalid tick configuration: %v", err)
	}

	// Create and initialize the map
	baseMap, err := game_maps.CreateMap(mapType)
	if err != nil {
		return nil, fmt.Errorf("failed to create map: %v", err)
	}

	// Create the room
	room := &GameRoom{
		ID:             id,
		Name:           name,
		Password:       password,
		RoomCode:       roomCode,
		EventManager:   NewGameEventManager(),
		ObjectManager:  NewGameObjectManager(baseMap),
		LastUpdateTime: time.Now(),
		Players:        make(map[string]*ConnectedPlayer),
		Map:            baseMap,
		TickInterval:   tickInterval,
		TickMultiplier: tickMultiplier,
	}

	// Initialize map objects
	for _, object := range baseMap.GetObjects() {
		room.EventManager.SubscribeToAll(object)
	}

	return room, nil
}

// AddPlayer adds a connected player to the game room
func (r *GameRoom) AddPlayer(playerID string, player *ConnectedPlayer) bool {
	// Check if player already exists
	if _, exists := r.Players[playerID]; exists {
		return false
	}

	r.LockObject.Lock()
	defer r.LockObject.Unlock()

	// Add player to the room
	r.Players[playerID] = player

	if !player.IsSpectator {
		gameObject := game_objects.NewPlayerGameObject(player.ID, player.Name, player.Token, r.Map.GetRespawnLocation, r.Map.WrapPosition)

		// Add player's GameObject to the object manager if it exists
		r.addObject(gameObject)
		log.Printf("Added player %s to room %s", player.ID, r.ID)
	} else {
		log.Printf("Added spectator %s to room %s", player.ID, r.ID)
	}

	return true
}

func (r *GameRoom) GetPlayer(playerID string) (*ConnectedPlayer, bool) {
	player, exists := r.Players[playerID]
	return player, exists
}

// RemovePlayer removes a player from the game room
func (r *GameRoom) RemovePlayer(playerID string) {

	// Remove player's GameObject from the object manager if it exists
	playerObject, exists := r.ObjectManager.GetObject(playerID)
	if exists {
		r.removeObject(playerObject)
	}

	r.LockObject.Lock()
	defer r.LockObject.Unlock()
	delete(r.Players, playerID)
}

// GetAllGameObjectStates returns the state of all game objects in the room
func (r *GameRoom) GetAllGameObjectStates() map[string]map[string]interface{} {
	r.LockObject.Lock()
	defer r.LockObject.Unlock()

	return r.ObjectManager.GetAllStates()
}

// Handle processes a batch of events and returns the game objects whose state has changed
func (r *GameRoom) Handle(events []*game_objects.GameEvent) *HandleEventResult {
	result := &HandleEventResult{
		UpdatedObjects: make(map[string]game_objects.GameObject),
		Events:         make([]*game_objects.GameEvent, 0),
	}
	handleResults := r.EventManager.Handle(events, r.ObjectManager.GetAllObjects())

	// Add updated objects
	maps.Copy(result.UpdatedObjects, handleResults.UpdatedObjects)

	// Add raised events
	result.Events = append(result.Events, handleResults.Events...)

	// Handle object related (e.g. created/destroyed) events
	for _, e := range result.Events {
		managerHandleResults, err := r.ObjectManager.HandleEvent(e)
		if err != nil {
			log.Printf("ObjectManager failed to handle event: %v", err)
			continue
		}
		for k, v := range managerHandleResults.AddedObjects {
			r.EventManager.SubscribeToAll(v)
			result.UpdatedObjects[k] = v
		}
		for k, v := range managerHandleResults.RemovedObjects {
			r.EventManager.UnsubscribeAll(k)
			result.UpdatedObjects[k] = v
		}
	}

	return result
}

func (r *GameRoom) addObject(object game_objects.GameObject) {
	r.ObjectManager.AddObject(object)
	r.EventManager.SubscribeToAll(object)
}

func (r *GameRoom) removeObject(object game_objects.GameObject) {
	r.ObjectManager.RemoveObject(object.GetID())
	r.EventManager.UnsubscribeAll(object.GetID())
}

func (r *GameRoom) GetNumberOfConnectedPlayers() int {
	r.LockObject.Lock()
	defer r.LockObject.Unlock()
	return len(r.Players)
}

func NewGameWithPlayer(roomName string, playerName string, mapType game_maps.MapType) (*GameRoom, *ConnectedPlayer, error) {
	return NewGameWithPlayerAndTickConfig(roomName, playerName, mapType, nil)
}

func NewGameWithPlayerAndTickConfig(roomName string, playerName string, mapType game_maps.MapType, tickConfig *TickConfig) (*GameRoom, *ConnectedPlayer, error) {
	// Generate room ID, password, and room code
	roomID := uuid.New().String()
	password := util.GeneratePassword()
	roomCode := util.GenerateRoomCode()

	// Create player
	playerID := uuid.New().String()
	playerToken := uuid.New().String()

	// Create room with default map and tick config
	room, err := NewGameRoomWithTickConfig(roomID, roomName, password, roomCode, mapType, tickConfig)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create room: %v", err)
	}

	// Create player
	player := &ConnectedPlayer{
		ID:    playerID,
		Name:  playerName,
		Token: playerToken,
	}

	// Add player to room
	added := room.AddPlayer(player.ID, player)
	if !added {
		return nil, nil, errors.New("failed to add player to room")
	}
	return room, player, nil
}

func AddPlayerToGame(room *GameRoom, playerName string, roomPassword string, isSpectator bool) (*ConnectedPlayer, error) {
	if room.Password != strings.ToUpper(roomPassword) {
		return nil, errors.New("invalid room password")
	}

	// Create player
	playerID := uuid.New().String()
	playerToken := uuid.New().String()

	// Create player
	player := &ConnectedPlayer{
		ID:          playerID,
		Name:        playerName,
		Token:       playerToken,
		IsSpectator: isSpectator,
	}

	// Add player to room
	added := room.AddPlayer(player.ID, player)
	if !added {
		return nil, errors.New("failed to add player to room")
	}

	return player, nil
}

func (r *GameRoom) GetSpectators() []string {
	r.LockObject.Lock()
	defer r.LockObject.Unlock()

	spectators := make([]string, 0)
	for _, player := range r.Players {
		if player.IsSpectator {
			spectators = append(spectators, player.Name)
		}
	}
	return spectators
}

// StartTickLoop starts the per-room tick goroutine
func (r *GameRoom) StartTickLoop(callback func(*GameRoom, *game_objects.GameEvent)) {
	r.tickCallback = callback
	r.tickStopChan = make(chan struct{})
	r.tickWg.Add(1)
	go func() {
		defer r.tickWg.Done()
		ticker := time.NewTicker(r.TickInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				if r.tickCallback != nil {
					r.tickCallback(r, game_objects.NewGameEvent(
						r.ID,
						game_objects.EventGameTick,
						nil,
						1,
						nil,
					))
				}
			case <-r.tickStopChan:
				return
			}
		}
	}()
}

// StopTickLoop stops the per-room tick goroutine and waits for it to complete
func (r *GameRoom) StopTickLoop() {
	if r.tickStopChan != nil {
		close(r.tickStopChan)
		r.tickWg.Wait()
	}
}

// calculateTickInterval computes the tick interval from TickConfig
func calculateTickInterval(config *TickConfig) (time.Duration, float64, error) {
	if config == nil {
		return DefaultTickInterval, 1.0, nil
	}

	var interval time.Duration
	var multiplier float64

	// TickInterval takes precedence over TickMultiplier
	if config.TickInterval > 0 {
		interval = config.TickInterval
		multiplier = float64(DefaultTickInterval) / float64(interval)
	} else if config.TickMultiplier > 0 {
		multiplier = config.TickMultiplier
		interval = time.Duration(float64(DefaultTickInterval) / multiplier)
	} else {
		return DefaultTickInterval, 1.0, nil
	}

	// Validate bounds
	if interval < MinTickInterval {
		return 0, 0, fmt.Errorf("tick interval %v is below minimum %v", interval, MinTickInterval)
	}
	if interval > MaxTickInterval {
		return 0, 0, fmt.Errorf("tick interval %v exceeds maximum %v", interval, MaxTickInterval)
	}

	return interval, multiplier, nil
}
