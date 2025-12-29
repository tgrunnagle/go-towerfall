package server

import (
	"errors"
	"fmt"
	"go-ws-server/pkg/server/constants"
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

// TrainingOptions holds configuration for training mode
type TrainingOptions struct {
	// Enabled indicates whether training mode is active
	Enabled bool
	// TickMultiplier for accelerated training (1.0 = normal, 10.0 = 10x speed)
	TickMultiplier float64
	// MaxGameDurationSec is the maximum game duration in seconds (0 = unlimited)
	MaxGameDurationSec int
	// DisableRespawnTimer enables instant respawn when true
	DisableRespawnTimer bool
	// MaxKills is the maximum number of kills before game ends (0 = unlimited)
	MaxKills int
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

	// Training mode configuration
	TrainingOptions *TrainingOptions
	// Training state tracking
	// trainingStartTime is set once during construction and never modified, so it's safe to read without locking
	trainingStartTime time.Time
	// trainingKillCount is modified during gameplay and must be accessed with LockObject held
	trainingKillCount int

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
	return NewGameRoomWithTrainingConfig(id, name, password, roomCode, mapType, tickConfig, nil)
}

// NewGameRoomWithTrainingConfig creates a new game room with training configuration.
//
// Tick rate precedence (highest to lowest):
//  1. tickConfig.TickInterval (explicit interval always wins)
//  2. tickConfig.TickMultiplier (explicit multiplier)
//  3. trainingOptions.TickMultiplier (training mode multiplier, only if no tickConfig set)
//  4. DefaultTickInterval (20ms, 1x speed)
func NewGameRoomWithTrainingConfig(id string, name string, password string, roomCode string, mapType game_maps.MapType, tickConfig *TickConfig, trainingOptions *TrainingOptions) (*GameRoom, error) {
	// Apply training tick multiplier only if no explicit tickConfig is provided
	effectiveTickConfig := tickConfig
	if trainingOptions != nil && trainingOptions.Enabled && trainingOptions.TickMultiplier > 0 {
		if effectiveTickConfig == nil {
			effectiveTickConfig = &TickConfig{}
		}
		// Only use training multiplier if tickConfig doesn't specify any tick rate
		if effectiveTickConfig.TickMultiplier == 0 && effectiveTickConfig.TickInterval == 0 {
			effectiveTickConfig.TickMultiplier = trainingOptions.TickMultiplier
		}
	}

	// Calculate tick interval from config
	tickInterval, tickMultiplier, err := calculateTickInterval(effectiveTickConfig)
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
		ID:                id,
		Name:              name,
		Password:          password,
		RoomCode:          roomCode,
		EventManager:      NewGameEventManager(),
		ObjectManager:     NewGameObjectManager(baseMap),
		LastUpdateTime:    time.Now(),
		Players:           make(map[string]*ConnectedPlayer),
		Map:               baseMap,
		TickInterval:      tickInterval,
		TickMultiplier:    tickMultiplier,
		TrainingOptions:   trainingOptions,
		trainingStartTime: time.Now(),
		trainingKillCount: 0,
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
		// Use room's configured respawn time (which may be 0 for training mode instant respawn)
		respawnTime := r.GetRespawnTimeSec()
		gameObject := game_objects.NewPlayerGameObjectWithRespawnTime(player.ID, player.Name, player.Token, r.Map.GetRespawnLocation, r.Map.WrapPosition, respawnTime)

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
	return NewGameWithPlayerAndTrainingConfig(roomName, playerName, mapType, tickConfig, nil)
}

func NewGameWithPlayerAndTrainingConfig(roomName string, playerName string, mapType game_maps.MapType, tickConfig *TickConfig, trainingOptions *TrainingOptions) (*GameRoom, *ConnectedPlayer, error) {
	// Generate room ID, password, and room code
	roomID := uuid.New().String()
	password := util.GeneratePassword()
	roomCode := util.GenerateRoomCode()

	// Create player
	playerID := uuid.New().String()
	playerToken := uuid.New().String()

	// Create room with training config
	room, err := NewGameRoomWithTrainingConfig(roomID, roomName, password, roomCode, mapType, tickConfig, trainingOptions)
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

// StartTickLoop starts the per-room tick goroutine.
// If the tick loop is already running, this method does nothing.
func (r *GameRoom) StartTickLoop(callback func(*GameRoom, *game_objects.GameEvent)) {
	if r.tickStopChan != nil {
		return // Already running
	}
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

// StopTickLoop stops the per-room tick goroutine and waits for it to complete.
// This method is idempotent and safe to call multiple times.
func (r *GameRoom) StopTickLoop() {
	if r.tickStopChan != nil {
		close(r.tickStopChan)
		r.tickWg.Wait()
		r.tickStopChan = nil // Prevent double-close panic
	}
}

// IncrementKillCount increments the training kill counter and returns the new count.
// This should be called when a player dies during training mode.
func (r *GameRoom) IncrementKillCount() int {
	r.LockObject.Lock()
	defer r.LockObject.Unlock()
	r.trainingKillCount++
	return r.trainingKillCount
}

// GetKillCount returns the current training kill count.
func (r *GameRoom) GetKillCount() int {
	r.LockObject.Lock()
	defer r.LockObject.Unlock()
	return r.trainingKillCount
}

// GetTrainingElapsedSeconds returns the elapsed time since training started in seconds.
func (r *GameRoom) GetTrainingElapsedSeconds() float64 {
	return time.Since(r.trainingStartTime).Seconds()
}

// IsTrainingComplete checks if training completion conditions have been met.
// Returns true if either max kills or max duration has been reached.
func (r *GameRoom) IsTrainingComplete() bool {
	if r.TrainingOptions == nil || !r.TrainingOptions.Enabled {
		return false
	}

	// Check max kills - requires lock since trainingKillCount is modified during gameplay
	if r.TrainingOptions.MaxKills > 0 {
		r.LockObject.Lock()
		killCount := r.trainingKillCount
		r.LockObject.Unlock()
		if killCount >= r.TrainingOptions.MaxKills {
			return true
		}
	}

	// Check max duration - no lock needed: trainingStartTime is immutable after construction
	if r.TrainingOptions.MaxGameDurationSec > 0 {
		elapsed := time.Since(r.trainingStartTime).Seconds()
		if int(elapsed) >= r.TrainingOptions.MaxGameDurationSec {
			return true
		}
	}

	return false
}

// IsTrainingMode returns true if the room is in training mode.
func (r *GameRoom) IsTrainingMode() bool {
	return r.TrainingOptions != nil && r.TrainingOptions.Enabled
}

// GetRespawnTimeSec returns the respawn time for this room.
// Returns 0 if training mode has instant respawn enabled, otherwise returns the default.
func (r *GameRoom) GetRespawnTimeSec() float64 {
	if r.TrainingOptions != nil && r.TrainingOptions.Enabled && r.TrainingOptions.DisableRespawnTimer {
		return 0.0
	}
	return constants.PlayerRespawnTimeSec
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
