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
	
	// Training mode extensions
	IsTraining       bool    `json:"isTraining"`
	SpeedMultiplier  float64 `json:"speedMultiplier"`
	HeadlessMode     bool    `json:"headlessMode"`
	DirectStateAccess bool   `json:"directStateAccess"`
	TrainingSessionID string `json:"trainingSessionId,omitempty"`
	CustomTickRate   time.Duration `json:"-"` // Custom tick rate for training
}

// NewGameRoom creates a new game room
func NewGameRoom(id string, name string, password string, roomCode string, mapType game_maps.MapType) (*GameRoom, error) {
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
	// Generate room ID, password, and room code
	roomID := uuid.New().String()
	password := util.GeneratePassword()
	roomCode := util.GenerateRoomCode()

	// Create player
	playerID := uuid.New().String()
	playerToken := uuid.New().String()

	// Create room with default map
	room, err := NewGameRoom(roomID, roomName, password, roomCode, mapType)
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

// NewTrainingGameWithPlayer creates a new training room with a player
func NewTrainingGameWithPlayer(roomName string, playerName string, mapType game_maps.MapType, trainingConfig TrainingConfig) (*GameRoom, *ConnectedPlayer, error) {
	// Generate room ID, password, and room code
	roomID := uuid.New().String()
	password := util.GeneratePassword()
	roomCode := util.GenerateRoomCode()

	// Create player
	playerID := uuid.New().String()
	playerToken := uuid.New().String()

	// Create room with training configuration
	room, err := NewTrainingGameRoom(roomID, roomName, password, roomCode, mapType, trainingConfig)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create training room: %v", err)
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
		return nil, nil, errors.New("failed to add player to training room")
	}
	
	return room, player, nil
}

// NewTrainingGameRoom creates a new game room with training capabilities
func NewTrainingGameRoom(id string, name string, password string, roomCode string, mapType game_maps.MapType, trainingConfig TrainingConfig) (*GameRoom, error) {
	// Create base room
	room, err := NewGameRoom(id, name, password, roomCode, mapType)
	if err != nil {
		return nil, err
	}
	
	// Configure training settings
	room.IsTraining = true
	room.SpeedMultiplier = trainingConfig.SpeedMultiplier
	room.HeadlessMode = trainingConfig.HeadlessMode
	room.DirectStateAccess = trainingConfig.DirectStateAccess
	room.TrainingSessionID = trainingConfig.SessionID
	
	// Set custom tick rate based on speed multiplier
	baseTickRate := time.Duration(20) * time.Millisecond // 20ms base tick rate
	room.CustomTickRate = time.Duration(float64(baseTickRate) / trainingConfig.SpeedMultiplier)
	
	log.Printf("Created training room %s with speed %.1fx, headless: %v", 
		id, trainingConfig.SpeedMultiplier, trainingConfig.HeadlessMode)
	
	return room, nil
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

// Training mode methods

// IsTrainingRoom returns true if this room supports training features
func (r *GameRoom) IsTrainingRoom() bool {
	return r.IsTraining
}

// GetSpeedMultiplier returns the current speed multiplier
func (r *GameRoom) GetSpeedMultiplier() float64 {
	return r.SpeedMultiplier
}

// IsHeadlessMode returns true if headless mode is enabled
func (r *GameRoom) IsHeadlessMode() bool {
	return r.HeadlessMode
}

// SetSpeedMultiplier sets the speed multiplier for the room
func (r *GameRoom) SetSpeedMultiplier(multiplier float64) error {
	if !r.IsTraining {
		return errors.New("room does not support speed control")
	}
	
	if multiplier <= 0 || multiplier > 100 {
		return errors.New("speed multiplier must be between 0.1 and 100")
	}
	
	r.LockObject.Lock()
	defer r.LockObject.Unlock()
	
	r.SpeedMultiplier = multiplier
	
	// Update tick rate based on speed multiplier
	baseTickRate := time.Duration(20) * time.Millisecond // 20ms base tick rate
	r.CustomTickRate = time.Duration(float64(baseTickRate) / multiplier)
	
	log.Printf("Set speed multiplier to %.1fx for room %s (tick rate: %v)", 
		multiplier, r.ID, r.CustomTickRate)
	
	return nil
}

// ConfigureTraining configures training-specific settings for the room
func (r *GameRoom) ConfigureTraining(trainingMode string, speedMultiplier float64, directStateAccess bool) error {
	if !r.IsTraining {
		return errors.New("room does not support training configuration")
	}
	
	r.LockObject.Lock()
	defer r.LockObject.Unlock()
	
	// Update headless mode based on training mode
	r.HeadlessMode = (trainingMode == "headless")
	r.DirectStateAccess = directStateAccess
	
	// Update speed multiplier if provided
	if speedMultiplier > 0 {
		if speedMultiplier > 100 {
			return errors.New("speed multiplier cannot exceed 100")
		}
		r.SpeedMultiplier = speedMultiplier
		
		// Update tick rate
		baseTickRate := time.Duration(20) * time.Millisecond
		r.CustomTickRate = time.Duration(float64(baseTickRate) / speedMultiplier)
	}
	
	return nil
}

// GetDirectGameState returns the current game state for direct access
func (r *GameRoom) GetDirectGameState() map[string]interface{} {
	r.LockObject.Lock()
	defer r.LockObject.Unlock()
	
	// Get all object states
	objectStates := r.ObjectManager.GetAllStates()
	
	// Build comprehensive game state
	gameState := map[string]interface{}{
		"objects": objectStates,
		"room": map[string]interface{}{
			"id":              r.ID,
			"name":            r.Name,
			"roomCode":        r.RoomCode,
			"isTraining":      r.IsTraining,
			"speedMultiplier": r.SpeedMultiplier,
			"headlessMode":    r.HeadlessMode,
		},
		"players": r.getPlayerStates(),
		"map": map[string]interface{}{
			"name":   r.Map.GetName(),
			"canvasSize": map[string]interface{}{
				"x": func() int { x, _ := r.Map.GetCanvasSize(); return x }(),
				"y": func() int { _, y := r.Map.GetCanvasSize(); return y }(),
			},
			"origin": r.Map.GetOriginCoordinates(),
		},
		"timestamp": time.Now().UnixMilli(),
	}
	
	return gameState
}

// ValidatePlayerToken validates a player token for this room
func (r *GameRoom) ValidatePlayerToken(token string) bool {
	r.LockObject.Lock()
	defer r.LockObject.Unlock()
	
	for _, player := range r.Players {
		if player.Token == token {
			return true
		}
	}
	return false
}

// GetCustomTickRate returns the custom tick rate for training mode
func (r *GameRoom) GetCustomTickRate() time.Duration {
	if r.CustomTickRate > 0 {
		return r.CustomTickRate
	}
	// Return default tick rate if not set
	return time.Duration(20) * time.Millisecond
}

// getPlayerStates returns player state information
func (r *GameRoom) getPlayerStates() []map[string]interface{} {
	playerStates := make([]map[string]interface{}, 0, len(r.Players))
	
	for _, player := range r.Players {
		playerState := map[string]interface{}{
			"id":          player.ID,
			"name":        player.Name,
			"isSpectator": player.IsSpectator,
		}
		
		// Add game object state if player has one
		if playerObject, exists := r.ObjectManager.GetObject(player.ID); exists {
			playerState["gameObject"] = playerObject.GetState()
		}
		
		playerStates = append(playerStates, playerState)
	}
	
	return playerStates
}
