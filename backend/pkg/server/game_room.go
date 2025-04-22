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
	ID    string
	Name  string
	Token string
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

	gameObject := game_objects.NewPlayerGameObject(player.ID, player.Name, player.Token, r.Map.GetRespawnLocation, r.Map.WrapPosition)

	// Add player's GameObject to the object manager if it exists
	r.addObject(gameObject)

	log.Printf("Added player %s to room %s", player.ID, r.ID)
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

func AddPlayerToGame(room *GameRoom, playerName string, roomPassword string) (*ConnectedPlayer, error) {
	if room.Password != strings.ToUpper(roomPassword) {
		return nil, errors.New("invalid room password")
	}

	// Create player
	playerID := uuid.New().String()
	playerToken := uuid.New().String()

	// Create player
	player := &ConnectedPlayer{
		ID:    playerID,
		Name:  playerName,
		Token: playerToken,
	}

	// Add player to room
	added := room.AddPlayer(player.ID, player)
	if !added {
		return nil, errors.New("failed to add player to room")
	}

	return player, nil
}
