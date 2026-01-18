package server

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"go-ws-server/pkg/server/game_objects"
	"go-ws-server/pkg/server/types"
	"go-ws-server/pkg/util"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// TODO move to constants.go
const (
	FULL_UPDATE_INTERVAL = 10 * 1000 * time.Millisecond
	// Time to keep inactive rooms before cleanup
	ROOM_CLEANUP_INTERVAL = 1 * time.Minute
	ROOM_INACTIVE_TIMEOUT = 10 * time.Minute
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for demo
	},
}

// GameUpdateQueueItem represents a request to update a game room
type GameUpdateQueueItem struct {
	RoomID string
	Update *types.GameUpdate
}

// SpectatorUpdateQueueItem represents a request to update spectators
type SpectatorUpdateQueueItem struct {
	RoomID string
}

// Connection represents a WebSocket connection
type Connection struct {
	ID         string
	connection *websocket.Conn
	RoomID     string
	PlayerID   string
	WriteMutex sync.Mutex
}

// Server handles WebSocket connections and game state
type Server struct {

	// Message queue for room updates
	gameStateUpdateQueue chan GameUpdateQueueItem

	// Message queue for spectator updates
	spectatorUpdateQueue chan SpectatorUpdateQueueItem

	// Map of roomID -> connectionID -> connection
	connectionsByRoom map[string]map[string]*Connection

	// Map of roomID -> last activity time
	lastActivity map[string]time.Time

	// Mutex for server-wide operations
	serverLock sync.Mutex

	// Track game rooms
	roomManager *RoomManager
}

// NewServer creates a new game server
func NewServer() *Server {
	server := &Server{
		gameStateUpdateQueue: make(chan GameUpdateQueueItem, 100), // Buffer size of 100
		spectatorUpdateQueue: make(chan SpectatorUpdateQueueItem, 100),
		connectionsByRoom:    make(map[string]map[string]*Connection),
		lastActivity:         make(map[string]time.Time),
		roomManager:          NewRoomManager(),
	}

	// Start the update worker
	go server.runProcessGameUpdateQueue()

	// Start the spectator update worker
	go server.runProcessSpectatorUpdateQueue()

	// Start a periodic check to ensure all rooms get updated occasionally
	go server.runPeriodicUpdates()

	// Start the inactive room cleanup worker
	go server.runCleanupInactiveRooms()

	// Note: Per-room tick loops are started when rooms are added via AddGameRoom

	return server
}

// HandleWebSocket handles incoming WebSocket connections
func (s *Server) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	// Upgrade HTTP connection to WebSocket
	ws, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("Error upgrading connection: %v", err)
		return
	}

	// Create new connection
	connID := uuid.New().String()
	conn := &Connection{
		ID:         connID,
		connection: ws,
	}

	// Handle connection
	go s.handleConnection(conn)
}

// handleConnection processes messages from a WebSocket connection
func (s *Server) handleConnection(conn *Connection) {
	defer func() {
		// Handle unexpected disconnection
		s.handleDisconnect(conn)
	}()

	for {
		// Read message from WebSocket
		var msg types.Message
		err := conn.connection.ReadJSON(&msg)
		if err != nil {
			// Check if this is a normal closure
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				// Don't log every normal closure, it clutters the logs
				if conn.RoomID != "" {
					log.Printf("Connection %s closed normally from room %s", conn.ID, conn.RoomID)
				}
			} else {
				log.Printf("Error reading message: %v", err)
			}
			break
		}

		// Process message based on type
		switch msg.Type {
		case "RejoinGame":
			var req types.RejoinGameRequest
			if err := json.Unmarshal(msg.Payload, &req); err != nil {
				log.Printf("Error unmarshalling RejoinGame payload: %v", err)
				continue
			}
			s.handleRejoinGame(conn, req)

		case "Key":
			var req types.KeyStatusRequest
			if err := json.Unmarshal(msg.Payload, &req); err != nil {
				log.Printf("Error unmarshalling KeyStatus payload: %v", err)
				continue
			}
			s.handleKeyStatus(conn, req)

		case "ClientState":
			var req types.ClientStateRequest
			if err := json.Unmarshal(msg.Payload, &req); err != nil {
				log.Printf("Error unmarshalling ClientState payload: %v", err)
				continue
			}
			s.handleClientState(conn, req)

		case "Click":
			var req types.PlayerClickRequest
			if err := json.Unmarshal(msg.Payload, &req); err != nil {
				log.Printf("Error unmarshalling ClickStatus payload: %v", err)
				continue
			}
			s.handlePlayerClick(conn, req)

		case "ExitGame":
			var req types.ExitGameRequest
			if err := json.Unmarshal(msg.Payload, &req); err != nil {
				log.Printf("Error unmarshalling ExitGame payload: %v", err)
				continue
			}
			s.handleExitGame(conn, req)

		default:
			log.Printf("Unknown message type: %s", msg.Type)
		}
	}
}

// handleDisconnect handles unexpected disconnections
func (s *Server) handleDisconnect(conn *Connection) {
	// If connection is not associated with a room, nothing to do
	if conn.RoomID == "" {
		return
	}

	log.Printf("Handling disconnect for connection %s from room %s", conn.ID, conn.RoomID)

	// Update last activity time
	s.serverLock.Lock()
	s.lastActivity[conn.RoomID] = time.Now()
	delete(s.connectionsByRoom[conn.RoomID], conn.ID)
	s.serverLock.Unlock()

	s.spectatorUpdateQueue <- SpectatorUpdateQueueItem{RoomID: conn.RoomID}
}

// runProcessGameUpdateQueue processes the game update queue
func (s *Server) runProcessGameUpdateQueue() {
	for {
		s.processGameUpdateQueue()
	}
}

func (s *Server) processGameUpdateQueue() {
	update := <-s.gameStateUpdateQueue
	go func() {
		_ = s.sendGameUpdate(update)
	}()
}

// sendGameUpdate sends the game state update to all connections to the room
func (s *Server) sendGameUpdate(update GameUpdateQueueItem) error {
	// Build the game state update from the GameObject states
	room, exists := s.roomManager.GetGameRoom(update.RoomID)
	if !exists {
		return nil
	}

	updateMessage := update.Update

	if update.Update.FullUpdate {
		updateMessage.ObjectStates = room.GetAllGameObjectStates()
	}

	// Include training info for training mode games
	if room.IsTrainingMode() {
		updateMessage.TrainingInfo = &types.TrainingStateInfo{
			Episode:        room.GetTrainingEpisode(),
			TotalKills:     room.GetKillCount(),
			ElapsedTime:    room.GetTrainingElapsedSeconds(),
			TickMultiplier: room.TickMultiplier,
		}
	}

	// Collect connections and check if throttling applies
	s.serverLock.Lock()
	connectionsToUpdate := make([]*Connection, 0)
	for _, conn := range s.connectionsByRoom[update.RoomID] {
		connectionsToUpdate = append(connectionsToUpdate, conn)
	}
	s.serverLock.Unlock()

	// If this is a training game with accelerated tick rate, check if we should throttle spectator updates
	shouldThrottle := room.ShouldThrottleSpectatorUpdate()

	for _, conn := range connectionsToUpdate {
		// For training games, check if this connection is a spectator
		// and if we should skip this update due to throttling
		if shouldThrottle {
			player, exists := room.GetPlayer(conn.PlayerID)
			if exists && player.IsSpectator {
				// Skip this update for spectators to maintain viewable frame rate
				continue
			}
		}

		// TODO remove this when the NaN issue is fixed
		_, err := json.Marshal(updateMessage)
		if err != nil {
			log.Printf("sendGameUpdate:Error marshalling GameState: %v. %v", err, updateMessage)
			continue
		}
		// Lock the connection before writing to prevent concurrent writes
		conn.WriteMutex.Lock()
		err = conn.connection.WriteJSON(types.Message{
			Type:    "GameState",
			Payload: util.Must(json.Marshal(updateMessage)),
		})
		conn.WriteMutex.Unlock()

		// Ignore write errors - connections may have been closed
		_ = err
	}

	return nil
}

func (s *Server) runProcessSpectatorUpdateQueue() {
	for {
		s.processSpectatorUpdateQueue()
	}
}

func (s *Server) processSpectatorUpdateQueue() {
	update := <-s.spectatorUpdateQueue
	go func() {
		_ = s.sendSpectatorUpdate(update)
	}()
}

func (s *Server) sendSpectatorUpdate(update SpectatorUpdateQueueItem) error {
	// Build the spectator update from the GameObject states
	room, exists := s.roomManager.GetGameRoom(update.RoomID)
	if !exists {
		return nil
	}

	spectators := room.GetSpectators()
	if len(spectators) == 0 {
		return nil
	}
	payload := types.SpectatorUpdate{
		Spectators: spectators,
	}

	// Send update to all connections in this room
	connectionsToUpdate := make([]*Connection, 0)
	s.serverLock.Lock()
	for _, conn := range s.connectionsByRoom[update.RoomID] {
		connectionsToUpdate = append(connectionsToUpdate, conn)
	}
	s.serverLock.Unlock()

	for _, conn := range connectionsToUpdate {
		// Lock the connection before writing to prevent concurrent writes
		conn.WriteMutex.Lock()
		err := conn.connection.WriteJSON(types.Message{
			Type:    "Spectators",
			Payload: util.Must(json.Marshal(payload)),
		})
		conn.WriteMutex.Unlock()

		if err != nil {
			log.Printf("sendSpectatorUpdate:Error sending Spectators to connection %s: %v. %v", conn.ID, err, spectators)
		}
	}

	return nil
}

// runPeriodicUpdates ensures that all rooms get updated occasionally
func (s *Server) runPeriodicUpdates() {
	for {
		time.Sleep(FULL_UPDATE_INTERVAL)
		s.triggerPeriodicUpdates()
	}
}

// triggerPeriodicUpdates triggers updates for all rooms
func (s *Server) triggerPeriodicUpdates() {

	// Send full updates to all rooms
	roomIDs := s.roomManager.GetGameRoomIDs()
	for _, roomID := range roomIDs {
		s.gameStateUpdateQueue <- GameUpdateQueueItem{
			RoomID: roomID,
			Update: &types.GameUpdate{FullUpdate: true},
		}
	}
}

// runCleanupInactiveRooms ensures that inactive rooms are cleaned up
func (s *Server) runCleanupInactiveRooms() {
	for {
		time.Sleep(ROOM_CLEANUP_INTERVAL)
		s.cleanupInactiveRooms()
	}
}

// cleanupInactiveRooms removes inactive rooms
func (s *Server) cleanupInactiveRooms() {
	roomIDs := s.roomManager.GetGameRoomIDs()

	s.serverLock.Lock()
	defer s.serverLock.Unlock()

	now := time.Now()
	for _, roomID := range roomIDs {
		if now.Sub(s.lastActivity[roomID]) > ROOM_INACTIVE_TIMEOUT {
			s.roomManager.RemoveGameRoom(roomID)
			log.Printf("Removed inactive game room: %s", roomID)
		} else if numConnectedPlayers, exists := s.roomManager.GetNumberOfConnectedPlayers(roomID); exists && numConnectedPlayers == 0 {
			s.roomManager.RemoveGameRoom(roomID)
			log.Printf("Removed empty game room: %s", roomID)
		}
	}
}

// AddGameRoomAndStartTick adds a room to the room manager and starts its tick loop
func (s *Server) AddGameRoomAndStartTick(room *GameRoom) {
	s.roomManager.AddGameRoom(room)
	room.StartTickLoop(func(r *GameRoom, event *game_objects.GameEvent) {
		go s.processEvent(r, event)
	})
}
