package server

import (
	"encoding/json"
	"errors"
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
	GAME_TICK_INTERVAL    = 20 * time.Millisecond
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
		connectionsByRoom:    make(map[string]map[string]*Connection),
		lastActivity:         make(map[string]time.Time),
		roomManager:          NewRoomManager(),
	}

	// Start the update worker
	go server.runProcessUpdateQueue()

	// Start a periodic check to ensure all rooms get updated occasionally
	go server.runPeriodicUpdates()

	// Start the inactive room cleanup worker
	go server.runCleanupInactiveRooms()

	// Start the game tick worker
	go server.runGameTick()

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
}

// runProcessUpdateQueue processes the update queue
func (s *Server) runProcessUpdateQueue() {
	for {
		s.processUpdateQueue()
	}
}

func (s *Server) processUpdateQueue() {
	update := <-s.gameStateUpdateQueue
	go s.notifyRoom(update)
}

// notifyRoom sends the game state update to all connections to the room
func (s *Server) notifyRoom(update GameUpdateQueueItem) error {
	// Build the game state update from the GameObject states
	room, exists := s.roomManager.GetGameRoom(update.RoomID)
	if !exists {
		return errors.New("room does not exist")
	}

	updateMessage := update.Update

	if update.Update.FullUpdate {
		updateMessage.ObjectStates = room.GetAllGameObjectStates()
	}

	// log.Printf("notifyRoom:Sending update to room %s: %v", update.RoomID, updateMessage)

	// Send update to all connections in this room
	connectionsToUpdate := make([]*Connection, 0)
	s.serverLock.Lock()
	for _, conn := range s.connectionsByRoom[update.RoomID] {
		connectionsToUpdate = append(connectionsToUpdate, conn)
	}
	s.serverLock.Unlock()

	for _, conn := range connectionsToUpdate {
		_, err := json.Marshal(updateMessage)
		if err != nil {
			log.Printf("notifyRoom:Error marshalling GameState: %v. %v", err, updateMessage)
			continue
		}
		// Lock the connection before writing to prevent concurrent writes
		conn.WriteMutex.Lock()
		err = conn.connection.WriteJSON(types.Message{
			Type:    "GameState",
			Payload: util.Must(json.Marshal(updateMessage)),
		})
		conn.WriteMutex.Unlock()

		if err != nil {
			log.Printf("notifyRoom:Error sending GameState to connection %s: %v. %v", conn.ID, err, updateMessage)
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

// runGameTick runs the game tick
func (s *Server) runGameTick() {
	for {
		time.Sleep(GAME_TICK_INTERVAL)
		s.triggerGameTick()
	}
}

// triggerGameTick triggers a game tick for all rooms
func (s *Server) triggerGameTick() {
	// Send tick events to all rooms
	roomIDs := s.roomManager.GetGameRoomIDs()
	for _, roomID := range roomIDs {
		room, exists := s.roomManager.GetGameRoom(roomID)
		if !exists {
			continue
		}
		go s.processEvent(room, game_objects.NewGameEvent(
			roomID,
			game_objects.EventGameTick,
			nil,
			1,
			nil,
		))
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
