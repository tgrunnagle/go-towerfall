package server

import (
	"encoding/json"
	"go-ws-server/pkg/server/game_objects"
	"go-ws-server/pkg/server/types"
	"go-ws-server/pkg/util"
	"log"
	"time"
)

// handleCreateGame creates a new game room
func (s *Server) handleCreateGame(conn *Connection, req types.CreateGameRequest) {
	room, player, err := NewGameWithPlayer(req.RoomName, req.PlayerName)
	if err != nil {
		// Send error response
		response := types.CreateGameResponse{
			Success: false,
			Error:   err.Error(),
		}
		conn.WriteMutex.Lock()
		conn.connection.WriteJSON(types.Message{
			Type:    "CreateGameResponse",
			Payload: util.Must(json.Marshal(response)),
		})
		conn.WriteMutex.Unlock()
		return
	}
	s.roomManager.AddGameRoom(room)

	// Associate connection with room
	conn.RoomID = room.ID
	conn.PlayerID = player.ID

	// Add connection to room
	s.addConnectionForRoom(conn, room)

	// Send response
	response := types.CreateGameResponse{
		Success:      true,
		RoomID:       room.ID,
		RoomName:     room.Name,
		RoomCode:     room.RoomCode,
		RoomPassword: room.Password,
		PlayerID:     player.ID,
		PlayerToken:  player.Token,
	}

	conn.WriteMutex.Lock()
	err = conn.connection.WriteJSON(types.Message{
		Type:    "CreateGameResponse",
		Payload: util.Must(json.Marshal(response)),
	})
	conn.WriteMutex.Unlock()

	if err != nil {
		log.Printf("Error sending CreateGameResponse to connection %s: %v", conn.ID, err)
	}

	log.Printf("Created game room %s with code %s via websocket", room.ID, room.RoomCode)
	log.Printf("Associated websocket connection %s with player %s", conn.ID, player.ID)
	log.Printf("Player %s joined game room %s via websocket", player.ID, room.ID)
}

// handleJoinGame joins an existing game room
func (s *Server) handleJoinGame(conn *Connection, req types.JoinGameRequest) {
	// Find room by code
	room, exists := s.roomManager.GetGameRoomByCode(req.RoomCode)
	if !exists {
		// Room not found
		conn.WriteMutex.Lock()
		conn.connection.WriteJSON(types.Message{
			Type:    "JoinGameResponse",
			Payload: util.Must(json.Marshal(types.JoinGameResponse{Success: false, Error: "Room not found"})),
		})
		conn.WriteMutex.Unlock()
		return
	}

	// Check password
	player, err := AddPlayerToGame(room, req.PlayerName, req.RoomPassword)
	if err != nil {
		// Send error response
		response := types.JoinGameResponse{
			Success: false,
			Error:   err.Error(),
		}
		conn.WriteMutex.Lock()
		conn.connection.WriteJSON(types.Message{
			Type:    "JoinGameResponse",
			Payload: util.Must(json.Marshal(response)),
		})
		conn.WriteMutex.Unlock()
		return
	}

	// Associate connection with room
	conn.RoomID = room.ID
	conn.PlayerID = player.ID

	// Add connection to room
	s.addConnectionForRoom(conn, room)

	// Send response
	response := types.JoinGameResponse{
		Success:     true,
		RoomID:      room.ID,
		RoomName:    room.Name,
		RoomCode:    room.RoomCode,
		PlayerID:    player.ID,
		PlayerToken: player.Token,
	}

	conn.WriteMutex.Lock()
	err = conn.connection.WriteJSON(types.Message{
		Type:    "JoinGameResponse",
		Payload: util.Must(json.Marshal(response)),
	})
	conn.WriteMutex.Unlock()

	if err != nil {
		log.Printf("Error sending JoinGameResponse to connection %s: %v", conn.ID, err)
	}

	log.Printf("Player %s joined game room %s", player.ID, room.ID)
}

// handleRejoinGame handles a player rejoining a game
func (s *Server) handleRejoinGame(conn *Connection, req types.RejoinGameRequest) {
	// Check if connection is already associated with a different player
	if conn.PlayerID != "" && conn.PlayerID != req.PlayerID {
		// cleanup
		oldRoom, exists := s.roomManager.GetGameRoom(conn.RoomID)
		if exists {
			log.Printf("handleRejoinGame: Removing player %s from room %s", conn.PlayerID, oldRoom.ID)
			oldRoom.RemovePlayer(conn.PlayerID)
		}
	}

	// Check if connection is already associated with a different room
	if conn.RoomID != "" && conn.RoomID != req.RoomID {
		// cleanup
		oldRoom, exists := s.roomManager.GetGameRoom(conn.RoomID)
		if exists {
			log.Printf("handleRejoinGame: Removing connection %s from room %s", conn.ID, oldRoom.ID)
			s.removeConnectionForRoom(conn, oldRoom)
		}
	}

	room, exists := s.roomManager.GetGameRoom(req.RoomID)
	if !exists {
		log.Printf("handleRejoinGame: Room %s does not exist", req.RoomID)
		conn.WriteMutex.Lock()
		err := conn.connection.WriteJSON(types.Message{
			Type:    "RejoinGameResponse",
			Payload: util.Must(json.Marshal(types.RejoinGameResponse{Success: false, Error: "Room not found"})),
		})
		conn.WriteMutex.Unlock()
		if err != nil {
			log.Printf("handleRejoinGame: Error sending RejoinGameResponse to connection %s: %v", conn.ID, err)
		}
		return
	}

	player, exists := room.GetPlayer(req.PlayerID)
	if !exists {
		log.Printf("handleRejoinGame: Player %s not found in room %s", req.PlayerID, req.RoomID)
		conn.WriteMutex.Lock()
		err := conn.connection.WriteJSON(types.Message{
			Type:    "RejoinGameResponse",
			Payload: util.Must(json.Marshal(types.RejoinGameResponse{Success: false, Error: "Player not found in room"})),
		})
		conn.WriteMutex.Unlock()
		if err != nil {
			log.Printf("handleRejoinGame: Error sending RejoinGameResponse to connection %s: %v", conn.ID, err)
		}
		return
	}

	if player.Token != req.PlayerToken {
		log.Printf("handleRejoinGame: Invalid player token for player %s in room %s", req.PlayerID, req.RoomID)
		conn.WriteMutex.Lock()
		err := conn.connection.WriteJSON(types.Message{
			Type:    "RejoinGameResponse",
			Payload: util.Must(json.Marshal(types.RejoinGameResponse{Success: false, Error: "Invalid player token"})),
		})
		conn.WriteMutex.Unlock()
		if err != nil {
			log.Printf("handleRejoinGame: Error sending RejoinGameResponse to connection %s: %v", conn.ID, err)
		}
		return
	}

	// Update connection with new room and player
	conn.RoomID = room.ID
	conn.PlayerID = player.ID

	// Add connection to room
	s.addConnectionForRoom(conn, room)

	// Send success response
	conn.WriteMutex.Lock()
	err := conn.connection.WriteJSON(types.Message{
		Type: "RejoinGameResponse",
		Payload: util.Must(
			json.Marshal(
				types.RejoinGameResponse{
					Success:      true,
					RoomName:     room.Name,
					RoomCode:     room.RoomCode,
					RoomPassword: room.Password,
					PlayerName:   player.Name,
					PlayerID:     player.ID,
				},
			),
		),
	})
	conn.WriteMutex.Unlock()
	if err != nil {
		log.Printf("handleRejoinGame: Error sending RejoinGameResponse to connection %s: %v", conn.ID, err)
	}

	log.Printf("Player %s rejoined game room %s", conn.PlayerID, conn.RoomID)

	s.gameStateUpdateQueue <- GameUpdateQueueItem{
		RoomID: conn.RoomID,
		Update: &types.GameUpdate{FullUpdate: true},
	}
}

// handleKeyStatus handles periodic key status updates
func (s *Server) handleKeyStatus(conn *Connection, req types.KeyStatusRequest) {
	room, _, exists := s.findRoomAndPlayer(conn, conn.RoomID, conn.PlayerID)

	if !exists {
		log.Printf("handleKeyStatus: Failed for find room %s or player %s", conn.RoomID, conn.PlayerID)
		return
	}

	s.updateConnectionActivity(conn)

	// Create a PlayerInput event
	eventData := map[string]interface{}{
		"playerId":    conn.PlayerID,
		"keysPressed": req.KeysPressed,
	}

	event := game_objects.NewGameEvent(
		conn.RoomID,
		game_objects.EventPlayerKeyStatus,
		eventData,
		1,   // Priority TODO centralize priority settings
		nil, // No source object for user input
	)

	s.processEvent(room, event)
}

func (s *Server) handleClientState(conn *Connection, req types.ClientStateRequest) {
	room, _, exists := s.findRoomAndPlayer(conn, conn.RoomID, conn.PlayerID)

	if !exists {
		log.Printf("handleClientState: Failed for find room %s or player %s", conn.RoomID, conn.PlayerID)
		return
	}

	s.updateConnectionActivity(conn)

	// Create a PlayerInput event
	eventData := map[string]interface{}{
		"playerId":  conn.PlayerID,
		"direction": req.Direction,
	}

	event := game_objects.NewGameEvent(
		conn.RoomID,
		game_objects.EventPlayerDirection,
		eventData,
		1,   // Priority TODO centralize priority settings
		nil, // No source object for user input
	)

	s.processEvent(room, event)
}

func (s *Server) handlePlayerClick(conn *Connection, req types.PlayerClickRequest) {
	room, _, exists := s.findRoomAndPlayer(conn, conn.RoomID, conn.PlayerID)

	if !exists {
		log.Printf("handlePlayerClick: Failed for find room %s or player %s", conn.RoomID, conn.PlayerID)
		return
	}

	s.updateConnectionActivity(conn)

	// Create a PlayerInput event
	eventData := map[string]interface{}{
		"playerId": conn.PlayerID,
		"x":        req.X,
		"y":        req.Y,
	}

	event := game_objects.NewGameEvent(
		conn.RoomID,
		game_objects.EventPlayerClickInput,
		eventData,
		1,   // Priority TODO centralize priority settings
		nil, // No source object for user input
	)

	s.processEvent(room, event)
}

func (s *Server) processEvent(room *GameRoom, event *game_objects.GameEvent) {
	// Process the event through the event manager
	roomHandleResult := room.Handle([]*game_objects.GameEvent{event})
	if len(roomHandleResult.UpdatedObjects) == 0 && len(roomHandleResult.Events) == 0 {
		return
	}

	updates := &types.GameUpdate{
		FullUpdate:   false,
		ObjectStates: make(map[string]map[string]interface{}),
		Events:       make([]types.GameUpdateEvent, 0),
	}
	for objectID, obj := range roomHandleResult.UpdatedObjects {
		if obj == nil {
			updates.ObjectStates[objectID] = nil
		} else {
			updates.ObjectStates[objectID] = obj.GetState()
		}
	}
	for _, event := range roomHandleResult.Events {
		updates.Events = append(updates.Events, types.GameUpdateEvent{
			Type: event.EventType,
			Data: event.Data,
		})
	}

	s.gameStateUpdateQueue <- GameUpdateQueueItem{RoomID: room.ID, Update: updates}
}

// handleExitGame handles a player exiting a game
func (s *Server) handleExitGame(conn *Connection, _ types.ExitGameRequest) {
	room, player, exists := s.findRoomAndPlayer(conn, conn.RoomID, conn.PlayerID)

	if !exists {
		log.Printf("handleExitGame: Failed for find room %s or player %s", conn.RoomID, conn.PlayerID)
		return
	}

	// Remove player from room
	room.RemovePlayer(player.ID)

	// Remove connection from room
	s.removeConnectionForRoom(conn, room)

	s.gameStateUpdateQueue <- GameUpdateQueueItem{RoomID: room.ID, Update: &types.GameUpdate{FullUpdate: true}}

	conn.WriteMutex.Lock()
	conn.RoomID = ""
	conn.connection.WriteJSON(types.Message{
		Type:    "ExitGameResponse",
		Payload: util.Must(json.Marshal(types.ExitGameResponse{Success: true})),
	})
	conn.WriteMutex.Unlock()

	log.Printf("Player %s exited game room %s", player.ID, room.ID)
}

func (s *Server) addConnectionForRoom(conn *Connection, room *GameRoom) {
	s.serverLock.Lock()
	if _, exists := s.connectionsByRoom[room.ID]; !exists {
		s.connectionsByRoom[room.ID] = make(map[string]*Connection)
	}
	s.connectionsByRoom[room.ID][conn.ID] = conn
	s.lastActivity[room.ID] = time.Now()
	s.serverLock.Unlock()
}

func (s *Server) removeConnectionForRoom(conn *Connection, room *GameRoom) {
	s.serverLock.Lock()
	if _, exists := s.connectionsByRoom[room.ID]; exists {
		delete(s.connectionsByRoom[room.ID], conn.ID)
		if len(s.connectionsByRoom[room.ID]) == 0 {
			delete(s.connectionsByRoom, room.ID)
		}
	}
	s.lastActivity[room.ID] = time.Now()
	s.serverLock.Unlock()
}

func (s *Server) updateConnectionActivity(conn *Connection) {
	s.serverLock.Lock()
	s.lastActivity[conn.RoomID] = time.Now()
	s.serverLock.Unlock()
}

func (s *Server) sendErrorMessage(conn *Connection, message string) {
	conn.WriteMutex.Lock()
	conn.connection.WriteJSON(types.Message{
		Type:    "ErrorMessage",
		Payload: util.Must(json.Marshal(types.ErrorMessage{Message: message})),
	})
	conn.WriteMutex.Unlock()
}

func (s *Server) findRoomAndPlayer(conn *Connection, roomID string, playerID string) (*GameRoom, *ConnectedPlayer, bool) {
	room, exists := s.roomManager.GetGameRoom(roomID)
	if !exists {
		s.sendErrorMessage(conn, "Room not found")
		return nil, nil, false
	}
	player, exists := room.GetPlayer(playerID)
	if !exists {
		s.sendErrorMessage(conn, "Player not found")
		return nil, nil, false
	}
	return room, player, true
}
