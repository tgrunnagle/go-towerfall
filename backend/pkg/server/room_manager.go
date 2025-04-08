package server

import (
	"strings"
	"sync"
)

type RoomManager struct {
	gameRooms map[string]*GameRoom
	roomLock  sync.Mutex
}

func NewRoomManager() *RoomManager {
	return &RoomManager{
		gameRooms: make(map[string]*GameRoom),
	}
}

func (s *RoomManager) AddGameRoom(room *GameRoom) {
	s.roomLock.Lock()
	s.gameRooms[room.ID] = room
	s.roomLock.Unlock()
}

func (s *RoomManager) RemoveGameRoom(roomID string) {
	s.roomLock.Lock()
	delete(s.gameRooms, roomID)
	s.roomLock.Unlock()
}

func (s *RoomManager) GetGameRoom(roomID string) (*GameRoom, bool) {
	s.roomLock.Lock()
	room, exists := s.gameRooms[roomID]
	s.roomLock.Unlock()
	return room, exists
}

func (s *RoomManager) GetNumberOfConnectedPlayers(roomID string) (int, bool) {
	s.roomLock.Lock()
	defer s.roomLock.Unlock()

	room, exists := s.gameRooms[roomID]
	if !exists {
		return 0, false
	}
	return room.GetNumberOfConnectedPlayers(), true
}

func (s *RoomManager) GetGameRoomByCode(roomCode string) (*GameRoom, bool) {
	s.roomLock.Lock()
	defer s.roomLock.Unlock()

	roomCode = strings.ToUpper(roomCode)
	for _, room := range s.gameRooms {
		if room.RoomCode == roomCode {
			return room, true
		}
	}
	return nil, false
}

func (s *RoomManager) GetGameRoomIDs() []string {
	s.roomLock.Lock()
	defer s.roomLock.Unlock()

	var ids []string
	for id := range s.gameRooms {
		ids = append(ids, id)
	}
	return ids
}
