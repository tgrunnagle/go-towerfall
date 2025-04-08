package util

import (
	"encoding/json"

	"github.com/google/uuid"
)

// GeneratePassword generates a random password for a game room
func GeneratePassword() string {
	const charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	password := make([]byte, 4)
	for i := range password {
		password[i] = charset[uint32(uuid.New().ID()&0xFF)%uint32(len(charset))]
	}
	return string(password)
}

// GenerateRoomCode generates a random room code for a game room
func GenerateRoomCode() string {
	const charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	code := make([]byte, 4)
	for i := range code {
		code[i] = charset[uint32(uuid.New().ID()&0xFF)%uint32(len(charset))]
	}
	return string(code)
}

// must is a helper function to simplify error handling
func Must(data []byte, err error) json.RawMessage {
	if err != nil {
		panic(err)
	}
	return data
}
