package main

import (
	"log"
	"net/http"
	"strings"

	"go-ws-server/pkg/server"
)

func main() {
	srv := server.NewServer()
	log.Printf("Starting server on :4000")

	// WebSocket endpoint
	http.HandleFunc("/ws", srv.HandleWebSocket)

	// HTTP API endpoints
	http.HandleFunc("/api/maps", srv.HandleGetMaps)
	http.HandleFunc("/api/createGame", srv.HandleCreateGame)
	http.HandleFunc("/api/joinGame", srv.HandleJoinGame)

	// Route /api/rooms/ requests based on path pattern
	http.HandleFunc("/api/rooms/", func(w http.ResponseWriter, r *http.Request) {
		// Check if this is a bot action endpoint: /api/rooms/{roomId}/players/{playerId}/action
		if strings.Contains(r.URL.Path, "/players/") && strings.HasSuffix(r.URL.Path, "/action") {
			srv.HandleBotAction(w, r)
			return
		}
		// Check if this is a reset endpoint: /api/rooms/{roomId}/reset
		if strings.HasSuffix(r.URL.Path, "/reset") {
			srv.HandleResetGame(w, r)
			return
		}
		// Check if this is a stats endpoint: /api/rooms/{roomId}/stats
		if strings.HasSuffix(r.URL.Path, "/stats") {
			srv.HandleGetRoomStats(w, r)
			return
		}
		// Otherwise, handle as room state endpoint: /api/rooms/{roomId}/state
		srv.HandleGetRoomState(w, r)
	})

	if err := http.ListenAndServe(":4000", nil); err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}
