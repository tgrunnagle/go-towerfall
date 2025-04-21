package main

import (
	"log"
	"net/http"

	"go-ws-server/pkg/server"
)

func main() {
	server := server.NewServer()
	log.Printf("Starting server on :4000")

	// WebSocket endpoint
	http.HandleFunc("/ws", server.HandleWebSocket)

	// HTTP API endpoints
	http.HandleFunc("/api/maps", server.HandleGetMaps)
	http.HandleFunc("/api/createGame", server.HandleCreateGame)
	http.HandleFunc("/api/joinGame", server.HandleJoinGame)

	if err := http.ListenAndServe(":4000", nil); err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}
