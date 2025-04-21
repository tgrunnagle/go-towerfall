package server

type MapInfo struct {
	Type        string `json:"type"`
	Name        string `json:"name"`
	CanvasSizeX int    `json:"canvas_size_x"`
	CanvasSizeY int    `json:"canvas_size_y"`
}

type GetMapsResponse struct {
	Maps []MapInfo `json:"maps"`
}

type CreateGameRequest struct {
	RoomName   string `json:"room_name"`
	PlayerName string `json:"player_name"`
	MapType    string `json:"map_type"`
}

type CreateGameResponse struct {
	RoomID       string `json:"room_id"`
	RoomCode     string `json:"room_code"`
	RoomName     string `json:"room_name"`
	PlayerID     string `json:"player_id"`
	PlayerToken  string `json:"player_token"`
	CanvasSizeX  int    `json:"canvas_size_x"`
	CanvasSizeY  int    `json:"canvas_size_y"`
}

type JoinGameRequest struct {
	RoomCode   string `json:"room_code"`
	PlayerName string `json:"player_name"`
}

type JoinGameResponse struct {
	RoomID       string `json:"room_id"`
	RoomName     string `json:"room_name"`
	PlayerID     string `json:"player_id"`
	PlayerToken  string `json:"player_token"`
	CanvasSizeX  int    `json:"canvas_size_x"`
	CanvasSizeY  int    `json:"canvas_size_y"`
}
