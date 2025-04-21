package game_maps

type Coordinate struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type ViewSize struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type MapMetadata struct {
	MapName        string       `json:"map_name"`
	LayoutFile     string       `json:"layout_file"`
	Origin         Coordinate   `json:"origin_coordinates"`
	SpawnLocations []Coordinate `json:"spawn_coordinates"`
	ViewSize       ViewSize     `json:"view_size"`
}
