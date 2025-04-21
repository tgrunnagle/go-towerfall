package game_maps

import "go-ws-server/pkg/server/game_objects"

type Coords struct {
	X float64
	Y float64
}

type Map interface {
	// Returns the name of the map
	GetName() string
	// Returns a map of game objects that should be placed on the map, keyed by object ID
	GetObjects() map[string]game_objects.GameObject
	// Returns a list of respawn locations for players in pixels
	GetRespawnLocations() []*Coords
	// Returns the size of the canvas in pixels
	GetCanvasSize() (int, int)
	// Returns the origin coordinates of the map in pixels
	GetOriginCoordinates() *Coords
}

type BaseMap struct {
	Name              string
	Objects           map[string]game_objects.GameObject
	RespawnLocations  []*Coords
	CanvasSizeX       int32
	CanvasSizeY       int32
	OriginCoordinates *Coords
}

func NewBaseMap(name string, objects []game_objects.GameObject, respawnLocations []*Coords, canvasSizeX int32, canvasSizeY int32, originCoordinates *Coords) *BaseMap {
	// Convert slice of objects to map keyed by object ID
	objectMap := make(map[string]game_objects.GameObject)
	for _, obj := range objects {
		objectMap[obj.GetID()] = obj
	}

	return &BaseMap{
		Name:              name,
		Objects:           objectMap,
		RespawnLocations:  respawnLocations,
		CanvasSizeX:       canvasSizeX,
		CanvasSizeY:       canvasSizeY,
		OriginCoordinates: originCoordinates,
	}
}

func (m *BaseMap) GetName() string {
	return m.Name
}

func (m *BaseMap) GetObjects() map[string]game_objects.GameObject {
	return m.Objects
}

func (m *BaseMap) GetRespawnLocations() []*Coords {
	return m.RespawnLocations
}

func (m *BaseMap) GetCanvasSize() (int, int) {
	return int(m.CanvasSizeX), int(m.CanvasSizeY)
}

func (m *BaseMap) GetOriginCoordinates() *Coords {
	return m.OriginCoordinates
}
