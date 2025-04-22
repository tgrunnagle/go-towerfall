package game_maps

import (
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/game_objects"
	"math/rand"
)

type Coords struct {
	X float64
	Y float64
}

type Map interface {
	// Returns the name of the map
	GetName() string
	// Returns a map of game objects that should be placed on the map, keyed by object ID
	GetObjects() map[string]game_objects.GameObject
	// Returns a random respawn location for a player
	GetRespawnLocation() (float64, float64)
	// Returns the size of the canvas in pixels
	GetCanvasSize() (int, int)
	// Returns the origin coordinates of the map in pixels
	GetOriginCoordinates() *Coords
	// Wraps a position to the map
	WrapPosition(x float64, y float64) (float64, float64)
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

func (m *BaseMap) GetRespawnLocation() (float64, float64) {
	index := rand.Intn(len(m.RespawnLocations))
	return m.RespawnLocations[index].X, m.RespawnLocations[index].Y
}

func (m *BaseMap) GetCanvasSize() (int, int) {
	return int(m.CanvasSizeX), int(m.CanvasSizeY)
}

func (m *BaseMap) GetOriginCoordinates() *Coords {
	return m.OriginCoordinates
}

func (m *BaseMap) WrapPosition(x float64, y float64) (float64, float64) {
	if x < -1.0*constants.RoomWrapDistancePx {
		x = float64(m.CanvasSizeX + constants.RoomWrapDistancePx)
	} else if x > float64(m.CanvasSizeX+constants.RoomWrapDistancePx) {
		x = -1.0 * constants.RoomWrapDistancePx
	}

	if y < -1.0*constants.RoomWrapDistancePx {
		y = float64(m.CanvasSizeY + constants.RoomWrapDistancePx)
	} else if y > float64(m.CanvasSizeY+constants.RoomWrapDistancePx) {
		y = -1.0 * constants.RoomWrapDistancePx
	}
	return x, y
}
