package game_maps

import (
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/game_objects"
	"log"

	"github.com/google/uuid"
)

type MapType string

const (
	MapDefault    MapType = "default"
	MapWithBlocks MapType = "with_blocks"
)

func CreateMap(mapType MapType) []game_objects.GameObject {
	switch mapType {
	case MapDefault:
		return nil
	case MapWithBlocks:
		return createMapWithBlocks()
	default:
		log.Printf("Unknown map type: %s", mapType)
		return nil
	}
}

func createMapWithBlocks() []game_objects.GameObject {
	floor := game_objects.NewBlockGameObject(
		uuid.New().String(),
		0,
		constants.RoomSizeY-game_objects.BlockSizeUnitPixels,
		constants.RoomSizeX,
		game_objects.BlockSizeUnitPixels,
	)
	return []game_objects.GameObject{floor}
}
