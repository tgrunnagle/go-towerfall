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
		constants.RoomSizePixelsY-game_objects.BlockSizeUnitPixels*3,
		constants.RoomSizePixelsX,
		game_objects.BlockSizeUnitPixels*3,
	)
	return []game_objects.GameObject{floor}
}
