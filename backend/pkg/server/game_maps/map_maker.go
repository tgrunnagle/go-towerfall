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
	// bottom left floor
	bottomLeft := game_objects.NewBlockGameObject(
		uuid.New().String(),
		0,
		constants.RoomSizePixelsY-game_objects.BlockSizeUnitPixels*3,
		constants.RoomSizePixelsX/2.0-100.,
		game_objects.BlockSizeUnitPixels*3,
	)
	// bottom right floor
	bottomRight := game_objects.NewBlockGameObject(
		uuid.New().String(),
		constants.RoomSizePixelsX/2.0+100.,
		constants.RoomSizePixelsY-game_objects.BlockSizeUnitPixels*3,
		constants.RoomSizePixelsX/2.0-100.,
		game_objects.BlockSizeUnitPixels*3,
	)
	// middle floor
	middleFloor := game_objects.NewBlockGameObject(
		uuid.New().String(),
		constants.RoomSizePixelsX/2.0-100.,
		constants.RoomSizePixelsY/2.0,
		200.,
		game_objects.BlockSizeUnitPixels,
	)
	// middle wall
	middleWall := game_objects.NewBlockGameObject(
		uuid.New().String(),
		constants.RoomSizePixelsX/2.0-game_objects.BlockSizeUnitPixels/2,
		constants.RoomSizePixelsY/2.0-200.,
		game_objects.BlockSizeUnitPixels,
		400.,
	)
	// top left floor
	topLeft := game_objects.NewBlockGameObject(
		uuid.New().String(),
		0,
		200.,
		200.,
		game_objects.BlockSizeUnitPixels,
	)
	// top right floor
	topRight := game_objects.NewBlockGameObject(
		uuid.New().String(),
		constants.RoomSizePixelsX-200.,
		200.,
		200.,
		game_objects.BlockSizeUnitPixels,
	)
	return []game_objects.GameObject{bottomLeft, bottomRight, middleFloor, middleWall, topLeft, topRight}
}
