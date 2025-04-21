package game_maps

import (
	"encoding/json"
	"fmt"
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/game_objects"
	"log"
	"os"
	"strings"

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

// CreateMapFromFile loads a map from a JSON metadata file and returns a BaseMap
func CreateMapFromFile(filePath string) (*BaseMap, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read map file: %v", err)
	}

	var metadata MapMetadata
	if err := json.Unmarshal(data, &metadata); err != nil {
		return nil, fmt.Errorf("failed to parse map metadata: %v", err)
	}

	// Convert layout to game objects
	objects := make([]game_objects.GameObject, 0)
	
	// Split the layout string into rows
	rows := strings.Split(strings.TrimSpace(metadata.Layout), "\n")
	
	// Process layout to create block objects
	// We'll scan the layout row by row looking for contiguous blocks
	for y, row := range rows {
		startX := -1 // Start of current block sequence
		for x, char := range row {
			isBlock := char == 'B'
			
			// If we find a block and haven't started a sequence, mark the start
			if isBlock && startX == -1 {
				startX = x
			}
			
			// If we hit a non-block or end of row and we're in a sequence, create the block
			if (!isBlock || x == len(row)-1) && startX != -1 {
				endX := x
				if isBlock && x == len(row)-1 {
					endX = x + 1
				}
				
				// Create a block for this sequence
				blockWidth := float64(endX-startX) * float64(game_objects.BlockSizeUnitPixels)
				blockX := float64(startX)*float64(game_objects.BlockSizeUnitPixels) - float64(metadata.Origin.X)*float64(game_objects.BlockSizeUnitPixels)
				blockY := float64(y)*float64(game_objects.BlockSizeUnitPixels) - float64(metadata.Origin.Y)*float64(game_objects.BlockSizeUnitPixels)
				
				block := game_objects.NewBlockGameObject(
					uuid.New().String(),
					blockX,
					blockY,
					blockWidth,
					float64(game_objects.BlockSizeUnitPixels),
				)
				objects = append(objects, block)
				startX = -1 // Reset for next sequence
			}
		}
	}

	// Convert spawn locations to pixel coordinates
	spawnLocations := make([]*Coords, len(metadata.SpawnLocations))
	for i, spawn := range metadata.SpawnLocations {
		spawnLocations[i] = &Coords{
			X: float64(spawn.X-metadata.Origin.X) * float64(game_objects.BlockSizeUnitPixels),
			Y: float64(spawn.Y-metadata.Origin.Y) * float64(game_objects.BlockSizeUnitPixels),
		}
	}

	// Convert origin to pixel coordinates
	originCoords := &Coords{
		X: float64(metadata.Origin.X) * float64(game_objects.BlockSizeUnitPixels),
		Y: float64(metadata.Origin.Y) * float64(game_objects.BlockSizeUnitPixels),
	}

	return NewBaseMap(
		metadata.MapName,
		objects,
		spawnLocations,
		int32(float64(metadata.ViewSize.X) * float64(game_objects.BlockSizeUnitPixels)),
		int32(float64(metadata.ViewSize.Y) * float64(game_objects.BlockSizeUnitPixels)),
		originCoords,
	), nil
}
