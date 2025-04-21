package game_maps

import (
	"encoding/json"
	"fmt"
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/game_objects"
	"go-ws-server/pkg/server/geo"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/google/uuid"
)

type MapType string

const (
	MapDefault MapType = "meta/default.json"
)

func CreateMap(mapType MapType) (*BaseMap, error) {
	// Get the directory of this source file
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return nil, fmt.Errorf("failed to get current file path")
	}
	// Get the directory containing this file
	dir := filepath.Dir(filename)
	// Construct path relative to this file
	mapPath := filepath.Join(dir, string(mapType))
	return CreateMapFromFile(mapPath)
}

// gets the corner points for a block at layout index (x, y)
func findBlockCorners(xIndex int, yIndex int) []*geo.Point {
	// Convert grid coordinates to pixel coordinates
	blockSize := constants.BlockSizeUnitPixels
	x := float64(xIndex) * blockSize
	y := float64(yIndex) * blockSize

	// Create a rectangle for this single block
	return []*geo.Point{
		{X: x, Y: y},                         // Top-left
		{X: x + blockSize, Y: y},             // Top-right
		{X: x + blockSize, Y: y + blockSize}, // Bottom-right
		{X: x, Y: y + blockSize},             // Bottom-left
	}
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

	// Read layout file
	layoutData, err := os.ReadFile(filepath.Join(filepath.Dir(filePath), metadata.LayoutFile))
	if err != nil {
		return nil, fmt.Errorf("failed to read layout file: %v", err)
	}

	// Split the layout string into rows
	rows := strings.Split(strings.TrimSpace(string(layoutData)), "\n")

	// Split the layout into a 2D grid
	grid := make([][]bool, 0)
	for _, row := range rows {
		row = strings.TrimSpace(row)
		if len(row) == 0 {
			continue
		}
		gridRow := make([]bool, len(row))
		for x, char := range row {
			gridRow[x] = char == 'B'
		}
		grid = append(grid, gridRow)
	}

	// Create blocks for each 'B' in the layout
	for y := range grid {
		for x := range grid[y] {
			if !grid[y][x] {
				continue
			}

			// Get the corner points for this block
			points := findBlockCorners(x, y)

			// Create a block object
			block := game_objects.NewBlockGameObject(
				uuid.New().String(),
				points,
			)
			objects = append(objects, block)
		}
	}

	// Convert spawn locations to pixel coordinates
	spawnLocations := make([]*Coords, len(metadata.SpawnLocations))
	for i, spawn := range metadata.SpawnLocations {
		spawnLocations[i] = &Coords{
			X: float64(spawn.X-metadata.Origin.X) * float64(constants.BlockSizeUnitPixels),
			Y: float64(spawn.Y-metadata.Origin.Y) * float64(constants.BlockSizeUnitPixels),
		}
	}

	// Convert origin to pixel coordinates
	originCoords := &Coords{
		X: float64(metadata.Origin.X) * float64(constants.BlockSizeUnitPixels),
		Y: float64(metadata.Origin.Y) * float64(constants.BlockSizeUnitPixels),
	}

	return NewBaseMap(
		metadata.MapName,
		objects,
		spawnLocations,
		int32(float64(metadata.ViewSize.X)*float64(constants.BlockSizeUnitPixels)),
		int32(float64(metadata.ViewSize.Y)*float64(constants.BlockSizeUnitPixels)),
		originCoords,
	), nil
}
