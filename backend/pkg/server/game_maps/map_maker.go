package game_maps

import (
	"container/list"
	"encoding/json"
	"fmt"
	"go-ws-server/pkg/server/game_objects"
	"go-ws-server/pkg/server/geo"
	"os"
	"sort"
	"strings"

	"github.com/google/uuid"
)

type MapType string

const (
	MapDefault    MapType = "meta/default.txt"
	MapWithBlocks MapType = "meta/with_blocks.txt"
)

func CreateMap(mapType MapType) (*BaseMap, error) {
	return CreateMapFromFile(string(mapType))
}

// Point represents a 2D point in grid coordinates
type Point struct {
	X, Y int
}

// findBlockShape uses BFS to find all connected block cells and returns their outline points
func findBlockShape(grid [][]bool, visited [][]bool, startX, startY int) []Point {
	// Directions for BFS: right, down, left, up
	directions := []Point{
		{X: 1, Y: 0},
		{X: 0, Y: 1},
		{X: -1, Y: 0},
		{X: 0, Y: -1},
	}

	// Use BFS to find all connected blocks
	queue := list.New()
	queue.PushBack(Point{X: startX, Y: startY})
	visited[startY][startX] = true

	// Keep track of outline points
	outline := make(map[Point]bool)

	// Process each point in the queue
	for queue.Len() > 0 {
		current := queue.Remove(queue.Front()).(Point)

		// Check if this is an outline point by looking at its neighbors
		isOutline := false
		for _, dir := range directions {
			nextX := current.X + dir.X
			nextY := current.Y + dir.Y

			// Check if the neighbor is within bounds
			if nextX >= 0 && nextX < len(grid[0]) && nextY >= 0 && nextY < len(grid) {
				// If the neighbor is empty, this is an outline point
				if !grid[nextY][nextX] {
					isOutline = true
				} else if !visited[nextY][nextX] {
					// If the neighbor is a block and unvisited, add it to the queue
					queue.PushBack(Point{X: nextX, Y: nextY})
					visited[nextY][nextX] = true
				}
			} else {
				// Points at the grid boundary are outline points
				isOutline = true
			}
		}

		if isOutline {
			outline[current] = true
		}
	}

	// Convert outline points to a slice and sort them
	points := make([]Point, 0, len(outline))
	for point := range outline {
		points = append(points, point)
	}

	// Sort points to ensure a consistent order (clockwise around the shape)
	sort.Slice(points, func(i, j int) bool {
		if points[i].Y != points[j].Y {
			return points[i].Y < points[j].Y
		}
		return points[i].X < points[j].X
	})

	return points
}

// processLayout converts a map metadata into game objects
func processLayout(metadata *MapMetadata) []game_objects.GameObject {
	// Convert layout to game objects
	objects := make([]game_objects.GameObject, 0)

	// Split the layout string into rows
	rows := strings.Split(strings.TrimSpace(metadata.Layout), "\n")

	// Split the layout into a 2D grid
	grid := make([][]bool, 0)
	for _, row := range rows {
		if len(strings.TrimSpace(row)) == 0 {
			continue
		}
		gridRow := make([]bool, len(strings.TrimSpace(row)))
		for x, char := range strings.TrimSpace(row) {
			gridRow[x] = char == 'B'
		}
		grid = append(grid, gridRow)
	}

	// Process layout to create block objects using BFS
	visited := make([][]bool, len(grid))
	for i := range visited {
		visited[i] = make([]bool, len(grid[0]))
	}

	// For each unvisited block cell, run BFS to find the shape
	for y := range grid {
		for x := range grid[y] {
			if !grid[y][x] || visited[y][x] {
				continue
			}

			// Found an unvisited block cell, run BFS to find the shape
			shape := findBlockShape(grid, visited, x, y)

			// Convert shape coordinates to game coordinates
			points := make([]*geo.Point, len(shape))
			for i, point := range shape {
				gameX := float64(point.X)*float64(game_objects.BlockSizeUnitPixels) - float64(metadata.Origin.X)*float64(game_objects.BlockSizeUnitPixels)
				gameY := float64(point.Y)*float64(game_objects.BlockSizeUnitPixels) - float64(metadata.Origin.Y)*float64(game_objects.BlockSizeUnitPixels)
				points[i] = geo.NewPoint(gameX, gameY)
			}

			// Create a block object for this shape
			block := game_objects.NewBlockGameObject(
				uuid.New().String(),
				points,
			)
			objects = append(objects, block)
		}
	}

	return objects
}

func createMapWithBlocks() []game_objects.GameObject {
	// Create a default map layout
	layout := `
		0000BB0000
		0000BB0000
		0000BB0000
		0BBBBBBBB0
		0000BB0000
		0000BB0000
		0000BB0000
	`

	// Create metadata for the default map
	metadata := MapMetadata{
		MapName: "default",
		Layout:  layout,
		Origin: Coordinate{
			X: 5,
			Y: 4,
		},
		SpawnLocations: []Coordinate{
			{X: 0, Y: 0},
			{X: 9, Y: 0},
			{X: 0, Y: 6},
			{X: 9, Y: 6},
		},
		ViewSize: ViewSize{
			X: 10,
			Y: 7,
		},
	}

	// Process the layout and create block objects
	return processLayout(&metadata)
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

	// Split the layout into a 2D grid
	grid := make([][]bool, 0)
	for _, row := range rows {
		if len(strings.TrimSpace(row)) == 0 {
			continue
		}
		gridRow := make([]bool, len(row))
		for x, char := range row {
			gridRow[x] = char == 'B'
		}
		grid = append(grid, gridRow)
	}

	// Process layout to create block objects using BFS
	visited := make([][]bool, len(grid))
	for i := range visited {
		visited[i] = make([]bool, len(grid[0]))
	}

	// For each unvisited block cell, run BFS to find the shape
	for y := range grid {
		for x := range grid[y] {
			if !grid[y][x] || visited[y][x] {
				continue
			}

			// Found an unvisited block cell, run BFS to find the shape
			shape := findBlockShape(grid, visited, x, y)

			// Convert shape coordinates to game coordinates
			points := make([]*geo.Point, len(shape))
			for i, point := range shape {
				gameX := float64(point.X)*float64(game_objects.BlockSizeUnitPixels) - float64(metadata.Origin.X)*float64(game_objects.BlockSizeUnitPixels)
				gameY := float64(point.Y)*float64(game_objects.BlockSizeUnitPixels) - float64(metadata.Origin.Y)*float64(game_objects.BlockSizeUnitPixels)
				points[i] = geo.NewPoint(gameX, gameY)
			}

			// Create a block object for this shape
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
		int32(float64(metadata.ViewSize.X)*float64(game_objects.BlockSizeUnitPixels)),
		int32(float64(metadata.ViewSize.Y)*float64(game_objects.BlockSizeUnitPixels)),
		originCoords,
	), nil
}
