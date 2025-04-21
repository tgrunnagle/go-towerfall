package game_maps

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

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

// GetAllMapsMetadata returns all map metadata found in the meta directory
func GetAllMapsMetadata() ([]*MapMetadata, error) {
	// Get the directory of this source file
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return nil, fmt.Errorf("failed to get current file path")
	}
	// Get the meta directory path
	metaDir := filepath.Join(filepath.Dir(filename), "meta")

	// List all JSON files in the meta directory
	files, err := os.ReadDir(metaDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read meta directory: %v", err)
	}

	var metadata []*MapMetadata
	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".json" {
			// Read and parse the metadata file
			data, err := os.ReadFile(filepath.Join(metaDir, file.Name()))
			if err != nil {
				return nil, fmt.Errorf("failed to read metadata file %s: %v", file.Name(), err)
			}

			var md MapMetadata
			if err := json.Unmarshal(data, &md); err != nil {
				return nil, fmt.Errorf("failed to parse metadata file %s: %v", file.Name(), err)
			}

			// Add the metadata to the list
			metadata = append(metadata, &md)
		}
	}

	return metadata, nil
}
