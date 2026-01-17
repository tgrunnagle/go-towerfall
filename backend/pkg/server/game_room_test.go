package server

import (
	"sync/atomic"
	"testing"
	"time"

	"go-ws-server/pkg/server/game_objects"
)

func TestCalculateTickInterval_Default(t *testing.T) {
	interval, multiplier, err := calculateTickInterval(nil)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if interval != DefaultTickInterval {
		t.Errorf("Expected interval %v, got %v", DefaultTickInterval, interval)
	}
	if multiplier != 1.0 {
		t.Errorf("Expected multiplier 1.0, got %v", multiplier)
	}
}

func TestCalculateTickInterval_WithInterval(t *testing.T) {
	tests := []struct {
		name           string
		config         *TickConfig
		wantInterval   time.Duration
		wantMultiplier float64
		wantErr        bool
	}{
		{
			name:           "Normal interval (10ms)",
			config:         &TickConfig{TickInterval: 10 * time.Millisecond},
			wantInterval:   10 * time.Millisecond,
			wantMultiplier: 2.0,
			wantErr:        false,
		},
		{
			name:           "Fast interval (2ms)",
			config:         &TickConfig{TickInterval: 2 * time.Millisecond},
			wantInterval:   2 * time.Millisecond,
			wantMultiplier: 10.0,
			wantErr:        false,
		},
		{
			name:           "Slow interval (40ms)",
			config:         &TickConfig{TickInterval: 40 * time.Millisecond},
			wantInterval:   40 * time.Millisecond,
			wantMultiplier: 0.5,
			wantErr:        false,
		},
		{
			name:         "Too fast (below minimum)",
			config:       &TickConfig{TickInterval: 500 * time.Microsecond},
			wantInterval: 0,
			wantErr:      true,
		},
		{
			name:         "Too slow (above maximum)",
			config:       &TickConfig{TickInterval: 2000 * time.Millisecond},
			wantInterval: 0,
			wantErr:      true,
		},
		{
			name:           "Minimum allowed (1ms)",
			config:         &TickConfig{TickInterval: 1 * time.Millisecond},
			wantInterval:   1 * time.Millisecond,
			wantMultiplier: 20.0,
			wantErr:        false,
		},
		{
			name:           "Maximum allowed (1000ms)",
			config:         &TickConfig{TickInterval: 1000 * time.Millisecond},
			wantInterval:   1000 * time.Millisecond,
			wantMultiplier: 0.02,
			wantErr:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			interval, multiplier, err := calculateTickInterval(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("calculateTickInterval() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if interval != tt.wantInterval {
					t.Errorf("calculateTickInterval() interval = %v, want %v", interval, tt.wantInterval)
				}
				if multiplier != tt.wantMultiplier {
					t.Errorf("calculateTickInterval() multiplier = %v, want %v", multiplier, tt.wantMultiplier)
				}
			}
		})
	}
}

func TestCalculateTickInterval_WithMultiplier(t *testing.T) {
	tests := []struct {
		name           string
		config         *TickConfig
		wantInterval   time.Duration
		wantMultiplier float64
		wantErr        bool
	}{
		{
			name:           "2x speed",
			config:         &TickConfig{TickMultiplier: 2.0},
			wantInterval:   10 * time.Millisecond,
			wantMultiplier: 2.0,
			wantErr:        false,
		},
		{
			name:           "10x speed",
			config:         &TickConfig{TickMultiplier: 10.0},
			wantInterval:   2 * time.Millisecond,
			wantMultiplier: 10.0,
			wantErr:        false,
		},
		{
			name:           "Half speed",
			config:         &TickConfig{TickMultiplier: 0.5},
			wantInterval:   40 * time.Millisecond,
			wantMultiplier: 0.5,
			wantErr:        false,
		},
		{
			name:         "Too fast multiplier (results in < 1ms)",
			config:       &TickConfig{TickMultiplier: 100.0},
			wantInterval: 0,
			wantErr:      true,
		},
		{
			name:         "Too slow multiplier (results in > 1000ms)",
			config:       &TickConfig{TickMultiplier: 0.01},
			wantInterval: 0,
			wantErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			interval, multiplier, err := calculateTickInterval(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("calculateTickInterval() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if interval != tt.wantInterval {
					t.Errorf("calculateTickInterval() interval = %v, want %v", interval, tt.wantInterval)
				}
				if multiplier != tt.wantMultiplier {
					t.Errorf("calculateTickInterval() multiplier = %v, want %v", multiplier, tt.wantMultiplier)
				}
			}
		})
	}
}

func TestCalculateTickInterval_IntervalTakesPrecedence(t *testing.T) {
	// When both TickInterval and TickMultiplier are set, TickInterval should take precedence
	config := &TickConfig{
		TickInterval:   5 * time.Millisecond,
		TickMultiplier: 10.0, // Would result in 2ms, but should be ignored
	}

	interval, _, err := calculateTickInterval(config)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if interval != 5*time.Millisecond {
		t.Errorf("Expected interval 5ms, got %v", interval)
	}
}

func TestNewGameRoomWithTickConfig(t *testing.T) {
	tests := []struct {
		name           string
		config         *TickConfig
		wantInterval   time.Duration
		wantMultiplier float64
		wantErr        bool
	}{
		{
			name:           "Default config (nil)",
			config:         nil,
			wantInterval:   DefaultTickInterval,
			wantMultiplier: 1.0,
			wantErr:        false,
		},
		{
			name:           "Custom interval 10ms",
			config:         &TickConfig{TickInterval: 10 * time.Millisecond},
			wantInterval:   10 * time.Millisecond,
			wantMultiplier: 2.0,
			wantErr:        false,
		},
		{
			name:           "Custom multiplier 5x",
			config:         &TickConfig{TickMultiplier: 5.0},
			wantInterval:   4 * time.Millisecond,
			wantMultiplier: 5.0,
			wantErr:        false,
		},
		{
			name:    "Invalid interval (too fast)",
			config:  &TickConfig{TickInterval: 100 * time.Microsecond},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			room, err := NewGameRoomWithTickConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewGameRoomWithTickConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if room.TickInterval != tt.wantInterval {
					t.Errorf("Room.TickInterval = %v, want %v", room.TickInterval, tt.wantInterval)
				}
				if room.TickMultiplier != tt.wantMultiplier {
					t.Errorf("Room.TickMultiplier = %v, want %v", room.TickMultiplier, tt.wantMultiplier)
				}
			}
		})
	}
}

func TestTickLoopStartsAndStops(t *testing.T) {
	room, err := NewGameRoomWithTickConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", &TickConfig{
		TickInterval: 5 * time.Millisecond, // Fast interval for quick testing
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	var tickCount int64
	room.StartTickLoop(func(r *GameRoom, event *game_objects.GameEvent) {
		atomic.AddInt64(&tickCount, 1)
	})

	// Wait for some ticks
	time.Sleep(30 * time.Millisecond)

	// Stop the tick loop
	room.StopTickLoop()

	// Capture the count after stopping
	countAfterStop := atomic.LoadInt64(&tickCount)

	// We should have at least a few ticks (30ms / 5ms = ~6 ticks)
	if countAfterStop < 3 {
		t.Errorf("Expected at least 3 ticks, got %d", countAfterStop)
	}

	// Wait a bit more and verify no more ticks happen
	time.Sleep(20 * time.Millisecond)
	countAfterWait := atomic.LoadInt64(&tickCount)

	if countAfterWait != countAfterStop {
		t.Errorf("Tick loop should have stopped, but tick count increased from %d to %d", countAfterStop, countAfterWait)
	}
}

func TestTickLoopCallsCallback(t *testing.T) {
	room, err := NewGameRoomWithTickConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", &TickConfig{
		TickInterval: 5 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	receivedEvents := make(chan *game_objects.GameEvent, 10)
	room.StartTickLoop(func(r *GameRoom, event *game_objects.GameEvent) {
		select {
		case receivedEvents <- event:
		default:
		}
	})
	defer room.StopTickLoop()

	// Wait for at least one event
	select {
	case event := <-receivedEvents:
		if event.EventType != game_objects.EventGameTick {
			t.Errorf("Expected EventGameTick, got %s", event.EventType)
		}
		if event.RoomID != "test-id" {
			t.Errorf("Expected room ID 'test-id', got %s", event.RoomID)
		}
	case <-time.After(50 * time.Millisecond):
		t.Error("Timeout waiting for tick event")
	}
}

func TestStopTickLoopIsIdempotent(t *testing.T) {
	room, err := NewGameRoomWithTickConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", nil)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	room.StartTickLoop(func(r *GameRoom, event *game_objects.GameEvent) {})

	// Stop multiple times - should not panic
	room.StopTickLoop()
	room.StopTickLoop() // Second stop should be safe
	room.StopTickLoop() // Third stop should also be safe
}

func TestStopTickLoopWithoutStart(t *testing.T) {
	room, err := NewGameRoomWithTickConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", nil)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Stopping a room that was never started should be safe
	room.StopTickLoop()
}

func TestStartTickLoopIgnoresDoubleStart(t *testing.T) {
	room, err := NewGameRoomWithTickConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", &TickConfig{
		TickInterval: 5 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	var tickCount1 int64
	var tickCount2 int64

	// Start with first callback
	room.StartTickLoop(func(r *GameRoom, event *game_objects.GameEvent) {
		atomic.AddInt64(&tickCount1, 1)
	})

	// Try to start again with a different callback - should be ignored
	room.StartTickLoop(func(r *GameRoom, event *game_objects.GameEvent) {
		atomic.AddInt64(&tickCount2, 1)
	})

	// Wait for some ticks
	time.Sleep(30 * time.Millisecond)

	room.StopTickLoop()

	// Only the first callback should have been called
	count1 := atomic.LoadInt64(&tickCount1)
	count2 := atomic.LoadInt64(&tickCount2)

	if count1 < 3 {
		t.Errorf("Expected at least 3 ticks on first callback, got %d", count1)
	}
	if count2 != 0 {
		t.Errorf("Expected 0 ticks on second callback (should be ignored), got %d", count2)
	}
}

// Training mode tests

func TestNewGameRoomWithTrainingConfig(t *testing.T) {
	tests := []struct {
		name                string
		trainingOptions     *TrainingOptions
		wantTrainingEnabled bool
		wantInterval        time.Duration
		wantMultiplier      float64
		wantErr             bool
	}{
		{
			name:                "No training options (nil)",
			trainingOptions:     nil,
			wantTrainingEnabled: false,
			wantInterval:        DefaultTickInterval,
			wantMultiplier:      1.0,
			wantErr:             false,
		},
		{
			name: "Training mode enabled with 10x speed",
			trainingOptions: &TrainingOptions{
				Enabled:        true,
				TickMultiplier: 10.0,
			},
			wantTrainingEnabled: true,
			wantInterval:        2 * time.Millisecond,
			wantMultiplier:      10.0,
			wantErr:             false,
		},
		{
			name: "Training mode with all options",
			trainingOptions: &TrainingOptions{
				Enabled:             true,
				TickMultiplier:      5.0,
				MaxGameDurationSec:  60,
				DisableRespawnTimer: true,
				MaxKills:            20,
			},
			wantTrainingEnabled: true,
			wantInterval:        4 * time.Millisecond,
			wantMultiplier:      5.0,
			wantErr:             false,
		},
		{
			name: "Training mode disabled (enabled=false)",
			trainingOptions: &TrainingOptions{
				Enabled:        false,
				TickMultiplier: 10.0,
			},
			wantTrainingEnabled: false,
			wantInterval:        DefaultTickInterval,
			wantMultiplier:      1.0,
			wantErr:             false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			room, err := NewGameRoomWithTrainingConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", nil, tt.trainingOptions)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewGameRoomWithTrainingConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if room.IsTrainingMode() != tt.wantTrainingEnabled {
					t.Errorf("Room.IsTrainingMode() = %v, want %v", room.IsTrainingMode(), tt.wantTrainingEnabled)
				}
				if room.TickInterval != tt.wantInterval {
					t.Errorf("Room.TickInterval = %v, want %v", room.TickInterval, tt.wantInterval)
				}
				if room.TickMultiplier != tt.wantMultiplier {
					t.Errorf("Room.TickMultiplier = %v, want %v", room.TickMultiplier, tt.wantMultiplier)
				}
			}
		})
	}
}

func TestTrainingModeKillTracking(t *testing.T) {
	room, err := NewGameRoomWithTrainingConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", nil, &TrainingOptions{
		Enabled:  true,
		MaxKills: 5,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Initial kill count should be 0
	if room.GetKillCount() != 0 {
		t.Errorf("Initial kill count should be 0, got %d", room.GetKillCount())
	}

	// Training should not be complete initially
	if room.IsTrainingComplete() {
		t.Error("Training should not be complete initially")
	}

	// Increment kills
	for i := 1; i <= 4; i++ {
		count := room.IncrementKillCount()
		if count != i {
			t.Errorf("IncrementKillCount() returned %d, want %d", count, i)
		}
		if room.IsTrainingComplete() {
			t.Errorf("Training should not be complete with %d kills (max is 5)", i)
		}
	}

	// Fifth kill should complete training
	count := room.IncrementKillCount()
	if count != 5 {
		t.Errorf("IncrementKillCount() returned %d, want 5", count)
	}
	if !room.IsTrainingComplete() {
		t.Error("Training should be complete after 5 kills")
	}
}

func TestTrainingModeDurationTracking(t *testing.T) {
	room, err := NewGameRoomWithTrainingConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", nil, &TrainingOptions{
		Enabled:            true,
		MaxGameDurationSec: 1, // 1 second for quick test
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Training should not be complete initially
	if room.IsTrainingComplete() {
		t.Error("Training should not be complete initially")
	}

	// Elapsed time should be very small initially
	elapsed := room.GetTrainingElapsedSeconds()
	if elapsed > 0.5 {
		t.Errorf("Initial elapsed time should be very small, got %v", elapsed)
	}

	// Wait for the duration to expire
	time.Sleep(1100 * time.Millisecond)

	// Training should be complete now
	if !room.IsTrainingComplete() {
		t.Error("Training should be complete after duration expires")
	}
}

func TestTrainingModeRespawnTime(t *testing.T) {
	tests := []struct {
		name               string
		trainingOptions    *TrainingOptions
		wantRespawnTimeSec float64
	}{
		{
			name:               "No training mode",
			trainingOptions:    nil,
			wantRespawnTimeSec: 5.0, // Default from constants
		},
		{
			name: "Training mode without instant respawn",
			trainingOptions: &TrainingOptions{
				Enabled:             true,
				DisableRespawnTimer: false,
			},
			wantRespawnTimeSec: 5.0, // Default
		},
		{
			name: "Training mode with instant respawn",
			trainingOptions: &TrainingOptions{
				Enabled:             true,
				DisableRespawnTimer: true,
			},
			wantRespawnTimeSec: 0.0, // Instant
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			room, err := NewGameRoomWithTrainingConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", nil, tt.trainingOptions)
			if err != nil {
				t.Fatalf("Failed to create room: %v", err)
			}
			if room.GetRespawnTimeSec() != tt.wantRespawnTimeSec {
				t.Errorf("Room.GetRespawnTimeSec() = %v, want %v", room.GetRespawnTimeSec(), tt.wantRespawnTimeSec)
			}
		})
	}
}

func TestIsTrainingModeWithNoOptions(t *testing.T) {
	room, err := NewGameRoom("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json")
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	if room.IsTrainingMode() {
		t.Error("Room should not be in training mode by default")
	}

	if room.IsTrainingComplete() {
		t.Error("Training should never be complete when not in training mode")
	}
}

func TestNewGameWithPlayerAndTrainingConfig(t *testing.T) {
	room, player, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, &TrainingOptions{
		Enabled:             true,
		TickMultiplier:      5.0,
		DisableRespawnTimer: true,
		MaxKills:            10,
	})
	if err != nil {
		t.Fatalf("Failed to create room with player: %v", err)
	}

	if player == nil {
		t.Fatal("Player should not be nil")
	}

	if !room.IsTrainingMode() {
		t.Error("Room should be in training mode")
	}

	if room.TickMultiplier != 5.0 {
		t.Errorf("Room.TickMultiplier = %v, want 5.0", room.TickMultiplier)
	}

	if room.TrainingOptions.MaxKills != 10 {
		t.Errorf("Room.TrainingOptions.MaxKills = %v, want 10", room.TrainingOptions.MaxKills)
	}

	// Verify player was added with instant respawn configuration
	if room.GetRespawnTimeSec() != 0.0 {
		t.Errorf("Room.GetRespawnTimeSec() = %v, want 0.0 for instant respawn", room.GetRespawnTimeSec())
	}
}

// Reset tests

func TestGameRoomReset(t *testing.T) {
	// Create a room with training mode enabled
	room, player, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, &TrainingOptions{
		Enabled:  true,
		MaxKills: 10,
	})
	if err != nil {
		t.Fatalf("Failed to create room with player: %v", err)
	}

	// Get the player game object
	obj, exists := room.ObjectManager.GetObject(player.ID)
	if !exists {
		t.Fatal("Player game object should exist")
	}
	playerObj := obj.(*game_objects.PlayerGameObject)

	// Increment kill count to simulate game progress
	room.IncrementKillCount()
	room.IncrementKillCount()
	room.IncrementKillCount()

	if room.GetKillCount() != 3 {
		t.Errorf("Kill count should be 3, got %d", room.GetKillCount())
	}

	// Modify player state to simulate gameplay (using short key names from constants)
	playerObj.SetState("dx", 100.0)
	playerObj.SetState("dy", -50.0)
	// Note: can't use "h" since Reset() uses constants.StateHealth which also uses "h"

	// Reset the game
	room.Reset()

	// Verify kill count is reset
	if room.GetKillCount() != 0 {
		t.Errorf("Kill count should be 0 after reset, got %d", room.GetKillCount())
	}

	// Verify player state is reset
	state := playerObj.GetState()

	// Velocity should be 0
	if dx, ok := state["dx"].(float64); !ok || dx != 0.0 {
		t.Errorf("Player dx should be 0 after reset, got %v", state["dx"])
	}
	if dy, ok := state["dy"].(float64); !ok || dy != 0.0 {
		t.Errorf("Player dy should be 0 after reset, got %v", state["dy"])
	}

	// Arrows should be reset to starting count (4)
	// Note: Arrow count is stored as int, not float64
	if arrows, ok := state["ac"].(int); !ok || arrows != 4 {
		t.Errorf("Player arrows should be 4 after reset, got %v", state["ac"])
	}

	// Dead should be false
	if dead, ok := state["dead"].(bool); !ok || dead {
		t.Errorf("Player should not be dead after reset, got %v", state["dead"])
	}

	// Player should still exist in the room
	_, exists = room.GetPlayer(player.ID)
	if !exists {
		t.Error("Player should still exist after reset")
	}
}

func TestGameRoomResetClearsNonPlayerObjects(t *testing.T) {
	// Create a room with a player
	room, player, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, &TrainingOptions{
		Enabled: true,
	})
	if err != nil {
		t.Fatalf("Failed to create room with player: %v", err)
	}

	// Manually add an arrow object to simulate gameplay
	// Get the player object for the arrow's source
	playerObj, _ := room.ObjectManager.GetObject(player.ID)
	arrow := game_objects.NewArrowGameObject("test-arrow", playerObj.(*game_objects.PlayerGameObject), 400, 400, 0.5, room.Map.WrapPosition)
	room.ObjectManager.AddObject(arrow)

	// Verify arrow exists
	_, arrowExists := room.ObjectManager.GetObject("test-arrow")
	if !arrowExists {
		t.Fatal("Arrow should exist before reset")
	}

	// Reset the game
	room.Reset()

	// Verify arrow is removed (nil in the Objects map)
	arrowObj, _ := room.ObjectManager.GetObject("test-arrow")
	if arrowObj != nil {
		t.Error("Arrow should be removed after reset")
	}

	// Verify player still exists
	playerObjAfterReset, playerExists := room.ObjectManager.GetObject(player.ID)
	if !playerExists || playerObjAfterReset == nil {
		t.Error("Player should still exist after reset")
	}
}

func TestGameRoomResetResetsTrainingStartTime(t *testing.T) {
	room, _, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, &TrainingOptions{
		Enabled: true,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Wait a bit so we can detect if the start time is reset
	time.Sleep(100 * time.Millisecond)

	// Get elapsed time before reset
	elapsedBefore := room.GetTrainingElapsedSeconds()
	if elapsedBefore < 0.05 {
		t.Error("Expected some elapsed time before reset")
	}

	// Reset the game
	room.Reset()

	// Get elapsed time after reset - should be very small
	elapsedAfter := room.GetTrainingElapsedSeconds()
	if elapsedAfter > 0.05 {
		t.Errorf("Elapsed time should be near 0 after reset, got %v", elapsedAfter)
	}
}

func TestGameRoomResetWithSpectator(t *testing.T) {
	// Create a room with a player
	room, _, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, &TrainingOptions{
		Enabled: true,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Add a spectator
	spectator, err := AddPlayerToGame(room, "spectator", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator: %v", err)
	}

	// Increment kill count
	room.IncrementKillCount()

	// Reset the game
	room.Reset()

	// Verify spectator is still in the room
	_, spectatorExists := room.GetPlayer(spectator.ID)
	if !spectatorExists {
		t.Error("Spectator should still exist after reset")
	}

	// Verify spectators list still has the spectator
	spectators := room.GetSpectators()
	found := false
	for _, name := range spectators {
		if name == "spectator" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Spectator should still be in spectators list after reset")
	}
}

// Player Stats tests

func TestPlayerStatsInitialization(t *testing.T) {
	room, player, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, nil)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Verify player stats are initialized
	stats, exists := room.GetPlayerStats(player.ID)
	if !exists {
		t.Fatal("Player stats should exist after player joins")
	}

	if stats.Kills != 0 {
		t.Errorf("Initial kills should be 0, got %d", stats.Kills)
	}
	if stats.Deaths != 0 {
		t.Errorf("Initial deaths should be 0, got %d", stats.Deaths)
	}
}

func TestPlayerStatsNotInitializedForSpectators(t *testing.T) {
	room, _, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, nil)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Add a spectator
	spectator, err := AddPlayerToGame(room, "spectator", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator: %v", err)
	}

	// Verify spectator does not have stats
	_, exists := room.GetPlayerStats(spectator.ID)
	if exists {
		t.Error("Spectator should not have player stats")
	}
}

func TestRecordKill(t *testing.T) {
	room, player, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, nil)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Record some kills
	room.RecordKill(player.ID)
	room.RecordKill(player.ID)
	room.RecordKill(player.ID)

	stats, exists := room.GetPlayerStats(player.ID)
	if !exists {
		t.Fatal("Player stats should exist")
	}

	if stats.Kills != 3 {
		t.Errorf("Kills should be 3, got %d", stats.Kills)
	}
	if stats.Deaths != 0 {
		t.Errorf("Deaths should still be 0, got %d", stats.Deaths)
	}
}

func TestRecordDeath(t *testing.T) {
	room, player, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, nil)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Record some deaths
	room.RecordDeath(player.ID)
	room.RecordDeath(player.ID)

	stats, exists := room.GetPlayerStats(player.ID)
	if !exists {
		t.Fatal("Player stats should exist")
	}

	if stats.Deaths != 2 {
		t.Errorf("Deaths should be 2, got %d", stats.Deaths)
	}
	if stats.Kills != 0 {
		t.Errorf("Kills should still be 0, got %d", stats.Kills)
	}
}

func TestRecordKillAndDeathForMultiplePlayers(t *testing.T) {
	room, player1, err := NewGameWithPlayerAndTrainingConfig("test-room", "player1", "meta/default.json", nil, nil)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Add a second player
	player2, err := AddPlayerToGame(room, "player2", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player2: %v", err)
	}

	// Simulate player1 killing player2
	room.RecordKill(player1.ID)
	room.RecordDeath(player2.ID)

	// Simulate player2 killing player1
	room.RecordKill(player2.ID)
	room.RecordDeath(player1.ID)

	// Check player1 stats
	stats1, exists := room.GetPlayerStats(player1.ID)
	if !exists {
		t.Fatal("Player1 stats should exist")
	}
	if stats1.Kills != 1 {
		t.Errorf("Player1 kills should be 1, got %d", stats1.Kills)
	}
	if stats1.Deaths != 1 {
		t.Errorf("Player1 deaths should be 1, got %d", stats1.Deaths)
	}

	// Check player2 stats
	stats2, exists := room.GetPlayerStats(player2.ID)
	if !exists {
		t.Fatal("Player2 stats should exist")
	}
	if stats2.Kills != 1 {
		t.Errorf("Player2 kills should be 1, got %d", stats2.Kills)
	}
	if stats2.Deaths != 1 {
		t.Errorf("Player2 deaths should be 1, got %d", stats2.Deaths)
	}
}

func TestGetAllPlayerStats(t *testing.T) {
	room, player1, err := NewGameWithPlayerAndTrainingConfig("test-room", "player1", "meta/default.json", nil, nil)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Add a second player
	player2, err := AddPlayerToGame(room, "player2", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player2: %v", err)
	}

	// Record some stats
	room.RecordKill(player1.ID)
	room.RecordKill(player1.ID)
	room.RecordDeath(player2.ID)
	room.RecordDeath(player2.ID)

	// Get all stats
	allStats := room.GetAllPlayerStats()

	if len(allStats) != 2 {
		t.Errorf("Expected 2 player stats, got %d", len(allStats))
	}

	// Check player1
	if stats, exists := allStats[player1.ID]; !exists {
		t.Error("Player1 stats should exist in all stats")
	} else {
		if stats.Kills != 2 {
			t.Errorf("Player1 kills should be 2, got %d", stats.Kills)
		}
	}

	// Check player2
	if stats, exists := allStats[player2.ID]; !exists {
		t.Error("Player2 stats should exist in all stats")
	} else {
		if stats.Deaths != 2 {
			t.Errorf("Player2 deaths should be 2, got %d", stats.Deaths)
		}
	}
}

func TestGetAllPlayerStatsReturnsCopy(t *testing.T) {
	room, player, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, nil)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Get stats and modify the returned copy
	allStats := room.GetAllPlayerStats()
	allStats[player.ID].Kills = 999

	// Verify original stats are unchanged
	stats, _ := room.GetPlayerStats(player.ID)
	if stats.Kills != 0 {
		t.Errorf("Original stats should be unchanged, got kills=%d", stats.Kills)
	}
}

func TestPlayerStatsResetOnGameReset(t *testing.T) {
	room, player1, err := NewGameWithPlayerAndTrainingConfig("test-room", "player1", "meta/default.json", nil, &TrainingOptions{
		Enabled: true,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Add a second player
	player2, err := AddPlayerToGame(room, "player2", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player2: %v", err)
	}

	// Record some stats
	room.RecordKill(player1.ID)
	room.RecordKill(player1.ID)
	room.RecordDeath(player1.ID)
	room.RecordKill(player2.ID)
	room.RecordDeath(player2.ID)
	room.RecordDeath(player2.ID)

	// Verify stats before reset
	stats1Before, _ := room.GetPlayerStats(player1.ID)
	stats2Before, _ := room.GetPlayerStats(player2.ID)
	if stats1Before.Kills != 2 || stats1Before.Deaths != 1 {
		t.Errorf("Player1 stats before reset should be kills=2 deaths=1, got kills=%d deaths=%d", stats1Before.Kills, stats1Before.Deaths)
	}
	if stats2Before.Kills != 1 || stats2Before.Deaths != 2 {
		t.Errorf("Player2 stats before reset should be kills=1 deaths=2, got kills=%d deaths=%d", stats2Before.Kills, stats2Before.Deaths)
	}

	// Reset the game
	room.Reset()

	// Verify stats are reset
	stats1After, _ := room.GetPlayerStats(player1.ID)
	stats2After, _ := room.GetPlayerStats(player2.ID)

	if stats1After.Kills != 0 || stats1After.Deaths != 0 {
		t.Errorf("Player1 stats after reset should be 0, got kills=%d deaths=%d", stats1After.Kills, stats1After.Deaths)
	}
	if stats2After.Kills != 0 || stats2After.Deaths != 0 {
		t.Errorf("Player2 stats after reset should be 0, got kills=%d deaths=%d", stats2After.Kills, stats2After.Deaths)
	}
}

func TestRecordKillForNonexistentPlayer(t *testing.T) {
	room, _, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, nil)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Should not panic when recording kill for nonexistent player
	room.RecordKill("nonexistent-player-id")

	// Verify no stats created for nonexistent player
	_, exists := room.GetPlayerStats("nonexistent-player-id")
	if exists {
		t.Error("Stats should not be created for nonexistent player")
	}
}

func TestRecordDeathForNonexistentPlayer(t *testing.T) {
	room, _, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, nil)
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Should not panic when recording death for nonexistent player
	room.RecordDeath("nonexistent-player-id")

	// Verify no stats created for nonexistent player
	_, exists := room.GetPlayerStats("nonexistent-player-id")
	if exists {
		t.Error("Stats should not be created for nonexistent player")
	}
}

// Spectator throttling tests

func TestSpectatorThrottler_NoThrottlingForNormalRoom(t *testing.T) {
	// Create a room without training mode
	room, err := NewGameRoom("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json")
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Without training mode, ShouldThrottleSpectatorUpdate should always return false
	for i := 0; i < 100; i++ {
		if room.ShouldThrottleSpectatorUpdate() {
			t.Error("ShouldThrottleSpectatorUpdate should return false for non-training rooms")
		}
	}
}

func TestSpectatorThrottler_NoThrottlingFor1xSpeed(t *testing.T) {
	// Create a training room with 1x speed (no acceleration)
	room, err := NewGameRoomWithTrainingConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", nil, &TrainingOptions{
		Enabled:        true,
		TickMultiplier: 1.0,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Without acceleration (1x speed), ShouldThrottleSpectatorUpdate should always return false
	for i := 0; i < 100; i++ {
		if room.ShouldThrottleSpectatorUpdate() {
			t.Error("ShouldThrottleSpectatorUpdate should return false for 1x speed training rooms")
		}
	}
}

func TestSpectatorThrottler_ThrottlesAcceleratedTraining(t *testing.T) {
	// Create a training room with 10x speed
	room, err := NewGameRoomWithTrainingConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", nil, &TrainingOptions{
		Enabled:        true,
		TickMultiplier: 10.0,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// First call should not throttle (allows the update)
	if room.ShouldThrottleSpectatorUpdate() {
		t.Error("First call to ShouldThrottleSpectatorUpdate should return false")
	}

	// Immediate subsequent calls should throttle
	throttled := false
	for i := 0; i < 10; i++ {
		if room.ShouldThrottleSpectatorUpdate() {
			throttled = true
			break
		}
	}
	if !throttled {
		t.Error("Immediate subsequent calls should be throttled")
	}
}

func TestSpectatorThrottler_AllowsUpdateAfterMinInterval(t *testing.T) {
	// Create a training room with 10x speed
	room, err := NewGameRoomWithTrainingConfig("test-id", "test-room", "PASSWORD", "ABCD", "meta/default.json", nil, &TrainingOptions{
		Enabled:        true,
		TickMultiplier: 10.0,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// First call should not throttle
	if room.ShouldThrottleSpectatorUpdate() {
		t.Error("First call should not throttle")
	}

	// Wait for longer than the minimum interval (16ms)
	time.Sleep(20 * time.Millisecond)

	// Next call should not throttle
	if room.ShouldThrottleSpectatorUpdate() {
		t.Error("Call after waiting should not throttle")
	}
}

func TestSpectatorCountAndPlayerCount(t *testing.T) {
	// Create a room with a player
	room, player1, err := NewGameWithPlayerAndTrainingConfig("test-room", "player1", "meta/default.json", nil, &TrainingOptions{
		Enabled: true,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Initially: 1 player, 0 spectators
	if room.GetPlayerCount() != 1 {
		t.Errorf("Expected 1 player, got %d", room.GetPlayerCount())
	}
	if room.GetSpectatorCount() != 0 {
		t.Errorf("Expected 0 spectators, got %d", room.GetSpectatorCount())
	}
	if room.HasSpectators() {
		t.Error("Expected HasSpectators to return false")
	}

	// Add a second player
	_, err = AddPlayerToGame(room, "player2", room.Password, false)
	if err != nil {
		t.Fatalf("Failed to add player2: %v", err)
	}

	// Now: 2 players, 0 spectators
	if room.GetPlayerCount() != 2 {
		t.Errorf("Expected 2 players, got %d", room.GetPlayerCount())
	}
	if room.GetSpectatorCount() != 0 {
		t.Errorf("Expected 0 spectators, got %d", room.GetSpectatorCount())
	}

	// Add a spectator
	_, err = AddPlayerToGame(room, "spectator1", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add spectator: %v", err)
	}

	// Now: 2 players, 1 spectator
	if room.GetPlayerCount() != 2 {
		t.Errorf("Expected 2 players, got %d", room.GetPlayerCount())
	}
	if room.GetSpectatorCount() != 1 {
		t.Errorf("Expected 1 spectator, got %d", room.GetSpectatorCount())
	}
	if !room.HasSpectators() {
		t.Error("Expected HasSpectators to return true")
	}

	// Add another spectator
	_, err = AddPlayerToGame(room, "spectator2", room.Password, true)
	if err != nil {
		t.Fatalf("Failed to add second spectator: %v", err)
	}

	// Now: 2 players, 2 spectators
	if room.GetPlayerCount() != 2 {
		t.Errorf("Expected 2 players, got %d", room.GetPlayerCount())
	}
	if room.GetSpectatorCount() != 2 {
		t.Errorf("Expected 2 spectators, got %d", room.GetSpectatorCount())
	}

	// Remove first player
	room.RemovePlayer(player1.ID)

	// Now: 1 player, 2 spectators
	if room.GetPlayerCount() != 1 {
		t.Errorf("Expected 1 player after removal, got %d", room.GetPlayerCount())
	}
	if room.GetSpectatorCount() != 2 {
		t.Errorf("Expected 2 spectators after player removal, got %d", room.GetSpectatorCount())
	}
}

func TestTrainingEpisodeCounter(t *testing.T) {
	// Create a training room
	room, _, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, &TrainingOptions{
		Enabled: true,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Initial episode should be 1
	if room.GetTrainingEpisode() != 1 {
		t.Errorf("Initial episode should be 1, got %d", room.GetTrainingEpisode())
	}

	// Reset should increment episode
	room.Reset()
	if room.GetTrainingEpisode() != 2 {
		t.Errorf("Episode after first reset should be 2, got %d", room.GetTrainingEpisode())
	}

	// Reset again
	room.Reset()
	if room.GetTrainingEpisode() != 3 {
		t.Errorf("Episode after second reset should be 3, got %d", room.GetTrainingEpisode())
	}
}

func TestGetRoomDuration(t *testing.T) {
	room, _, err := NewGameWithPlayerAndTrainingConfig("test-room", "test-player", "meta/default.json", nil, &TrainingOptions{
		Enabled: true,
	})
	if err != nil {
		t.Fatalf("Failed to create room: %v", err)
	}

	// Duration should be very small initially
	duration := room.GetRoomDuration()
	if duration > 100*time.Millisecond {
		t.Errorf("Initial duration should be very small, got %v", duration)
	}

	// Wait a bit
	time.Sleep(50 * time.Millisecond)

	// Duration should increase
	duration = room.GetRoomDuration()
	if duration < 40*time.Millisecond {
		t.Errorf("Duration should be at least 40ms after waiting, got %v", duration)
	}
}
