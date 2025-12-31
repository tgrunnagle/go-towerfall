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
