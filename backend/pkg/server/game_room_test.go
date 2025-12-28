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
