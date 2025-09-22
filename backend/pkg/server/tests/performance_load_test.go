package tests

import (
	"fmt"
	"go-ws-server/pkg/server"
	"go-ws-server/pkg/server/game_maps"
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestServerPerformanceWithMultipleConcurrentRooms tests server performance with many concurrent rooms
func TestServerPerformanceWithMultipleConcurrentRooms(t *testing.T) {
	// Create server instance
	gameServer := server.NewServer()
	roomManager := gameServer.GetRoomManager()

	tests := []struct {
		name           string
		numRooms       int
		playersPerRoom int
		testDuration   time.Duration
	}{
		{
			name:           "Small load - 10 rooms, 2 players each",
			numRooms:       10,
			playersPerRoom: 2,
			testDuration:   5 * time.Second,
		},
		{
			name:           "Medium load - 50 rooms, 4 players each",
			numRooms:       50,
			playersPerRoom: 4,
			testDuration:   10 * time.Second,
		},
		{
			name:           "High load - 100 rooms, 6 players each",
			numRooms:       100,
			playersPerRoom: 6,
			testDuration:   15 * time.Second,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startTime := time.Now()

			// Record initial memory stats
			var initialMemStats runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&initialMemStats)

			// Create rooms concurrently
			var wg sync.WaitGroup
			rooms := make([]*server.GameRoom, tt.numRooms)

			for i := 0; i < tt.numRooms; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()

					roomName := fmt.Sprintf("Load Test Room %d", index)
					playerName := fmt.Sprintf("Player %d", index)

					room, _, err := server.NewGameWithPlayer(roomName, playerName, game_maps.MapDefault)
					if err != nil {
						t.Errorf("Failed to create room %d: %v", index, err)
						return
					}

					rooms[index] = room
					roomManager.AddGameRoom(room)

					// Add additional players to the room
					for j := 1; j < tt.playersPerRoom; j++ {
						playerName := fmt.Sprintf("Player %d-%d", index, j)
						_, err := server.AddPlayerToGame(room, playerName, room.Password, false)
						if err != nil {
							t.Errorf("Failed to add player %s to room %d: %v", playerName, index, err)
						}
					}
				}(i)
			}

			wg.Wait()
			creationTime := time.Since(startTime)

			// Verify all rooms were created
			roomIDs := roomManager.GetGameRoomIDs()
			if len(roomIDs) < tt.numRooms {
				t.Errorf("Expected at least %d rooms, got %d", tt.numRooms, len(roomIDs))
			}

			// Let the server run under load for the test duration
			time.Sleep(tt.testDuration)

			// Record final memory stats
			var finalMemStats runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&finalMemStats)

			// Calculate memory usage
			var memoryIncrease uint64
			if finalMemStats.Alloc > initialMemStats.Alloc {
				memoryIncrease = finalMemStats.Alloc - initialMemStats.Alloc
			} else {
				memoryIncrease = 0 // Memory usage decreased or stayed the same
			}

			// Performance metrics
			t.Logf("Performance metrics for %s:", tt.name)
			t.Logf("  Room creation time: %v", creationTime)
			t.Logf("  Average time per room: %v", creationTime/time.Duration(tt.numRooms))
			t.Logf("  Total players: %d", tt.numRooms*tt.playersPerRoom)
			t.Logf("  Memory increase: %d bytes (%.2f MB)", memoryIncrease, float64(memoryIncrease)/(1024*1024))
			t.Logf("  Memory per room: %d bytes", memoryIncrease/uint64(tt.numRooms))

			// Performance assertions
			maxCreationTimePerRoom := 100 * time.Millisecond
			avgCreationTime := creationTime / time.Duration(tt.numRooms)
			if avgCreationTime > maxCreationTimePerRoom {
				t.Errorf("Room creation too slow: %v per room (max: %v)", avgCreationTime, maxCreationTimePerRoom)
			}

			// Memory usage should be reasonable (less than 10MB per room for small loads)
			if tt.numRooms <= 10 {
				maxMemoryPerRoom := uint64(10 * 1024 * 1024) // 10MB per room
				memoryPerRoom := memoryIncrease / uint64(tt.numRooms)
				if memoryPerRoom > maxMemoryPerRoom {
					t.Errorf("Memory usage too high: %d bytes per room (max: %d)", memoryPerRoom, maxMemoryPerRoom)
				}
			}

			// Cleanup - remove all rooms
			for _, room := range rooms {
				if room != nil {
					roomManager.RemoveGameRoom(room.ID)
				}
			}
		})
	}
}

// TestMemoryUsageAndCleanupUnderLoad tests memory management and cleanup
func TestMemoryUsageAndCleanupUnderLoad(t *testing.T) {
	gameServer := server.NewServer()
	roomManager := gameServer.GetRoomManager()

	// Record baseline memory
	var baselineMemStats runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&baselineMemStats)

	numCycles := 5
	roomsPerCycle := 20
	playersPerRoom := 4

	for cycle := 0; cycle < numCycles; cycle++ {
		t.Logf("Memory test cycle %d/%d", cycle+1, numCycles)

		// Create rooms
		rooms := make([]*server.GameRoom, roomsPerCycle)
		for i := 0; i < roomsPerCycle; i++ {
			roomName := fmt.Sprintf("Memory Test Room %d-%d", cycle, i)
			playerName := fmt.Sprintf("Player %d-%d", cycle, i)

			room, _, err := server.NewGameWithPlayer(roomName, playerName, game_maps.MapDefault)
			if err != nil {
				t.Fatalf("Failed to create room: %v", err)
			}

			rooms[i] = room
			roomManager.AddGameRoom(room)

			// Add additional players
			for j := 1; j < playersPerRoom; j++ {
				playerName := fmt.Sprintf("Player %d-%d-%d", cycle, i, j)
				_, err := server.AddPlayerToGame(room, playerName, room.Password, false)
				if err != nil {
					t.Errorf("Failed to add player: %v", err)
				}
			}
		}

		// Let rooms run for a bit
		time.Sleep(2 * time.Second)

		// Record memory after creation
		var afterCreationMemStats runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&afterCreationMemStats)

		// Remove all rooms
		for _, room := range rooms {
			roomManager.RemoveGameRoom(room.ID)
		}

		// Force garbage collection and wait
		runtime.GC()
		time.Sleep(1 * time.Second)
		runtime.GC()

		// Record memory after cleanup
		var afterCleanupMemStats runtime.MemStats
		runtime.ReadMemStats(&afterCleanupMemStats)

		// Calculate memory metrics
		var memoryAfterCreation, memoryAfterCleanup, memoryReclaimed uint64

		if afterCreationMemStats.Alloc > baselineMemStats.Alloc {
			memoryAfterCreation = afterCreationMemStats.Alloc - baselineMemStats.Alloc
		}

		if afterCleanupMemStats.Alloc > baselineMemStats.Alloc {
			memoryAfterCleanup = afterCleanupMemStats.Alloc - baselineMemStats.Alloc
		}

		if memoryAfterCreation > memoryAfterCleanup {
			memoryReclaimed = memoryAfterCreation - memoryAfterCleanup
		}

		t.Logf("  Memory after creation: %.2f MB", float64(memoryAfterCreation)/(1024*1024))
		t.Logf("  Memory after cleanup: %.2f MB", float64(memoryAfterCleanup)/(1024*1024))
		t.Logf("  Memory reclaimed: %.2f MB (%.1f%%)",
			float64(memoryReclaimed)/(1024*1024),
			float64(memoryReclaimed)/float64(memoryAfterCreation)*100)

		// Verify that reasonable memory is reclaimed (at least 50%)
		// Note: Go's garbage collector may not immediately reclaim all memory
		if memoryAfterCreation > 0 {
			reclaimPercentage := float64(memoryReclaimed) / float64(memoryAfterCreation) * 100
			if reclaimPercentage < 50 {
				t.Errorf("Insufficient memory cleanup: only %.1f%% reclaimed (expected at least 50%%)", reclaimPercentage)
			}
		}

		// Verify room count is back to 0
		roomIDs := roomManager.GetGameRoomIDs()
		if len(roomIDs) != 0 {
			t.Errorf("Expected 0 rooms after cleanup, got %d", len(roomIDs))
		}
	}
}

// TestWebSocketConnectionLimitsAndHandling tests WebSocket connection management
func TestWebSocketConnectionLimitsAndHandling(t *testing.T) {
	gameServer := server.NewServer()
	roomManager := gameServer.GetRoomManager()

	tests := []struct {
		name                string
		numRooms            int
		connectionsPerRoom  int
		maxTotalConnections int
	}{
		{
			name:                "Moderate connections - 10 rooms, 10 connections each",
			numRooms:            10,
			connectionsPerRoom:  10,
			maxTotalConnections: 100,
		},
		{
			name:                "High connections - 20 rooms, 25 connections each",
			numRooms:            20,
			connectionsPerRoom:  25,
			maxTotalConnections: 500,
		},
		{
			name:                "Stress test - 50 rooms, 20 connections each",
			numRooms:            50,
			connectionsPerRoom:  20,
			maxTotalConnections: 1000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startTime := time.Now()

			// Create rooms with multiple players (simulating connections)
			rooms := make([]*server.GameRoom, tt.numRooms)
			totalPlayers := 0

			for i := 0; i < tt.numRooms; i++ {
				roomName := fmt.Sprintf("Connection Test Room %d", i)
				playerName := fmt.Sprintf("Host Player %d", i)

				room, _, err := server.NewGameWithPlayer(roomName, playerName, game_maps.MapDefault)
				if err != nil {
					t.Fatalf("Failed to create room %d: %v", i, err)
				}

				rooms[i] = room
				roomManager.AddGameRoom(room)
				totalPlayers++

				// Add additional players (simulating connections)
				for j := 1; j < tt.connectionsPerRoom; j++ {
					playerName := fmt.Sprintf("Player %d-%d", i, j)
					isSpectator := j > tt.connectionsPerRoom/2 // Half are spectators

					_, err := server.AddPlayerToGame(room, playerName, room.Password, isSpectator)
					if err != nil {
						t.Errorf("Failed to add player %s: %v", playerName, err)
						continue
					}
					totalPlayers++
				}
			}

			setupTime := time.Since(startTime)

			// Verify all connections were created
			actualRooms := roomManager.GetGameRoomIDs()
			if len(actualRooms) != tt.numRooms {
				t.Errorf("Expected %d rooms, got %d", tt.numRooms, len(actualRooms))
			}

			// Verify player counts
			totalConnectedPlayers := 0
			for _, room := range rooms {
				if room != nil {
					totalConnectedPlayers += room.GetNumberOfConnectedPlayers()
				}
			}

			expectedPlayers := tt.numRooms * tt.connectionsPerRoom
			if totalConnectedPlayers != expectedPlayers {
				t.Errorf("Expected %d total players, got %d", expectedPlayers, totalConnectedPlayers)
			}

			// Test connection stability under load
			stabilityTestDuration := 10 * time.Second
			time.Sleep(stabilityTestDuration)

			// Verify connections are still stable
			finalConnectedPlayers := 0
			for _, room := range rooms {
				if room != nil {
					finalConnectedPlayers += room.GetNumberOfConnectedPlayers()
				}
			}

			if finalConnectedPlayers != totalConnectedPlayers {
				t.Errorf("Connection instability detected: started with %d players, ended with %d",
					totalConnectedPlayers, finalConnectedPlayers)
			}

			// Performance metrics
			t.Logf("Connection test metrics for %s:", tt.name)
			t.Logf("  Setup time: %v", setupTime)
			t.Logf("  Total connections: %d", totalConnectedPlayers)
			t.Logf("  Average setup time per connection: %v", setupTime/time.Duration(totalConnectedPlayers))
			t.Logf("  Connections per room: %d", tt.connectionsPerRoom)

			// Performance assertions
			maxSetupTimePerConnection := 10 * time.Millisecond
			avgSetupTime := setupTime / time.Duration(totalConnectedPlayers)
			if avgSetupTime > maxSetupTimePerConnection {
				t.Errorf("Connection setup too slow: %v per connection (max: %v)",
					avgSetupTime, maxSetupTimePerConnection)
			}

			// Cleanup
			for _, room := range rooms {
				if room != nil {
					roomManager.RemoveGameRoom(room.ID)
				}
			}
		})
	}
}

// TestTrainingModePerformanceAtHighSpeedMultipliers tests training mode performance
func TestTrainingModePerformanceAtHighSpeedMultipliers(t *testing.T) {
	gameServer := server.NewServer()
	roomManager := gameServer.GetRoomManager()

	speedMultipliers := []float64{1.0, 5.0, 10.0, 25.0, 50.0, 100.0}
	testDuration := 5 * time.Second

	for _, speed := range speedMultipliers {
		t.Run(fmt.Sprintf("Speed %.1fx", speed), func(t *testing.T) {
			// Create training room with specific speed
			trainingConfig := server.TrainingConfig{
				SpeedMultiplier:   speed,
				HeadlessMode:      speed >= 10.0, // Use headless for high speeds
				TrainingMode:      true,
				SessionID:         fmt.Sprintf("perf-test-%.1fx", speed),
				DirectStateAccess: true,
			}

			roomName := fmt.Sprintf("Speed Test %.1fx", speed)
			room, _, err := server.NewTrainingGameWithPlayer(roomName, "Test Player", game_maps.MapDefault, trainingConfig)
			if err != nil {
				t.Fatalf("Failed to create training room: %v", err)
			}

			roomManager.AddGameRoom(room)

			// Add a few more players for realistic load
			for i := 1; i < 4; i++ {
				playerName := fmt.Sprintf("Bot Player %d", i)
				_, err := server.AddPlayerToGame(room, playerName, room.Password, false)
				if err != nil {
					t.Errorf("Failed to add bot player: %v", err)
				}
			}

			// Record initial metrics
			var initialMemStats runtime.MemStats
			runtime.ReadMemStats(&initialMemStats)
			startTime := time.Now()

			// Let the room run at high speed
			time.Sleep(testDuration)

			// Record final metrics
			var finalMemStats runtime.MemStats
			runtime.ReadMemStats(&finalMemStats)
			endTime := time.Now()

			// Calculate performance metrics
			actualDuration := endTime.Sub(startTime)
			var memoryUsed uint64
			if finalMemStats.Alloc > initialMemStats.Alloc {
				memoryUsed = finalMemStats.Alloc - initialMemStats.Alloc
			} else {
				memoryUsed = 0 // Memory usage decreased or stayed the same
			}

			// Test direct state access performance
			stateAccessStart := time.Now()
			for i := 0; i < 100; i++ {
				gameState := room.GetDirectGameState()
				if gameState == nil {
					t.Errorf("Direct state access failed at iteration %d", i)
					break
				}
			}
			stateAccessDuration := time.Since(stateAccessStart)
			avgStateAccessTime := stateAccessDuration / 100

			// Verify room is still functional
			if room.GetNumberOfConnectedPlayers() != 4 {
				t.Errorf("Player count changed during test: expected 4, got %d", room.GetNumberOfConnectedPlayers())
			}

			// Verify speed multiplier is still correct
			if room.GetSpeedMultiplier() != speed {
				t.Errorf("Speed multiplier changed: expected %.1f, got %.1f", speed, room.GetSpeedMultiplier())
			}

			// Performance metrics
			t.Logf("Performance metrics for speed %.1fx:", speed)
			t.Logf("  Test duration: %v", actualDuration)
			t.Logf("  Memory used: %.2f MB", float64(memoryUsed)/(1024*1024))
			t.Logf("  Average state access time: %v", avgStateAccessTime)
			t.Logf("  Headless mode: %v", room.IsHeadlessMode())
			t.Logf("  Custom tick rate: %v", room.GetCustomTickRate())

			// Performance assertions
			maxStateAccessTime := 1 * time.Millisecond
			if avgStateAccessTime > maxStateAccessTime {
				t.Errorf("State access too slow at speed %.1fx: %v (max: %v)",
					speed, avgStateAccessTime, maxStateAccessTime)
			}

			// Memory usage should not grow excessively with speed
			maxMemoryUsage := uint64(50 * 1024 * 1024) // 50MB max
			if memoryUsed > maxMemoryUsage {
				t.Errorf("Excessive memory usage at speed %.1fx: %.2f MB (max: %.2f MB)",
					speed, float64(memoryUsed)/(1024*1024), float64(maxMemoryUsage)/(1024*1024))
			}

			// Cleanup
			roomManager.RemoveGameRoom(room.ID)
		})
	}
}

// TestServerStabilityDuringExtendedOperation tests long-running server stability
func TestServerStabilityDuringExtendedOperation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping extended operation test in short mode")
	}

	gameServer := server.NewServer()
	roomManager := gameServer.GetRoomManager()

	// Test configuration
	testDuration := 2 * time.Minute // Reduced for CI/testing
	numRooms := 10
	playersPerRoom := 3
	operationInterval := 5 * time.Second

	t.Logf("Starting extended stability test for %v", testDuration)

	// Create initial rooms
	rooms := make([]*server.GameRoom, numRooms)
	for i := 0; i < numRooms; i++ {
		roomName := fmt.Sprintf("Stability Test Room %d", i)
		playerName := fmt.Sprintf("Player %d", i)

		room, _, err := server.NewGameWithPlayer(roomName, playerName, game_maps.MapDefault)
		if err != nil {
			t.Fatalf("Failed to create initial room %d: %v", i, err)
		}

		rooms[i] = room
		roomManager.AddGameRoom(room)

		// Add additional players
		for j := 1; j < playersPerRoom; j++ {
			playerName := fmt.Sprintf("Player %d-%d", i, j)
			_, err := server.AddPlayerToGame(room, playerName, room.Password, false)
			if err != nil {
				t.Errorf("Failed to add player: %v", err)
			}
		}
	}

	// Record initial state
	var initialMemStats runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&initialMemStats)
	startTime := time.Now()

	// Run stability test with periodic operations
	ticker := time.NewTicker(operationInterval)
	defer ticker.Stop()

	operationCount := 0
	maxOperations := int(testDuration / operationInterval)

	for {
		select {
		case <-ticker.C:
			operationCount++

			// Perform various operations to stress the server
			t.Logf("Stability test operation %d/%d", operationCount, maxOperations)

			// 1. Create and destroy temporary rooms
			tempRoomName := fmt.Sprintf("Temp Room %d", operationCount)
			tempRoom, _, err := server.NewGameWithPlayer(tempRoomName, "Temp Player", game_maps.MapDefault)
			if err != nil {
				t.Errorf("Failed to create temporary room: %v", err)
			} else {
				roomManager.AddGameRoom(tempRoom)
				time.Sleep(1 * time.Second)
				roomManager.RemoveGameRoom(tempRoom.ID)
			}

			// 2. Add and remove players from existing rooms
			if len(rooms) > 0 {
				roomIndex := operationCount % len(rooms)
				room := rooms[roomIndex]
				if room != nil {
					playerName := fmt.Sprintf("Dynamic Player %d", operationCount)
					player, err := server.AddPlayerToGame(room, playerName, room.Password, false)
					if err != nil {
						t.Errorf("Failed to add dynamic player: %v", err)
					} else {
						time.Sleep(500 * time.Millisecond)
						room.RemovePlayer(player.ID)
					}
				}
			}

			// 3. Test room state access
			roomIDs := roomManager.GetGameRoomIDs()
			for _, roomID := range roomIDs {
				room, exists := roomManager.GetGameRoom(roomID)
				if exists {
					_ = room.GetAllGameObjectStates()
				}
			}

			// 4. Check memory usage
			var currentMemStats runtime.MemStats
			runtime.ReadMemStats(&currentMemStats)
			var currentMemoryUsage uint64
			if currentMemStats.Alloc > initialMemStats.Alloc {
				currentMemoryUsage = currentMemStats.Alloc - initialMemStats.Alloc
			}

			if operationCount%5 == 0 { // Log every 5 operations
				t.Logf("  Current memory usage: %.2f MB", float64(currentMemoryUsage)/(1024*1024))
				t.Logf("  Active rooms: %d", len(roomIDs))
			}

			// Check for excessive memory growth
			maxMemoryGrowth := uint64(100 * 1024 * 1024) // 100MB max growth
			if currentMemoryUsage > maxMemoryGrowth {
				t.Errorf("Excessive memory growth detected: %.2f MB", float64(currentMemoryUsage)/(1024*1024))
			}

			if operationCount >= maxOperations {
				goto testComplete
			}

		case <-time.After(testDuration + 10*time.Second):
			t.Error("Test timeout exceeded")
			goto testComplete
		}
	}

testComplete:
	endTime := time.Now()
	actualDuration := endTime.Sub(startTime)

	// Final verification
	var finalMemStats runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&finalMemStats)

	finalRoomIDs := roomManager.GetGameRoomIDs()
	totalPlayers := 0
	for _, roomID := range finalRoomIDs {
		if count, exists := roomManager.GetNumberOfConnectedPlayers(roomID); exists {
			totalPlayers += count
		}
	}

	// Verify server stability
	expectedRooms := numRooms
	if len(finalRoomIDs) != expectedRooms {
		t.Errorf("Room count changed during stability test: expected %d, got %d", expectedRooms, len(finalRoomIDs))
	}

	expectedPlayers := numRooms * playersPerRoom
	if totalPlayers != expectedPlayers {
		t.Errorf("Player count changed during stability test: expected %d, got %d", expectedPlayers, totalPlayers)
	}

	// Final metrics
	var finalMemoryUsage uint64
	if finalMemStats.Alloc > initialMemStats.Alloc {
		finalMemoryUsage = finalMemStats.Alloc - initialMemStats.Alloc
	}
	t.Logf("Extended stability test completed:")
	t.Logf("  Actual duration: %v", actualDuration)
	t.Logf("  Operations completed: %d", operationCount)
	t.Logf("  Final memory usage: %.2f MB", float64(finalMemoryUsage)/(1024*1024))
	t.Logf("  Final room count: %d", len(finalRoomIDs))
	t.Logf("  Final player count: %d", totalPlayers)

	// Cleanup
	for _, room := range rooms {
		if room != nil {
			roomManager.RemoveGameRoom(room.ID)
		}
	}
}

// BenchmarkRoomCreation benchmarks room creation performance
func BenchmarkRoomCreation(b *testing.B) {
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		roomName := fmt.Sprintf("Benchmark Room %d", i)
		playerName := fmt.Sprintf("Benchmark Player %d", i)

		room, _, err := server.NewGameWithPlayer(roomName, playerName, game_maps.MapDefault)
		if err != nil {
			b.Fatalf("Failed to create room: %v", err)
		}

		// Clean up immediately to avoid memory issues
		_ = room
	}
}

// BenchmarkTrainingRoomCreation benchmarks training room creation performance
func BenchmarkTrainingRoomCreation(b *testing.B) {
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   10.0,
		HeadlessMode:      true,
		TrainingMode:      true,
		SessionID:         "benchmark-session",
		DirectStateAccess: true,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		roomName := fmt.Sprintf("Benchmark Training Room %d", i)
		playerName := fmt.Sprintf("Benchmark Player %d", i)

		room, _, err := server.NewTrainingGameWithPlayer(roomName, playerName, game_maps.MapDefault, trainingConfig)
		if err != nil {
			b.Fatalf("Failed to create training room: %v", err)
		}

		// Clean up immediately
		_ = room
	}
}

// BenchmarkDirectStateAccess benchmarks direct state access performance
func BenchmarkDirectStateAccess(b *testing.B) {
	trainingConfig := server.TrainingConfig{
		SpeedMultiplier:   50.0,
		HeadlessMode:      true,
		TrainingMode:      true,
		SessionID:         "benchmark-state-session",
		DirectStateAccess: true,
	}

	room, _, err := server.NewTrainingGameWithPlayer("Benchmark Room", "Benchmark Player", game_maps.MapDefault, trainingConfig)
	if err != nil {
		b.Fatalf("Failed to create training room: %v", err)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		gameState := room.GetDirectGameState()
		if gameState == nil {
			b.Fatal("Direct state access returned nil")
		}
	}
}
