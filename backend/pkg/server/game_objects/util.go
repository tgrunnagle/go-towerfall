package game_objects

import (
	"errors"
	"go-ws-server/pkg/server/constants"
	"log"
	"math"
	"time"
)

func GetExtrapolatedPosition(p GameObject) (float64, float64, float64, float64, error) {
	lastUpdateTIme, exists := p.GetStateValue(constants.StateLastLocUpdateTime)
	if !exists {
		return 0.0, 0.0, 0.0, 0.0, errors.New("missing lastLocUpdateTime state")
	}
	deltaTime := time.Since(lastUpdateTIme.(time.Time)).Seconds()

	// Cap deltaTime to prevent extreme calculations
	if deltaTime > 1.0 { // Cap at 1 second
		log.Printf("GetExtrapolatedPosition: Capping excessive deltaTime from %v to 1.0 seconds", deltaTime)
		deltaTime = 1.0
	}
	if deltaTime < 0 {
		log.Printf("GetExtrapolatedPosition: Negative deltaTime %v detected, resetting to 0", deltaTime)
		deltaTime = 0
	}

	dx, exists := p.GetStateValue(constants.StateDx)
	if !exists {
		return 0.0, 0.0, 0.0, 0.0, errors.New("missing dx state")
	}
	dy, exists := p.GetStateValue(constants.StateDy)
	if !exists {
		return 0.0, 0.0, 0.0, 0.0, errors.New("missing dy state")
	}

	_, exists = p.GetProperty(GameObjectPropertyMassKg)
	nextDy := dy.(float64)
	if exists {
		// Apply gravity
		nextDy += deltaTime * constants.AccelerationDueToGravityMetersPerSec2 * constants.PxPerMeter
	}

	// Cap velocity
	// TODO cap velocity vector magnitude, rather than each direction
	if nextDy > 0 {
		nextDy = math.Min(nextDy, constants.MaxVelocityMetersPerSec*constants.PxPerMeter)
	} else if nextDy < 0 {
		nextDy = math.Max(nextDy, -constants.MaxVelocityMetersPerSec*constants.PxPerMeter)
	}
	nextDx := dx.(float64)
	if nextDx > 0 {
		nextDx = math.Min(nextDx, constants.MaxVelocityMetersPerSec*constants.PxPerMeter)
	} else if nextDx < 0 {
		nextDx = math.Max(nextDx, -constants.MaxVelocityMetersPerSec*constants.PxPerMeter)
	}

	x, exists := p.GetStateValue(constants.StateX)
	if !exists {
		return 0.0, 0.0, 0.0, 0.0, errors.New("missing x state")
	}
	y, exists := p.GetStateValue(constants.StateY)
	if !exists {
		return 0.0, 0.0, 0.0, 0.0, errors.New("missing y state")
	}

	nextX := x.(float64) + float64(dx.(float64))*deltaTime
	nextY := y.(float64) + float64(dy.(float64))*deltaTime

	// Check for NaN or infinite values and return current position if invalid
	if math.IsNaN(nextX) || math.IsInf(nextX, 0) {
		log.Printf("GetExtrapolatedPosition: Invalid nextX=%v for object, using current x=%v (deltaTime=%v, dx=%v)",
			nextX, x.(float64), deltaTime, dx.(float64))
		nextX = x.(float64)
	}
	if math.IsNaN(nextY) || math.IsInf(nextY, 0) {
		log.Printf("GetExtrapolatedPosition: Invalid nextY=%v for object, using current y=%v (deltaTime=%v, dy=%v)",
			nextY, y.(float64), deltaTime, dy.(float64))
		nextY = y.(float64)
	}
	if math.IsNaN(nextDx) || math.IsInf(nextDx, 0) {
		log.Printf("GetExtrapolatedPosition: Invalid nextDx=%v for object, resetting to 0 (original dx=%v)",
			nextDx, dx.(float64))
		nextDx = 0
	}
	if math.IsNaN(nextDy) || math.IsInf(nextDy, 0) {
		log.Printf("GetExtrapolatedPosition: Invalid nextDy=%v for object, resetting to 0 (original dy=%v)",
			nextDy, dy.(float64))
		nextDy = 0
	}

	return nextX, nextY, nextDx, nextDy, nil
}
