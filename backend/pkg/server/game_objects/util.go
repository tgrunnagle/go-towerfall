package game_objects

import (
	"errors"
	"go-ws-server/pkg/server/constants"
	"math"
	"time"
)

func GetExtrapolatedPositionForDxDy(p GameObject, dX float64, dY float64) (float64, float64, error) {
	lastUpdateTIme, exists := p.GetStateValue(constants.StateLastLocUpdateTime)
	if !exists {
		return 0.0, 0.0, errors.New("missing lastLocUpdateTime state")
	}
	deltaTime := time.Since(lastUpdateTIme.(time.Time)).Seconds()

	x, exists := p.GetStateValue(constants.StateX)
	if !exists {
		return 0.0, 0.0, errors.New("missing x state")
	}
	y, exists := p.GetStateValue(constants.StateY)
	if !exists {
		return 0.0, 0.0, errors.New("missing y state")
	}

	nextX := x.(float64) + dX*deltaTime
	nextY := y.(float64) + dY*deltaTime
	return nextX, nextY, nil
}

func GetExtrapolatedPosition(p GameObject) (float64, float64, float64, float64, error) {
	lastUpdateTIme, exists := p.GetStateValue(constants.StateLastLocUpdateTime)
	if !exists {
		return 0.0, 0.0, 0.0, 0.0, errors.New("missing lastLocUpdateTime state")
	}
	deltaTime := time.Since(lastUpdateTIme.(time.Time)).Seconds()

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
	return nextX, nextY, nextDx, nextDy, nil
}
