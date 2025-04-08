package game_objects

import (
	"errors"
	"go-ws-server/pkg/server/constants"
	"math"
	"time"
)

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
	if exists {
		// Apply gravity
		dy = dy.(float64) + deltaTime*constants.AccelerationDueToGravity*constants.PxPerMeter
	}

	dy = math.Min(dy.(float64), constants.MaxVelocityMetersPerSec*constants.PxPerMeter)

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
	return nextX, nextY, float64(dx.(float64)), float64(dy.(float64)), nil
}
