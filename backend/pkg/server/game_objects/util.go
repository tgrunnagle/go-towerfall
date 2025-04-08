package game_objects

import (
	"errors"
	"go-ws-server/pkg/server/constants"
	"time"
)

func GetExtrapolatedPosition(p GameObject) (float64, float64, error) {
	dx, exists := p.GetStateValue(constants.StateDx)
	if !exists {
		return 0.0, 0.0, errors.New("missing dx state")
	}
	dy, exists := p.GetStateValue(constants.StateDy)
	if !exists {
		return 0.0, 0.0, errors.New("missing dy state")
	}
	x, exists := p.GetStateValue(constants.StateX)
	if !exists {
		return 0.0, 0.0, errors.New("missing x state")
	}
	y, exists := p.GetStateValue(constants.StateY)
	if !exists {
		return 0.0, 0.0, errors.New("missing y state")
	}
	lastUpdateTIme, exists := p.GetStateValue(constants.StateLastLocUpdateTime)
	if !exists {
		return 0.0, 0.0, errors.New("missing lastLocUpdateTime state")
	}
	
	deltaTime := time.Since(lastUpdateTIme.(time.Time)).Seconds()
	nextX := x.(float64) + float64(dx.(float64))*deltaTime
	nextY := y.(float64) + float64(dy.(float64))*deltaTime
	return nextX, nextY, nil
}
