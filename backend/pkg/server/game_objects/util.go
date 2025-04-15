package game_objects

import (
	"errors"
	"go-ws-server/pkg/server/constants"
	"math"
	"math/rand"
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
	if exists {
		// Apply gravity
		dy = dy.(float64) + deltaTime*constants.AccelerationDueToGravityMetersPerSec2*constants.PxPerMeter
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

func WrapPosition(x float64, y float64) (float64, float64) {
	if x < -1.0*constants.RoomWrapDistancePx {
		x = constants.RoomSizePixelsX + constants.RoomWrapDistancePx
	} else if x > constants.RoomSizePixelsX+constants.RoomWrapDistancePx {
		x = -1.0 * constants.RoomWrapDistancePx
	}

	if y < -1.0*constants.RoomWrapDistancePx {
		y = constants.RoomSizePixelsY + constants.RoomWrapDistancePx
	} else if y > constants.RoomSizePixelsY+constants.RoomWrapDistancePx {
		y = -1.0 * constants.RoomWrapDistancePx
	}
	return x, y
}

// TODO: move this to map maker
func GetRespawnLocation() (float64, float64) {
	index := rand.Intn(len(constants.RespawnLocationsPx))
	return constants.RespawnLocationsPx[index].X, constants.RespawnLocationsPx[index].Y
}
