package game_objects

import (
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/geo"
)

const (
	BlockSizeUnitMeters float64 = 1.0
	BlockSizeUnitPixels float64 = BlockSizeUnitMeters * constants.PxPerMeter
)

type BlockGameObject struct {
	*BaseGameObject
}

func NewBlockGameObject(id string, x float64, y float64, width float64, height float64) *BlockGameObject {
	base := NewBaseGameObject(id, constants.ObjectTypeBlock)
	block := &BlockGameObject{
		BaseGameObject: base,
	}
	block.SetState(constants.StateID, id)
	block.SetState(constants.StateX, x)
	block.SetState(constants.StateY, y)
	block.SetState(constants.StateWidth, width)
	block.SetState(constants.StateHeight, height)
	return block
}

func (b *BlockGameObject) GetBoundingShape() geo.Shape {
	x, _ := b.GetStateValue(constants.StateX)
	y, _ := b.GetStateValue(constants.StateY)
	w, _ := b.GetStateValue(constants.StateWidth)
	h, _ := b.GetStateValue(constants.StateHeight)
	xf := x.(float64)
	yf := y.(float64)
	wf := w.(float64)
	hf := h.(float64)
	points := []*geo.Point{
		geo.NewPoint(xf, yf),
		geo.NewPoint(xf+wf, yf),
		geo.NewPoint(xf+wf, yf+hf),
		geo.NewPoint(xf, yf+hf),
	}
	return geo.NewPolygon(points)
}

func (b *BlockGameObject) GetProperty(key GameObjectProperty) (interface{}, bool) {
	switch key {
	case GameObjectPropertyIsSolid:
		return true, true
	default:
		return nil, false
	}
}
