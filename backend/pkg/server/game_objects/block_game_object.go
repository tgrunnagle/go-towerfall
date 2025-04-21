package game_objects

import (
	"go-ws-server/pkg/server/constants"
	"go-ws-server/pkg/server/geo"
)

type BlockGameObject struct {
	*BaseGameObject
}

func NewBlockGameObject(id string, points []*geo.Point) *BlockGameObject {
	base := NewBaseGameObject(id, constants.ObjectTypeBlock)
	block := &BlockGameObject{
		BaseGameObject: base,
	}
	block.SetState(constants.StateID, id)
	block.SetState(constants.StatePoints, points)
	return block
}

func (b *BlockGameObject) GetBoundingShape() geo.Shape {
	points, _ := b.GetStateValue(constants.StatePoints)
	return geo.NewPolygon(points.([]*geo.Point))
}

func (b *BlockGameObject) GetProperty(key GameObjectProperty) (interface{}, bool) {
	switch key {
	case GameObjectPropertyIsSolid:
		return true, true
	default:
		return nil, false
	}
}
