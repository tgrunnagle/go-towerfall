package geo

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"reflect"
)

type Point struct {
	X float64
	Y float64
}

func NewPoint(x float64, y float64) *Point {
	return &Point{
		X: x,
		Y: y,
	}
}

type Shape interface {
	GetCenter() *Point
	CollidesWith(other Shape) (bool, []*Point)
	DistanceTo(other Shape) float64
}

type Line struct {
	A *Point
	B *Point
}

func NewLine(a *Point, b *Point) *Line {
	return &Line{
		A: a,
		B: b,
	}
}

func (l *Line) GetCenter() *Point {
	return NewPoint((l.A.X + l.B.X) / 2, (l.A.Y + l.B.Y) / 2)
}

func (l *Line) CollidesWith(other Shape) (bool, []*Point) {
	if reflect.TypeOf(other) == reflect.TypeOf(&Line{}) {
		return checkLineLineCollision(l, other.(*Line))
	}
	if reflect.TypeOf(other) == reflect.TypeOf(&Circle{}) {
		return checkLineCircleCollision(l, other.(*Circle))
	}
	return false, nil
}

func (l *Line) DistanceTo(other Shape) float64 {
	center := l.GetCenter()
	otherCenter := other.GetCenter()
	return math.Sqrt(math.Pow(otherCenter.X-center.X, 2) + math.Pow(otherCenter.Y-center.Y, 2))
}

type Circle struct {
	C *Point
	R float64
}

func NewCircle(c *Point, r float64) *Circle {
	return &Circle{
		C: c,
		R: r,
	}
}

func (c *Circle) GetCenter() *Point {
	return NewPoint(c.C.X, c.C.Y)
}

func (c *Circle) CollidesWith(other Shape) (bool, []*Point) {
	// TODO maybe move to static function
	if reflect.TypeOf(other) == reflect.TypeOf(&Line{}) {
		return checkLineCircleCollision(other.(*Line), c)
	}
	if reflect.TypeOf(other) == reflect.TypeOf(&Circle{}) {
		return checkCircleCircleCollision(c, other.(*Circle))
	}
	return false, nil
}

func (c *Circle) DistanceTo(other Shape) float64 {
	// TODO make more accurate based on the other shapes, maybe move to static function
	center := c.GetCenter()
	otherCenter := other.GetCenter()
	return math.Sqrt(math.Pow(otherCenter.X-center.X, 2) + math.Pow(otherCenter.Y-center.Y, 2))
}

func checkLineLineCollision(l1 *Line, l2 *Line) (bool, []*Point) {
	// https://stackoverflow.com/a/563275
	a := mat.NewVecDense(2, []float64{l1.A.X, l1.A.Y})
	b := mat.NewVecDense(2, []float64{l1.B.X, l1.B.Y})
	c := mat.NewVecDense(2, []float64{l2.A.X, l2.A.Y})
	d := mat.NewVecDense(2, []float64{l2.B.X, l2.B.Y})

	var e mat.VecDense
	e.SubVec(b, a)
	var f mat.VecDense
	f.SubVec(d, c)
	p := mat.NewVecDense(2, []float64{-e.AtVec(1), e.AtVec(0)})

	fDotP := mat.Dot(&f, p)
	// If lines are parallel, fDotP will be 0
	if math.Abs(fDotP) < 1e-10 {
		return false, nil
	}

	var aMinusC mat.VecDense
	aMinusC.SubVec(a, c)
	aMinusCDotP := mat.Dot(&aMinusC, p)
	h := aMinusCDotP / fDotP

	if h >= 0 && h <= 1 {
		// lines intersect at c + h * f
		hTimesF := mat.NewVecDense(2, []float64{h * f.AtVec(0), h * f.AtVec(1)})
		var intersect mat.VecDense
		intersect.AddVec(c, hTimesF)

		return true, []*Point{{X: intersect.AtVec(0), Y: intersect.AtVec(1)}}
	}

	return false, nil
}

func checkCircleCircleCollision(c1 *Circle, c2 *Circle) (bool, []*Point) {
	x1 := c1.C.X
	y1 := c1.C.Y
	r1 := c1.R
	x2 := c2.C.X
	y2 := c2.C.Y
	r2 := c2.R

	// https://math.stackexchange.com/a/1033561
	d := math.Sqrt(math.Pow(x1-x2, 2) + math.Pow(y1-y2, 2))
	if d > r1+r2 {
		return false, nil
	}
	l := (math.Pow(r1, 2) - math.Pow(r2, 2) + math.Pow(d, 2)) / (2 * d)
	h := math.Sqrt(math.Pow(r1, 2) - math.Pow(l, 2))

	xIntersect1 := l/d*(x2-x1) + h/d*(y2-y1) + x1
	yIntersect1 := l/d*(y2-y1) - h/d*(x2-x1) + y1

	xIntersect2 := l/d*(x2-x1) - h/d*(y2-y1) + x1
	yIntersect2 := l/d*(y2-y1) + h/d*(x2-x1) + y1

	return true, []*Point{{X: xIntersect1, Y: yIntersect1}, {X: xIntersect2, Y: yIntersect2}}
}

func checkLineCircleCollision(l *Line, c *Circle) (bool, []*Point) {
	aX := l.A.X
	aY := l.A.Y
	bX := l.B.X
	bY := l.B.Y
	cX := c.C.X
	cY := c.C.Y
	cR := c.R

	// https://stackoverflow.com/a/1088058
	// compute the euclidean distance between A and B
	lab := math.Sqrt(math.Pow(bX-aX, 2) + math.Pow(bY-aY, 2))

	// compute the direction vector D from A to B
	dX := (bX - aX) / lab
	dY := (bY - aY) / lab

	// the equation of the line AB is x = Dx*t + Ax, y = Dy*t + Ay with 0 <= t <= LAB.

	// compute the distance between the points A and E, where
	// E is the point of AB closest the circle center (Cx, Cy)
	t := dX*(cX-aX) + dY*(cY-aY)

	// compute the coordinates of the point E
	eX := t*dX + aX
	eY := t*dY + aY

	// compute the euclidean distance between E and C
	lec := math.Sqrt(math.Pow(eX-cX, 2) + math.Pow(eY-cY, 2))

	// test if the line intersects the circle
	if lec < cR {
		// compute distance from t to circle intersection point
		dt := math.Sqrt(math.Pow(cR, 2) - math.Pow(lec, 2))
		// compute first intersection point
		fX := (t-dt)*dX + aX
		fY := (t-dt)*dY + aY
		// compute second intersection point
		gX := (t+dt)*dX + aX
		gY := (t+dt)*dY + aY
		return true, []*Point{{X: fX, Y: fY}, {X: gX, Y: gY}}
	} else if lec == cR {
		// tangent point to circle is E
		return true, []*Point{{X: eX, Y: eY}}
	}
	// line doesn't touch circle
	return false, nil
}
