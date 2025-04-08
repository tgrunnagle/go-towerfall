package geo

import (
	"math"
	"reflect"

	"gonum.org/v1/gonum/mat"
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
	*BaseShape
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
	return NewPoint((l.A.X+l.B.X)/2, (l.A.Y+l.B.Y)/2)
}

type Circle struct {
	*BaseShape
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

type Polygon struct {
	*BaseShape
	Points []*Point
}

func NewPolygon(points []*Point) *Polygon {
	return &Polygon{
		Points: points,
	}
}

func (p *Polygon) GetCenter() *Point {
	var sumX, sumY float64
	for _, point := range p.Points {
		sumX += point.X
		sumY += point.Y
	}
	return NewPoint(sumX/float64(len(p.Points)), sumY/float64(len(p.Points)))
}

func (p *Polygon) GetLines() []*Line {
	lines := []*Line{}
	for i, point := range p.Points {
		lines = append(lines, NewLine(point, p.Points[(i+1)%len(p.Points)]))
	}
	return lines
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

func checkPolygonPolygonCollision(p1 *Polygon, p2 *Polygon) (bool, []*Point) {
	p1Lines := p1.GetLines()
	p2Lines := p2.GetLines()
	collides := false
	collisionPoints := []*Point{}

	for _, p1Line := range p1Lines {
		for _, p2Line := range p2Lines {
			linesCollide, lineCollisionPoints := checkLineLineCollision(p1Line, p2Line)
			if linesCollide {
				collides = true
				collisionPoints = append(collisionPoints, lineCollisionPoints...)
			}
		}
	}
	return collides, collisionPoints

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

func checkLinePolygonCollision(l *Line, p *Polygon) (bool, []*Point) {
	pLines := p.GetLines()
	return checkMultipleLinesShapeCollision(pLines, l)
}

func checkCirclePolygonCollision(c *Circle, p *Polygon) (bool, []*Point) {
	pLines := p.GetLines()
	return checkMultipleLinesShapeCollision(pLines, c)
}

func checkMultipleLinesShapeCollision(lines []*Line, other Shape) (bool, []*Point) {
	collides := false
	collisionPoints := []*Point{}
	for _, line := range lines {
		collides, points := line.CollidesWith(other)
		if collides {
			collides = true
			collisionPoints = append(collisionPoints, points...)
		}
	}
	return collides, collisionPoints
}

type BaseShape struct {
}

func (s *BaseShape) GetCenter() *Point {
	return nil
}

func (s *BaseShape) CollidesWith(other Shape) (bool, []*Point) {
	lines := []Shape{}
	circles := []Shape{}
	polygons := []Shape{}

	switch reflect.TypeOf(s) {
	case reflect.TypeOf(&Line{}):
		lines = append(lines, s)
	case reflect.TypeOf(&Circle{}):
		circles = append(circles, s)
	case reflect.TypeOf(&Polygon{}):
		polygons = append(polygons, s)
	}
	switch reflect.TypeOf(other) {
	case reflect.TypeOf(&Line{}):
		lines = append(lines, other)
	case reflect.TypeOf(&Circle{}):
		circles = append(circles, other)
	case reflect.TypeOf(&Polygon{}):
		polygons = append(polygons, other)
	}

	if len(lines) == 2 {
		return checkLineLineCollision(lines[0].(*Line), lines[1].(*Line))
	}
	if len(circles) == 2 {
		return checkCircleCircleCollision(circles[0].(*Circle), circles[1].(*Circle))
	}
	if len(polygons) == 2 {
		return checkPolygonPolygonCollision(polygons[0].(*Polygon), polygons[1].(*Polygon))
	}
	if len(lines) == 1 && len(circles) == 1 {
		return checkLineCircleCollision(lines[0].(*Line), circles[0].(*Circle))
	}
	if len(lines) == 1 && len(polygons) == 1 {
		return checkLinePolygonCollision(lines[0].(*Line), polygons[0].(*Polygon))
	}
	if len(circles) > 0 && len(polygons) > 0 {
		return checkCirclePolygonCollision(circles[0].(*Circle), polygons[0].(*Polygon))
	}
	return false, nil
}

func (s *BaseShape) DistanceTo(other Shape) float64 {
	center := s.GetCenter()
	otherCenter := other.GetCenter()
	return math.Sqrt(math.Pow(otherCenter.X-center.X, 2) + math.Pow(otherCenter.Y-center.Y, 2))
}
