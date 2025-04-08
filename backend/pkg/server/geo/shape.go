package geo

import (
	"math"
	"sort"

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

func (l *Line) CollidesWith(other Shape) (bool, []*Point) {
	switch other.(type) {
	case *Line:
		return checkLineLineCollision(l, other.(*Line))
	case *Circle:
		return checkLineCircleCollision(l, other.(*Circle))
	case *Polygon:
		return checkLinePolygonCollision(l, other.(*Polygon))
	}
	return false, nil
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

func (c *Circle) CollidesWith(other Shape) (bool, []*Point) {
	switch other.(type) {
	case *Line:
		return checkLineCircleCollision(other.(*Line), c)
	case *Circle:
		return checkCircleCircleCollision(c, other.(*Circle))
	case *Polygon:
		return checkCirclePolygonCollision(c, other.(*Polygon))
	}
	return false, nil
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

func (p *Polygon) CollidesWith(other Shape) (bool, []*Point) {
	switch other.(type) {
	case *Line:
		return checkLinePolygonCollision(other.(*Line), p)
	case *Circle:
		return checkCirclePolygonCollision(other.(*Circle), p)
	case *Polygon:
		return checkPolygonPolygonCollision(p, other.(*Polygon))
	}
	return false, nil
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

	// Check each line from p1 against each line from p2
	for _, p1Line := range p1Lines {
		for _, p2Line := range p2Lines {
			linesCollide, lineCollisionPoints := checkLineLineCollision(p1Line, p2Line)
			if linesCollide {
				collides = true
				// Only add points that are actually on both line segments
				for _, point := range lineCollisionPoints {
					if pointOnLineSegment(point, p1Line) && pointOnLineSegment(point, p2Line) {
						collisionPoints = append(collisionPoints, point)
					}
				}
			}
		}
	}
	return collides, deduplicatePoints(collisionPoints)
}

func checkLineCircleCollision(line *Line, circle *Circle) (bool, []*Point) {
	// Convert line segment to parametric form: P = A + t(B-A), where t in [0,1]
	// We want to ensure that the t values are in [0,1], so we need to orient
	// the line segment correctly based on its relationship to the circle center
	var a, b *Point
	if math.Abs(line.A.X-line.B.X) > math.Abs(line.A.Y-line.B.Y) {
		// Line is more horizontal
		if line.A.X <= line.B.X {
			a, b = line.A, line.B
		} else {
			a, b = line.B, line.A
		}
	} else {
		// Line is more vertical
		if line.A.Y <= line.B.Y {
			a, b = line.A, line.B
		} else {
			a, b = line.B, line.A
		}
	}

	dx := b.X - a.X
	dy := b.Y - a.Y

	// Compute coefficients of quadratic equation
	// (x - cx)^2 + (y - cy)^2 = r^2
	// where x = ax + t*dx and y = ay + t*dy
	aa := dx*dx + dy*dy
	bb := 2 * (dx*(a.X-circle.C.X) + dy*(a.Y-circle.C.Y))
	cc := math.Pow(a.X-circle.C.X, 2) + math.Pow(a.Y-circle.C.Y, 2) - circle.R*circle.R

	// Handle special case where line is a point
	if math.Abs(aa) < 1e-10 {
		// Check if the point is on the circle
		dist := math.Sqrt(math.Pow(a.X-circle.C.X, 2) + math.Pow(a.Y-circle.C.Y, 2))
		if math.Abs(dist-circle.R) < 1e-10 {
			return true, []*Point{{X: a.X, Y: a.Y}}
		}
		return false, []*Point{}
	}

	// Solve quadratic equation
	discriminant := bb*bb - 4*aa*cc

	if discriminant < -1e-10 { // No intersection
		return false, []*Point{}
	}

	// Handle tangent case (discriminant ≈ 0)
	if math.Abs(discriminant) < 1e-10 {
		t := -bb / (2 * aa)
		// Check if the point lies on the line segment
		if t >= -1e-10 && t <= 1+1e-10 {
			x := a.X + t*dx
			y := a.Y + t*dy
			return true, []*Point{{X: x, Y: y}}
		}
		return false, []*Point{}
	}

	// Handle intersection case (discriminant > 0)
	sqrtD := math.Sqrt(discriminant)
	t1 := (-bb - sqrtD) / (2 * aa)
	t2 := (-bb + sqrtD) / (2 * aa)

	points := []*Point{}
	// Check first intersection point
	if t1 >= -1e-10 && t1 <= 1+1e-10 { // Use epsilon to handle floating point errors
		x := a.X + t1*dx
		y := a.Y + t1*dy
		points = append(points, &Point{X: x, Y: y})
	}
	// Check second intersection point
	if t2 >= -1e-10 && t2 <= 1+1e-10 { // Use epsilon to handle floating point errors
		x := a.X + t2*dx
		y := a.Y + t2*dy
		points = append(points, &Point{X: x, Y: y})
	}

	// Handle case where line is tangent to circle at an endpoint
	if len(points) == 0 {
		// Check if either endpoint is on the circle
		dist1 := math.Sqrt(math.Pow(a.X-circle.C.X, 2) + math.Pow(a.Y-circle.C.Y, 2))
		if math.Abs(dist1-circle.R) < 1e-10 {
			points = append(points, &Point{X: a.X, Y: a.Y})
		}
		dist2 := math.Sqrt(math.Pow(b.X-circle.C.X, 2) + math.Pow(b.Y-circle.C.Y, 2))
		if math.Abs(dist2-circle.R) < 1e-10 {
			points = append(points, &Point{X: b.X, Y: b.Y})
		}
	}

	return len(points) > 0, points
}

func checkLinePolygonCollision(line *Line, polygon *Polygon) (bool, []*Point) {
	collides := false
	collisionPoints := []*Point{}

	for _, polygonLine := range polygon.GetLines() {
		lineCollides, points := checkLineLineCollision(line, polygonLine)
		if lineCollides {
			collides = true
			collisionPoints = append(collisionPoints, points...)
		}
	}
	return collides, deduplicatePoints(collisionPoints)
}

func checkCirclePolygonCollision(circle *Circle, polygon *Polygon) (bool, []*Point) {
	lines := polygon.GetLines()
	collides := false
	collisionPoints := []*Point{}

	// Check each edge of the polygon for intersection with the circle
	for _, line := range lines {
		linesCollide, lineCollisionPoints := checkLineCircleCollision(line, circle)
		if linesCollide {
			collides = true
			// For each edge, keep all intersection points
			collisionPoints = append(collisionPoints, lineCollisionPoints...)
		}
	}

	// Check if the circle's center is inside the polygon
	// This is needed for cases where the circle is entirely inside the polygon
	if pointInPolygon(circle.C, polygon) {
		collides = true
	}

	// Check if any polygon vertex is inside the circle
	for _, vertex := range polygon.Points {
		dist := math.Sqrt(math.Pow(vertex.X-circle.C.X, 2) + math.Pow(vertex.Y-circle.C.Y, 2))
		if dist <= circle.R+1e-10 {
			collisionPoints = append(collisionPoints, vertex)
		}
	}

	return collides, deduplicatePoints(collisionPoints)
}

// pointInPolygon checks if a point is inside a polygon using the ray casting algorithm
func pointInPolygon(point *Point, polygon *Polygon) bool {
	inside := false
	j := len(polygon.Points) - 1
	for i := 0; i < len(polygon.Points); i++ {
		if ((polygon.Points[i].Y > point.Y) != (polygon.Points[j].Y > point.Y)) &&
			(point.X < (polygon.Points[j].X-polygon.Points[i].X)*(point.Y-polygon.Points[i].Y)/(polygon.Points[j].Y-polygon.Points[i].Y)+polygon.Points[i].X) {
			inside = !inside
		}
		j = i
	}
	return inside
}

// Helper function to deduplicate points with floating-point tolerance
func deduplicatePoints(points []*Point) []*Point {
	if len(points) == 0 {
		return points
	}

	// Sort points by X coordinate, then Y coordinate
	sorted := make([]*Point, len(points))
	copy(sorted, points)
	sort.Slice(sorted, func(i, j int) bool {
		if math.Abs(sorted[i].X-sorted[j].X) < 1e-10 {
			return sorted[i].Y < sorted[j].Y
		}
		return sorted[i].X < sorted[j].X
	})

	// Keep only unique points within epsilon
	const epsilon = 0.1 // Use a larger epsilon for circle-polygon collisions
	result := []*Point{sorted[0]}
	for i := 1; i < len(sorted); i++ {
		curr := sorted[i]
		// Check if this point is approximately equal to any previous point
		isDuplicate := false
		for j := 0; j < len(result); j++ {
			prev := result[j]
			// Two points are considered duplicates if:
			// 1. They are very close to each other
			// 2. They lie on the same horizontal or vertical line and are close in one coordinate
			dx := math.Abs(curr.X - prev.X)
			dy := math.Abs(curr.Y - prev.Y)
			if dx*dx+dy*dy < epsilon*epsilon || // Points are very close to each other
				(dx < epsilon && math.Abs(curr.Y-prev.Y) < 2*epsilon) || // Points are on same vertical line
				(dy < epsilon && math.Abs(curr.X-prev.X) < 2*epsilon) { // Points are on same horizontal line
				isDuplicate = true
				break
			}
		}
		if !isDuplicate {
			result = append(result, curr)
		}
	}
	return result
}

// Helper function to check if a point is close to a line segment
func pointOnLineSegment(point *Point, line *Line) bool {
	const epsilon = 1e-10
	// Calculate the distance from the point to both endpoints of the line
	d1 := math.Sqrt(math.Pow(point.X-line.A.X, 2) + math.Pow(point.Y-line.A.Y, 2))
	d2 := math.Sqrt(math.Pow(point.X-line.B.X, 2) + math.Pow(point.Y-line.B.Y, 2))
	lineLength := math.Sqrt(math.Pow(line.B.X-line.A.X, 2) + math.Pow(line.B.Y-line.A.Y, 2))

	// If the sum of distances from the point to both endpoints is approximately equal to the line length,
	// then the point lies on the line segment
	return math.Abs(d1+d2-lineLength) < epsilon
}

// approximatePointsEqual checks if two points are approximately equal within a small epsilon
func approximatePointsEqual(p1 *Point, p2 *Point) bool {
	if p1 == nil && p2 == nil {
		return true
	}
	if p1 == nil || p2 == nil {
		return false
	}
	const epsilon = 1e-10
	dx := math.Abs(p1.X - p2.X)
	dy := math.Abs(p1.Y - p2.Y)
	return dx < epsilon && dy < epsilon
}

type BaseShape struct {
}

func (s *BaseShape) GetCenter() *Point {
	return nil
}

func (s *BaseShape) CollidesWith(other Shape) (bool, []*Point) {
	switch other.(type) {
	case *Line:
		return false, nil
	case *Circle:
		return false, nil
	case *Polygon:
		return false, nil
	}
	return false, nil
}

func (s *BaseShape) DistanceTo(other Shape) float64 {
	center := s.GetCenter()
	otherCenter := other.GetCenter()
	return math.Sqrt(math.Pow(otherCenter.X-center.X, 2) + math.Pow(otherCenter.Y-center.Y, 2))
}
