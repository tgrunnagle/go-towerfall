package geo

import (
	"math"
	"testing"
)

func TestLineLineCollision(t *testing.T) {
	tests := []struct {
		name     string
		line1    Line
		line2    Line
		want     bool
		wantPoint *Point
	}{
		{
			name:  "Intersecting lines",
			line1: Line{A: &Point{X: 0, Y: 0}, B: &Point{X: 2, Y: 2}},
			line2: Line{A: &Point{X: 0, Y: 2}, B: &Point{X: 2, Y: 0}},
			want:  true,
			wantPoint: &Point{X: 1, Y: 1},
		},
		{
			name:  "Parallel lines",
			line1: Line{A: &Point{X: 0, Y: 0}, B: &Point{X: 2, Y: 2}},
			line2: Line{A: &Point{X: 0, Y: 1}, B: &Point{X: 2, Y: 3}},
			want:  false,
			wantPoint: nil,
		},
		{
			name:  "Non-intersecting lines",
			line1: Line{A: &Point{X: 0, Y: 0}, B: &Point{X: 1, Y: 1}},
			line2: Line{A: &Point{X: 3, Y: 3}, B: &Point{X: 4, Y: 4}},
			want:  false,
			wantPoint: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, points := checkLineLineCollision(&tt.line1, &tt.line2)
			if got != tt.want {
				t.Errorf("Line.CollidesWith() collision = %v, want %v", got, tt.want)
			}
			if tt.want && (len(points) != 1 || !approximatelyEqual(points[0], tt.wantPoint)) {
				t.Errorf("Line.CollidesWith() point = %v, want %v", points[0], tt.wantPoint)
			}
		})
	}
}

func TestCircleCircleCollision(t *testing.T) {
	tests := []struct {
		name      string
		circle1   Circle
		circle2   Circle
		want      bool
		numPoints int
	}{
		{
			name:      "Overlapping circles",
			circle1:   Circle{C: &Point{X: 0, Y: 0}, R: 2},
			circle2:   Circle{C: &Point{X: 2, Y: 0}, R: 2},
			want:      true,
			numPoints: 2,
		},
		{
			name:      "Non-overlapping circles",
			circle1:   Circle{C: &Point{X: 0, Y: 0}, R: 1},
			circle2:   Circle{C: &Point{X: 4, Y: 0}, R: 1},
			want:      false,
			numPoints: 0,
		},
		{
			name:      "Tangent circles",
			circle1:   Circle{C: &Point{X: 0, Y: 0}, R: 1},
			circle2:   Circle{C: &Point{X: 2, Y: 0}, R: 1},
			want:      true,
			numPoints: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, points := checkCircleCircleCollision(&tt.circle1, &tt.circle2)
			if got != tt.want {
				t.Errorf("Circle.CollidesWith() collision = %v, want %v", got, tt.want)
			}
			if len(points) != tt.numPoints {
				t.Errorf("Circle.CollidesWith() got %v intersection points, want %v", len(points), tt.numPoints)
			}
		})
	}
}

func TestCircleLineCollision(t *testing.T) {
	tests := []struct {
		name      string
		circle    Circle
		line      Line
		want      bool
		numPoints int
	}{
		{
			name:      "Line intersects circle at two points",
			circle:    Circle{C: &Point{X: 0, Y: 0}, R: 1},
			line:      Line{A: &Point{X: -2, Y: 0}, B: &Point{X: 2, Y: 0}},
			want:      true,
			numPoints: 2,
		},
		{
			name:      "Line is tangent to circle",
			circle:    Circle{C: &Point{X: 0, Y: 0}, R: 1},
			line:      Line{A: &Point{X: -2, Y: 1}, B: &Point{X: 2, Y: 1}},
			want:      true,
			numPoints: 1,
		},
		{
			name:      "Line does not intersect circle",
			circle:    Circle{C: &Point{X: 0, Y: 0}, R: 1},
			line:      Line{A: &Point{X: -2, Y: 2}, B: &Point{X: 2, Y: 2}},
			want:      false,
			numPoints: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, points := checkLineCircleCollision(&tt.line, &tt.circle)
			if got != tt.want {
				t.Errorf("Circle.CollidesWith() collision = %v, want %v", got, tt.want)
			}
			if len(points) != tt.numPoints {
				t.Errorf("Circle.CollidesWith() got %v intersection points, want %v", len(points), tt.numPoints)
			}
		})
	}
}

// Helper function to compare points with floating-point values
func approximatelyEqual(p1, p2 *Point) bool {
	if p1 == nil && p2 == nil {
		return true
	}
	if p1 == nil || p2 == nil {
		return false
	}
	const epsilon = 1e-10
	return math.Abs(p1.X-p2.X) < epsilon && math.Abs(p1.Y-p2.Y) < epsilon
}
