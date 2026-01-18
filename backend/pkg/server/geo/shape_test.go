package geo

import (
	"testing"
)

func TestLineLineCollision(t *testing.T) {
	tests := []struct {
		name      string
		line1     Line
		line2     Line
		want      bool
		wantPoint *Point
	}{
		{
			name:      "Intersecting lines",
			line1:     Line{A: &Point{X: 0, Y: 0}, B: &Point{X: 2, Y: 2}},
			line2:     Line{A: &Point{X: 0, Y: 2}, B: &Point{X: 2, Y: 0}},
			want:      true,
			wantPoint: &Point{X: 1, Y: 1},
		},
		{
			name:      "Parallel lines",
			line1:     Line{A: &Point{X: 0, Y: 0}, B: &Point{X: 2, Y: 2}},
			line2:     Line{A: &Point{X: 0, Y: 1}, B: &Point{X: 2, Y: 3}},
			want:      false,
			wantPoint: nil,
		},
		{
			name:      "Non-intersecting lines",
			line1:     Line{A: &Point{X: 0, Y: 0}, B: &Point{X: 1, Y: 1}},
			line2:     Line{A: &Point{X: 3, Y: 3}, B: &Point{X: 4, Y: 4}},
			want:      false,
			wantPoint: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, points := checkLineLineCollision(&tt.line1, &tt.line2)
			if got != tt.want {
				t.Errorf("Line.CollidesWith() collision = %v, want %v", got, tt.want)
			}
			if tt.want && (len(points) != 1 || !approximatePointsEqual(points[0], tt.wantPoint)) {
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

func TestPolygonPolygonCollision(t *testing.T) {
	tests := []struct {
		name      string
		polygon1  Polygon
		polygon2  Polygon
		want      bool
		numPoints int
	}{
		{
			name: "Overlapping squares",
			polygon1: Polygon{
				Points: []*Point{
					{X: 0, Y: 0},
					{X: 2, Y: 0},
					{X: 2, Y: 2},
					{X: 0, Y: 2},
				},
			},
			polygon2: Polygon{
				Points: []*Point{
					{X: 1, Y: 1},
					{X: 3, Y: 1},
					{X: 3, Y: 3},
					{X: 1, Y: 3},
				},
			},
			want:      true,
			numPoints: 2,
		},
		{
			name: "Non-overlapping squares",
			polygon1: Polygon{
				Points: []*Point{
					{X: 0, Y: 0},
					{X: 1, Y: 0},
					{X: 1, Y: 1},
					{X: 0, Y: 1},
				},
			},
			polygon2: Polygon{
				Points: []*Point{
					{X: 3, Y: 3},
					{X: 4, Y: 3},
					{X: 4, Y: 4},
					{X: 3, Y: 4},
				},
			},
			want:      false,
			numPoints: 0,
		},
		{
			name: "Touching squares at vertex",
			polygon1: Polygon{
				Points: []*Point{
					{X: 0, Y: 0},
					{X: 1, Y: 0},
					{X: 1, Y: 1},
					{X: 0, Y: 1},
				},
			},
			polygon2: Polygon{
				Points: []*Point{
					{X: 1, Y: 1},
					{X: 2, Y: 1},
					{X: 2, Y: 2},
					{X: 1, Y: 2},
				},
			},
			want:      true,
			numPoints: 1,
		},
		{
			name: "Squares with overlapping edge",
			polygon1: Polygon{
				Points: []*Point{
					{X: 0, Y: 0},
					{X: 2, Y: 0},
					{X: 2, Y: 2},
					{X: 0, Y: 2},
				},
			},
			polygon2: Polygon{
				Points: []*Point{
					{X: 2, Y: 1},
					{X: 4, Y: 1},
					{X: 4, Y: 3},
					{X: 2, Y: 3},
				},
			},
			want:      true,
			numPoints: 2, // Two points: one at each end of the overlapping edge (2,1) and (2,2)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, points := checkPolygonPolygonCollision(&tt.polygon1, &tt.polygon2)
			if got != tt.want {
				t.Errorf("Polygon.CollidesWith() collision = %v, want %v", got, tt.want)
			}
			if len(points) != tt.numPoints {
				t.Errorf("Polygon.CollidesWith() got %v intersection points, want %v", len(points), tt.numPoints)
			}
		})
	}
}

func TestPolygonLineCollision(t *testing.T) {
	tests := []struct {
		name      string
		polygon   Polygon
		line      Line
		want      bool
		numPoints int
	}{
		{
			name: "Line intersecting square",
			polygon: Polygon{
				Points: []*Point{
					{X: 0, Y: 0},
					{X: 2, Y: 0},
					{X: 2, Y: 2},
					{X: 0, Y: 2},
				},
			},
			line: Line{
				A: &Point{X: -1, Y: 1},
				B: &Point{X: 3, Y: 1},
			},
			want:      true,
			numPoints: 2,
		},
		{
			name: "Line not intersecting square",
			polygon: Polygon{
				Points: []*Point{
					{X: 0, Y: 0},
					{X: 2, Y: 0},
					{X: 2, Y: 2},
					{X: 0, Y: 2},
				},
			},
			line: Line{
				A: &Point{X: -1, Y: 3},
				B: &Point{X: 3, Y: 3},
			},
			want:      false,
			numPoints: 0,
		},
		{
			name: "Line touching square at vertex",
			polygon: Polygon{
				Points: []*Point{
					{X: 0, Y: 0},
					{X: 2, Y: 0},
					{X: 2, Y: 2},
					{X: 0, Y: 2},
				},
			},
			line: Line{
				A: &Point{X: -1, Y: 0},
				B: &Point{X: 3, Y: 0},
			},
			want:      true,
			numPoints: 2,
		},
		{
			name: "Line far above rectangle",
			polygon: Polygon{
				Points: []*Point{
					{X: 0, Y: 540},
					{X: 800, Y: 540},
					{X: 800, Y: 600},
					{X: 0, Y: 600},
				},
			},
			line: Line{
				A: &Point{X: 275.983354, Y: 522.589787},
				B: &Point{X: 290.013471, Y: 508.336513},
			},
			want:      false,
			numPoints: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, points := checkLinePolygonCollision(&tt.line, &tt.polygon)
			if got != tt.want {
				t.Errorf("Polygon.CollidesWith() collision = %v, want %v", got, tt.want)
			}
			if len(points) != tt.numPoints {
				t.Errorf("Polygon.CollidesWith() got %v intersection points, want %v", len(points), tt.numPoints)
			}
		})
	}
}

func TestPolygonCircleCollision(t *testing.T) {
	tests := []struct {
		name      string
		polygon   Polygon
		circle    Circle
		want      bool
		numPoints int
	}{
		{
			name: "Circle intersecting square",
			polygon: Polygon{
				Points: []*Point{
					{X: 0, Y: 0},
					{X: 2, Y: 0},
					{X: 2, Y: 2},
					{X: 0, Y: 2},
				},
			},
			circle: Circle{
				C: &Point{X: 1, Y: 1},
				R: 2,
			},
			want:      true,
			numPoints: 4,
		},
		{
			name: "Circle not intersecting square",
			polygon: Polygon{
				Points: []*Point{
					{X: 0, Y: 0},
					{X: 2, Y: 0},
					{X: 2, Y: 2},
					{X: 0, Y: 2},
				},
			},
			circle: Circle{
				C: &Point{X: 5, Y: 5},
				R: 1,
			},
			want:      false,
			numPoints: 0,
		},
		{
			name: "Circle touching square at edge",
			polygon: Polygon{
				Points: []*Point{
					{X: 0, Y: 0},
					{X: 2, Y: 0},
					{X: 2, Y: 2},
					{X: 0, Y: 2},
				},
			},
			circle: Circle{
				C: &Point{X: 3, Y: 1},
				R: 1,
			},
			want:      true,
			numPoints: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, points := checkCirclePolygonCollision(&tt.circle, &tt.polygon)
			if got != tt.want {
				t.Errorf("Polygon.CollidesWith() collision = %v, want %v", got, tt.want)
			}
			if len(points) != tt.numPoints {
				t.Errorf("Polygon.CollidesWith() got %v intersection points, want %v", len(points), tt.numPoints)
			}
		})
	}
}

func TestCollidesWith(t *testing.T) {
	tests := []struct {
		name       string
		shape1     Shape
		shape2     Shape
		want       bool
		wantPoints int // Number of collision points expected
	}{
		{
			name:       "Line-Line collision",
			shape1:     &Line{BaseShape: &BaseShape{}, A: &Point{X: 0, Y: 0}, B: &Point{X: 2, Y: 2}},
			shape2:     &Line{BaseShape: &BaseShape{}, A: &Point{X: 0, Y: 2}, B: &Point{X: 2, Y: 0}},
			want:       true,
			wantPoints: 1,
		},
		{
			name:       "Line-Circle collision",
			shape1:     &Line{BaseShape: &BaseShape{}, A: &Point{X: -2, Y: 0}, B: &Point{X: 2, Y: 0}},
			shape2:     &Circle{BaseShape: &BaseShape{}, C: &Point{X: 0, Y: 0.5}, R: 1},
			want:       true,
			wantPoints: 2,
		},
		{
			name:   "Line-Polygon collision",
			shape1: &Line{BaseShape: &BaseShape{}, A: &Point{X: 0, Y: 0}, B: &Point{X: 2, Y: 2}},
			shape2: &Polygon{
				BaseShape: &BaseShape{},
				Points: []*Point{
					&Point{X: 1, Y: 0},
					&Point{X: 2, Y: 1},
					&Point{X: 1, Y: 2},
					&Point{X: 0, Y: 1},
				},
			},
			want:       true,
			wantPoints: 2,
		},
		{
			name:       "Circle-Circle collision",
			shape1:     &Circle{BaseShape: &BaseShape{}, C: &Point{X: 0, Y: 0}, R: 1},
			shape2:     &Circle{BaseShape: &BaseShape{}, C: &Point{X: 1, Y: 0}, R: 1},
			want:       true,
			wantPoints: 2, // Circle-Circle intersection produces 2 points
		},
		{
			name:   "Circle-Polygon collision",
			shape1: &Circle{BaseShape: &BaseShape{}, C: &Point{X: 0, Y: 0}, R: 1},
			shape2: &Polygon{
				BaseShape: &BaseShape{},
				Points: []*Point{
					&Point{X: 0.5, Y: 0},
					&Point{X: 1.5, Y: 0},
					&Point{X: 1.5, Y: 1},
					&Point{X: 0.5, Y: 1},
				},
			},
			want:       true,
			wantPoints: 3, // Circle-Polygon intersection can produce multiple points
		},
		{
			name: "Polygon-Polygon collision",
			shape1: &Polygon{
				BaseShape: &BaseShape{},
				Points: []*Point{
					&Point{X: 0, Y: 0},
					&Point{X: 2, Y: 0},
					&Point{X: 2, Y: 2},
					&Point{X: 0, Y: 2},
				},
			},
			shape2: &Polygon{
				BaseShape: &BaseShape{},
				Points: []*Point{
					&Point{X: 1, Y: 1},
					&Point{X: 3, Y: 1},
					&Point{X: 3, Y: 3},
					&Point{X: 1, Y: 3},
				},
			},
			want:       true,
			wantPoints: 2, // Overlapping squares produce 2 intersection points
		},
		{
			name:       "No collision - Line-Line parallel",
			shape1:     &Line{BaseShape: &BaseShape{}, A: &Point{X: 0, Y: 0}, B: &Point{X: 2, Y: 2}},
			shape2:     &Line{BaseShape: &BaseShape{}, A: &Point{X: 0, Y: 1}, B: &Point{X: 2, Y: 3}},
			want:       false,
			wantPoints: 0,
		},
		{
			name:       "No collision - Circle-Circle distant",
			shape1:     &Circle{BaseShape: &BaseShape{}, C: &Point{X: 0, Y: 0}, R: 1},
			shape2:     &Circle{BaseShape: &BaseShape{}, C: &Point{X: 3, Y: 3}, R: 1},
			want:       false,
			wantPoints: 0,
		},
		{
			name: "No collision - Polygon-Polygon distant",
			shape1: &Polygon{
				BaseShape: &BaseShape{},
				Points: []*Point{
					&Point{X: 0, Y: 0},
					&Point{X: 1, Y: 0},
					&Point{X: 1, Y: 1},
					&Point{X: 0, Y: 1},
				},
			},
			shape2: &Polygon{
				BaseShape: &BaseShape{},
				Points: []*Point{
					&Point{X: 3, Y: 3},
					&Point{X: 4, Y: 3},
					&Point{X: 4, Y: 4},
					&Point{X: 3, Y: 4},
				},
			},
			want:       false,
			wantPoints: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, points := tt.shape1.CollidesWith(tt.shape2)
			if got != tt.want {
				t.Errorf("CollidesWith() collision = %v, want %v", got, tt.want)
			}
			if len(points) != tt.wantPoints {
				t.Errorf("CollidesWith() number of collision points = %v, want %v", len(points), tt.wantPoints)
			}
		})
	}
}
