package math

import (
	"testing"
)

func TestNewMatrixAndGetSet(t *testing.T) {
	m := NewMatrix(2, 3)
	if m.Rows != 2 || m.Cols != 3 || m.Stride != 3 {
		t.Fatalf("unexpected matrix dimensions: got %dx%d stride %d", m.Rows, m.Cols, m.Stride)
	}
	m.Set(1, 2, 42.5)
	if got := m.Get(1, 2); got != 42.5 {
		t.Errorf("Get/Set failed: want 42.5, got %v", got)
	}
}

func TestAdd(t *testing.T) {
	a := NewMatrix(2, 2)
	b := NewMatrix(2, 2)
	a.Set(0, 0, 1)
	a.Set(0, 1, 2)

	a.Set(1, 0, 3)

	a.Set(1, 1, 4)
	b.Set(0, 0, 5)
	b.Set(0, 1, 6)
	b.Set(1, 0, 7)
	b.Set(1, 1, 8)
	c := Add(a, b)
	want := [][]float32{{6, 8}, {10, 12}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if got := c.Get(i, j); got != want[i][j] {
				t.Errorf("Add: c[%d][%d]=%v, want %v", i, j, got, want[i][j])
			}
		}
	}
}

func TestMul(t *testing.T) {
	a := NewMatrix(2, 3)
	b := NewMatrix(3, 2)
	// a = [1 2 3; 4 5 6]
	a.Set(0, 0, 1)
	a.Set(0, 1, 2)
	a.Set(0, 2, 3)
	a.Set(1, 0, 4)
	a.Set(1, 1, 5)
	a.Set(1, 2, 6)
	// b = [7 8; 9 10; 11 12]
	b.Set(0, 0, 7)
	b.Set(0, 1, 8)
	b.Set(1, 0, 9)
	b.Set(1, 1, 10)
	b.Set(2, 0, 11)
	b.Set(2, 1, 12)
	c := Mul(a, b)
	// c = [58 64; 139 154]
	want := [][]float32{{58, 64}, {139, 154}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if got := c.Get(i, j); got != want[i][j] {
				t.Errorf("Mul: c[%d][%d]=%v, want %v", i, j, got, want[i][j])
			}
		}
	}
}

func TestNewVectorAndDotProduct(t *testing.T) {
	a := NewVector(3)
	b := NewVector(3)
	a.Data[0], a.Data[1], a.Data[2] = 1, 2, 3
	b.Data[0], b.Data[1], b.Data[2] = 4, 5, 6
	if got := DotProduct(a, b); got != 32 {
		t.Errorf("DotProduct: got %v, want 32", got)
	}
}
