package math

import (
	"testing"
)

func TestNewMatrixAndGetSet(t *testing.T) {
	m := NewMatrix(2, 3)
	if m.Rows != 2 || m.Cols != 3 || m.Stride != 3 {
		t.Fatalf("unexpected matrix dimensions: got %dx%d stride %d", m.Rows, m.Cols, m.Stride)
	}
	m.Set(1, 2, 1)
	if got := m.Get(1, 2); got != 1 {
		t.Errorf("Get/Set failed: want 1, got %v", got)
	}
}

func TestMatrix_GetSet(t *testing.T) {
	m := NewMatrix(2, 2)
	m.Set(0, 0, 1)
	m.Set(0, 1, -1)
	m.Set(1, 0, 0)
	m.Set(1, 1, 1)

	if m.Get(0, 0) != 1 {
		t.Errorf("Get(0, 0) = %v, want 1", m.Get(0, 0))
	}
	if m.Get(0, 1) != -1 {
		t.Errorf("Get(0, 1) = %v, want -1", m.Get(0, 1))
	}
	if m.Get(1, 0) != 0 {
		t.Errorf("Get(1, 0) = %v, want 0", m.Get(1, 0))
	}
	if m.Get(1, 1) != 1 {
		t.Errorf("Get(1, 1) = %v, want 1", m.Get(1, 1))
	}
}

func TestMatrix_Add(t *testing.T) {
	a := NewMatrix(2, 2)
	b := NewMatrix(2, 2)

	// Initialize matrices
	a.Set(0, 0, 1)
	a.Set(0, 1, -1)
	a.Set(1, 0, 0)
	a.Set(1, 1, 1)

	b.Set(0, 0, 1)
	b.Set(0, 1, 1)
	b.Set(1, 0, 1)
	b.Set(1, 1, 1)

	// Test addition
	result := Add(a, b)
	want := [][]int8{{1, 0}, {1, 1}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if result.Get(i, j) != want[i][j] {
				t.Errorf("Add() at (%d,%d) = %v, want %v", i, j, result.Get(i, j), want[i][j])
			}
		}
	}

	// Test clamping
	a.Set(0, 0, 1)
	b.Set(0, 0, 1)
	result = Add(a, b)
	if result.Get(0, 0) != 1 {
		t.Errorf("Add() clamping = %v, want 1", result.Get(0, 0))
	}

	a.Set(0, 0, -1)
	b.Set(0, 0, -1)
	result = Add(a, b)
	if result.Get(0, 0) != -1 {
		t.Errorf("Add() clamping = %v, want -1", result.Get(0, 0))
	}
}

func TestMatrix_Mul(t *testing.T) {
	a := NewMatrix(2, 3)
	b := NewMatrix(3, 2)

	// Initialize matrices
	a.Set(0, 0, 1)
	a.Set(0, 1, -1)
	a.Set(0, 2, 0)
	a.Set(1, 0, 1)
	a.Set(1, 1, 1)
	a.Set(1, 2, 1)

	b.Set(0, 0, 1)
	b.Set(0, 1, 1)
	b.Set(1, 0, 1)
	b.Set(1, 1, 1)
	b.Set(2, 0, 1)
	b.Set(2, 1, 1)

	// Test multiplication
	result := Mul(a, b)
	want := [][]int8{{0, 0}, {1, 1}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if result.Get(i, j) != want[i][j] {
				t.Errorf("Mul() at (%d,%d) = %v, want %v", i, j, result.Get(i, j), want[i][j])
			}
		}
	}

	// Test clamping
	a.Set(0, 0, 1)
	a.Set(0, 1, 1)
	a.Set(0, 2, 1)
	b.Set(0, 0, 1)
	b.Set(1, 0, 1)
	b.Set(2, 0, 1)
	result = Mul(a, b)
	if result.Get(0, 0) != 1 {
		t.Errorf("Mul() clamping = %v, want 1", result.Get(0, 0))
	}
}

func TestNewVectorAndDotProduct(t *testing.T) {
	a := NewVector(3)
	b := NewVector(3)
	a.Data[0], a.Data[1], a.Data[2] = 1, 1, 1
	b.Data[0], b.Data[1], b.Data[2] = 1, 1, 1
	if got := DotProduct(a, b); got != 1 {
		t.Errorf("DotProduct: got %v, want 1", got)
	}
}

func TestVector_DotProduct(t *testing.T) {
	a := NewVector(3)
	b := NewVector(3)

	// Initialize vectors
	a.Data[0] = 1
	a.Data[1] = -1
	a.Data[2] = 0

	b.Data[0] = 1
	b.Data[1] = 1
	b.Data[2] = 1

	// Test dot product
	result := DotProduct(a, b)
	if result != 0 {
		t.Errorf("DotProduct() = %v, want 0", result)
	}

	// Test clamping
	a.Data[0] = 1
	a.Data[1] = 1
	a.Data[2] = 1
	b.Data[0] = 1
	b.Data[1] = 1
	b.Data[2] = 1
	result = DotProduct(a, b)
	if result != 1 {
		t.Errorf("DotProduct() clamping = %v, want 1", result)
	}

	a.Data[0] = -1
	a.Data[1] = -1
	a.Data[2] = -1
	result = DotProduct(a, b)
	if result != -1 {
		t.Errorf("DotProduct() clamping = %v, want -1", result)
	}
}

func TestMatrix_Dimensions(t *testing.T) {
	// Test invalid dimensions for Add
	a := NewMatrix(2, 2)
	b := NewMatrix(2, 3)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Add() did not panic with mismatched dimensions")
		}
	}()
	Add(a, b)

	// Test invalid dimensions for Mul
	a = NewMatrix(2, 2)
	b = NewMatrix(3, 2)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Mul() did not panic with mismatched dimensions")
		}
	}()
	Mul(a, b)
}

func TestVector_Dimensions(t *testing.T) {
	a := NewVector(2)
	b := NewVector(3)
	defer func() {
		if r := recover(); r == nil {
			t.Error("DotProduct() did not panic with mismatched dimensions")
		}
	}()
	DotProduct(a, b)
}
