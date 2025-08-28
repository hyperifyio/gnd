package vector

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func TestNewVectorAndDotProduct(t *testing.T) {
	a := NewVector(3)
	b := NewVector(3)
	a.Data[0], a.Data[1], a.Data[2] = 1, 1, 1
	b.Data[0], b.Data[1], b.Data[2] = 1, 1, 1
	got, err := DotProduct(a, b)
	require.NoError(t, err)
	if got != 1 {
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
	result, err := DotProduct(a, b)
	require.NoError(t, err)
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
	result, err = DotProduct(a, b)
	require.NoError(t, err)
	if result != 1 {
		t.Errorf("DotProduct() clamping = %v, want 1", result)
	}

	a.Data[0] = -1
	a.Data[1] = -1
	a.Data[2] = -1
	result, err = DotProduct(a, b)
	require.NoError(t, err)
	if result != -1 {
		t.Errorf("DotProduct() clamping = %v, want -1", result)
	}
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
