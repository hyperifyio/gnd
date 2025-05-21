package tensor

import (
	"testing"
)

func TestRawTensor(t *testing.T) {
	tests := []struct {
		name     string
		rows     int
		cols     int
		setup    func(*rawTensor)
		expected [][]int8
	}{
		{
			name: "basic 2x2 operations",
			rows: 2,
			cols: 2,
			setup: func(rt *rawTensor) {
				rt.Set(0, 0, 1)
				rt.Set(0, 1, 2)
				rt.Set(1, 0, 3)
				rt.Set(1, 1, 4)
			},
			expected: [][]int8{
				{1, 2},
				{3, 4},
			},
		},
		{
			name: "full int8 range",
			rows: 2,
			cols: 2,
			setup: func(rt *rawTensor) {
				rt.Set(0, 0, -128)
				rt.Set(0, 1, 127)
				rt.Set(1, 0, 0)
				rt.Set(1, 1, 42)
			},
			expected: [][]int8{
				{-128, 127},
				{0, 42},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create raw tensor
			rt := newRawTensor(tt.rows, tt.cols)

			// Setup values
			tt.setup(rt)

			// Verify values
			for i := 0; i < tt.rows; i++ {
				for j := 0; j < tt.cols; j++ {
					got := rt.At(i, j)
					want := tt.expected[i][j]
					if got != want {
						t.Errorf("At(%d, %d) = %d, want %d", i, j, got, want)
					}
				}
			}

			// Verify Shape
			rows, cols := rt.Shape()
			if rows != tt.rows || cols != tt.cols {
				t.Errorf("Shape() = (%d, %d), want (%d, %d)", rows, cols, tt.rows, tt.cols)
			}
		})
	}
}

func TestNewRawTensorFrom(t *testing.T) {
	tests := []struct {
		name     string
		input    [][]int8
		expected [][]int8
	}{
		{
			name: "2x2 tensor",
			input: [][]int8{
				{1, 2},
				{3, 4},
			},
			expected: [][]int8{
				{1, 2},
				{3, 4},
			},
		},
		{
			name: "full int8 range",
			input: [][]int8{
				{-128, 127},
				{0, 42},
			},
			expected: [][]int8{
				{-128, 127},
				{0, 42},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input tensor
			input := NewTensor(len(tt.input), len(tt.input[0]))
			for i := range tt.input {
				for j := range tt.input[i] {
					input.setRaw(tt.input[i][j], i, j)
				}
			}

			// Convert to raw tensor
			rt := newRawTensorFrom(input)

			// Verify values
			for i := 0; i < len(tt.expected); i++ {
				for j := 0; j < len(tt.expected[i]); j++ {
					got := rt.At(i, j)
					want := tt.expected[i][j]
					if got != want {
						t.Errorf("At(%d, %d) = %d, want %d", i, j, got, want)
					}
				}
			}
		})
	}
}

func TestRawTensorPanics(t *testing.T) {
	tests := []struct {
		name string
		fn   func()
	}{
		{
			name: "1D tensor",
			fn: func() {
				t := NewTensor(2)
				newRawTensorFrom(t)
			},
		},
		{
			name: "3D tensor",
			fn: func() {
				t := NewTensor(2, 2, 2)
				newRawTensorFrom(t)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic")
				}
			}()
			tt.fn()
		})
	}
}
