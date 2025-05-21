package tensor

import (
	"testing"
)

func TestBitLinear(t *testing.T) {
	tests := []struct {
		name     string
		input    [][]int8
		weights  [][]int8
		expected [][]int8
	}{
		{
			name: "simple 2x2 matrix multiplication",
			input: [][]int8{
				{1, 2},
				{3, 4},
			},
			weights: [][]int8{
				{1, -1},
				{0, 1},
			},
			expected: [][]int8{
				{-1, 2},
				{-1, 7},
			},
		},
		{
			name: "larger matrix with mixed values",
			input: [][]int8{
				{10, 20, 30},
				{40, 50, 60},
			},
			weights: [][]int8{
				{1, 0, -1},
				{-1, 1, 0},
				{0, -1, 1},
			},
			expected: [][]int8{
				{-20, 10, 10},
				{-20, 40, 40},
			},
		},
		{
			name: "clamping test",
			input: [][]int8{
				{100, 100},
			},
			weights: [][]int8{
				{1, 1},
			},
			expected: [][]int8{
				{127},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input tensor
			input := NewTensor(len(tt.input), len(tt.input[0]))
			for i := range tt.input {
				for j := range tt.input[i] {
					input.Set(tt.input[i][j], i, j)
				}
			}

			// Create weights tensor
			weights := NewTensor(len(tt.weights), len(tt.weights[0]))
			for i := range tt.weights {
				for j := range tt.weights[i] {
					weights.Set(tt.weights[i][j], i, j)
				}
			}

			// Run BitLinear
			output := BitLinear(input, weights)

			// Verify output
			for i := range tt.expected {
				for j := range tt.expected[i] {
					got := output.Get(i, j)
					if got != tt.expected[i][j] {
						t.Errorf("output[%d][%d] = %d, want %d", i, j, got, tt.expected[i][j])
					}
				}
			}
		})
	}
}

func TestBitLinearPanics(t *testing.T) {
	tests := []struct {
		name    string
		input   *Tensor
		weights *Tensor
	}{
		{
			name:    "nil input",
			input:   nil,
			weights: NewTensor(2, 2),
		},
		{
			name:    "nil weights",
			input:   NewTensor(2, 2),
			weights: nil,
		},
		{
			name:    "1D input",
			input:   NewTensor(2),
			weights: NewTensor(2, 2),
		},
		{
			name:    "1D weights",
			input:   NewTensor(2, 2),
			weights: NewTensor(2),
		},
		{
			name:    "dimension mismatch",
			input:   NewTensor(2, 3),
			weights: NewTensor(2, 2),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic")
				}
			}()
			BitLinear(tt.input, tt.weights)
		})
	}
}
