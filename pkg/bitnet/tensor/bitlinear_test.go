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
				{-1, 4},
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
					input.setRaw(tt.input[i][j], i, j)
				}
			}

			// Create weights tensor
			weights := NewTensor(len(tt.weights), len(tt.weights[0]))
			for i := range tt.weights {
				for j := range tt.weights[i] {
					weights.setRaw(tt.weights[i][j], i, j)
				}
			}

			// Run BitLinear
			output, err := BitLinear(input, weights)
			if err != nil {
				t.Fatalf("BitLinear failed: %v", err)
			}
			defer output.Close()

			// Debug: print output matrix for the first test case
			if tt.name == "simple 2x2 matrix multiplication" {
				t.Logf("Actual output matrix:")
				for i := range tt.expected {
					row := make([]int8, len(tt.expected[i]))
					for j := range tt.expected[i] {
						row[j] = output.Get(i, j)
					}
					t.Logf("%v", row)
				}
			}

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

func TestMax(t *testing.T) {
	tests := []struct {
		name     string
		a        int32
		b        int32
		expected int32
	}{
		{
			name:     "a greater than b",
			a:        10,
			b:        5,
			expected: 10,
		},
		{
			name:     "b greater than a",
			a:        5,
			b:        10,
			expected: 10,
		},
		{
			name:     "equal values",
			a:        10,
			b:        10,
			expected: 10,
		},
		{
			name:     "negative values",
			a:        -10,
			b:        -5,
			expected: -5,
		},
		{
			name:     "zero values",
			a:        0,
			b:        0,
			expected: 0,
		},
		{
			name:     "large values",
			a:        1000000,
			b:        999999,
			expected: 1000000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := max(tt.a, tt.b)
			if got != tt.expected {
				t.Errorf("max(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.expected)
			}
		})
	}
}

func TestBitLinear_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		batchSize   int
		inFeatures  int
		outFeatures int
		setup       func(*Tensor, *Tensor)
		wantErr     bool
	}{
		{
			name:        "zero batch size",
			batchSize:   0,
			inFeatures:  10,
			outFeatures: 10,
			wantErr:     true,
		},
		{
			name:        "zero input features",
			batchSize:   10,
			inFeatures:  0,
			outFeatures: 10,
			wantErr:     true,
		},
		{
			name:        "zero output features",
			batchSize:   10,
			inFeatures:  10,
			outFeatures: 0,
			wantErr:     true,
		},
		{
			name:        "all ones input",
			batchSize:   2,
			inFeatures:  3,
			outFeatures: 2,
			setup: func(input, weights *Tensor) {
				// Set all input values to 1
				for i := 0; i < input.shape[0]; i++ {
					for j := 0; j < input.shape[1]; j++ {
						input.Set(1, i, j)
					}
				}
				// Set all weights to 1
				for i := 0; i < weights.shape[0]; i++ {
					for j := 0; j < weights.shape[1]; j++ {
						weights.Set(1, i, j)
					}
				}
			},
			wantErr: false,
		},
		{
			name:        "all negative input",
			batchSize:   2,
			inFeatures:  3,
			outFeatures: 2,
			setup: func(input, weights *Tensor) {
				// Set all input values to -1
				for i := 0; i < input.shape[0]; i++ {
					for j := 0; j < input.shape[1]; j++ {
						input.Set(-1, i, j)
					}
				}
				// Set all weights to -1
				for i := 0; i < weights.shape[0]; i++ {
					for j := 0; j < weights.shape[1]; j++ {
						weights.Set(-1, i, j)
					}
				}
			},
			wantErr: false,
		},
		{
			name:        "mixed values",
			batchSize:   2,
			inFeatures:  3,
			outFeatures: 2,
			setup: func(input, weights *Tensor) {
				// Set alternating values
				for i := 0; i < input.shape[0]; i++ {
					for j := 0; j < input.shape[1]; j++ {
						input.Set(int8((i+j)%3-1), i, j)
					}
				}
				// Set alternating weights
				for i := 0; i < weights.shape[0]; i++ {
					for j := 0; j < weights.shape[1]; j++ {
						weights.Set(int8((i+j)%3-1), i, j)
					}
				}
			},
			wantErr: false,
		},
		{
			name:        "large dimensions",
			batchSize:   100,
			inFeatures:  100,
			outFeatures: 100,
			setup: func(input, weights *Tensor) {
				// Set pattern of values
				for i := 0; i < input.shape[0]; i++ {
					for j := 0; j < input.shape[1]; j++ {
						input.Set(int8((i+j)%3-1), i, j)
					}
				}
				// Set pattern of weights
				for i := 0; i < weights.shape[0]; i++ {
					for j := 0; j < weights.shape[1]; j++ {
						weights.Set(int8((i+j)%3-1), i, j)
					}
				}
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.wantErr {
				defer func() {
					if r := recover(); r == nil {
						t.Error("BitLinear did not panic as expected")
					}
				}()
			}

			input := NewTensor(tt.batchSize, tt.inFeatures)
			weights := NewTensor(tt.outFeatures, tt.inFeatures)

			if tt.setup != nil {
				tt.setup(input, weights)
			}

			output, err := BitLinear(input, weights)
			if err != nil {
				t.Fatalf("BitLinear failed: %v", err)
			}
			defer output.Close()

			if !tt.wantErr {
				if output == nil {
					t.Fatal("BitLinear returned nil")
				}

				// Verify output shape
				shape := output.Shape()
				if len(shape) != 2 || shape[0] != tt.batchSize || shape[1] != tt.outFeatures {
					t.Errorf("Output shape = %v, want [%d %d]", shape, tt.batchSize, tt.outFeatures)
				}

				// Verify output values are within int8 range
				data := output.Data()
				for i, v := range data {
					if v < -128 || v > 127 {
						t.Errorf("Output[%d] = %d, out of int8 range", i, v)
					}
				}
			}
		})
	}
}
