package tensor

import (
	"fmt"
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
			input, err := NewTensor(len(tt.input), len(tt.input[0]))
			if err != nil {
				t.Fatalf("Failed to create input tensor: %v", err)
			}
			for i := range tt.input {
				for j := range tt.input[i] {
					if err := input.setRaw(tt.input[i][j], i, j); err != nil {
						t.Fatalf("Failed to set input value: %v", err)
					}
				}
			}

			// Create weights tensor
			weights, err := NewTensor(len(tt.weights), len(tt.weights[0]))
			if err != nil {
				t.Fatalf("Failed to create weights tensor: %v", err)
			}
			for i := range tt.weights {
				for j := range tt.weights[i] {
					if err := weights.setRaw(tt.weights[i][j], i, j); err != nil {
						t.Fatalf("Failed to set weight value: %v", err)
					}
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
						val, err := output.Get(i, j)
						if err != nil {
							t.Fatalf("Failed to get output value: %v", err)
						}
						row[j] = val
					}
					t.Logf("%v", row)
				}
			}

			// Verify output
			for i := range tt.expected {
				for j := range tt.expected[i] {
					got, err := output.Get(i, j)
					if err != nil {
						t.Fatalf("Failed to get output value: %v", err)
					}
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
		wantErr error
	}{
		{
			name:    "1D_input",
			input:   func() *Tensor { t, _ := NewTensor(10); return t }(),
			weights: func() *Tensor { t, _ := NewTensor(10, 20); return t }(),
			wantErr: ErrInvalidShape,
		},
		{
			name:    "1D_weights",
			input:   func() *Tensor { t, _ := NewTensor(10, 20); return t }(),
			weights: func() *Tensor { t, _ := NewTensor(10); return t }(),
			wantErr: ErrInvalidShape,
		},
		{
			name:    "dimension_mismatch",
			input:   func() *Tensor { t, _ := NewTensor(10, 20); return t }(),
			weights: func() *Tensor { t, _ := NewTensor(30, 40); return t }(),
			wantErr: ErrDimensionMismatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BitLinear(tt.input, tt.weights)
			if err != tt.wantErr {
				t.Errorf("BitLinear() error = %v, wantErr %v", err, tt.wantErr)
			}
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
	// Test with empty tensors (using 1x1 instead of 0x0 since zero dimensions are invalid)
	input, err := NewTensor(1, 1)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	weights, err := NewTensor(1, 1)
	if err != nil {
		t.Fatalf("Failed to create weights tensor: %v", err)
	}

	output, err := BitLinear(input, weights)
	if err != nil {
		t.Fatalf("BitLinear failed: %v", err)
	}
	defer output.Close()

	shape, err := output.Shape()
	if err != nil {
		t.Fatalf("Failed to get output shape: %v", err)
	}
	if len(shape) != 2 || shape[0] != 1 || shape[1] != 1 {
		t.Errorf("Expected 1x1 tensor, got shape %v", shape)
	}

	data, err := output.Data()
	if err != nil {
		t.Fatalf("Failed to get output data: %v", err)
	}
	if len(data) != 1 {
		t.Errorf("Expected data length 1, got length %d", len(data))
	}

	// Test with nil tensors
	_, err = BitLinear(nil, weights)
	if err == nil {
		t.Error("Expected error with nil input tensor")
	}

	_, err = BitLinear(input, nil)
	if err == nil {
		t.Error("Expected error with nil weights tensor")
	}

	// Test with closed tensors
	err = input.Close()
	if err != nil {
		t.Fatalf("Failed to close input tensor: %v", err)
	}
	_, err = BitLinear(input, weights)
	if err == nil {
		t.Error("Expected error with closed input tensor")
	}

	err = weights.Close()
	if err != nil {
		t.Fatalf("Failed to close weights tensor: %v", err)
	}
	_, err = BitLinear(input, weights)
	if err == nil {
		t.Error("Expected error with closed weights tensor")
	}
}

func TestBitLinear_ConcurrentAccess(t *testing.T) {
	// Create input and weights tensors
	input, err := NewTensor(10, 10)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	weights, err := NewTensor(10, 10)
	if err != nil {
		t.Fatalf("Failed to create weights tensor: %v", err)
	}

	// Fill with test data
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			if err := input.setRaw(1, i, j); err != nil {
				t.Fatalf("Failed to set input value: %v", err)
			}
			if err := weights.setRaw(1, i, j); err != nil {
				t.Fatalf("Failed to set weight value: %v", err)
			}
		}
	}

	// Run multiple BitLinear operations concurrently
	const numGoroutines = 10
	results := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			output, err := BitLinear(input, weights)
			if err != nil {
				results <- err
				return
			}
			defer output.Close()

			// Verify output
			for i := 0; i < 10; i++ {
				for j := 0; j < 10; j++ {
					val, err := output.Get(i, j)
					if err != nil {
						results <- err
						return
					}
					if val != 10 { // 10 * 1 = 10
						results <- fmt.Errorf("unexpected value at [%d,%d]: got %d, want 10", i, j, val)
						return
					}
				}
			}
			results <- nil
		}()
	}

	// Check results
	for i := 0; i < numGoroutines; i++ {
		if err := <-results; err != nil {
			t.Errorf("Concurrent BitLinear failed: %v", err)
		}
	}
}
