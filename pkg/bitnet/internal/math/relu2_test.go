package math

import (
	"testing"
)

func TestReLU2(t *testing.T) {
	tests := []struct {
		name     string
		input    []int8
		expected []int8
	}{
		{
			name:     "empty input",
			input:    []int8{},
			expected: []int8{},
		},
		{
			name:     "all negative",
			input:    []int8{-10, -5, -1},
			expected: []int8{0, 0, 0},
		},
		{
			name:     "all positive",
			input:    []int8{1, 2, 3, 4, 5},
			expected: []int8{1, 4, 9, 16, 25},
		},
		{
			name:     "mixed values",
			input:    []int8{-3, -2, -1, 0, 1, 2, 3},
			expected: []int8{0, 0, 0, 0, 1, 4, 9},
		},
		{
			name:     "clamping test",
			input:    []int8{12, 13, 14, 15},
			expected: []int8{127, 127, 127, 127}, // 15² = 225 > 127, so clamped
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := ReLU2(tt.input)
			if len(output) != len(tt.expected) {
				t.Errorf("expected length %d, got %d", len(tt.expected), len(output))
				return
			}
			for i := range output {
				if output[i] != tt.expected[i] {
					t.Errorf("output[%d] = %d, want %d", i, output[i], tt.expected[i])
				}
			}
		})
	}
}

func TestReLU2Batch(t *testing.T) {
	tests := []struct {
		name     string
		input    [][]int8
		expected [][]int8
	}{
		{
			name:     "empty batch",
			input:    [][]int8{},
			expected: [][]int8{},
		},
		{
			name: "single vector",
			input: [][]int8{
				{-2, -1, 0, 1, 2},
			},
			expected: [][]int8{
				{0, 0, 0, 1, 4},
			},
		},
		{
			name: "multiple vectors",
			input: [][]int8{
				{-3, -2, -1},
				{0, 1, 2},
				{3, 4, 5},
			},
			expected: [][]int8{
				{0, 0, 0},
				{0, 1, 4},
				{9, 16, 25},
			},
		},
		{
			name: "clamping test",
			input: [][]int8{
				{12, 13},
				{14, 15},
			},
			expected: [][]int8{
				{127, 127},
				{127, 127},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := ReLU2Batch(tt.input)
			if len(output) != len(tt.expected) {
				t.Errorf("expected batch size %d, got %d", len(tt.expected), len(output))
				return
			}
			for i := range output {
				if len(output[i]) != len(tt.expected[i]) {
					t.Errorf("vector %d: expected length %d, got %d", i, len(tt.expected[i]), len(output[i]))
					continue
				}
				for j := range output[i] {
					if output[i][j] != tt.expected[i][j] {
						t.Errorf("output[%d][%d] = %d, want %d", i, j, output[i][j], tt.expected[i][j])
					}
				}
			}
		})
	}
}

func BenchmarkReLU2(b *testing.B) {
	// Create test data
	input := make([]int8, 1024)
	for i := range input {
		input[i] = int8(i - 512) // Range from -512 to 511
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ReLU2(input)
	}
}

func BenchmarkReLU2Batch(b *testing.B) {
	// Create test data
	batchSize := 32
	vectorSize := 1024
	input := make([][]int8, batchSize)
	for i := range input {
		input[i] = make([]int8, vectorSize)
		for j := range input[i] {
			input[i][j] = int8(j - 512) // Range from -512 to 511
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ReLU2Batch(input)
	}
}
