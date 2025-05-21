package math

import (
	"math"
	"testing"
)

func TestNewSubLN(t *testing.T) {
	hiddenSize := 256
	epsilon := float32(1e-5)
	subln := NewSubLN(hiddenSize, epsilon)

	if subln == nil {
		t.Fatal("NewSubLN returned nil")
	}

	if subln.epsilon != epsilon {
		t.Errorf("expected epsilon %v, got %v", epsilon, subln.epsilon)
	}

	if len(subln.gamma) != hiddenSize {
		t.Errorf("expected gamma length %d, got %d", hiddenSize, len(subln.gamma))
	}

	// Check that gamma is initialized with ones
	for i, g := range subln.gamma {
		if g != 1.0 {
			t.Errorf("expected gamma[%d] to be 1.0, got %v", i, g)
		}
	}
}

func TestSubLNNormalize(t *testing.T) {
	tests := []struct {
		name      string
		input     [][]float32
		epsilon   float32
		expected  [][]float32
		checkFunc func(t *testing.T, got, want [][]float32)
	}{
		{
			name:     "empty input",
			input:    [][]float32{},
			epsilon:  1e-5,
			expected: [][]float32{},
			checkFunc: func(t *testing.T, got, want [][]float32) {
				if len(got) != 0 {
					t.Errorf("expected empty output, got length %d", len(got))
				}
			},
		},
		{
			name: "single vector",
			input: [][]float32{
				{1.0, 2.0, 3.0, 4.0},
			},
			epsilon: 1e-5,
			expected: [][]float32{
				{-1.3416, -0.4472, 0.4472, 1.3416},
			},
			checkFunc: func(t *testing.T, got, want [][]float32) {
				for i := range got[0] {
					if math.Abs(float64(got[0][i]-want[0][i])) > 1e-4 {
						t.Errorf("expected %v, got %v", want[0][i], got[0][i])
					}
				}
			},
		},
		{
			name: "batch of vectors",
			input: [][]float32{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
			},
			epsilon: 1e-5,
			expected: [][]float32{
				{-1.2247, 0.0, 1.2247},
				{-1.2247, 0.0, 1.2247},
			},
			checkFunc: func(t *testing.T, got, want [][]float32) {
				for i := range got {
					for j := range got[i] {
						if math.Abs(float64(got[i][j]-want[i][j])) > 1e-4 {
							t.Errorf("expected %v, got %v", want[i][j], got[i][j])
						}
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.input) == 0 {
				subln := NewSubLN(1, tt.epsilon) // hiddenSize doesn't matter for empty input
				got := subln.Normalize(tt.input)
				tt.checkFunc(t, got, tt.expected)
				return
			}
			subln := NewSubLN(len(tt.input[0]), tt.epsilon)
			got := subln.Normalize(tt.input)
			tt.checkFunc(t, got, tt.expected)
		})
	}
}

func TestSubLNGamma(t *testing.T) {
	hiddenSize := 4
	subln := NewSubLN(hiddenSize, 1e-5)

	// Test setting gamma
	newGamma := []float32{2.0, 3.0, 4.0, 5.0}
	subln.SetGamma(newGamma)

	// Test getting gamma
	got := subln.GetGamma()
	if len(got) != len(newGamma) {
		t.Errorf("expected gamma length %d, got %d", len(newGamma), len(got))
	}
	for i, g := range got {
		if g != newGamma[i] {
			t.Errorf("expected gamma[%d] to be %v, got %v", i, newGamma[i], g)
		}
	}

	// Test gamma dimension mismatch
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for gamma dimension mismatch")
		}
	}()
	subln.SetGamma([]float32{1.0, 2.0}) // Should panic
}

func BenchmarkSubLNNormalize(b *testing.B) {
	// Create test data
	hiddenSize := 256
	batchSize := 32
	input := make([][]float32, batchSize)
	for i := range input {
		input[i] = make([]float32, hiddenSize)
		for j := range input[i] {
			input[i][j] = float32(i+j) / float32(hiddenSize)
		}
	}

	subln := NewSubLN(hiddenSize, 1e-5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		subln.Normalize(input)
	}
}
