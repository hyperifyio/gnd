package math

import (
	"math"
	"testing"
)

func TestNewRoPE(t *testing.T) {
	base := 10000.0
	maxSeqLen := 4096
	dim := 256

	rope := NewRoPE(base, maxSeqLen, dim)
	if rope == nil {
		t.Fatal("NewRoPE returned nil")
	}

	// Check initialization
	if rope.base != base {
		t.Errorf("expected base %f, got %f", base, rope.base)
	}
	if rope.maxSeqLen != maxSeqLen {
		t.Errorf("expected maxSeqLen %d, got %d", maxSeqLen, rope.maxSeqLen)
	}
	if rope.dim != dim {
		t.Errorf("expected dim %d, got %d", dim, rope.dim)
	}
	if len(rope.rotations) != maxSeqLen {
		t.Errorf("expected %d rotation matrices, got %d", maxSeqLen, len(rope.rotations))
	}

	// Check rotation matrix values
	for pos := 0; pos < maxSeqLen; pos++ {
		if len(rope.rotations[pos]) != dim/2 {
			t.Errorf("position %d: expected %d dimensions, got %d", pos, dim/2, len(rope.rotations[pos]))
		}
		for i := 0; i < dim/2; i++ {
			expected := float64(pos) * math.Pow(base, -float64(2*i)/float64(dim))
			if math.Abs(rope.rotations[pos][i]-expected) > 1e-10 {
				t.Errorf("position %d, dim %d: expected angle %f, got %f", pos, i, expected, rope.rotations[pos][i])
			}
		}
	}
}

func TestApplyRoPE(t *testing.T) {
	base := 10000.0
	maxSeqLen := 4
	dim := 4

	rope := NewRoPE(base, maxSeqLen, dim)

	// Test vector with known values
	vector := []float32{1.0, 0.0, 0.0, 1.0}
	position := 1

	result := rope.ApplyRoPE(vector, position)

	// Check dimensions
	if len(result) != dim {
		t.Errorf("expected result length %d, got %d", dim, len(result))
	}

	// Check rotation properties
	// For position 1, the rotation should be approximately:
	// [cos(θ), sin(θ), cos(2θ), sin(2θ)]
	// where θ = 1/10000
	theta := 1.0 / base
	expected := []float32{
		float32(math.Cos(theta)),
		float32(math.Sin(theta)),
		float32(-math.Sin(2 * theta)),
		float32(math.Cos(2 * theta)),
	}

	for i := 0; i < dim; i++ {
		actual := result[i]
		exp := expected[i]
		if math.Abs(float64(actual-exp)) > 1e-2 {
			t.Errorf("dimension %d: expected %f, got %f", i, exp, actual)
		}
	}
}

func TestApplyRoPEBatch(t *testing.T) {
	base := 10000.0
	maxSeqLen := 4
	dim := 4

	rope := NewRoPE(base, maxSeqLen, dim)

	// Test batch of vectors
	vectors := [][]float32{
		{1.0, 0.0, 0.0, 1.0},
		{0.0, 1.0, 1.0, 0.0},
	}
	startPos := 0

	result := rope.ApplyRoPEBatch(vectors, startPos)

	// Check batch size
	if len(result) != len(vectors) {
		t.Errorf("expected %d results, got %d", len(vectors), len(result))
	}

	// Check each vector in the batch
	for i, vector := range vectors {
		expected := rope.ApplyRoPE(vector, startPos+i)
		for j := 0; j < dim; j++ {
			if math.Abs(float64(result[i][j]-expected[j])) > 1e-5 {
				t.Errorf("vector %d, dimension %d: expected %f, got %f", i, j, expected[j], result[i][j])
			}
		}
	}
}

func TestApplyRoPEInvalidInput(t *testing.T) {
	base := 10000.0
	maxSeqLen := 4
	dim := 4

	rope := NewRoPE(base, maxSeqLen, dim)

	// Test invalid position
	vector := []float32{1.0, 0.0, 0.0, 1.0}
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for invalid position")
		}
	}()
	rope.ApplyRoPE(vector, maxSeqLen)

	// Test invalid vector dimension
	invalidVector := []float32{1.0, 0.0}
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for invalid vector dimension")
		}
	}()
	rope.ApplyRoPE(invalidVector, 0)
}

func BenchmarkApplyRoPE(b *testing.B) {
	base := 10000.0
	maxSeqLen := 4096
	dim := 256

	rope := NewRoPE(base, maxSeqLen, dim)
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = float32(i) / float32(dim)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rope.ApplyRoPE(vector, i%maxSeqLen)
	}
}

func BenchmarkApplyRoPEBatch(b *testing.B) {
	base := 10000.0
	maxSeqLen := 4096
	dim := 256
	batchSize := 32

	rope := NewRoPE(base, maxSeqLen, dim)
	vectors := make([][]float32, batchSize)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			vectors[i][j] = float32(j) / float32(dim)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rope.ApplyRoPEBatch(vectors, i%maxSeqLen)
	}
}
