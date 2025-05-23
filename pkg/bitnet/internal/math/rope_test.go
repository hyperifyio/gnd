package math

import (
	"math"
	"testing"
)

func TestNewRoPE(t *testing.T) {
	tests := []struct {
		name        string
		base        float64
		maxSeqLen   int
		dim         int
		shouldPanic bool
	}{
		{
			name:        "valid parameters",
			base:        10000.0,
			maxSeqLen:   4096,
			dim:         256,
			shouldPanic: false,
		},
		{
			name:        "odd dimension",
			base:        10000.0,
			maxSeqLen:   4,
			dim:         5,
			shouldPanic: false,
		},
		{
			name:        "zero maxSeqLen",
			base:        10000.0,
			maxSeqLen:   0,
			dim:         256,
			shouldPanic: true,
		},
		{
			name:        "zero dimension",
			base:        10000.0,
			maxSeqLen:   4,
			dim:         0,
			shouldPanic: true,
		},
		{
			name:        "negative maxSeqLen",
			base:        10000.0,
			maxSeqLen:   -1,
			dim:         256,
			shouldPanic: true,
		},
		{
			name:        "negative dimension",
			base:        10000.0,
			maxSeqLen:   4,
			dim:         -1,
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.shouldPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Error("expected panic")
					}
				}()
			}

			rope := NewRoPE(tt.base, tt.maxSeqLen, tt.dim)
			if tt.shouldPanic {
				return
			}

			if rope == nil {
				t.Fatal("NewRoPE returned nil")
			}

			// Check initialization
			if rope.base != tt.base {
				t.Errorf("expected base %f, got %f", tt.base, rope.base)
			}
			if rope.maxSeqLen != tt.maxSeqLen {
				t.Errorf("expected maxSeqLen %d, got %d", tt.maxSeqLen, rope.maxSeqLen)
			}
			if rope.dim != tt.dim {
				t.Errorf("expected dim %d, got %d", tt.dim, rope.dim)
			}
			if len(rope.rotations) != tt.maxSeqLen {
				t.Errorf("expected %d rotation matrices, got %d", tt.maxSeqLen, len(rope.rotations))
			}

			// Check rotation matrix values
			for pos := 0; pos < tt.maxSeqLen; pos++ {
				if len(rope.rotations[pos]) != tt.dim/2 {
					t.Errorf("position %d: expected %d dimensions, got %d", pos, tt.dim/2, len(rope.rotations[pos]))
				}
				for i := 0; i < tt.dim/2; i++ {
					expected := float64(pos) * math.Pow(tt.base, -float64(2*i)/float64(tt.dim))
					if math.Abs(rope.rotations[pos][i]-expected) > 1e-10 {
						t.Errorf("position %d, dim %d: expected angle %f, got %f", pos, i, expected, rope.rotations[pos][i])
					}
				}
			}
		})
	}
}

func TestApplyRoPE(t *testing.T) {
	tests := []struct {
		name        string
		base        float64
		maxSeqLen   int
		dim         int
		vector      []float32
		position    int
		expected    []float32
		shouldPanic bool
	}{
		{
			name:      "basic rotation",
			base:      10000.0,
			maxSeqLen: 4,
			dim:       4,
			vector:    []float32{1.0, 0.0, 0.0, 1.0},
			position:  1,
			expected: []float32{
				float32(math.Cos(1.0)),
				float32(math.Sin(1.0)),
				-float32(math.Sin(0.01)),
				float32(math.Cos(0.01)),
			},
			shouldPanic: false,
		},
		{
			name:        "zero vector",
			base:        10000.0,
			maxSeqLen:   4,
			dim:         4,
			vector:      []float32{0.0, 0.0, 0.0, 0.0},
			position:    0,
			expected:    []float32{0.0, 0.0, 0.0, 0.0},
			shouldPanic: false,
		},
		{
			name:      "odd dimension",
			base:      10000.0,
			maxSeqLen: 4,
			dim:       5,
			vector:    []float32{1.0, 0.0, 0.0, 1.0, 0.5},
			position:  1,
			expected: func() []float32 {
				// Create a temporary RoPE to get the correct angles
				rope := NewRoPE(10000.0, 4, 5)
				// Get the actual angles used in the implementation
				angle0 := rope.rotations[1][0] // angle for first pair
				angle1 := rope.rotations[1][1] // angle for second pair
				cos0 := float32(math.Cos(angle0))
				sin0 := float32(math.Sin(angle0))
				cos1 := float32(math.Cos(angle1))
				sin1 := float32(math.Sin(angle1))
				v := []float32{1.0, 0.0, 0.0, 1.0, 0.5}
				result := make([]float32, 5)
				// First pair
				result[0] = v[0]*cos0 - v[1]*sin0
				result[1] = v[0]*sin0 + v[1]*cos0
				// Second pair
				result[2] = v[2]*cos1 - v[3]*sin1
				result[3] = v[2]*sin1 + v[3]*cos1
				// Odd last element
				result[4] = v[4]
				return result
			}(),
			shouldPanic: false,
		},
		{
			name:        "invalid position",
			base:        10000.0,
			maxSeqLen:   4,
			dim:         4,
			vector:      []float32{1.0, 0.0, 0.0, 1.0},
			position:    5,
			shouldPanic: true,
		},
		{
			name:        "invalid vector dimension",
			base:        10000.0,
			maxSeqLen:   4,
			dim:         4,
			vector:      []float32{1.0, 0.0},
			position:    0,
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rope := NewRoPE(tt.base, tt.maxSeqLen, tt.dim)

			if tt.shouldPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Error("expected panic")
					}
				}()
			}

			result := rope.ApplyRoPE(tt.vector, tt.position)

			if tt.shouldPanic {
				return
			}

			// Check dimensions
			if len(result) != tt.dim {
				t.Errorf("expected result length %d, got %d", tt.dim, len(result))
			}

			// Check values
			for i := 0; i < tt.dim; i++ {
				actual := result[i]
				exp := tt.expected[i]
				if math.Abs(float64(actual-exp)) > 1e-2 {
					t.Errorf("dimension %d: expected %f, got %f", i, exp, actual)
				}
			}
		})
	}
}

func TestApplyRoPEBatch(t *testing.T) {
	tests := []struct {
		name        string
		base        float64
		maxSeqLen   int
		dim         int
		vectors     [][]float32
		startPos    int
		shouldPanic bool
	}{
		{
			name:      "valid batch",
			base:      10000.0,
			maxSeqLen: 4,
			dim:       4,
			vectors: [][]float32{
				{1.0, 0.0, 0.0, 1.0},
				{0.0, 1.0, 1.0, 0.0},
			},
			startPos:    0,
			shouldPanic: false,
		},
		{
			name:        "empty batch",
			base:        10000.0,
			maxSeqLen:   4,
			dim:         4,
			vectors:     [][]float32{},
			startPos:    0,
			shouldPanic: false,
		},
		{
			name:      "invalid start position",
			base:      10000.0,
			maxSeqLen: 4,
			dim:       4,
			vectors: [][]float32{
				{1.0, 0.0, 0.0, 1.0},
				{0.0, 1.0, 1.0, 0.0},
			},
			startPos:    5,
			shouldPanic: true,
		},
		{
			name:      "invalid vector dimension",
			base:      10000.0,
			maxSeqLen: 4,
			dim:       4,
			vectors: [][]float32{
				{1.0, 0.0},
				{0.0, 1.0},
			},
			startPos:    0,
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rope := NewRoPE(tt.base, tt.maxSeqLen, tt.dim)

			if tt.shouldPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Error("expected panic")
					}
				}()
			}

			result := rope.ApplyRoPEBatch(tt.vectors, tt.startPos)

			if tt.shouldPanic {
				return
			}

			// Check batch size
			if len(result) != len(tt.vectors) {
				t.Errorf("expected %d results, got %d", len(tt.vectors), len(result))
			}

			// Check each vector in the batch
			for i, vector := range tt.vectors {
				expected := rope.ApplyRoPE(vector, tt.startPos+i)
				for j := 0; j < tt.dim; j++ {
					if math.Abs(float64(result[i][j]-expected[j])) > 1e-5 {
						t.Errorf("vector %d, dimension %d: expected %f, got %f", i, j, expected[j], result[i][j])
					}
				}
			}
		})
	}
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
		rope.ApplyRoPEBatch(vectors, i%(maxSeqLen-batchSize))
	}
}
