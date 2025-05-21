package math

import (
	"math"
)

// RoPE implements Rotary Positional Encoding for attention mechanisms
type RoPE struct {
	// Base for the rotary encoding (theta)
	base float64
	// Maximum sequence length supported
	maxSeqLen int
	// Dimension of the key/query vectors
	dim int
	// Pre-computed rotation matrices for each position
	rotations [][]float64
}

// NewRoPE creates a new RoPE instance with the given parameters
func NewRoPE(base float64, maxSeqLen, dim int) *RoPE {
	rope := &RoPE{
		base:      base,
		maxSeqLen: maxSeqLen,
		dim:       dim,
		rotations: make([][]float64, maxSeqLen),
	}

	// Pre-compute rotation matrices for each position
	for pos := 0; pos < maxSeqLen; pos++ {
		rope.rotations[pos] = make([]float64, dim/2) // Only need half the dimensions for angles
		for i := 0; i < dim/2; i++ {
			// Calculate rotation angle for this dimension
			angle := float64(pos) / math.Pow(base, float64(2*i)/float64(dim))
			rope.rotations[pos][i] = angle
		}
	}

	return rope
}

// ApplyRoPE applies rotary positional encoding to a query or key vector
func (r *RoPE) ApplyRoPE(vector []float32, position int) []float32 {
	if position >= r.maxSeqLen {
		panic("position exceeds maximum sequence length")
	}
	if len(vector) != r.dim {
		panic("vector dimension does not match RoPE dimension")
	}

	result := make([]float32, r.dim)
	for i := 0; i < r.dim; i += 2 {
		if i+1 >= r.dim {
			// Handle odd dimensions
			result[i] = vector[i]
			break
		}

		// Get rotation angle for this position and dimension pair
		angle := r.rotations[position][i/2]

		// Apply rotation to the pair of dimensions
		cos := float32(math.Cos(angle))
		sin := float32(math.Sin(angle))

		// Rotate the vector pair
		result[i] = vector[i]*cos - vector[i+1]*sin
		result[i+1] = vector[i]*sin + vector[i+1]*cos
	}

	return result
}

// ApplyRoPEBatch applies rotary positional encoding to a batch of vectors
func (r *RoPE) ApplyRoPEBatch(vectors [][]float32, startPos int) [][]float32 {
	result := make([][]float32, len(vectors))
	for i, vector := range vectors {
		result[i] = r.ApplyRoPE(vector, startPos+i)
	}
	return result
}
