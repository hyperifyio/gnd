package rope

import (
	"errors"
	"math"
)

var (
	ErrRoPEInvalidParams    = errors.New("rope: invalid parameters")
	ErrRoPEInvalidPosition  = errors.New("rope: position exceeds maximum sequence length")
	ErrRoPEInvalidDimension = errors.New("rope: vector dimension does not match RoPE dimension")
)

// Package rope implements Rotary Positional Encoding (RoPE) for BitNet attention.
//
// # Rotary Positional Encoding (RoPE) for BitNet
//
// This package provides RoPE for attention mechanisms, as described in the BitNet paper (https://arxiv.org/abs/2310.11453).
//
// Key aspects:
//   - Implements rotary positional encoding for query/key vectors
//   - Supports configurable base, sequence length, and dimension
//   - Optimized for CPU efficiency and low memory use
//   - Not suitable for training or float32 inference
//
// Implementation details:
//   - Pre-computes rotation angles for each position and dimension
//   - Supports both single vector and batch application
//   - Handles odd and even dimensions
//
// Related tasks and dependencies:
//   - #177: Implement Rotary Positional Encoding (RoPE) (Core implementation)
//   - #182: Compute Scaled Dot-Product Attention (Depends on #177)
//   - #186: Integrate Attention Sublayer (Pre-Norm & Residual) (Depends on #177)
//
// Usage:
//   - Used in BitNet attention blocks for positional encoding
//   - Maintainers should not change encoding logic without full pipeline review
//
// Caveats:
//   - Any change must be validated against end-to-end BitNet inference
//   - Performance critical - changes should be benchmarked against existing implementation
//   - Memory management is important - tensors should be properly closed after use
//
// For more details, see BitNet issue #190 and the BitNet project documentation.

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
func NewRoPE(base float64, maxSeqLen, dim int) (*RoPE, error) {
	// Validate input parameters
	if maxSeqLen <= 0 {
		return nil, ErrRoPEInvalidParams
	}
	if dim <= 0 {
		return nil, ErrRoPEInvalidParams
	}

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

	return rope, nil
}

// ApplyRoPE applies rotary positional encoding to a query or key vector
func (r *RoPE) ApplyRoPE(vector []float32, position int) ([]float32, error) {
	if position >= r.maxSeqLen {
		return nil, ErrRoPEInvalidPosition
	}
	if len(vector) != r.dim {
		return nil, ErrRoPEInvalidDimension
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

	return result, nil
}

// ApplyRoPEBatch applies rotary positional encoding to a batch of vectors
func (r *RoPE) ApplyRoPEBatch(vectors [][]float32, startPos int) ([][]float32, error) {
	if startPos < 0 || startPos+len(vectors) > r.maxSeqLen {
		return nil, ErrRoPEInvalidPosition
	}

	result := make([][]float32, len(vectors))
	for i, vector := range vectors {
		if len(vector) != r.dim {
			return nil, ErrRoPEInvalidDimension
		}
		encoded, err := r.ApplyRoPE(vector, startPos+i)
		if err != nil {
			return nil, err
		}
		result[i] = encoded
	}
	return result, nil
}
