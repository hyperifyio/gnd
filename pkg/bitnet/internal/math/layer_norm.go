// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"math"
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// LayerNorm implements layer normalization for BitNet.
// It normalizes each token's hidden state across the feature dimension
// and scales with a learnable parameter gamma (no bias).
//
// The normalization process:
// 1. Calculates mean and variance across the feature dimension
// 2. Normalizes using: (x - mean) / sqrt(variance + epsilon)
// 3. Scales with learnable parameter gamma
//
// The implementation supports both 2D [batch_size, hidden_dim] and
// 3D [batch_size, seq_len, hidden_dim] inputs, with parallel processing
// for efficient computation on multi-core systems.
type LayerNorm struct {
	// Hidden dimension of the model
	hiddenDim int
	// Epsilon for numerical stability (default: 1e-5)
	epsilon float32
	// Learnable scale parameter (gamma) [hidden_dim]
	gamma *tensor.Tensor
}

// NewLayerNorm creates a new layer normalization instance.
//
// Parameters:
//   - hiddenDim: Size of the hidden dimension
//
// The layer is initialized with:
// - gamma: Vector of ones [hidden_dim]
// - epsilon: 1e-5 for numerical stability
//
// The layer supports both single-token and multi-token inputs,
// with automatic shape detection and appropriate processing.
func NewLayerNorm(hiddenDim int) *LayerNorm {
	// Initialize gamma with ones
	gamma := tensor.NewTensor(hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		gamma.Set(1, i)
	}

	return &LayerNorm{
		hiddenDim: hiddenDim,
		epsilon:   1e-5,
		gamma:     gamma,
	}
}

// Forward performs layer normalization on the input tensor.
//
// Input tensor can be either:
//   - 2D [batch_size, hidden_dim] for single-token inputs
//   - 3D [batch_size, seq_len, hidden_dim] for multi-token inputs
//
// The function:
// 1. Validates input shape and dimensions
// 2. Calculates mean and variance for each token
// 3. Normalizes using (x - mean) / sqrt(variance + epsilon)
// 4. Scales with gamma parameter
// 5. Clamps values to int8 range
//
// Returns a tensor with the same shape as the input.
// The implementation uses parallel processing with chunked computation
// for better performance on multi-core systems.
func (l *LayerNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Validate input shape
	if err := ValidateShape(x, 2, 3); err != nil {
		DebugLog("input shape validation failed: %v", err)
		return nil, err
	}

	// Get input dimensions
	batchSize := x.Shape()[0]
	seqLen := 1
	if len(x.Shape()) == 3 {
		seqLen = x.Shape()[1]
	}
	hiddenDim := x.Shape()[len(x.Shape())-1]

	if hiddenDim != l.hiddenDim {
		DebugLog("input hidden dimension (%d) does not match layer hidden dimension (%d)", hiddenDim, l.hiddenDim)
		return nil, ErrHiddenDimMismatch
	}

	// Create output tensor
	output := tensor.NewTensor(x.Shape()...)

	// Process in parallel chunks
	var wg sync.WaitGroup
	chunkSize := batchSize / runtime.NumCPU()
	if chunkSize < 1 {
		chunkSize = 1
	}

	for b := 0; b < batchSize; b += chunkSize {
		wg.Add(1)
		go func(startBatch int) {
			defer wg.Done()
			endBatch := startBatch + chunkSize
			if endBatch > batchSize {
				endBatch = batchSize
			}

			// For each batch element
			for b := startBatch; b < endBatch; b++ {
				// For each sequence position
				for s := 0; s < seqLen; s++ {
					// Calculate mean
					var sum float32
					for d := 0; d < hiddenDim; d++ {
						var val float32
						if len(x.Shape()) == 2 {
							val = float32(x.Get(b, d))
						} else {
							val = float32(x.Get(b, s, d))
						}
						sum += val
					}
					mean := sum / float32(hiddenDim)

					// Calculate variance
					var variance float32
					for d := 0; d < hiddenDim; d++ {
						var val float32
						if len(x.Shape()) == 2 {
							val = float32(x.Get(b, d))
						} else {
							val = float32(x.Get(b, s, d))
						}
						diff := val - mean
						variance += diff * diff
					}
					variance /= float32(hiddenDim)

					// Normalize and scale
					stdDev := float32(math.Sqrt(float64(variance + l.epsilon)))
					for d := 0; d < hiddenDim; d++ {
						var val float32
						if len(x.Shape()) == 2 {
							val = float32(x.Get(b, d))
						} else {
							val = float32(x.Get(b, s, d))
						}
						normalized := (val - mean) / stdDev
						scaled := normalized * float32(l.gamma.Get(d))
						// Clamp to int8 range and convert back to int8
						if len(x.Shape()) == 2 {
							output.Set(int8(min(max(int32(math.Round(float64(scaled))), -128), 127)), b, d)
						} else {
							output.Set(int8(min(max(int32(math.Round(float64(scaled))), -128), 127)), b, s, d)
						}
					}
				}
			}
		}(b)
	}

	wg.Wait()
	return output, nil
}

// SetGamma sets the learnable scale parameter.
//
// Parameters:
//   - gamma: Scale parameter [hidden_dim]
//
// Returns an error if the gamma tensor has incorrect shape.
// The gamma parameter must match the layer's hidden dimension.
func (l *LayerNorm) SetGamma(gamma *tensor.Tensor) error {
	if len(gamma.Shape()) != 1 || gamma.Shape()[0] != l.hiddenDim {
		DebugLog("gamma shape %v does not match required shape [%d]", gamma.Shape(), l.hiddenDim)
		return ErrInvalidGammaShape
	}
	l.gamma = gamma
	return nil
}

// GetGamma returns the current scale parameter.
//
// Returns the gamma tensor with shape [hidden_dim].
// This is the learnable parameter used for scaling the normalized values.
func (l *LayerNorm) GetGamma() *tensor.Tensor {
	return l.gamma
}
