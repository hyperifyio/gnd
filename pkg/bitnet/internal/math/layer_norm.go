// Package math implements mathematical operations for the BitNet model.
package math

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// LayerNorm implements layer normalization for BitNet.
// It normalizes each token's hidden state across the feature dimension
// and scales with a learnable parameter gamma (no bias).
type LayerNorm struct {
	// Hidden dimension
	hiddenDim int
	// Epsilon for numerical stability
	epsilon float32
	// Learnable scale parameter (gamma)
	gamma *tensor.Tensor
}

// NewLayerNorm creates a new LayerNorm instance.
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
// Input tensor can be either:
//   - 2D [batch_size, hidden_dim]
//   - 3D [batch_size, seq_len, hidden_dim]
//
// Returns a tensor with the same shape as the input.
func (l *LayerNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Validate input shape
	if err := ValidateShape(x, 2, 3); err != nil {
		return nil, fmt.Errorf("input must be 2D or 3D tensor: %w", err)
	}

	// Get input dimensions
	batchSize := x.Shape()[0]
	seqLen := 1
	if len(x.Shape()) == 3 {
		seqLen = x.Shape()[1]
	}
	hiddenDim := x.Shape()[len(x.Shape())-1]

	if hiddenDim != l.hiddenDim {
		return nil, fmt.Errorf("input hidden dimension (%d) must match layer hidden dimension (%d)", hiddenDim, l.hiddenDim)
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
func (l *LayerNorm) SetGamma(gamma *tensor.Tensor) error {
	if len(gamma.Shape()) != 1 || gamma.Shape()[0] != l.hiddenDim {
		return fmt.Errorf("gamma must be 1D tensor with shape [%d], got %v", l.hiddenDim, gamma.Shape())
	}
	l.gamma = gamma
	return nil
}

// GetGamma returns the current scale parameter.
func (l *LayerNorm) GetGamma() *tensor.Tensor {
	return l.gamma
}
