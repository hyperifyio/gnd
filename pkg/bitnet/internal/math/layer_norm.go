// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"fmt"
	"math"

	"github.com/hyperifyio/gnd/pkg/bitnet/errors"
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// LayerNorm represents a layer normalization component.
// It normalizes the input tensor along the last dimension.
type LayerNorm struct {
	hiddenDim int
	gamma     *tensor.Tensor
	closed    bool
	epsilon   float32
}

// NewLayerNorm creates a new layer normalization component.
func NewLayerNorm(hiddenDim int) *LayerNorm {
	return &LayerNorm{
		hiddenDim: hiddenDim,
		epsilon:   1e-5,
	}
}

// Forward applies layer normalization to the input tensor.
// Returns a normalized tensor with the same shape as input.
func (l *LayerNorm) Forward(x tensor.TensorReader) (*tensor.Tensor, error) {
	if l.closed {
		return nil, errors.ErrLayerClosed
	}

	// Validate input shape
	shape := x.Shape()
	if len(shape) < 2 {
		return nil, errors.ErrInvalidShape
	}

	// Get input dimensions
	var batchSize, seqLen int
	if len(shape) == 2 {
		batchSize, seqLen = shape[0], 1
	} else {
		batchSize, seqLen = shape[0], shape[1]
	}

	// Validate hidden dimension
	hiddenDim := shape[len(shape)-1]
	if hiddenDim != l.hiddenDim {
		return nil, fmt.Errorf("tensor: invalid hidden dimension, got %d, want %d", hiddenDim, l.hiddenDim)
	}

	// Create output tensor
	output := tensor.NewTensor(batchSize, seqLen, l.hiddenDim)

	// Apply layer normalization
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			// Calculate mean
			var sum float32
			for d := 0; d < l.hiddenDim; d++ {
				var val int8
				if len(shape) == 2 {
					val = x.Get(b, d)
				} else {
					val = x.Get(b, s, d)
				}
				sum += float32(val)
			}
			mean := sum / float32(l.hiddenDim)

			// Calculate variance
			var variance float32
			for d := 0; d < l.hiddenDim; d++ {
				var val int8
				if len(shape) == 2 {
					val = x.Get(b, d)
				} else {
					val = x.Get(b, s, d)
				}
				diff := float32(val) - mean
				variance += diff * diff
			}
			variance /= float32(l.hiddenDim)

			// Normalize and scale
			for d := 0; d < l.hiddenDim; d++ {
				var val int8
				if len(shape) == 2 {
					val = x.Get(b, d)
				} else {
					val = x.Get(b, s, d)
				}
				normalized := (float32(val) - mean) / float32(math.Sqrt(float64(variance+l.epsilon)))
				if l.gamma != nil {
					normalized *= float32(l.gamma.Get(d))
				}
				if normalized > 127 {
					output.Set(127, b, s, d)
				} else if normalized < -128 {
					output.Set(-128, b, s, d)
				} else {
					output.Set(int8(normalized), b, s, d)
				}
			}
		}
	}

	return output, nil
}

// SetGamma sets the scale parameter for layer normalization.
// LayerNorm takes ownership of the gamma tensor and will close it when LayerNorm is closed.
// The caller must not close the tensor after passing it to SetGamma.
func (l *LayerNorm) SetGamma(gamma tensor.TensorOperations) error {
	if l.closed {
		return errors.ErrLayerClosed
	}

	if gamma == nil {
		return errors.ErrNilTensor
	}

	// Convert to concrete type
	g, ok := gamma.(*tensor.Tensor)
	if !ok {
		return errors.ErrInvalidShape
	}

	// Validate shape
	shape := g.Shape()
	if len(shape) != 1 || shape[0] != l.hiddenDim {
		return fmt.Errorf("tensor: invalid gamma shape, got %v, want [%d]", shape, l.hiddenDim)
	}

	if l.gamma != nil {
		l.gamma.Close()
	}
	l.gamma = g
	return nil
}

// GetGamma returns the gamma parameter.
// The returned tensor is owned by LayerNorm and should not be closed by the caller.
func (l *LayerNorm) GetGamma() tensor.TensorReader {
	if l.closed {
		panic("layer is closed")
	}
	return l.gamma
}

// Close releases all resources associated with the layer normalization.
// This includes closing all tensors and cleaning up memory.
func (l *LayerNorm) Close() {
	if !l.closed {
		if l.gamma != nil {
			l.gamma.Close()
		}
		l.closed = true
	}
}
