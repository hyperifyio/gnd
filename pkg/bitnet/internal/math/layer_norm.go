// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"math"

	bitneterrors "github.com/hyperifyio/gnd/pkg/bitnet/errors"
	"github.com/hyperifyio/gnd/pkg/bitnet/logging"
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
func NewLayerNorm(hiddenDim int) (*LayerNorm, error) {
	if hiddenDim <= 0 {
		logging.DebugLogf("layer_norm: invalid hidden dimension %d", hiddenDim)
		return nil, bitneterrors.ErrInvalidHiddenDim
	}
	return &LayerNorm{
		hiddenDim: hiddenDim,
		epsilon:   1e-5,
	}, nil
}

// Forward applies layer normalization to the input tensor.
// Returns a normalized tensor with the same shape as input.
func (l *LayerNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if l.closed {
		return nil, bitneterrors.ErrLayerClosed
	}

	// Validate input shape
	shape, err := x.Shape()
	if err != nil {
		logging.DebugLogf("failed to get input shape: %v", err)
		return nil, bitneterrors.ErrInvalidShape
	}
	if len(shape) < 2 {
		return nil, bitneterrors.ErrInvalidShape
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
		logging.DebugLogf("tensor: invalid hidden dimension, got %d, want %d", hiddenDim, l.hiddenDim)
		return nil, bitneterrors.ErrInvalidHiddenDim
	}

	// Create output tensor
	output, err := tensor.NewTensor(batchSize, seqLen, l.hiddenDim)
	if err != nil {
		logging.DebugLogf("failed to create output tensor: %v", err)
		return nil, err
	}

	// Apply layer normalization
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			// Calculate mean
			var sum float32
			for d := 0; d < l.hiddenDim; d++ {
				var val int8
				var err error
				if len(shape) == 2 {
					val, err = x.Get(b, d)
				} else {
					val, err = x.Get(b, s, d)
				}
				if err != nil {
					logging.DebugLogf("failed to get input value: %v", err)
					return nil, err
				}
				sum += float32(val)
			}
			mean := sum / float32(l.hiddenDim)

			// Calculate variance
			var variance float32
			for d := 0; d < l.hiddenDim; d++ {
				var val int8
				var err error
				if len(shape) == 2 {
					val, err = x.Get(b, d)
				} else {
					val, err = x.Get(b, s, d)
				}
				if err != nil {
					logging.DebugLogf("failed to get input value: %v", err)
					return nil, err
				}
				diff := float32(val) - mean
				variance += diff * diff
			}
			variance /= float32(l.hiddenDim)

			// Normalize and scale
			for d := 0; d < l.hiddenDim; d++ {
				var val int8
				var err error
				if len(shape) == 2 {
					val, err = x.Get(b, d)
				} else {
					val, err = x.Get(b, s, d)
				}
				if err != nil {
					logging.DebugLogf("failed to get input value: %v", err)
					return nil, err
				}
				normalized := (float32(val) - mean) / float32(math.Sqrt(float64(variance+l.epsilon)))
				if l.gamma != nil {
					gammaVal, err := l.gamma.Get(d)
					if err != nil {
						logging.DebugLogf("failed to get gamma value: %v", err)
						return nil, err
					}
					normalized *= float32(gammaVal)
				}
				if normalized > 127 {
					if err := output.Set(127, b, s, d); err != nil {
						logging.DebugLogf("failed to set output value: %v", err)
						return nil, err
					}
				} else if normalized < -128 {
					if err := output.Set(-128, b, s, d); err != nil {
						logging.DebugLogf("failed to set output value: %v", err)
						return nil, err
					}
				} else {
					if err := output.Set(int8(normalized), b, s, d); err != nil {
						logging.DebugLogf("failed to set output value: %v", err)
						return nil, err
					}
				}
			}
		}
	}

	return output, nil
}

// SetGamma sets the scale parameter for layer normalization.
// LayerNorm takes ownership of the gamma tensor and will close it when LayerNorm is closed.
// The caller must not close the tensor after passing it to SetGamma.
func (l *LayerNorm) SetGamma(gamma *tensor.Tensor) error {
	if l.closed {
		return bitneterrors.ErrLayerClosed
	}

	if gamma == nil {
		return bitneterrors.ErrNilTensor
	}

	// Validate shape
	shape, err := gamma.Shape()
	if err != nil {
		logging.DebugLogf("failed to get gamma shape: %v", err)
		return bitneterrors.ErrInvalidShape
	}
	if len(shape) != 1 || shape[0] != l.hiddenDim {
		logging.DebugLogf("tensor: invalid gamma shape, got %v, want [%d]", shape, l.hiddenDim)
		return ErrInvalidGammaShape
	}

	if l.gamma != nil {
		if err := l.gamma.Close(); err != nil {
			logging.DebugLogf("failed to close existing gamma tensor: %v", err)
			return err
		}
	}
	l.gamma = gamma
	return nil
}

// GetGamma returns the gamma parameter.
// The returned tensor is owned by LayerNorm and should not be closed by the caller.
func (l *LayerNorm) GetGamma() (*tensor.Tensor, error) {
	if l.closed {
		return nil, bitneterrors.ErrLayerClosed
	}
	return l.gamma, nil
}

// Close releases all resources associated with the layer normalization.
// This includes closing all tensors and cleaning up memory.
func (l *LayerNorm) Close() error {
	if !l.closed {
		if l.gamma != nil {
			if err := l.gamma.Close(); err != nil {
				logging.DebugLogf("failed to close gamma tensor: %v", err)
				return err
			}
		}
		l.closed = true
	}
	return nil
}
