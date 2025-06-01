// Package layer_norm provides normalization functions for BitNet math operations.
//
// # LayerNorm for Quantized BitNet Inference
//
// This package provides a LayerNorm implementation specifically designed for the BitNet model's
// quantized (int8) inference pipeline. The normalization math is performed in float32 for accuracy,
// but the output is quantized to int8 to match the model's memory and performance requirements.
//
// Key aspects:
//   - All input, output, and gamma tensors are int8, as required by BitNet's quantized architecture
//   - The normalization step computes mean/variance in float32, but the result is rounded and clamped to int8
//   - This design enables high-throughput, low-memory inference on CPUs, at the cost of some precision
//   - The gamma parameter is also quantized (int8), and is applied as a scale after normalization
//   - The implementation is optimized for 2D and 3D tensors, matching transformer batch/sequence/hidden layouts
//   - Uses epsilon=1e-5 for numerical stability as specified in BitNet config
//
// Implementation details:
//   - Pre-norm architecture with epsilon=1e-5 for numerical stability
//   - Efficient computation of mean and variance in float32
//   - Scaling factor of sqrt(0.5) applied to match BitNet's requirements
//   - Support for both 2D [batch, hidden] and 3D [batch, seq_len, hidden] inputs
//   - No bias term as per BitNet architecture
//   - Handles 4096-token context length
//
// Related tasks and dependencies:
//   - #179: Implement Sub-Layer Normalization (SubLN)
//   - #186: Integrate Attention Sublayer (Pre-Norm & Residual)
//   - #187: Integrate Feed-Forward Sublayer (Pre-Norm & Residual)
//   - #182: Compute Scaled Dot-Product Attention
//   - #185: Feed-Forward Network (FFN) Sublayer
//
// Usage:
//   - Used as a sublayer in the BitNet transformer block during inference
//   - Not intended for training or high-precision floating-point use
//   - Maintainers should avoid changing output type to float32, as this would break model compatibility
//
// Caveats:
//   - If you need floating-point normalization for testing, implement a separate float version for test-only use
//   - Do not change the quantization logic unless updating the entire BitNet inference pipeline
//   - Always validate changes against end-to-end BitNet inference and quantized model outputs
//   - Performance critical - changes should be benchmarked against existing implementation
//   - Memory management is important - tensors should be properly closed after use
//   - Must maintain compatibility with BitNet's binary-weight quantization
//
// For more details, see BitNet issue #170 and the BitNet project documentation.
package layer_norm

import (
	"errors"
	"math"

	"github.com/hyperifyio/gnd/pkg/bitnet/logging"
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

var (

	// ErrInvalidHiddenDim is returned when the hidden dimension is invalid
	ErrInvalidHiddenDim = errors.New("invalid hidden dimension")

	// ErrLayerClosed is returned when a bitnet layer is closed
	ErrLayerClosed = errors.New("bitnet: layer is closed")

	// ErrInvalidShape is returned when a tensor has an invalid shape
	ErrInvalidShape = errors.New("invalid tensor shape")
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
		return nil, ErrInvalidHiddenDim
	}

	// Initialize gamma with ones
	gamma, err := tensor.NewTensor(hiddenDim)
	if err != nil {
		return nil, err
	}
	for i := 0; i < hiddenDim; i++ {
		if err := gamma.Set(1, i); err != nil {
			gamma.Close()
			return nil, err
		}
	}

	return &LayerNorm{
		hiddenDim: hiddenDim,
		gamma:     gamma,
		epsilon:   1e-5,
	}, nil
}

// Forward applies layer normalization to the input tensor.
// Returns a normalized tensor with the same shape as input.
func (l *LayerNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if l.closed {
		return nil, ErrLayerClosed
	}

	// Validate input shape
	shape, err := x.Shape()
	if err != nil {
		logging.DebugLogf("failed to get input shape: %v", err)
		return nil, ErrInvalidShape
	}
	if len(shape) < 2 {
		return nil, ErrInvalidShape
	}

	// Validate hidden dimension
	hiddenDim := shape[len(shape)-1]
	if hiddenDim != l.hiddenDim {
		logging.DebugLogf("tensor: invalid hidden dimension, got %d, want %d", hiddenDim, l.hiddenDim)
		return nil, ErrInvalidHiddenDim
	}

	// Create output tensor with same shape as input, but float32 type
	output, err := tensor.NewTensor(shape...)
	if err != nil {
		logging.DebugLogf("failed to create output tensor: %v", err)
		return nil, err
	}

	if len(shape) == 2 {
		for b := 0; b < shape[0]; b++ {
			var sum float32
			for d := 0; d < l.hiddenDim; d++ {
				val, err := x.Get(b, d)
				if err != nil {
					logging.DebugLogf("failed to get input value: %v", err)
					return nil, err
				}
				sum += float32(val)
			}
			mean := sum / float32(l.hiddenDim)

			var variance float32
			for d := 0; d < l.hiddenDim; d++ {
				val, err := x.Get(b, d)
				if err != nil {
					logging.DebugLogf("failed to get input value: %v", err)
					return nil, err
				}
				diff := float32(val) - mean
				variance += diff * diff
			}
			variance /= float32(l.hiddenDim)
			// Apply sqrt(0.5) scaling to variance as per BitNet requirements
			variance *= 0.5

			for d := 0; d < l.hiddenDim; d++ {
				val, err := x.Get(b, d)
				if err != nil {
					logging.DebugLogf("failed to get input value: %v", err)
					return nil, err
				}
				normalized := (float32(val) - mean) / float32(math.Sqrt(float64(variance+l.epsilon)))
				gammaVal, err := l.gamma.Get(d)
				if err != nil {
					logging.DebugLogf("failed to get gamma value: %v", err)
					return nil, err
				}
				// Apply gamma scaling
				normalized *= float32(gammaVal)
				// Convert to int8 and clamp
				intVal := int8(math.Round(float64(normalized)))
				if intVal > 127 {
					intVal = 127
				} else if intVal < -128 {
					intVal = -128
				}
				if err := output.Set(intVal, b, d); err != nil {
					logging.DebugLogf("failed to set output value: %v", err)
					return nil, err
				}
			}
		}
	} else {
		for b := 0; b < shape[0]; b++ {
			for s := 0; s < shape[1]; s++ {
				var sum float32
				for d := 0; d < l.hiddenDim; d++ {
					val, err := x.Get(b, s, d)
					if err != nil {
						logging.DebugLogf("failed to get input value: %v", err)
						return nil, err
					}
					sum += float32(val)
				}
				mean := sum / float32(l.hiddenDim)

				var variance float32
				for d := 0; d < l.hiddenDim; d++ {
					val, err := x.Get(b, s, d)
					if err != nil {
						logging.DebugLogf("failed to get input value: %v", err)
						return nil, err
					}
					diff := float32(val) - mean
					variance += diff * diff
				}
				variance /= float32(l.hiddenDim)
				// Apply sqrt(0.5) scaling to variance as per BitNet requirements
				variance *= 0.5

				for d := 0; d < l.hiddenDim; d++ {
					val, err := x.Get(b, s, d)
					if err != nil {
						logging.DebugLogf("failed to get input value: %v", err)
						return nil, err
					}
					normalized := (float32(val) - mean) / float32(math.Sqrt(float64(variance+l.epsilon)))
					gammaVal, err := l.gamma.Get(d)
					if err != nil {
						logging.DebugLogf("failed to get gamma value: %v", err)
						return nil, err
					}
					// Apply gamma scaling
					normalized *= float32(gammaVal)
					// Convert to int8 and clamp
					intVal := int8(math.Round(float64(normalized)))
					if intVal > 127 {
						intVal = 127
					} else if intVal < -128 {
						intVal = -128
					}
					if err := output.Set(intVal, b, s, d); err != nil {
						logging.DebugLogf("failed to set output value: %v", err)
						return nil, err
					}
				}
			}
		}
	}

	return output, nil
}

// SetGamma sets the gamma parameter of the layer normalization.
func (l *LayerNorm) SetGamma(gamma *tensor.Tensor) error {
	if l.closed {
		return ErrLayerClosed
	}

	// Validate gamma shape
	shape, err := gamma.Shape()
	if err != nil {
		logging.DebugLogf("failed to get gamma shape: %v", err)
		return ErrInvalidShape
	}
	if len(shape) != 1 || shape[0] != l.hiddenDim {
		logging.DebugLogf("tensor: invalid gamma shape, got %v, want [%d]", shape, l.hiddenDim)
		return ErrInvalidShape
	}

	// Close old gamma tensor
	if l.gamma != nil {
		if err := l.gamma.Close(); err != nil {
			logging.DebugLogf("failed to close old gamma tensor: %v", err)
			return err
		}
	}

	l.gamma = gamma
	return nil
}

// GetGamma returns the gamma parameter of the layer normalization.
func (l *LayerNorm) GetGamma() (*tensor.Tensor, error) {
	if l.closed {
		return nil, ErrLayerClosed
	}
	return l.gamma, nil
}

// Close closes the layer normalization and releases its resources.
func (l *LayerNorm) Close() error {
	if l.closed {
		return nil
	}
	l.closed = true
	if l.gamma != nil {
		if err := l.gamma.Close(); err != nil {
			logging.DebugLogf("failed to close gamma tensor: %v", err)
			return err
		}
	}
	return nil
}
