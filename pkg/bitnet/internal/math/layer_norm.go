// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"errors"
	"math"
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

var (
	// ErrInvalidHiddenDim is returned when the hidden dimension is invalid
	ErrInvalidHiddenDim = errors.New("invalid hidden dimension")
	// ErrNilTensor is returned when a nil tensor is provided
	ErrNilTensor = errors.New("nil tensor provided")
	// ErrInvalidShape is returned when a tensor has an invalid shape
	ErrInvalidShape = errors.New("invalid tensor shape")
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
	// Mutex to protect concurrent access to gamma
	mu sync.RWMutex
	// Flag to track if the layer is closed
	closed bool
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
	// Check if layer is closed
	if l.closed {
		panic("layer is closed")
	}

	// Validate input shape
	if err := ValidateShape(x, 2, 3); err != nil {
		return nil, err
	}

	// Get input dimensions
	var batchSize, seqLen, hiddenDim int
	if len(x.Shape()) == 2 {
		batchSize, hiddenDim = x.Shape()[0], x.Shape()[1]
		seqLen = 1
	} else {
		batchSize, seqLen, hiddenDim = x.Shape()[0], x.Shape()[1], x.Shape()[2]
	}

	if hiddenDim != l.hiddenDim {
		return nil, ErrHiddenDimMismatch
	}

	// Create output tensor with same shape as input (int8)
	var output *tensor.Tensor
	if len(x.Shape()) == 2 {
		output = tensor.NewTensor(batchSize, hiddenDim)
	} else {
		output = tensor.NewTensor(batchSize, seqLen, hiddenDim)
	}

	// Process in parallel chunks with a reasonable chunk size
	var wg sync.WaitGroup
	numCPU := runtime.NumCPU()
	chunkSize := (batchSize + numCPU - 1) / numCPU
	if chunkSize < 1 {
		chunkSize = 1
	}

	// Create a channel to collect errors
	errChan := make(chan error, numCPU)

	for i := 0; i < batchSize; i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > batchSize {
				end = batchSize
			}

			// Process each batch element
			for b := start; b < end; b++ {
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
					var sumSq float32
					for d := 0; d < hiddenDim; d++ {
						var val float32
						if len(x.Shape()) == 2 {
							val = float32(x.Get(b, d))
						} else {
							val = float32(x.Get(b, s, d))
						}
						diff := val - mean
						sumSq += diff * diff
					}
					variance := sumSq / float32(hiddenDim)

					// Normalize and scale
					stdDev := float32(math.Sqrt(float64(variance + l.epsilon)))
					for d := 0; d < hiddenDim; d++ {
						var val float32
						if len(x.Shape()) == 2 {
							val = float32(x.Get(b, d))
						} else {
							val = float32(x.Get(b, s, d))
						}

						// Normalize: (x - mean) / sqrt(variance + epsilon)
						normalized := (val - mean) / stdDev

						// Scale with gamma (with read lock)
						l.mu.RLock()
						gammaVal := l.gamma.Get(d)
						l.mu.RUnlock()
						scaled := normalized * float32(gammaVal)

						// Clamp to int8 range
						if scaled >= 127 {
							scaled = 127
						} else if scaled <= -128 {
							scaled = -128
						}

						// Store as int8
						if len(x.Shape()) == 2 {
							output.Set(int8(scaled), b, d)
						} else {
							output.Set(int8(scaled), b, s, d)
						}
					}
				}
			}
		}(i)
	}

	// Wait for all goroutines to complete
	wg.Wait()

	// Check for errors
	select {
	case err := <-errChan:
		output.Close()
		return nil, err
	default:
		return output, nil
	}
}

// SetGamma sets the gamma parameter for layer normalization.
func (l *LayerNorm) SetGamma(gamma *tensor.Tensor) error {
	// Check if layer is closed
	if l.closed {
		panic("layer is closed")
	}

	if gamma == nil {
		return ErrNilTensor
	}
	if len(gamma.Shape()) != 1 || gamma.Shape()[0] != l.hiddenDim {
		return ErrInvalidShape
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	l.gamma = gamma
	return nil
}

// GetGamma returns the gamma parameter.
func (l *LayerNorm) GetGamma() *tensor.Tensor {
	// Check if layer is closed
	if l.closed {
		panic("layer is closed")
	}

	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.gamma
}

// Close releases all resources associated with the layer normalization.
// This includes closing all tensors and cleaning up memory.
func (l *LayerNorm) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.gamma != nil {
		l.gamma.Close()
	}
	l.closed = true
}
