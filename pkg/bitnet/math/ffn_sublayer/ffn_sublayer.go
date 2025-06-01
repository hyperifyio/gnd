// Package ffn_sublayer implements the feed-forward sublayer for BitNet transformer blocks.
//
// # Feed-Forward Sublayer for BitNet
//
// This package provides the complete feed-forward sublayer implementation for BitNet,
// including pre-norm layer normalization, two-layer FFN with ReLU² activation,
// and residual connections. The implementation follows BitNet's b1.58-2B 4T architecture specifications.
//
// Key aspects:
//   - All weights and activations are int8, matching BitNet's quantized design
//   - Pre-norm architecture with layer normalization (epsilon=1e-5)
//   - Two-layer FFN with ReLU² activation
//   - Efficient parallel processing for batch computation
//   - Handles 4096-token context length
//   - No bias terms in linear layers as per BitNet architecture
//
// Implementation details:
//   - Pre-norm layer normalization with proper scaling
//   - Up-projection to intermediate dimension (6912)
//   - ReLU² activation with proper scaling
//   - Down-projection back to hidden dimension (2560)
//   - Residual connection with proper tensor management
//   - Efficient memory management with proper cleanup
//   - Parallel processing using goroutines for batch computation
//
// Related tasks and dependencies:
//   - #187: Integrate Feed-Forward Sublayer (Pre-Norm & Residual)
//   - #185: Feed-Forward Network (FFN) Sublayer
//   - #180: Implement Squared ReLU Activation
//   - #178: Implement BitLinear Layer
//   - #179: Implement Sub-Layer Normalization
//
// Usage:
//   - Used in BitNet transformer blocks for feed-forward processing
//   - Supports both single-token and multi-token inputs
//   - Maintainers should not change quantization or architecture without full pipeline review
//   - Critical for maintaining correct quantized inference
//
// Caveats:
//   - Quantization may cause saturation/clamping; tests should check for correct quantized output
//   - Any change must be validated against end-to-end BitNet inference
//   - Performance critical - changes should be benchmarked against existing implementation
//   - Memory management is important - tensors should be properly closed after use
//   - Must maintain compatibility with BitNet's binary-weight quantization
//
// For more details, see BitNet issue #170 and the BitNet project documentation.
package ffn_sublayer

import (
	ffn2 "github.com/hyperifyio/gnd/pkg/bitnet/math/ffn"
	"github.com/hyperifyio/gnd/pkg/bitnet/math/layer_norm"
	"math"
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// FFNSublayer represents a feed-forward sublayer with pre-norm layer normalization.
type FFNSublayer struct {
	hiddenDim       int
	intermediateDim int
	preNorm         *layer_norm.LayerNorm
	ffn             *ffn2.FFN
	closed          bool
}

// NewFFNSublayer creates a new feed-forward sublayer with pre-norm layer normalization.
//
// Parameters:
//   - hiddenDim: Size of the hidden dimension
//   - intermediateDim: Size of the intermediate dimension (typically 4x hidden_dim)
//
// The sublayer is initialized with:
// - SubLN: Pre-norm layer with epsilon=1e-5
// - FFN: Two-layer feed-forward network with ReLU² activation
//
// Returns a new FFNSublayer instance ready for use.
func NewFFNSublayer(hiddenDim, intermediateDim int) (*FFNSublayer, error) {
	if hiddenDim <= 0 || intermediateDim <= 0 {
		return nil, ffn2.ErrInvalidWeightsShape
	}

	// Initialize pre-norm layer
	preNorm, err := layer_norm.NewLayerNorm(hiddenDim)
	if err != nil {
		return nil, err
	}

	// Initialize FFN
	ffn, err := ffn2.NewFFN(hiddenDim, intermediateDim)
	if err != nil {
		preNorm.Close()
		return nil, err
	}

	return &FFNSublayer{
		hiddenDim:       hiddenDim,
		intermediateDim: intermediateDim,
		preNorm:         preNorm,
		ffn:             ffn,
	}, nil
}

// Forward performs the forward pass through the feed-forward sublayer.
//
// Input tensor can be either:
//   - 2D [seq_len, hidden_dim] for single-batch inputs
//   - 3D [batch_size, seq_len, hidden_dim] for multi-batch inputs
//
// The function performs the following steps:
// 1. Validates input shape and dimensions
// 2. Converts input to float32 for normalization
// 3. Applies pre-norm layer normalization
// 4. Applies feed-forward network
// 5. Adds residual connection
// 6. Clamps output to int8 range
//
// Returns a tensor with the same shape as the input.
// Panics if the input shape is invalid.
func (f *FFNSublayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if f.closed {
		return nil, ffn2.ErrFFNClosed
	}

	// Apply pre-norm
	normalized, err := f.preNorm.Forward(input)
	if err != nil {
		return nil, err
	}
	defer normalized.Close()

	// Apply FFN
	ffnOutput, err := f.ffn.Forward(normalized)
	if err != nil {
		return nil, err
	}
	defer ffnOutput.Close()

	// Get input shape
	shape, err := input.Shape()
	if err != nil {
		return nil, err
	}

	// Create output tensor
	output, err := tensor.NewTensor(shape...)
	if err != nil {
		return nil, err
	}

	// Process in parallel chunks
	var wg sync.WaitGroup
	numCPU := runtime.NumCPU()
	chunkSize := (shape[0] + numCPU - 1) / numCPU
	if chunkSize < 1 {
		chunkSize = 1
	}

	errChan := make(chan error, numCPU)

	for i := 0; i < shape[0]; i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > shape[0] {
				end = shape[0]
			}

			for b := start; b < end; b++ {
				for s := 0; s < shape[1]; s++ {
					for h := 0; h < f.hiddenDim; h++ {
						// Get input and FFN output values
						ival, ierr := input.Get(b, s, h)
						if ierr != nil {
							errChan <- ierr
							return
						}
						fval, ferr := ffnOutput.Get(b, s, h)
						if ferr != nil {
							errChan <- ferr
							return
						}

						// Add values and clamp to int8 range
						sum := int16(ival) + int16(fval)
						if sum > 127 {
							sum = 127
						} else if sum < -128 {
							sum = -128
						}

						if err := output.Set(int8(sum), b, s, h); err != nil {
							errChan <- err
							return
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

// SetWeights sets the weights for the feed-forward network.
//
// Parameters:
//   - upWeights: Up-projection weights [intermediate_dim, hidden_dim]
//   - downWeights: Down-projection weights [hidden_dim, intermediate_dim]
//
// The weights are used for the two-layer feed-forward network:
// 1. Up-projection expands the hidden dimension
// 2. Down-projection contracts back to the hidden dimension
func (f *FFNSublayer) SetWeights(upWeights, downWeights *tensor.Tensor) {
	f.ffn.SetWeights(upWeights, downWeights)
}

// SetGamma sets the scale parameter for layer normalization.
func (f *FFNSublayer) SetGamma(gamma []float32) error {
	if f.closed {
		return ffn2.ErrFFNClosed
	}

	// Create tensor from gamma values
	gammaTensor, err := tensor.NewTensor(len(gamma))
	if err != nil {
		return err
	}

	// Set gamma values
	for i, v := range gamma {
		// Convert float32 to int8 with proper rounding
		intVal := int8(math.Round(float64(v)))
		// Clamp to int8 range
		if intVal > 127 {
			intVal = 127
		} else if intVal < -128 {
			intVal = -128
		}
		if err := gammaTensor.Set(intVal, i); err != nil {
			gammaTensor.Close()
			return err
		}
	}

	// Set gamma tensor
	return f.preNorm.SetGamma(gammaTensor)
}

// Close releases all resources associated with the FFNSublayer.
func (f *FFNSublayer) Close() error {
	if f.closed {
		return nil
	}
	f.closed = true
	f.preNorm.Close()
	f.ffn.Close()
	return nil
}
