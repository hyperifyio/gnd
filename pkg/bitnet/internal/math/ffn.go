// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// FFN represents a two-layer feed-forward network with ReLU² activation.
// This is a key component of the transformer architecture that processes
// each position independently through two linear transformations with
// a non-linear activation in between.
//
// The network consists of:
// 1. An up-projection layer that expands the hidden dimension
// 2. A ReLU² activation function
// 3. A down-projection layer that contracts back to the hidden dimension
//
// The implementation is optimized for parallel processing and includes
// scaling to prevent numerical overflow in the ReLU² activation.
type FFN struct {
	// Hidden dimension of the model
	hiddenDim int
	// Intermediate dimension (typically 4x hidden_dim)
	intermediateDim int
	// First layer weights (up-projection) [intermediate_dim, hidden_dim]
	upProj *tensor.Tensor
	// Second layer weights (down-projection) [hidden_dim, intermediate_dim]
	downProj *tensor.Tensor
}

// NewFFN creates a new feed-forward network instance.
//
// Parameters:
//   - hiddenDim: Size of the hidden dimension
//   - intermediateDim: Size of the intermediate dimension (typically 4x hidden_dim)
//
// The network is initialized with two weight matrices:
// - upProj: [intermediate_dim, hidden_dim] for expansion
// - downProj: [hidden_dim, intermediate_dim] for contraction
func NewFFN(hiddenDim, intermediateDim int) *FFN {
	// Create weight matrices
	upProj := tensor.NewTensor(intermediateDim, hiddenDim)
	downProj := tensor.NewTensor(hiddenDim, intermediateDim)

	return &FFN{
		hiddenDim:       hiddenDim,
		intermediateDim: intermediateDim,
		upProj:          upProj,
		downProj:        downProj,
	}
}

// Forward performs the forward pass through the feed-forward network.
//
// Input tensor must be 3D with shape [batch_size, seq_len, hidden_dim].
// The function:
// 1. Reshapes input for efficient linear projection
// 2. Applies up-projection to expand dimensions
// 3. Applies ReLU² activation with scaling
// 4. Applies down-projection to contract dimensions
// 5. Reshapes output back to original dimensions
//
// Returns a 3D tensor with shape [batch_size, seq_len, hidden_dim].
//
// The implementation uses BitLinear for efficient computation with
// ternary weights and includes parallel processing for the activation.
func (f *FFN) Forward(input *tensor.Tensor) *tensor.Tensor {
	if len(input.Shape()) != 3 {
		panic("input must be 3D tensor [batch_size, seq_len, hidden_dim]")
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]

	// Reshape input for linear projection
	flatInput := input.Reshape(batchSize*seqLen, f.hiddenDim)

	// First linear layer (up-projection)
	intermediate := tensor.BitLinear(flatInput, f.upProj)

	// Apply ReLU² activation
	intermediate = f.applyReLU2(intermediate)

	// Second linear layer (down-projection)
	output := tensor.BitLinear(intermediate, f.downProj)

	// Reshape back to [batch_size, seq_len, hidden_dim]
	return output.Reshape(batchSize, seqLen, f.hiddenDim)
}

// applyReLU2 applies the ReLU² activation function to the intermediate outputs.
//
// Input tensor must be 2D with shape [batch_size * seq_len, intermediate_dim].
// The function:
// 1. Applies ReLU²: max(0, x)²
// 2. Scales down by 16 to prevent overflow
// 3. Clamps values to int8 range
//
// Returns a 2D tensor with shape [batch_size * seq_len, intermediate_dim].
//
// The implementation uses parallel processing with chunked computation
// for better performance on multi-core systems.
func (f *FFN) applyReLU2(input *tensor.Tensor) *tensor.Tensor {
	if len(input.Shape()) != 2 {
		panic("input must be 2D tensor [batch_size * seq_len, intermediate_dim]")
	}

	batchSize := input.Shape()[0]
	intermediateDim := input.Shape()[1]

	// Create output tensor
	output := tensor.NewTensor(batchSize, intermediateDim)

	// Process in parallel chunks
	var wg sync.WaitGroup
	chunkSize := batchSize / runtime.NumCPU()
	if chunkSize < 1 {
		chunkSize = 1
	}

	for i := 0; i < batchSize; i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > batchSize {
				end = batchSize
			}

			// Process each element
			for b := start; b < end; b++ {
				for d := 0; d < intermediateDim; d++ {
					// Get input value
					val := float32(input.Get(b, d))
					// Apply ReLU²: max(0, x)²
					if val > 0 {
						val = val * val
						// Scale down to prevent overflow
						val = val / 16.0
					} else {
						val = 0
					}
					// Convert to int8 with clamping
					if val > 127 {
						val = 127
					} else if val < -128 {
						val = -128
					}
					output.Set(int8(val), b, d)
				}
			}
		}(i)
	}

	wg.Wait()
	return output
}

// SetWeights sets the feed-forward network weights.
//
// Parameters:
//   - upWeights: Up-projection weights [intermediate_dim, hidden_dim]
//   - downWeights: Down-projection weights [hidden_dim, intermediate_dim]
//
// Panics if either weight matrix has incorrect dimensions.
// The weights must match the network's hidden and intermediate dimensions.
func (f *FFN) SetWeights(upWeights, downWeights *tensor.Tensor) {
	if upWeights.Shape()[0] != f.intermediateDim || upWeights.Shape()[1] != f.hiddenDim {
		panic("invalid up-projection weights shape")
	}
	if downWeights.Shape()[0] != f.hiddenDim || downWeights.Shape()[1] != f.intermediateDim {
		panic("invalid down-projection weights shape")
	}

	f.upProj = upWeights
	f.downProj = downWeights
}
