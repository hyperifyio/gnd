// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// Linear represents a linear transformation layer.
// It performs the operation: output = input * weights
//
// The layer supports both 2D [batch_size, in_dim] and 3D [batch_size, seq_len, in_dim]
// inputs, automatically handling the reshaping required for efficient matrix multiplication.
// The implementation uses BitLinear for efficient computation with ternary weights.
type Linear struct {
	// Input dimension of the layer
	inDim int
	// Output dimension of the layer
	outDim int
	// Weight matrix [out_dim, in_dim]
	weights *tensor.Tensor
}

// NewLinear creates a new linear transformation layer.
//
// Parameters:
//   - inDim: Size of the input dimension
//   - outDim: Size of the output dimension
//
// The layer is initialized with a weight matrix of shape [out_dim, in_dim].
// The weights are used for the linear transformation: output = input * weights.
func NewLinear(inDim, outDim int) *Linear {
	// Create weight matrix
	weights := tensor.NewTensor(outDim, inDim)

	return &Linear{
		inDim:   inDim,
		outDim:  outDim,
		weights: weights,
	}
}

// Forward performs the linear transformation on the input tensor.
//
// Input tensor can be either:
//   - 2D [batch_size, in_dim] for single-token inputs
//   - 3D [batch_size, seq_len, in_dim] for multi-token inputs
//
// The function:
// 1. Validates input shape and dimensions
// 2. Reshapes input to 2D for efficient matrix multiplication
// 3. Performs linear transformation using BitLinear
// 4. Reshapes output back to match input dimensions
//
// Returns a tensor with the same shape as input but with out_dim as the last dimension.
// The implementation handles both single-token and multi-token cases efficiently.
func (l *Linear) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Validate input shape
	if err := ValidateShape(x, 2, 3); err != nil {
		tensor.DebugLog("input shape validation failed: %v", err)
		return nil, ErrLinearInputShape
	}

	// Get input dimensions
	var batchSize, seqLen, inDim int
	if len(x.Shape()) == 2 {
		batchSize, inDim = x.Shape()[0], x.Shape()[1]
		seqLen = 1
	} else {
		batchSize, seqLen, inDim = x.Shape()[0], x.Shape()[1], x.Shape()[2]
	}

	if inDim != l.inDim {
		tensor.DebugLog("input dimension (%d) must match layer input dimension (%d)", inDim, l.inDim)
		return nil, ErrLinearInputDimension
	}

	// Create 2D view of input tensor for matrix multiplication
	input2d := tensor.NewTensor(batchSize*seqLen, inDim)
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < inDim; d++ {
				var val int8
				if len(x.Shape()) == 2 {
					val = x.Get(b, d)
				} else {
					val = x.Get(b, s, d)
				}
				input2d.Set(val, b*seqLen+s, d)
			}
		}
	}

	// Perform linear transformation
	output2d := tensor.BitLinear(input2d, l.weights)

	// Reshape output back to original shape
	if len(x.Shape()) == 2 {
		// For 2D input, output is already 2D
		return output2d, nil
	}

	// For 3D input, reshape output to 3D
	output := tensor.NewTensor(batchSize, seqLen, l.outDim)
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < l.outDim; d++ {
				val := output2d.Get(b*seqLen+s, d)
				output.Set(val, b, s, d)
			}
		}
	}
	return output, nil
}

// SetWeights sets the weight matrix for the linear transformation.
//
// Parameters:
//   - weights: Weight matrix [out_dim, in_dim]
//
// Returns an error if the weights tensor has incorrect shape.
// The weights must match the layer's input and output dimensions.
func (l *Linear) SetWeights(weights *tensor.Tensor) error {
	if len(weights.Shape()) != 2 || weights.Shape()[0] != l.outDim || weights.Shape()[1] != l.inDim {
		tensor.DebugLog("weights must be 2D tensor with shape [%d, %d], got %v", l.outDim, l.inDim, weights.Shape())
		return ErrLinearWeightsShape
	}
	l.weights = weights
	return nil
}

// GetWeights returns the current weight matrix.
//
// Returns the weight tensor with shape [out_dim, in_dim].
// This is the matrix used for the linear transformation.
func (l *Linear) GetWeights() *tensor.Tensor {
	return l.weights
}
