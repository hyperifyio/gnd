// Package math implements mathematical operations for the BitNet model.
package math

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// Linear represents a linear transformation layer.
// It performs the operation: output = input * weights
type Linear struct {
	// Input dimension
	inDim int
	// Output dimension
	outDim int
	// Weight matrix [out_dim, in_dim]
	weights *tensor.Tensor
}

// NewLinear creates a new linear layer.
func NewLinear(inDim, outDim int) *Linear {
	// Create weight matrix
	weights := tensor.NewTensor(outDim, inDim)

	return &Linear{
		inDim:   inDim,
		outDim:  outDim,
		weights: weights,
	}
}

// Forward performs the linear transformation.
// Input tensor can be either:
//   - 2D [batch_size, in_dim]
//   - 3D [batch_size, seq_len, in_dim]
//
// Returns a tensor with the same shape as input but with out_dim as the last dimension.
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

// SetWeights sets the weight matrix.
func (l *Linear) SetWeights(weights *tensor.Tensor) error {
	if len(weights.Shape()) != 2 || weights.Shape()[0] != l.outDim || weights.Shape()[1] != l.inDim {
		tensor.DebugLog("weights must be 2D tensor with shape [%d, %d], got %v", l.outDim, l.inDim, weights.Shape())
		return ErrLinearWeightsShape
	}
	l.weights = weights
	return nil
}

// GetWeights returns the current weight matrix.
func (l *Linear) GetWeights() *tensor.Tensor {
	return l.weights
}
