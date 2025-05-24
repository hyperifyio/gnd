// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// Linear represents a linear transformation layer.
// It applies a weight matrix to the input tensor.
type Linear struct {
	inDim   int
	outDim  int
	weights *tensor.Tensor
	closed  bool
}

// NewLinear creates a new linear layer with the given input and output dimensions.
func NewLinear(inDim, outDim int) *Linear {
	return &Linear{
		inDim:  inDim,
		outDim: outDim,
	}
}

// Forward applies the linear transformation to the input tensor.
// Returns a tensor with the same shape as input but with out_dim as the last dimension.
// The implementation handles both single-token and multi-token cases efficiently.
func (l *Linear) Forward(x tensor.TensorReader) (*tensor.Tensor, error) {
	if l.closed {
		panic("Linear layer has been closed")
	}

	// Convert to concrete type for validation
	t, ok := x.(*tensor.Tensor)
	if !ok {
		return nil, ErrLinearInputShape
	}

	// Validate input shape
	if err := ValidateShape(t, 2, 3); err != nil {
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
	defer input2d.Close()

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

	// Apply linear transformation
	output2d, err := tensor.BitLinear(input2d, l.weights)
	if err != nil {
		return nil, err
	}
	defer output2d.Close()

	// Create output tensor with correct shape
	var output *tensor.Tensor
	if len(x.Shape()) == 2 {
		output = tensor.NewTensor(batchSize, l.outDim)
	} else {
		output = tensor.NewTensor(batchSize, seqLen, l.outDim)
	}

	// Copy data from output2d to output
	if len(x.Shape()) == 2 {
		// Input was 2D, output should be 2D
		for b := 0; b < batchSize; b++ {
			for d := 0; d < l.outDim; d++ {
				output.Set(output2d.Get(b, d), b, d)
			}
		}
	} else {
		// Input was 3D, output should be 3D
		for b := 0; b < batchSize; b++ {
			for s := 0; s < seqLen; s++ {
				for d := 0; d < l.outDim; d++ {
					val := output2d.Get(b*seqLen+s, d)
					output.Set(val, b, s, d)
				}
			}
		}
	}

	return output, nil
}

// SetWeights sets the weight matrix for the linear transformation.
// Linear takes ownership of the weights tensor and will close it when Linear is closed.
// The caller must not close the tensor after passing it to SetWeights.
func (l *Linear) SetWeights(weights tensor.TensorOperations) error {
	if l.closed {
		panic("Linear layer has been closed")
	}
	if weights == nil {
		return ErrLinearWeightsShape
	}
	if len(weights.Shape()) != 2 || weights.Shape()[0] != l.outDim || weights.Shape()[1] != l.inDim {
		tensor.DebugLog("weights must be 2D tensor with shape [%d, %d], got %v", l.outDim, l.inDim, weights.Shape())
		return ErrLinearWeightsShape
	}
	if l.weights != nil {
		l.weights.Close()
	}
	l.weights = weights.(*tensor.Tensor)
	return nil
}

// GetWeights returns the current weight matrix.
//
// Returns the weight tensor with shape [out_dim, in_dim].
// This is the matrix used for the linear transformation.
func (l *Linear) GetWeights() tensor.TensorReader {
	if l.closed {
		panic("Linear layer has been closed")
	}
	return l.weights
}

// Close releases all resources associated with the linear layer.
// This includes closing all tensors and cleaning up memory.
func (l *Linear) Close() {
	if !l.closed {
		if l.weights != nil {
			l.weights.Close()
		}
		l.closed = true
	}
}
