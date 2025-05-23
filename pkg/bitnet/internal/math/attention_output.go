// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

// AttentionOutputProjection represents the output projection layer for multi-head attention.
// This layer projects the concatenated attention outputs from all heads back to the
// model's hidden dimension.
//
// The projection is performed using a linear transformation:
//
//	output = input * W
//
// where W is a [hidden_dim, hidden_dim] weight matrix.
//
// The layer handles both single-token and multi-token cases efficiently,
// with special optimizations for the single-token case to avoid unnecessary
// reshaping operations.
type AttentionOutputProjection struct {
	// Hidden dimension of the model
	hiddenDim int
	// Number of attention heads
	numHeads int
	// Output projection weights [hidden_dim, hidden_dim]
	outProj *tensor.Tensor
}

// NewAttentionOutputProjection creates a new attention output projection layer.
//
// Parameters:
//   - hiddenDim: Size of the hidden dimension
//   - numHeads: Number of attention heads
//
// The projection matrix is initialized as a [hidden_dim, hidden_dim] tensor.
// The layer is optimized for efficient computation with both single-token
// and multi-token inputs.
func NewAttentionOutputProjection(hiddenDim, numHeads int) *AttentionOutputProjection {
	// Create output projection matrix
	outProj := tensor.NewTensor(hiddenDim, hiddenDim)

	return &AttentionOutputProjection{
		hiddenDim: hiddenDim,
		numHeads:  numHeads,
		outProj:   outProj,
	}
}

// Project performs the output projection on the concatenated attention contexts.
//
// Input tensor must be 3D with shape [batch_size, seq_len, num_heads * head_dim].
// The function:
//  1. Reshapes input if needed for efficient computation
//  2. Applies linear projection
//  3. Reshapes output to [batch_size, seq_len, hidden_dim]
//
// Returns a 3D tensor with shape [batch_size, seq_len, hidden_dim].
//
// The function includes special optimizations for single-token inputs
// (batch_size=1, seq_len=1) to avoid unnecessary reshaping operations.
// For multi-token inputs, it uses efficient reshaping and linear projection.
func (out *AttentionOutputProjection) Project(input *tensor.Tensor) (*tensor.Tensor, error) {
	if len(input.Shape()) != 3 {
		return nil, ErrInvalidInputShape
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]
	hiddenIn := input.Shape()[2]
	headDim := hiddenIn / out.numHeads

	loggers.Printf(loggers.Debug, "AttentionOutputProjection input shape: %v", input.Shape())

	flatSize := batchSize * seqLen
	if flatSize*out.numHeads*headDim != len(input.Data()) {
		return nil, ErrInvalidInputShape
	}

	var flatInput *tensor.Tensor
	if batchSize == 1 && seqLen == 1 {
		// Single-token case: manually flatten
		data := input.Data()
		flatInput = tensor.NewTensor(1, out.numHeads*headDim)
		defer flatInput.Close()
		for i := 0; i < out.numHeads*headDim; i++ {
			flatInput.Set(data[i], 0, i)
		}
	} else {
		flatInput = input.Reshape(flatSize, out.numHeads*headDim)
		defer flatInput.Close()
	}

	loggers.Printf(loggers.Debug, "AttentionOutputProjection flat input shape: %v", flatInput.Shape())

	// Apply linear transformation
	output, err := tensor.BitLinear(flatInput, out.outProj)
	if err != nil {
		return nil, err
	}
	defer output.Close()

	if batchSize == 1 && seqLen == 1 {
		// Single-token case: manually reshape
		reshaped := tensor.NewTensor(1, 1, out.hiddenDim)
		outData := output.Data()
		for i := 0; i < out.hiddenDim; i++ {
			reshaped.Set(outData[i], 0, 0, i)
		}
		loggers.Printf(loggers.Debug, "AttentionOutputProjection output shape: %v", reshaped.Shape())
		return reshaped, nil
	}

	reshaped := output.Reshape(batchSize, seqLen, out.hiddenDim)
	loggers.Printf(loggers.Debug, "AttentionOutputProjection output shape: %v", reshaped.Shape())
	return reshaped, nil
}

// SetWeights sets the output projection weights.
//
// Parameters:
//   - weights: Output projection weights [hidden_dim, hidden_dim]
//
// Returns an error if the weights tensor has incorrect dimensions.
// The weights must match the layer's hidden dimension for both input and output.
func (out *AttentionOutputProjection) SetWeights(weights *tensor.Tensor) error {
	if out.outProj == nil {
		panic("projection is closed")
	}
	if weights == nil {
		panic("weights cannot be nil")
	}
	if len(weights.Shape()) != 2 || weights.Shape()[0] != out.hiddenDim || weights.Shape()[1] != out.hiddenDim {
		panic("invalid weights shape")
	}
	out.outProj = weights
	return nil
}

// Close releases all resources associated with the attention output projection.
// This includes closing all tensors and cleaning up memory.
func (out *AttentionOutputProjection) Close() {
	if out.outProj != nil {
		out.outProj.Close()
		out.outProj = nil
	}
}
