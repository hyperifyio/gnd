// Package math implements mathematical operations for the BitNet model.
package math

import (
	"fmt"
	"os"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
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
func (out *AttentionOutputProjection) Project(input *tensor.Tensor) *tensor.Tensor {
	if len(input.Shape()) != 3 {
		panic("input must be 3D tensor [batch_size, seq_len, num_heads * head_dim]")
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]
	hiddenIn := input.Shape()[2]
	headDim := hiddenIn / out.numHeads

	fmt.Fprintf(os.Stderr, "[DEBUG] AttentionOutputProjection input shape: %v\n", input.Shape())

	flatSize := batchSize * seqLen
	if flatSize*out.numHeads*headDim != len(input.Data()) {
		panic("AttentionOutputProjection: shape product does not match data length")
	}

	var flatInput *tensor.Tensor
	if batchSize == 1 && seqLen == 1 {
		// Single-token case: manually flatten
		data := input.Data()
		flatInput = tensor.NewTensor(1, out.numHeads*headDim)
		for i := 0; i < out.numHeads*headDim; i++ {
			flatInput.Set(data[i], 0, i)
		}
	} else {
		flatInput = input.Reshape(flatSize, out.numHeads*headDim)
	}

	fmt.Fprintf(os.Stderr, "[DEBUG] AttentionOutputProjection flat input shape: %v\n", flatInput.Shape())

	output := tensor.BitLinear(flatInput, out.outProj)

	if batchSize == 1 && seqLen == 1 {
		// Single-token case: manually reshape
		reshaped := tensor.NewTensor(1, 1, out.hiddenDim)
		outData := output.Data()
		for i := 0; i < out.hiddenDim; i++ {
			reshaped.Set(outData[i], 0, 0, i)
		}
		fmt.Fprintf(os.Stderr, "[DEBUG] AttentionOutputProjection output shape: %v\n", reshaped.Shape())
		return reshaped
	}

	output = output.Reshape(batchSize, seqLen, out.hiddenDim)
	fmt.Fprintf(os.Stderr, "[DEBUG] AttentionOutputProjection output shape: %v\n", output.Shape())
	return output
}

// SetWeights sets the output projection weights.
//
// Parameters:
//   - weights: Output projection weights [hidden_dim, hidden_dim]
//
// Panics if the weights tensor has incorrect dimensions.
func (out *AttentionOutputProjection) SetWeights(weights *tensor.Tensor) {
	if weights.Shape()[0] != out.hiddenDim || weights.Shape()[1] != out.hiddenDim {
		panic("invalid output projection weights shape")
	}
	out.outProj = weights
}
