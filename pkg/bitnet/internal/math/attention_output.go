package math

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// AttentionOutputProjection represents the output projection layer for multi-head attention
type AttentionOutputProjection struct {
	// Hidden dimension
	hiddenDim int
	// Number of attention heads
	numHeads int
	// Output projection weights
	outProj *tensor.Tensor
}

// NewAttentionOutputProjection creates a new attention output projection layer
func NewAttentionOutputProjection(hiddenDim, numHeads int) *AttentionOutputProjection {
	// Create output projection matrix
	outProj := tensor.NewTensor(hiddenDim, hiddenDim)

	return &AttentionOutputProjection{
		hiddenDim: hiddenDim,
		numHeads:  numHeads,
		outProj:   outProj,
	}
}

// Project performs the output projection on the concatenated attention contexts
// input: [batch_size, seq_len, num_heads * head_dim]
// Returns: [batch_size, seq_len, hidden_dim]
func (out *AttentionOutputProjection) Project(input *tensor.Tensor) *tensor.Tensor {
	if len(input.Shape()) != 3 {
		panic("input must be 3D tensor [batch_size, seq_len, num_heads * head_dim]")
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]
	headDim := input.Shape()[2] / out.numHeads

	// Reshape input for linear projection
	flatInput := input.Reshape(batchSize*seqLen, out.numHeads*headDim)

	// Apply output projection
	output := tensor.BitLinear(flatInput, out.outProj)

	// Reshape back to [batch_size, seq_len, hidden_dim]
	return output.Reshape(batchSize, seqLen, out.hiddenDim)
}

// SetWeights sets the output projection weights
func (out *AttentionOutputProjection) SetWeights(weights *tensor.Tensor) {
	if weights.Shape()[0] != out.hiddenDim || weights.Shape()[1] != out.hiddenDim {
		panic("invalid output projection weights shape")
	}
	out.outProj = weights
}
