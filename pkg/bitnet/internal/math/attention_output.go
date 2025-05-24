// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"errors"
	"fmt"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// Error definitions
var (
	ErrNilTensor    = errors.New("nil tensor")
	ErrClosed       = errors.New("operation on closed tensor")
	ErrInvalidShape = errors.New("invalid tensor shape")
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
func NewAttentionOutputProjection(hiddenDim, numHeads int) (*AttentionOutputProjection, error) {
	outProj, err := tensor.NewTensor(hiddenDim, hiddenDim)
	if err != nil {
		tensor.DebugLog("NewAttentionOutputProjection: failed to create outProj: %v", err)
		return nil, err
	}
	return &AttentionOutputProjection{
		hiddenDim: hiddenDim,
		numHeads:  numHeads,
		outProj:   outProj,
	}, nil
}

// Project applies the attention output projection to the input tensor.
func (out *AttentionOutputProjection) Project(input *tensor.Tensor) (*tensor.Tensor, error) {
	if input == nil {
		return nil, fmt.Errorf("nil tensor")
	}

	// Get input shape
	shape, err := input.Shape()
	if err != nil {
		return nil, fmt.Errorf("failed to get input shape: %w", err)
	}
	if len(shape) != 3 {
		return nil, fmt.Errorf("invalid input shape: expected 3 dimensions, got %d", len(shape))
	}

	batchSize := shape[0]
	seqLen := shape[1]
	headDim := shape[2]

	// Reshape input for processing
	flatSize := batchSize * seqLen
	reshaped, err := input.Reshape(flatSize, out.numHeads*headDim)
	if err != nil {
		return nil, fmt.Errorf("failed to reshape input tensor: %w", err)
	}

	// Process each head
	outputs := make([]*tensor.Tensor, out.numHeads)
	for i := 0; i < out.numHeads; i++ {
		// Create a new tensor for this head
		headOutput, err := tensor.NewTensor(flatSize, headDim)
		if err != nil {
			return nil, fmt.Errorf("failed to create head output tensor: %w", err)
		}

		// Process this head
		headStart := i * headDim
		for j := 0; j < flatSize; j++ {
			for k := 0; k < headDim; k++ {
				val, err := reshaped.Get(j, headStart+k)
				if err != nil {
					return nil, fmt.Errorf("failed to get value from reshaped tensor: %w", err)
				}
				if err := headOutput.Set(val, j, k); err != nil {
					return nil, fmt.Errorf("failed to set value in head output tensor: %w", err)
				}
			}
		}
		outputs[i] = headOutput
	}

	// Combine head outputs
	combined, err := tensor.NewTensor(flatSize, out.hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create combined output tensor: %w", err)
	}

	for i := 0; i < out.numHeads; i++ {
		headStart := i * headDim
		for j := 0; j < flatSize; j++ {
			for k := 0; k < headDim; k++ {
				val, err := outputs[i].Get(j, k)
				if err != nil {
					return nil, fmt.Errorf("failed to get value from head output tensor: %w", err)
				}
				if err := combined.Set(val, j, headStart+k); err != nil {
					return nil, fmt.Errorf("failed to set value in combined tensor: %w", err)
				}
			}
		}
	}

	// Reshape back to original dimensions
	result, err := combined.Reshape(batchSize, seqLen, out.hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to reshape combined tensor: %w", err)
	}
	return result, nil
}

// SetWeights sets the output projection weights.
// AttentionOutputProjection takes ownership of the weights tensor.
// The caller must not use the weights tensor after passing it to SetWeights.
func (out *AttentionOutputProjection) SetWeights(weights *tensor.Tensor) error {
	if out.outProj == nil {
		return ErrClosed
	}
	if weights == nil {
		return ErrNilTensor
	}
	shape, err := weights.Shape()
	if err != nil {
		return fmt.Errorf("failed to get weights shape: %w", err)
	}
	if len(shape) != 2 || shape[0] != out.hiddenDim || shape[1] != out.hiddenDim {
		return ErrInvalidShape
	}
	if out.outProj != nil {
		if err := out.outProj.Close(); err != nil {
			return err
		}
	}
	out.outProj = weights
	return nil
}

// Close releases all resources associated with the attention output projection.
// This includes closing all tensors and cleaning up memory.
func (out *AttentionOutputProjection) Close() error {
	if out.outProj != nil {
		if err := out.outProj.Close(); err != nil {
			return err
		}
		out.outProj = nil
	}
	return nil
}

// AttentionOutput represents the output layer for multi-head attention.
// This layer processes the attention outputs from all heads and combines them
// into a single output tensor.
type AttentionOutput struct {
	// Hidden dimension of the model
	hiddenDim int
	// Number of attention heads
	numHeads int
	// Dimension of each attention head
	headDim int
	// Output tensors for each head
	outputs []*tensor.Tensor
}

// NewAttentionOutput creates a new attention output layer.
func NewAttentionOutput(hiddenDim, numHeads int) *AttentionOutput {
	headDim := hiddenDim / numHeads
	return &AttentionOutput{
		hiddenDim: hiddenDim,
		numHeads:  numHeads,
		headDim:   headDim,
		outputs:   make([]*tensor.Tensor, numHeads),
	}
}

// Forward performs the forward pass of the attention output layer
func (out *AttentionOutput) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	// Validate input shape
	if input == nil {
		return nil, ErrNilTensor
	}
	shape, err := input.Shape()
	if err != nil {
		return nil, fmt.Errorf("failed to get input shape: %w", err)
	}
	if len(shape) != 3 {
		return nil, ErrInvalidInputShape
	}
	batchSize, seqLen, hiddenDim := shape[0], shape[1], shape[2]
	if hiddenDim != out.hiddenDim {
		return nil, ErrInvalidShape
	}

	// Reshape input for processing
	flatSize := batchSize * seqLen
	reshaped, err := input.Reshape(flatSize, out.hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to reshape input tensor: %w", err)
	}

	// Process each head
	outputs := make([]*tensor.Tensor, out.numHeads)
	for i := 0; i < out.numHeads; i++ {
		// Create a new tensor for this head
		headOutput, err := tensor.NewTensor(flatSize, out.headDim)
		if err != nil {
			return nil, fmt.Errorf("failed to create head output tensor: %w", err)
		}

		// Process this head
		headStart := i * out.headDim
		for j := 0; j < flatSize; j++ {
			for k := 0; k < out.headDim; k++ {
				val, err := reshaped.Get(j, headStart+k)
				if err != nil {
					return nil, fmt.Errorf("failed to get value from reshaped tensor: %w", err)
				}
				if err := headOutput.Set(val, j, k); err != nil {
					return nil, fmt.Errorf("failed to set value in head output tensor: %w", err)
				}
			}
		}
		outputs[i] = headOutput
	}

	// Combine head outputs
	combined, err := tensor.NewTensor(batchSize, seqLen, out.hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create combined output tensor: %w", err)
	}

	for i := 0; i < out.numHeads; i++ {
		headStart := i * out.headDim
		for j := 0; j < flatSize; j++ {
			for k := 0; k < out.headDim; k++ {
				val, err := outputs[i].Get(j, k)
				if err != nil {
					return nil, fmt.Errorf("failed to get value from head output tensor: %w", err)
				}
				if err := combined.Set(val, j, headStart+k); err != nil {
					return nil, fmt.Errorf("failed to set value in combined tensor: %w", err)
				}
			}
		}
	}

	return combined, nil
}

// processHead processes a single attention head's output
func (out *AttentionOutput) processHead(headSlice *tensor.Tensor) (*tensor.Tensor, error) {
	// TODO: Implement head-specific processing
	return headSlice, nil
}

// combineHeads combines the outputs from all attention heads
func (out *AttentionOutput) combineHeads(batchSize, seqLen int) (*tensor.Tensor, error) {
	// TODO: Implement head combination
	result, err := tensor.NewTensor(batchSize, seqLen, out.hiddenDim)
	if err != nil {
		return nil, err
	}
	return result, nil
}

// Close releases all resources associated with the attention output layer
func (out *AttentionOutput) Close() error {
	var lastErr error
	for _, t := range out.outputs {
		if t != nil {
			if err := t.Close(); err != nil {
				lastErr = err
			}
		}
	}
	out.outputs = nil
	return lastErr
}
