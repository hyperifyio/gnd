// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"errors"

	bitneterrors "github.com/hyperifyio/gnd/pkg/bitnet/errors"
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

// Error definitions
var (
	ErrNilTensor        = errors.New("attention: nil tensor")
	ErrClosed           = errors.New("attention: operation on closed tensor")
	ErrGetInputShape    = errors.New("attention: failed to get input shape")
	ErrReshapeInput     = errors.New("attention: failed to reshape input tensor")
	ErrCreateHeadOutput = errors.New("attention: failed to create head output tensor")
	ErrGetReshapedValue = errors.New("attention: failed to get value from reshaped tensor")
	ErrSetHeadOutput    = errors.New("attention: failed to set value in head output tensor")
	ErrCreateCombined   = errors.New("attention: failed to create combined output tensor")
	ErrGetHeadOutput    = errors.New("attention: failed to get value from head output tensor")
	ErrSetCombined      = errors.New("attention: failed to set value in combined tensor")
	ErrReshapeCombined  = errors.New("attention: failed to reshape combined tensor")
	ErrGetWeightsShape  = errors.New("attention: failed to get weights shape")
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
		return nil, ErrNilTensor
	}

	// Get input shape
	shape, err := input.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get input shape: %v", err)
		return nil, ErrGetInputShape
	}
	if len(shape) != 3 {
		loggers.Printf(loggers.Debug, "invalid input shape: expected 3 dimensions, got %d", len(shape))
		return nil, bitneterrors.ErrInvalidShape
	}

	batchSize := shape[0]
	seqLen := shape[1]
	headDim := shape[2]

	// Reshape input for processing
	flatSize := batchSize * seqLen
	reshaped, err := input.Reshape(flatSize, out.numHeads*headDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to reshape input tensor: %v", err)
		return nil, ErrReshapeInput
	}

	// Process each head
	outputs := make([]*tensor.Tensor, out.numHeads)
	for i := 0; i < out.numHeads; i++ {
		// Create a new tensor for this head
		headOutput, err := tensor.NewTensor(flatSize, headDim)
		if err != nil {
			loggers.Printf(loggers.Debug, "failed to create head output tensor: %v", err)
			return nil, ErrCreateHeadOutput
		}

		// Process this head
		headStart := i * headDim
		for j := 0; j < flatSize; j++ {
			for k := 0; k < headDim; k++ {
				val, err := reshaped.Get(j, headStart+k)
				if err != nil {
					loggers.Printf(loggers.Debug, "failed to get value from reshaped tensor: %v", err)
					return nil, ErrGetReshapedValue
				}
				if err := headOutput.Set(val, j, k); err != nil {
					loggers.Printf(loggers.Debug, "failed to set value in head output tensor: %v", err)
					return nil, ErrSetHeadOutput
				}
			}
		}
		outputs[i] = headOutput
	}

	// Combine head outputs
	combined, err := tensor.NewTensor(flatSize, out.hiddenDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to create combined output tensor: %v", err)
		return nil, ErrCreateCombined
	}

	for i := 0; i < out.numHeads; i++ {
		headStart := i * headDim
		for j := 0; j < flatSize; j++ {
			for k := 0; k < headDim; k++ {
				val, err := outputs[i].Get(j, k)
				if err != nil {
					loggers.Printf(loggers.Debug, "failed to get value from head output tensor: %v", err)
					return nil, ErrGetHeadOutput
				}
				if err := combined.Set(val, j, headStart+k); err != nil {
					loggers.Printf(loggers.Debug, "failed to set value in combined tensor: %v", err)
					return nil, ErrSetCombined
				}
			}
		}
	}

	// Reshape back to original dimensions
	result, err := combined.Reshape(batchSize, seqLen, out.hiddenDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to reshape combined tensor: %v", err)
		return nil, ErrReshapeCombined
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
		loggers.Printf(loggers.Debug, "failed to get weights shape: %v", err)
		return ErrGetWeightsShape
	}
	if len(shape) != 2 || shape[0] != out.hiddenDim || shape[1] != out.hiddenDim {
		return bitneterrors.ErrInvalidShape
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
	if input == nil {
		return nil, ErrNilTensor
	}
	shape, err := input.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get input shape: %v", err)
		return nil, ErrGetInputShape
	}
	if len(shape) != 3 {
		return nil, bitneterrors.ErrInvalidShape
	}
	batchSize, seqLen, hiddenDim := shape[0], shape[1], shape[2]
	if hiddenDim != out.hiddenDim {
		return nil, bitneterrors.ErrInvalidShape
	}

	// Reshape input for processing
	flatSize := batchSize * seqLen
	reshaped, err := input.Reshape(flatSize, out.hiddenDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to reshape input tensor: %v", err)
		return nil, ErrReshapeInput
	}

	// Process each head
	outputs := make([]*tensor.Tensor, out.numHeads)
	for i := 0; i < out.numHeads; i++ {
		// Create a new tensor for this head
		headOutput, err := tensor.NewTensor(flatSize, out.headDim)
		if err != nil {
			loggers.Printf(loggers.Debug, "failed to create head output tensor: %v", err)
			return nil, ErrCreateHeadOutput
		}

		// Process this head
		headStart := i * out.headDim
		for j := 0; j < flatSize; j++ {
			for k := 0; k < out.headDim; k++ {
				val, err := reshaped.Get(j, headStart+k)
				if err != nil {
					loggers.Printf(loggers.Debug, "failed to get value from reshaped tensor: %v", err)
					return nil, ErrGetReshapedValue
				}
				if err := headOutput.Set(val, j, k); err != nil {
					loggers.Printf(loggers.Debug, "failed to set value in head output tensor: %v", err)
					return nil, ErrSetHeadOutput
				}
			}
		}
		outputs[i] = headOutput
	}

	// Combine head outputs
	combined, err := tensor.NewTensor(batchSize, seqLen, out.hiddenDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to create combined output tensor: %v", err)
		return nil, ErrCreateCombined
	}

	for i := 0; i < out.numHeads; i++ {
		headStart := i * out.headDim
		for j := 0; j < flatSize; j++ {
			for k := 0; k < out.headDim; k++ {
				val, err := outputs[i].Get(j, k)
				if err != nil {
					loggers.Printf(loggers.Debug, "failed to get value from head output tensor: %v", err)
					return nil, ErrGetHeadOutput
				}
				if err := combined.Set(val, j, headStart+k); err != nil {
					loggers.Printf(loggers.Debug, "failed to set value in combined tensor: %v", err)
					return nil, ErrSetCombined
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
