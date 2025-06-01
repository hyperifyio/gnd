// Package lm_head implements the quantized language model (LM) head for BitNet inference.
//
// # Quantized LM Head for BitNet
//
// This package provides the final output layer, projecting hidden states to logits using int8 weights.
// It implements the output layer described in the BitNet paper (https://arxiv.org/abs/2310.11453).
//
// Key aspects:
//   - All weights and activations are int8, matching BitNet's quantized design
//   - No bias is used, as per BitNet architecture
//   - Optimized for CPU efficiency and low memory use
//   - Not suitable for training or float32 inference
//
// Implementation details:
//   - Linear projection from hidden dimension to vocabulary size
//   - Uses transposed embedding weights for efficiency
//   - Efficient batch processing with proper reshaping
//   - Proper tensor cleanup and resource management
//
// Related tasks and dependencies:
//   - #189: Final Output Layer (LM Head) (Core implementation)
//   - #178: Implement BitLinear Layer (Required by #189)
//   - #190: Token Decoding (Inference Loop) (Depends on #189)
//   - #188: Stack Transformer Blocks (Required by #189)
//
// Usage:
//   - Used as the final output layer in BitNet inference
//   - Supports both single-token and multi-token inputs
//   - Maintainers should not change quantization or projection logic without full pipeline review
//
// Caveats:
//   - Quantization may cause saturation/clamping; tests should check for correct quantized output
//   - Any change must be validated against end-to-end BitNet inference
//   - Performance critical - changes should be benchmarked against existing implementation
//   - Memory management is important - tensors should be properly closed after use
//
// For more details, see BitNet issue #190 and the BitNet project documentation.
package lm_head

import (
	"errors"
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

var (
	// ErrLMHeadPanic is returned when a panic occurs in the LMHead.Forward method
	ErrLMHeadPanic = errors.New("lmhead: panic in forward pass")
	// ErrLMHeadClosed is returned when operations are performed on a closed LMHead
	ErrLMHeadClosed = errors.New("lmhead: operation called on closed layer")
	// ErrLMHeadInvalidParams is returned when invalid parameters are provided to NewLMHead
	ErrLMHeadInvalidParams = errors.New("lmhead: invalid parameters")

	ErrInvalidInputShape = errors.New("lm_head: invalid input shape")
	ErrWeightsNotSet     = errors.New("lm_head: weights not set")
	ErrWeightsShape      = errors.New("lm_head: invalid weights shape")
)

// LMHead represents the final output layer of the BitNet model.
// It produces logits for each token in the vocabulary by applying
// a linear transformation using the transposed embedding weights.
//
// The layer:
// 1. Takes hidden states as input (8-bit)
// 2. Uses transposed embedding weights (ternary)
// 3. Produces logits for each token in the vocabulary
// 4. No bias is used
type LMHead struct {
	// Hidden dimension of the model
	hiddenDim int
	// Vocabulary size
	vocabSize int
	// Transposed embedding weights [vocab_size, hidden_dim]
	weights *tensor.Tensor
	// Flag indicating if the layer has been closed
	closed bool
}

// NewLMHead creates a new LM Head layer.
//
// Parameters:
//   - hiddenDim: Size of the hidden dimension
//   - vocabSize: Size of the vocabulary
//
// The layer is initialized with nil weights, which must be set
// using SetWeights before use.
func NewLMHead(hiddenDim, vocabSize int) (*LMHead, error) {
	if hiddenDim <= 0 {
		return nil, ErrLMHeadInvalidParams
	}
	if vocabSize <= 0 {
		return nil, ErrLMHeadInvalidParams
	}
	return &LMHead{
		hiddenDim: hiddenDim,
		vocabSize: vocabSize,
	}, nil
}

// Forward performs the forward pass through the LM Head layer.
//
// Input tensor must be 3D with shape [batch_size, seq_len, hidden_dim].
// The function:
// 1. Reshapes input for efficient linear projection
// 2. Applies linear transformation using transposed embedding weights
// 3. Reshapes output back to original dimensions
//
// Returns a 3D tensor with shape [batch_size, seq_len, vocab_size].
func (l *LMHead) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if l.closed {
		return nil, ErrLMHeadClosed
	}
	if l.weights == nil {
		return nil, ErrWeightsNotSet
	}
	shape, err := input.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 3 {
		return nil, ErrInvalidInputShape
	}
	if shape[2] != l.hiddenDim {
		return nil, ErrInvalidInputShape
	}

	batchSize := shape[0]
	seqLen := shape[1]

	var reshaped *tensor.Tensor
	var output *tensor.Tensor
	defer func() {
		if r := recover(); r != nil {
			err = ErrLMHeadPanic
			reshaped = nil
			output = nil
		}
	}()

	// Reshape input for linear projection
	flatInput, err := input.Reshape(batchSize*seqLen, l.hiddenDim)
	if err != nil {
		return nil, err
	}
	defer flatInput.Close()

	// Apply linear transformation
	output, err = tensor.BitLinear(flatInput, l.weights)
	if err != nil {
		return nil, err
	}
	defer output.Close()

	// Reshape back to [batch_size, seq_len, vocab_size]
	reshaped, err = output.Reshape(batchSize, seqLen, l.vocabSize)
	if err != nil {
		return nil, err
	}
	return reshaped, nil
}

// SetWeights sets the transposed embedding weights for the layer.
//
// Parameters:
//   - weights: Transposed embedding weights [vocab_size, hidden_dim]
//
// Returns an error if the weights tensor has incorrect shape.
func (l *LMHead) SetWeights(weights *tensor.Tensor) error {
	if l.closed {
		return ErrLMHeadClosed
	}
	if weights == nil {
		return ErrWeightsNotSet
	}
	shape, err := weights.Shape()
	if err != nil {
		return err
	}
	if len(shape) != 2 || shape[0] != l.vocabSize || shape[1] != l.hiddenDim {
		return ErrWeightsShape
	}
	l.weights = weights
	return nil
}

// GetWeights returns the current weights.
//
// Returns the weight tensor with shape [vocab_size, hidden_dim].
func (l *LMHead) GetWeights() (*tensor.Tensor, error) {
	if l.closed {
		return nil, ErrLMHeadClosed
	}
	return l.weights, nil
}

// Close releases all resources associated with the layer.
func (l *LMHead) Close() error {
	if !l.closed {
		if l.weights != nil {
			if err := l.weights.Close(); err != nil {
				return err
			}
		}
		l.closed = true
	}
	return nil
}
