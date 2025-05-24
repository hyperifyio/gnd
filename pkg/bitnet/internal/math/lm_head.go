// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

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
	if len(input.Shape()) != 3 {
		return nil, ErrInvalidInputShape
	}
	if input.Shape()[2] != l.hiddenDim {
		return nil, ErrInvalidInputShape
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]

	var reshaped *tensor.Tensor
	var output *tensor.Tensor
	var err error
	defer func() {
		if r := recover(); r != nil {
			err = ErrLMHeadPanic
			reshaped = nil
			output = nil
		}
	}()

	// Reshape input for linear projection
	flatInput := input.Reshape(batchSize*seqLen, l.hiddenDim)
	defer flatInput.Close()

	// Apply linear transformation
	output, err = tensor.BitLinear(flatInput, l.weights)
	if err != nil {
		return nil, err
	}
	defer output.Close()

	// Reshape back to [batch_size, seq_len, vocab_size]
	reshaped = output.Reshape(batchSize, seqLen, l.vocabSize)
	return reshaped, err
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
	if len(weights.Shape()) != 2 || weights.Shape()[0] != l.vocabSize || weights.Shape()[1] != l.hiddenDim {
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
