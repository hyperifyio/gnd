// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"errors"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

// DebugLog logs debug information with formatting.
// Used for internal debugging and diagnostics in the math package.
func DebugLog(format string, args ...interface{}) {
	loggers.Printf(loggers.Debug, format, args...)
}

var (
	// ErrLMHeadPanic is returned when a panic occurs in the LMHead.Forward method
	ErrLMHeadPanic = errors.New("lmhead: panic in forward pass")
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
func NewLMHead(hiddenDim, vocabSize int) *LMHead {
	if hiddenDim <= 0 {
		panic("hiddenDim must be positive")
	}
	if vocabSize <= 0 {
		panic("vocabSize must be positive")
	}
	return &LMHead{
		hiddenDim: hiddenDim,
		vocabSize: vocabSize,
	}
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
		panic("LMHead has been closed")
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
			DebugLog("panic in LMHead.Forward: %v", r)
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
		panic("LMHead has been closed")
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
func (l *LMHead) GetWeights() *tensor.Tensor {
	if l.closed {
		panic("LMHead has been closed")
	}
	return l.weights
}

// Close releases all resources associated with the layer.
func (l *LMHead) Close() {
	if !l.closed {
		if l.weights != nil {
			l.weights.Close()
		}
		l.closed = true
	}
}
