// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// FFNSublayer implements the feed-forward sublayer with pre-norm and residual connection.
// It is a key component of the transformer architecture that processes each position
// independently through a feed-forward network after normalization.
//
// The sublayer consists of:
// 1. Pre-norm layer normalization
// 2. Two-layer feed-forward network with ReLU² activation
// 3. Residual connection
//
// The implementation supports both 2D [seq_len, hidden_dim] and 3D [batch_size, seq_len, hidden_dim]
// inputs, with automatic shape detection and appropriate processing.
type FFNSublayer struct {
	// Sub-layer normalization for pre-norm
	subln *SubLN
	// Feed-forward network for position-wise processing
	ffn *FFN
	// Hidden dimension of the model
	hiddenDim int
	// Intermediate dimension (typically 4x hidden_dim)
	intermediateDim int
}

// NewFFNSublayer creates a new feed-forward sublayer instance.
//
// Parameters:
//   - hiddenDim: Size of the hidden dimension
//   - intermediateDim: Size of the intermediate dimension (typically 4x hidden_dim)
//
// The sublayer is initialized with:
// - SubLN: Pre-norm layer with epsilon=1e-5
// - FFN: Two-layer feed-forward network with ReLU² activation
//
// Returns a new FFNSublayer instance ready for use.
func NewFFNSublayer(hiddenDim, intermediateDim int) *FFNSublayer {
	return &FFNSublayer{
		subln:           NewSubLN(hiddenDim, 1e-5),
		ffn:             NewFFN(hiddenDim, intermediateDim),
		hiddenDim:       hiddenDim,
		intermediateDim: intermediateDim,
	}
}

// Forward performs the forward pass through the feed-forward sublayer.
//
// Input tensor can be either:
//   - 2D [seq_len, hidden_dim] for single-batch inputs
//   - 3D [batch_size, seq_len, hidden_dim] for multi-batch inputs
//
// The function performs the following steps:
// 1. Validates input shape and dimensions
// 2. Converts input to float32 for normalization
// 3. Applies pre-norm layer normalization
// 4. Applies feed-forward network
// 5. Adds residual connection
// 6. Clamps output to int8 range
//
// Returns a tensor with the same shape as the input.
// Panics if the input shape is invalid.
func (f *FFNSublayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	// Get input dimensions
	var batchSize, seqLen, hiddenDim int
	if len(input.Shape()) == 2 {
		// [seq_len, hidden_dim]
		seqLen, hiddenDim = input.Shape()[0], input.Shape()[1]
		batchSize = 1
	} else if len(input.Shape()) == 3 {
		// [batch_size, seq_len, hidden_dim]
		batchSize, seqLen, hiddenDim = input.Shape()[0], input.Shape()[1], input.Shape()[2]
	} else {
		return nil, ErrInvalidInputShape
	}

	if hiddenDim != f.hiddenDim {
		return nil, ErrHiddenDimMismatch
	}

	// Convert input to float32 for normalization
	inputFloat := make([][]float32, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			idx := i*seqLen + j
			inputFloat[idx] = make([]float32, hiddenDim)
			for k := 0; k < hiddenDim; k++ {
				var val int8
				if len(input.Shape()) == 2 {
					val = input.Get(j, k)
				} else {
					val = input.Get(i, j, k)
				}
				inputFloat[idx][k] = float32(val)
			}
		}
	}

	// Apply pre-norm
	normalized := f.subln.Normalize(inputFloat)

	// Reshape normalized output back to tensor
	var normalizedTensor *tensor.Tensor
	if len(input.Shape()) == 2 {
		normalizedTensor = tensor.NewTensor(seqLen, hiddenDim)
		for j := 0; j < seqLen; j++ {
			for k := 0; k < hiddenDim; k++ {
				normalizedTensor.Set(int8(normalized[j][k]), j, k)
			}
		}
	} else {
		normalizedTensor = tensor.NewTensor(batchSize, seqLen, hiddenDim)
		for i := 0; i < batchSize; i++ {
			for j := 0; j < seqLen; j++ {
				idx := i*seqLen + j
				for k := 0; k < hiddenDim; k++ {
					normalizedTensor.Set(int8(normalized[idx][k]), i, j, k)
				}
			}
		}
	}
	defer normalizedTensor.Close()

	// Apply feed-forward network
	ffnOutput, err := f.ffn.Forward(normalizedTensor)
	if err != nil {
		return nil, err
	}
	defer ffnOutput.Close()

	// Add residual connection
	var result *tensor.Tensor
	if len(input.Shape()) == 2 {
		result = tensor.NewTensor(seqLen, hiddenDim)
		for j := 0; j < seqLen; j++ {
			for k := 0; k < hiddenDim; k++ {
				// Get input value
				inputVal := input.Get(j, k)
				// Get FFN output value
				ffnVal := ffnOutput.Get(j, k)
				// Add residual connection
				sum := inputVal + ffnVal
				// Clamp to int8 range
				if sum > 127 {
					sum = 127
				} else if sum < -128 {
					sum = -128
				}
				// Set final value
				result.Set(int8(sum), j, k)
			}
		}
	} else {
		result = tensor.NewTensor(batchSize, seqLen, hiddenDim)
		for i := 0; i < batchSize; i++ {
			for j := 0; j < seqLen; j++ {
				for k := 0; k < hiddenDim; k++ {
					// Get input value
					inputVal := input.Get(i, j, k)
					// Get FFN output value
					ffnVal := ffnOutput.Get(i, j, k)
					// Add residual connection
					sum := inputVal + ffnVal
					// Clamp to int8 range
					if sum > 127 {
						sum = 127
					} else if sum < -128 {
						sum = -128
					}
					// Set final value
					result.Set(int8(sum), i, j, k)
				}
			}
		}
	}

	return result, nil
}

// SetWeights sets the weights for the feed-forward network.
//
// Parameters:
//   - upWeights: Up-projection weights [intermediate_dim, hidden_dim]
//   - downWeights: Down-projection weights [hidden_dim, intermediate_dim]
//
// The weights are used for the two-layer feed-forward network:
// 1. Up-projection expands the hidden dimension
// 2. Down-projection contracts back to the hidden dimension
func (f *FFNSublayer) SetWeights(upWeights, downWeights *tensor.Tensor) {
	f.ffn.SetWeights(upWeights, downWeights)
}

// SetGamma sets the scale parameter for sublayer normalization.
//
// Parameters:
//   - gamma: Scale parameter vector [hidden_dim]
//
// The gamma parameter is used to scale the normalized values
// after the pre-norm layer normalization step.
func (f *FFNSublayer) SetGamma(gamma []float32) {
	f.subln.SetGamma(gamma)
}

// Close releases all resources associated with the feed-forward sublayer.
// This includes closing all tensors and cleaning up memory.
func (f *FFNSublayer) Close() {
	if f.ffn != nil {
		f.ffn.Close()
		f.ffn = nil
	}
	if f.subln != nil {
		f.subln.Close()
		f.subln = nil
	}
}
