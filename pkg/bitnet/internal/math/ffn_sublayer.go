package math

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// FFNSublayer implements the feed-forward sublayer with pre-norm and residual connection
type FFNSublayer struct {
	// Sub-layer normalization
	subln *SubLN
	// Feed-forward network
	ffn *FFN
	// Hidden dimension
	hiddenDim int
	// Intermediate dimension
	intermediateDim int
}

// NewFFNSublayer creates a new feed-forward sublayer
func NewFFNSublayer(hiddenDim, intermediateDim int) *FFNSublayer {
	return &FFNSublayer{
		subln:           NewSubLN(hiddenDim, 1e-5),
		ffn:             NewFFN(hiddenDim, intermediateDim),
		hiddenDim:       hiddenDim,
		intermediateDim: intermediateDim,
	}
}

// Forward performs the forward pass through the feed-forward sublayer
func (f *FFNSublayer) Forward(input *tensor.Tensor) *tensor.Tensor {
	// Get input dimensions
	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]
	hiddenDim := input.Shape()[2]

	// Convert input to float32 for normalization
	inputFloat := make([][]float32, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			idx := i*seqLen + j
			inputFloat[idx] = make([]float32, hiddenDim)
			for k := 0; k < hiddenDim; k++ {
				inputFloat[idx][k] = float32(input.Get(i, j, k))
			}
		}
	}

	// Apply pre-norm
	normalized := f.subln.Normalize(inputFloat)

	// Reshape normalized output back to 3D tensor
	normalizedTensor := tensor.NewTensor(batchSize, seqLen, hiddenDim)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			idx := i*seqLen + j
			for k := 0; k < hiddenDim; k++ {
				normalizedTensor.Set(int8(normalized[idx][k]), i, j, k)
			}
		}
	}

	// Apply feed-forward network
	ffnOutput := f.ffn.Forward(normalizedTensor)

	// Add residual connection
	result := tensor.NewTensor(batchSize, seqLen, hiddenDim)
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

	return result
}

// SetWeights sets the weights for the feed-forward network
func (f *FFNSublayer) SetWeights(upWeights, downWeights *tensor.Tensor) {
	f.ffn.SetWeights(upWeights, downWeights)
}

// SetGamma sets the scale parameter for sublayer normalization
func (f *FFNSublayer) SetGamma(gamma []float32) {
	f.subln.SetGamma(gamma)
}
