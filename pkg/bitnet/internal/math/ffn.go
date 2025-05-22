package math

import (
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// FFN represents a two-layer feed-forward network with ReLU² activation
type FFN struct {
	// Hidden dimension
	hiddenDim int
	// Intermediate dimension
	intermediateDim int
	// First layer weights (up-projection)
	upProj *tensor.Tensor
	// Second layer weights (down-projection)
	downProj *tensor.Tensor
}

// NewFFN creates a new FFN instance
func NewFFN(hiddenDim, intermediateDim int) *FFN {
	// Create weight matrices
	upProj := tensor.NewTensor(intermediateDim, hiddenDim)
	downProj := tensor.NewTensor(hiddenDim, intermediateDim)

	return &FFN{
		hiddenDim:       hiddenDim,
		intermediateDim: intermediateDim,
		upProj:          upProj,
		downProj:        downProj,
	}
}

// Forward performs the forward pass through the FFN
// input: [batch_size, seq_len, hidden_dim]
// Returns: [batch_size, seq_len, hidden_dim]
func (f *FFN) Forward(input *tensor.Tensor) *tensor.Tensor {
	if len(input.Shape()) != 3 {
		panic("input must be 3D tensor [batch_size, seq_len, hidden_dim]")
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]

	// Reshape input for linear projection
	flatInput := input.Reshape(batchSize*seqLen, f.hiddenDim)

	// First linear layer (up-projection)
	intermediate := tensor.BitLinear(flatInput, f.upProj)

	// Apply ReLU² activation
	intermediate = f.applyReLU2(intermediate)

	// Second linear layer (down-projection)
	output := tensor.BitLinear(intermediate, f.downProj)

	// Reshape back to [batch_size, seq_len, hidden_dim]
	return output.Reshape(batchSize, seqLen, f.hiddenDim)
}

// applyReLU2 applies the ReLU² activation function to the intermediate outputs
// input: [batch_size * seq_len, intermediate_dim]
// Returns: [batch_size * seq_len, intermediate_dim]
func (f *FFN) applyReLU2(input *tensor.Tensor) *tensor.Tensor {
	if len(input.Shape()) != 2 {
		panic("input must be 2D tensor [batch_size * seq_len, intermediate_dim]")
	}

	batchSize := input.Shape()[0]
	intermediateDim := input.Shape()[1]

	// Create output tensor
	output := tensor.NewTensor(batchSize, intermediateDim)

	// Process in parallel chunks
	var wg sync.WaitGroup
	chunkSize := batchSize / runtime.NumCPU()
	if chunkSize < 1 {
		chunkSize = 1
	}

	for i := 0; i < batchSize; i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > batchSize {
				end = batchSize
			}

			// Process each element
			for b := start; b < end; b++ {
				for d := 0; d < intermediateDim; d++ {
					// Get input value
					val := float32(input.Get(b, d))
					// Apply ReLU²: max(0, x)²
					if val > 0 {
						val = val * val
					} else {
						val = 0
					}
					// Clamp to int8 range and convert back to int8
					output.Set(int8(min(max(int32(val), -128), 127)), b, d)
				}
			}
		}(i)
	}

	wg.Wait()
	return output
}

// SetWeights sets the FFN weights
func (f *FFN) SetWeights(upWeights, downWeights *tensor.Tensor) {
	if upWeights.Shape()[0] != f.intermediateDim || upWeights.Shape()[1] != f.hiddenDim {
		panic("invalid up-projection weights shape")
	}
	if downWeights.Shape()[0] != f.hiddenDim || downWeights.Shape()[1] != f.intermediateDim {
		panic("invalid down-projection weights shape")
	}

	f.upProj = upWeights
	f.downProj = downWeights
}
