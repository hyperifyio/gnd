package tensor

import (
	"runtime"
	"sync"
)

// BitLinear performs a linear transformation using 1.58-bit weights
// input: 8-bit activations [batch_size, in_features]
// weights: 1.58-bit weights [out_features, in_features]
// Returns: 8-bit output [batch_size, out_features]
func BitLinear(input, weights *Tensor) *Tensor {
	if len(input.shape) != 2 || len(weights.shape) != 2 {
		panic("bitlinear: input and weights must be 2D tensors")
	}
	if input.shape[1] != weights.shape[1] {
		panic("bitlinear: input and weight dimensions must match")
	}

	// Convert to rawTensor for efficient computation
	rawInput := newRawTensorFrom(input)
	rawWeights := newRawTensorFrom(weights)

	batchSize := input.shape[0]
	inFeatures := input.shape[1]
	outFeatures := weights.shape[0]

	// Create raw output tensor
	rawOutput := newRawTensor(batchSize, outFeatures)

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

			// Process each batch element
			for b := start; b < end; b++ {
				// Process each output feature
				for o := 0; o < outFeatures; o++ {
					var sum int32
					// Compute dot product
					for f := 0; f < inFeatures; f++ {
						// Get input activation (8-bit)
						act := rawInput.At(b, f)
						// Get weight (1.58-bit, stored as -1, 0, +1)
						w := rawWeights.At(o, f)
						// Multiply and accumulate
						sum += int32(act) * int32(w)
					}
					// Clamp to int8 range and store
					if sum > 127 {
						sum = 127
					} else if sum < -128 {
						sum = -128
					}
					rawOutput.Set(b, o, int8(sum))
				}
			}
		}(i)
	}

	wg.Wait()

	// Convert result back to Tensor
	output := NewTensor(batchSize, outFeatures)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < outFeatures; j++ {
			output.setRaw(rawOutput.At(i, j), i, j)
		}
	}

	return output
}
