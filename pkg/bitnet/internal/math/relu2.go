package math

import (
	"runtime"
	"sync"
)

// ReLU2 applies the squared ReLU activation function: y = max(0, x)²
// The input and output are 8-bit integers (-128 to 127)
// The function ensures the output can be quantized back to 8-bit
func ReLU2(input []int8) []int8 {
	if len(input) == 0 {
		return input
	}

	output := make([]int8, len(input))

	// Process in parallel chunks
	var wg sync.WaitGroup
	chunkSize := len(input) / runtime.NumCPU()
	if chunkSize < 1 {
		chunkSize = 1
	}

	for i := 0; i < len(input); i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > len(input) {
				end = len(input)
			}

			// Process each element
			for j := start; j < end; j++ {
				x := int32(input[j])
				// Apply ReLU: max(0, x)
				if x < 0 {
					x = 0
				}
				// Square the result
				x = x * x
				// Clamp to int8 range
				if x > 127 {
					x = 127
				}
				output[j] = int8(x)
			}
		}(i)
	}

	wg.Wait()
	return output
}

// ReLU2Batch applies the squared ReLU activation function to a batch of vectors
func ReLU2Batch(input [][]int8) [][]int8 {
	if len(input) == 0 {
		return input
	}

	output := make([][]int8, len(input))
	for i := range output {
		output[i] = make([]int8, len(input[i]))
	}

	// Process in parallel chunks
	var wg sync.WaitGroup
	chunkSize := len(input) / runtime.NumCPU()
	if chunkSize < 1 {
		chunkSize = 1
	}

	for i := 0; i < len(input); i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > len(input) {
				end = len(input)
			}

			// Process each vector in the batch
			for j := start; j < end; j++ {
				output[j] = ReLU2(input[j])
			}
		}(i)
	}

	wg.Wait()
	return output
}
