// Package relu2 provides activation functions for BitNet math operations.
//
// # Squared ReLU Activation for BitNet
//
// This package implements the squared ReLU activation function (ReLU²) used in BitNet's
// feed-forward networks. The implementation is optimized for quantized (int8) inference
// and parallel processing on CPU.
//
// Key aspects:
//   - Implements y = max(0, x)² with int8 input/output
//   - Optimized for parallel processing using goroutines
//   - Automatic chunking based on CPU count
//   - Supports both single vector and batch processing
//
// Implementation details:
//   - Efficient parallel processing with dynamic chunk sizing
//   - Direct int8 arithmetic to avoid float conversions
//   - Automatic clamping to int8 range (-128 to 127)
//   - Zero-copy for empty inputs
//
// Related tasks and dependencies:
//   - #180: Implement Squared ReLU Activation (Core implementation)
//   - #185: Feed-Forward Network (FFN) Sublayer (Depends on #180)
//   - #187: Integrate Feed-Forward Sublayer (Pre-Norm & Residual) (Depends on #185)
//   - #179: Implement Sub-Layer Normalization (Required by #187)
//   - #178: Implement BitLinear Layer (Required by #185)
//
// Usage:
//   - Used in BitNet's feed-forward networks for non-linear activation
//   - Supports both single vector and batch processing
//   - Maintainers should not change the activation formula or quantization
//
// Caveats:
//   - Performance critical - changes should be benchmarked
//   - Output range is limited to [0, 127] due to int8 quantization
//   - Parallel processing overhead may not be beneficial for very small inputs
//
// For more details, see BitNet issue #190 and the BitNet project documentation.
package relu2

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
