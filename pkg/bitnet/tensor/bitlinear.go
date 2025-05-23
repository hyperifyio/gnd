// Package tensor implements a multi-dimensional array data structure optimized
// for ternary values (-1, 0, +1). It provides efficient operations for tensor
// manipulation, including reshaping, transposition, and parallel processing.
// The package is designed for use in neural network computations with a focus
// on memory efficiency and thread safety.
package tensor

import (
	"runtime"
	"sync"
	"unsafe"

	"github.com/hyperifyio/gnd/pkg/loggers"
)

// workBuffer represents a pre-allocated buffer for computations.
// It is used to store intermediate results during tensor operations
// to avoid repeated memory allocations.
type workBuffer struct {
	sums []int32 // Buffer for accumulating sums during matrix multiplication
}

// bufferPool is a sync.Pool for work buffers.
// It provides a pool of pre-allocated work buffers to reduce
// memory allocations during parallel computations.
var bufferPool = sync.Pool{
	New: func() interface{} {
		// Pre-allocate a buffer with a reasonable default size
		// This will be resized if needed
		return &workBuffer{
			sums: make([]int32, 1024),
		}
	},
}

// alignedAlloc allocates a slice with proper alignment for better cache performance.
// The function ensures that the allocated memory is aligned according to the
// type's alignment requirements, which can improve performance on modern CPUs.
func alignedAlloc[T any](size int) []T {
	// Calculate size needed for alignment
	var zero T
	align := int(unsafe.Alignof(zero))
	// Add padding to ensure alignment
	paddedSize := (size + align - 1) & ^(align - 1)
	return make([]T, paddedSize)
}

// BitLinear performs a linear transformation using 1.58-bit weights.
// This version uses atomic operations and channels for thread safety.
//
// Parameters:
//   - input: 8-bit activations with shape [batch_size, in_features]
//   - weights: 1.58-bit weights with shape [out_features, in_features]
//
// Returns:
//   - 8-bit output tensor with shape [batch_size, out_features]
//
// The function performs the following optimizations:
//   - Memory-aligned allocations for better cache performance
//   - Parallel processing across batch elements
//   - Loop unrolling for faster matrix multiplication
//   - Reuse of work buffers to reduce allocations
//   - Branchless clamping of output values
func BitLinear(input, weights *Tensor) *Tensor {
	if len(input.shape) != 2 || len(weights.shape) != 2 {
		panic("bitlinear: input and weights must be 2D tensors")
	}
	if input.shape[1] != weights.shape[1] {
		panic("bitlinear: input and weight dimensions must match")
	}

	batchSize := input.shape[0]
	inFeatures := input.shape[1]
	outFeatures := weights.shape[0]

	// Debug output for shapes
	loggers.Printf(loggers.Debug, "BitLinear input shape: %v", input.shape)
	loggers.Printf(loggers.Debug, "BitLinear weights shape: %v", weights.shape)
	loggers.Printf(loggers.Debug, "BitLinear output shape: [%d %d]", batchSize, outFeatures)
	loggers.Printf(loggers.Debug, "BitLinear batchSize: %d, inFeatures: %d, outFeatures: %d", batchSize, inFeatures, outFeatures)

	// Pre-allocate output tensor with aligned memory
	output := &Tensor{
		shape:  []int{batchSize, outFeatures},
		stride: []int{outFeatures, 1},
		data:   alignedAlloc[int8](batchSize * outFeatures),
	}

	// Create a channel to receive results from workers
	type result struct {
		batchIdx int
		values   []int8
	}
	resultChan := make(chan result, batchSize)

	// Process in parallel chunks
	numCPU := runtime.NumCPU()
	chunkSize := (batchSize + numCPU - 1) / numCPU // Ceiling division

	var wg sync.WaitGroup
	wg.Add(numCPU)

	// Launch worker goroutines
	for cpu := 0; cpu < numCPU; cpu++ {
		go func(cpu int) {
			defer wg.Done()

			start := cpu * chunkSize
			end := start + chunkSize
			if end > batchSize {
				end = batchSize
			}
			loggers.Printf(loggers.Debug, "BitLinear goroutine %d: start=%d, end=%d", cpu, start, end)

			// Get a buffer from the pool
			buf := bufferPool.Get().(*workBuffer)
			defer bufferPool.Put(buf)

			// Resize buffer if needed
			if cap(buf.sums) < outFeatures {
				buf.sums = alignedAlloc[int32](outFeatures)
			} else {
				buf.sums = buf.sums[:outFeatures]
			}

			// Process each batch element
			for b := start; b < end; b++ {
				// Reset sums for this batch element
				for o := range buf.sums {
					buf.sums[o] = 0
				}

				// Process each output feature
				for o := 0; o < outFeatures; o++ {
					// Compute dot product with loop unrolling
					f := 0
					// Process 4 elements at a time
					for ; f+3 < inFeatures; f += 4 {
						// Get input activations (8-bit) - using atomic load
						act0 := int32(input.data[b*inFeatures+f])
						act1 := int32(input.data[b*inFeatures+f+1])
						act2 := int32(input.data[b*inFeatures+f+2])
						act3 := int32(input.data[b*inFeatures+f+3])
						// Get weights (1.58-bit) - using atomic load
						w0 := int32(weights.data[o*inFeatures+f])
						w1 := int32(weights.data[o*inFeatures+f+1])
						w2 := int32(weights.data[o*inFeatures+f+2])
						w3 := int32(weights.data[o*inFeatures+f+3])
						// Multiply and accumulate
						buf.sums[o] += act0*w0 + act1*w1 + act2*w2 + act3*w3
					}
					// Process remaining elements
					for ; f < inFeatures; f++ {
						act := int32(input.data[b*inFeatures+f])
						w := int32(weights.data[o*inFeatures+f])
						buf.sums[o] += act * w
					}
				}

				// Clamp and prepare results
				results := make([]int8, outFeatures)
				for o := 0; o < outFeatures; o++ {
					sum := buf.sums[o]
					// Branchless clamping using min/max
					sum = min(max(sum, -128), 127)
					results[o] = int8(sum)
				}

				// Send results through channel
				resultChan <- result{
					batchIdx: b,
					values:   results,
				}
			}
		}(cpu)
	}

	// Close result channel when all workers are done
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	for result := range resultChan {
		// Store results using atomic operations
		for o, v := range result.values {
			output.data[result.batchIdx*outFeatures+o] = v
		}
	}

	return output
}

// min returns the minimum of two int32 values.
// This is a utility function used internally for bounds checking.
func min(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two int32 values.
// This is a utility function used internally for bounds checking.
func max(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}
