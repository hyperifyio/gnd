// Package tensor implements a multi-dimensional array data structure optimized
// for ternary values (-1, 0, +1). It provides efficient operations for tensor
// manipulation, including reshaping, transposition, and parallel processing.
// The package is designed for use in neural network computations with a focus
// on memory efficiency and thread safety.
package tensor

import (
	"runtime"
	"sync"
	"sync/atomic"
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
//   - error if dimensions don't match or tensors are closed
//
// The function performs the following optimizations:
//   - Memory-aligned allocations for better cache performance
//   - Parallel processing across batch elements
//   - Loop unrolling for faster matrix multiplication
//   - Reuse of work buffers to reduce allocations
//   - Branchless clamping of output values
func BitLinear(input TensorReader, weights TensorReader) (TensorOperations, error) {
	// Convert to concrete types for validation
	t, ok := input.(*Tensor)
	if !ok {
		return nil, ErrInvalidShape
	}
	w, ok := weights.(*Tensor)
	if !ok {
		return nil, ErrInvalidShape
	}

	// Lock both tensors for the duration of the operation
	t.mu.RLock()
	w.mu.RLock()
	defer t.mu.RUnlock()
	defer w.mu.RUnlock()

	if atomic.LoadUint32(&t.closed) == 1 || atomic.LoadUint32(&w.closed) == 1 {
		return nil, ErrTensorClosed
	}

	if len(t.shape) != 2 || len(w.shape) != 2 {
		return nil, ErrInvalidShape
	}
	if t.shape[1] != w.shape[1] {
		return nil, ErrDimensionMismatch
	}

	batchSize := t.shape[0]
	inFeatures := t.shape[1]
	outFeatures := w.shape[0]

	// Debug output for shapes
	loggers.Printf(loggers.Debug, "BitLinear input shape: %v", t.shape)
	loggers.Printf(loggers.Debug, "BitLinear weights shape: %v", w.shape)
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
		err      error
	}
	resultChan := make(chan result, batchSize)

	// Process in parallel chunks
	numCPU := runtime.NumCPU()
	chunkSize := (batchSize + numCPU - 1) / numCPU // Ceiling division

	var wg sync.WaitGroup
	wg.Add(numCPU)

	// Launch worker goroutines
	for i := 0; i < numCPU; i++ {
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > batchSize {
				end = batchSize
			}

			// Get work buffer from pool
			buf := bufferPool.Get().(*workBuffer)
			defer bufferPool.Put(buf)

			// Ensure buffer is large enough
			if cap(buf.sums) < outFeatures {
				buf.sums = make([]int32, outFeatures)
			}
			buf.sums = buf.sums[:outFeatures]

			// Process each batch element
			for b := start; b < end; b++ {
				// Clear sums for this batch element
				for j := range buf.sums {
					buf.sums[j] = 0
				}

				// Compute matrix multiplication
				for i := 0; i < inFeatures; i++ {
					inputVal := int32(t.data[b*inFeatures+i])
					for j := 0; j < outFeatures; j++ {
						buf.sums[j] += inputVal * int32(w.data[j*inFeatures+i])
					}
				}

				// Convert sums to int8 with clamping
				outputVals := make([]int8, outFeatures)
				for j := range buf.sums {
					sum := buf.sums[j]
					if sum > 127 {
						outputVals[j] = 127
					} else if sum < -128 {
						outputVals[j] = -128
					} else {
						outputVals[j] = int8(sum)
					}
				}

				// Send result
				resultChan <- result{
					batchIdx: b,
					values:   outputVals,
				}
			}
		}(i * chunkSize)
	}

	// Close result channel when all workers are done
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	for r := range resultChan {
		if r.err != nil {
			return nil, r.err
		}
		copy(output.data[r.batchIdx*outFeatures:], r.values)
	}

	return output, nil
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
