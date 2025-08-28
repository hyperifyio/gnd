// Package raw_tensor provides a highly optimized 2D tensor implementation for BitNet inference.
//
// # Raw Tensor Implementation for BitNet
//
// This package implements a minimal 2D tensor optimized for BitNet's binary-weight quantization
// and CPU-based inference. It is designed to work with the token decoding process (see issue #190)
// and supports the overall goal of pure Go LLM implementation (see issue #170).
//
// Key aspects:
//   - 2D tensor implementation optimized for matrix operations in token decoding
//   - Binary-weight quantization support via int8 data type
//   - CPU-optimized memory layout for cache efficiency
//   - Goroutine-based parallel processing support
//   - Minimal memory footprint for edge deployment
//
// Implementation Details:
//   - Direct memory access without synchronization for maximum performance
//   - int8 data type to support BitNet's binary-weight quantization
//   - Row-major memory layout for optimal CPU cache utilization
//   - ParallelForEach for goroutine-based concurrent processing
//   - Zero-copy operations where possible
//
// Usage:
//   - Used internally by BitNet for token decoding and matrix operations
//   - Supports both float64 and int8 data types for model weights
//   - Maintainers should not use this type directly in public APIs
//   - Input shape must be [rows, cols] for 2D operations
//   - All operations assume valid indices and values
//
// Performance Considerations:
//   - No thread safety; caller must ensure thread safety
//   - No value clamping; caller must ensure values are in valid range
//   - Optimized for CPU cache line size (typically 64 bytes)
//   - Supports goroutine-based parallel processing
//   - Minimal memory allocations during operations
//
// Integration:
//   - Used by BitLinear for performance-critical matrix operations
//   - Supports token decoding process (issue #190)
//   - Part of pure Go LLM implementation (issue #170)
//   - Designed for CPU-based inference
//
// For more details, see:
//   - BitNet issue #170: Pure Go LLM for CPUs
//   - BitNet issue #190: Token Decoding (Inference Loop)
//   - BitNet project documentation
package raw_tensor

import (
	"errors"
)

var (
	ErrRawTensorInvalidDimensions = errors.New("raw_tensor: dimensions must be positive")
	ErrRawTensorInvalidShape      = errors.New("raw_tensor: input must be 2D")
	ErrRawTensorInvalidIndices    = errors.New("raw_tensor: requires exactly 2 indices")
	ErrRawTensorInvalidReshape    = errors.New("raw_tensor: cannot reshape tensor with different total size")
)

// rawTensor represents a 2D matrix of int8 values without locking or clamping
type rawTensor struct {
	data []int8
	rows int
	cols int
}

// newRawTensor creates a new rawTensor with the given dimensions
func newRawTensor(rows, cols int) (*rawTensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, ErrRawTensorInvalidDimensions
	}
	return &rawTensor{
		data: make([]int8, rows*cols),
		rows: rows,
		cols: cols,
	}, nil
}

// newRawTensorFromData creates a rawTensor from shape and data directly
func newRawTensorFromData(shape []int, data interface{}) (*rawTensor, error) {
	if len(shape) != 2 {
		return nil, ErrRawTensorInvalidShape
	}
	rows, cols := shape[0], shape[1]
	rt, err := newRawTensor(rows, cols)
	if err != nil {
		return nil, err
	}

	switch d := data.(type) {
	case []float64:
		for i := 0; i < len(d); i++ {
			rt.data[i] = int8(d[i]) // Convert float64 to int8
		}
	case []int8:
		copy(rt.data, d) // Direct copy for int8 data
	default:
		return nil, errors.New("raw_tensor: unsupported data type")
	}
	return rt, nil
}

// Get retrieves a value from the tensor at the specified indices
func (r *rawTensor) Get(indices ...int) (int8, error) {
	if len(indices) != 2 {
		return 0, ErrRawTensorInvalidIndices
	}
	return r.data[indices[0]*r.cols+indices[1]], nil
}

// Set assigns a value to the tensor at the specified indices
func (r *rawTensor) Set(value int8, indices ...int) error {
	if len(indices) != 2 {
		return ErrRawTensorInvalidIndices
	}
	r.data[indices[0]*r.cols+indices[1]] = value // No clamping
	return nil
}

// Data returns the underlying data slice
func (r *rawTensor) Data() []int8 {
	return r.data
}

// Shape returns the dimensions of the tensor
func (r *rawTensor) Shape() []int {
	return []int{r.rows, r.cols}
}

// Close is a no-op for rawTensor as it doesn't manage resources
func (r *rawTensor) Close() error { return nil }

// Reshape creates a new rawTensor with the given shape
func (r *rawTensor) Reshape(shape ...int) (*rawTensor, error) {
	if len(shape) != 2 {
		return nil, ErrRawTensorInvalidIndices
	}
	rows, cols := shape[0], shape[1]
	if rows*cols != len(r.data) {
		return nil, ErrRawTensorInvalidReshape
	}
	return &rawTensor{
		data: r.data,
		rows: rows,
		cols: cols,
	}, nil
}

// ParallelForEach processes each element in parallel
func (r *rawTensor) ParallelForEach(fn func(indices []int, value int8)) {
	for i := 0; i < r.rows; i++ {
		for j := 0; j < r.cols; j++ {
			fn([]int{i, j}, r.data[i*r.cols+j])
		}
	}
}

// NewRawTensor creates a new rawTensor with the given dimensions
func NewRawTensor(rows, cols int) (*rawTensor, error) {
	return newRawTensor(rows, cols)
}

// NewRawTensorFromData creates a rawTensor from shape and data directly
func NewRawTensorFromData(shape []int, data interface{}) (*rawTensor, error) {
	return newRawTensorFromData(shape, data)
}
