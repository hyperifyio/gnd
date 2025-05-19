package tensor

import (
	"sync"
)

// TensorType defines the core operations for a tensor
type TensorType interface {
	// Get returns the value at the specified indices
	Get(indices ...int) float32
	// Set sets the value at the specified indices
	Set(value float32, indices ...int)
	// Shape returns the dimensions of the tensor
	Shape() []int
	// Data returns the underlying data array
	Data() []float32
}

// ParallelProcessor defines operations that can be executed in parallel
type ParallelProcessor interface {
	// ParallelForEach applies the given function to each element in parallel
	ParallelForEach(fn func(value float32, indices ...int) float32)
}

// Tensor represents a multi-dimensional array
type Tensor struct {
	data   []float32
	shape  []int
	stride []int
}

// Verify interface implementation
var (
	_ TensorType        = &Tensor{}
	_ ParallelProcessor = &Tensor{}
)

// NewTensor creates a new tensor with the given shape
func NewTensor(shape ...int) *Tensor {
	if len(shape) == 0 {
		panic("tensor must have at least one dimension")
	}

	// Calculate total size and strides
	size := 1
	stride := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		stride[i] = size
		size *= shape[i]
	}

	return &Tensor{
		data:   make([]float32, size),
		shape:  shape,
		stride: stride,
	}
}

// Get returns the value at the specified indices
func (t *Tensor) Get(indices ...int) float32 {
	idx := 0
	for i, pos := range indices {
		idx += pos * t.stride[i]
	}
	return t.data[idx]
}

// Set sets the value at the specified indices
func (t *Tensor) Set(value float32, indices ...int) {
	idx := 0
	for i, pos := range indices {
		idx += pos * t.stride[i]
	}
	t.data[idx] = value
}

// Shape returns the dimensions of the tensor
func (t *Tensor) Shape() []int {
	return t.shape
}

// Data returns the underlying data array
func (t *Tensor) Data() []float32 {
	return t.data
}

// ParallelForEach applies the given function to each element in parallel
func (t *Tensor) ParallelForEach(fn func(value float32, indices ...int) float32) {
	var wg sync.WaitGroup
	chunkSize := len(t.data) / 4 // Divide work into 4 chunks
	if chunkSize == 0 {
		chunkSize = 1
	}

	for i := 0; i < len(t.data); i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > len(t.data) {
				end = len(t.data)
			}

			for j := start; j < end; j++ {
				// Convert linear index to multi-dimensional indices
				indices := make([]int, len(t.shape))
				remaining := j
				for k := len(t.shape) - 1; k >= 0; k-- {
					indices[k] = remaining / t.stride[k]
					remaining %= t.stride[k]
				}
				t.data[j] = fn(t.data[j], indices...)
			}
		}(i)
	}
	wg.Wait()
}
