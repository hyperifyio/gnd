package tensor

import (
	"sync"
)

// Tensor represents a multi-dimensional array
type Tensor struct {
	Data   []float32
	Shape  []int
	Stride []int
}

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
		Data:   make([]float32, size),
		Shape:  shape,
		Stride: stride,
	}
}

// Get returns the value at the specified indices
func (t *Tensor) Get(indices ...int) float32 {
	idx := 0
	for i, pos := range indices {
		idx += pos * t.Stride[i]
	}
	return t.Data[idx]
}

// Set sets the value at the specified indices
func (t *Tensor) Set(value float32, indices ...int) {
	idx := 0
	for i, pos := range indices {
		idx += pos * t.Stride[i]
	}
	t.Data[idx] = value
}

// ParallelForEach applies the given function to each element in parallel
func (t *Tensor) ParallelForEach(fn func(value float32, indices ...int) float32) {
	var wg sync.WaitGroup
	chunkSize := len(t.Data) / 4 // Divide work into 4 chunks
	if chunkSize == 0 {
		chunkSize = 1
	}

	for i := 0; i < len(t.Data); i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > len(t.Data) {
				end = len(t.Data)
			}

			for j := start; j < end; j++ {
				// Convert linear index to multi-dimensional indices
				indices := make([]int, len(t.Shape))
				remaining := j
				for k := len(t.Shape) - 1; k >= 0; k-- {
					indices[k] = remaining / t.Stride[k]
					remaining %= t.Stride[k]
				}
				t.Data[j] = fn(t.Data[j], indices...)
			}
		}(i)
	}
	wg.Wait()
}
