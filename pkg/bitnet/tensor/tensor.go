package tensor

import (
	"runtime"
	"sync"
)

// TensorType defines the core tensor operations
type TensorType interface {
	Get(indices ...int) float64
	Set(value float64, indices ...int)
	Shape() []int
	Data() []float64
}

// ParallelProcessor defines operations that can be executed in parallel
type ParallelProcessor interface {
	ParallelForEach(fn func(indices []int, value float64))
}

// Tensor represents a multi-dimensional array
type Tensor struct {
	data   []float64
	shape  []int
	stride []int
}

// workerPool manages a pool of worker goroutines
var workerPool = sync.Pool{
	New: func() interface{} {
		return make(chan struct{}, 1)
	},
}

// NewTensor creates a new tensor with the given shape
func NewTensor(shape ...int) *Tensor {
	if len(shape) == 0 {
		return nil
	}

	// Calculate total size and stride
	size := 1
	stride := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		stride[i] = size
		size *= shape[i]
	}

	// Create tensor
	return &Tensor{
		data:   make([]float64, size),
		shape:  shape,
		stride: stride,
	}
}

// Get returns the value at the given indices
func (t *Tensor) Get(indices ...int) float64 {
	if len(indices) != len(t.shape) {
		panic("invalid number of indices")
	}

	// Calculate linear index
	idx := 0
	for i, v := range indices {
		if v < 0 || v >= t.shape[i] {
			panic("index out of range")
		}
		idx += v * t.stride[i]
	}

	return t.data[idx]
}

// Set sets the value at the given indices
func (t *Tensor) Set(value float64, indices ...int) {
	if len(indices) != len(t.shape) {
		panic("invalid number of indices")
	}

	// Calculate linear index
	idx := 0
	for i, v := range indices {
		if v < 0 || v >= t.shape[i] {
			panic("index out of range")
		}
		idx += v * t.stride[i]
	}

	t.data[idx] = value
}

// Shape returns the shape of the tensor
func (t *Tensor) Shape() []int {
	return t.shape
}

// Data returns the underlying data array
func (t *Tensor) Data() []float64 {
	return t.data
}

// ParallelForEach applies the given function to each element in parallel
func (t *Tensor) ParallelForEach(fn func(indices []int, value float64)) {
	// Get number of CPU cores
	numCPU := runtime.NumCPU()
	if numCPU < 2 {
		// Fall back to sequential processing for single CPU
		t.forEach(fn)
		return
	}

	// Create work channels
	workChan := make(chan []int, numCPU*2)
	doneChan := make(chan struct{}, numCPU)

	// Start worker goroutines
	var wg sync.WaitGroup
	for i := 0; i < numCPU; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for indices := range workChan {
				fn(indices, t.Get(indices...))
			}
			doneChan <- struct{}{}
		}()
	}

	// Generate work
	go func() {
		t.forEach(func(indices []int, _ float64) {
			workChan <- indices
		})
		close(workChan)
	}()

	// Wait for completion
	go func() {
		wg.Wait()
		close(doneChan)
	}()

	// Wait for all workers to finish
	for range doneChan {
	}
}

// forEach applies the given function to each element sequentially
func (t *Tensor) forEach(fn func(indices []int, value float64)) {
	indices := make([]int, len(t.shape))
	t.forEachRecursive(0, indices, fn)
}

// forEachRecursive recursively traverses the tensor
func (t *Tensor) forEachRecursive(dim int, indices []int, fn func(indices []int, value float64)) {
	if dim == len(t.shape) {
		fn(indices, t.Get(indices...))
		return
	}

	for i := 0; i < t.shape[dim]; i++ {
		indices[dim] = i
		t.forEachRecursive(dim+1, indices, fn)
	}
}

// Verify interface implementation
var (
	_ TensorType        = (*Tensor)(nil)
	_ ParallelProcessor = (*Tensor)(nil)
)
