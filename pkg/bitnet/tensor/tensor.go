package tensor

import (
	"runtime"
	"sync"
)

// TensorType defines the core tensor operations
type TensorType interface {
	Get(indices ...int) int8
	Set(value int8, indices ...int)
	Shape() []int
	Data() []int8
	Close()
}

// ParallelProcessor defines operations that can be executed in parallel
type ParallelProcessor interface {
	ParallelForEach(fn func(indices []int, value int8))
}

// Tensor represents a multi-dimensional array of ternary values (-1, 0, +1)
type Tensor struct {
	data   []int8
	shape  []int
	stride []int
	mu     sync.RWMutex
	closed bool
}

// tensorOp represents a tensor operation
type tensorOp struct {
	opType   string // "get" or "set"
	indices  []int
	value    int8
	resultCh chan int8
	doneCh   chan struct{}
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
	t := &Tensor{
		data:   make([]int8, size),
		shape:  shape,
		stride: stride,
	}

	return t
}

// Get retrieves a value from the tensor
func (t *Tensor) Get(indices ...int) int8 {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		panic("tensor: Get called on closed tensor")
	}

	if len(indices) != len(t.shape) {
		panic("tensor: invalid number of indices")
	}

	index := t.calculateIndex(indices)
	if index < 0 || index >= len(t.data) {
		panic("tensor: index out of range")
	}

	return t.data[index]
}

// Set assigns a value to the tensor
func (t *Tensor) Set(value int8, indices ...int) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		panic("tensor: Set called on closed tensor")
	}

	if len(indices) != len(t.shape) {
		panic("tensor: invalid number of indices")
	}

	index := t.calculateIndex(indices)
	if index < 0 || index >= len(t.data) {
		panic("tensor: index out of range")
	}

	// Clamp value to ternary range
	if value > 1 {
		value = 1
	} else if value < -1 {
		value = -1
	}

	t.data[index] = value
}

// setRaw assigns a value to the tensor without clamping (for internal use only)
func (t *Tensor) setRaw(value int8, indices ...int) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		panic("tensor: Set called on closed tensor")
	}

	if len(indices) != len(t.shape) {
		panic("tensor: invalid number of indices")
	}

	index := t.calculateIndex(indices)
	if index < 0 || index >= len(t.data) {
		panic("tensor: index out of range")
	}

	t.data[index] = value // No clamping
}

// Shape returns the tensor's dimensions
func (t *Tensor) Shape() []int {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		panic("tensor: Shape called on closed tensor")
	}

	shape := make([]int, len(t.shape))
	copy(shape, t.shape)
	return shape
}

// Data returns the underlying data array
func (t *Tensor) Data() []int8 {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		panic("tensor: Data called on closed tensor")
	}

	data := make([]int8, len(t.data))
	copy(data, t.data)
	return data
}

// ParallelForEach processes each element in parallel
func (t *Tensor) ParallelForEach(fn func(indices []int, value int8)) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		panic("tensor: ParallelForEach called on closed tensor")
	}

	var wg sync.WaitGroup
	chunkSize := len(t.data) / runtime.NumCPU()
	if chunkSize < 1 {
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
				indices := t.calculateIndices(j)
				fn(indices, t.data[j])
			}
		}(i)
	}

	wg.Wait()
}

// Close marks the tensor as closed and frees its resources
// The write-lock is only held in Close(), which is called very rarely
// (only when tearing down or freeing the tensor), so the per-access
// RLock overhead remains negligible.
func (t *Tensor) Close() {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.closed {
		t.closed = true
		t.data = nil
	}
}

// calculateIndex converts multi-dimensional indices to a flat index
func (t *Tensor) calculateIndex(indices []int) int {
	if len(indices) != len(t.shape) {
		panic("number of indices does not match tensor rank")
	}
	index := 0
	for i, idx := range indices {
		if idx < 0 || idx >= t.shape[i] {
			return -1
		}
		index = index*t.shape[i] + idx
	}
	return index
}

// calculateIndices converts a flat index to multi-dimensional indices
func (t *Tensor) calculateIndices(index int) []int {
	indices := make([]int, len(t.shape))
	stride := 1

	for i := len(t.shape) - 1; i >= 0; i-- {
		indices[i] = (index / stride) % t.shape[i]
		stride *= t.shape[i]
	}

	return indices
}

// Reshape creates a new tensor with the same data but different shape
func (t *Tensor) Reshape(shape ...int) *Tensor {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		panic("tensor: Reshape called on closed tensor")
	}

	// Calculate total size of new shape
	newSize := 1
	for _, dim := range shape {
		if dim <= 0 {
			panic("tensor: invalid shape dimension")
		}
		newSize *= dim
	}

	// Verify total size matches
	if newSize != len(t.data) {
		panic("tensor: total size must match")
	}

	// Create new tensor with same data but new shape
	newTensor := &Tensor{
		data:   make([]int8, len(t.data)),
		shape:  shape,
		stride: make([]int, len(shape)),
	}

	// Copy data
	copy(newTensor.data, t.data)

	// Calculate new strides
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		newTensor.stride[i] = stride
		stride *= shape[i]
	}

	return newTensor
}

// Verify interface implementation
var (
	_ TensorType        = (*Tensor)(nil)
	_ ParallelProcessor = (*Tensor)(nil)
)
