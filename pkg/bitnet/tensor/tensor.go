package tensor

import (
	"fmt"
	"os"
	"runtime"
	"sync"
)

// DebugLog logs debug information to stderr
func DebugLog(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "[DEBUG] "+format+"\n", args...)
}

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

	// Clamp value to int8 range
	if value > 127 {
		value = 127
	} else if value < -128 {
		value = -128
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

	// Create a copy of the data to avoid race conditions
	data := make([]int8, len(t.data))
	copy(data, t.data)

	var wg sync.WaitGroup
	chunkSize := len(data) / runtime.NumCPU()
	if chunkSize < 1 {
		chunkSize = 1
	}

	for i := 0; i < len(data); i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > len(data) {
				end = len(data)
			}

			for j := start; j < end; j++ {
				indices := t.calculateIndices(j)
				fn(indices, data[j])
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
		index += idx * t.stride[i]
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
			fmt.Fprintf(os.Stderr, "[DEBUG] Invalid shape dimension encountered: %v\n", shape)
			panic("tensor: invalid shape dimension")
		}
		newSize *= dim
	}

	// Verify total size matches
	if newSize != len(t.data) {
		panic("tensor: total size must match")
	}

	// Debug output for current shape, stride, and data length
	fmt.Fprintf(os.Stderr, "[DEBUG] Current shape: %v, stride: %v, data length: %d\n", t.shape, t.stride, len(t.data))
	fmt.Fprintf(os.Stderr, "[DEBUG] Target shape: %v, product: %d\n", shape, newSize)

	// Check if the data is contiguous (C-order: stride[i] == product(shape[i+1:]))
	isContiguous := true
	expectedStride := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		if t.stride[i] != expectedStride {
			isContiguous = false
			break
		}
		expectedStride *= t.shape[i]
	}

	// If not contiguous, copy data into a new contiguous tensor
	if !isContiguous {
		contiguousData := make([]int8, len(t.data))
		for i := 0; i < len(t.data); i++ {
			indices := t.calculateIndices(i)
			contiguousData[i] = t.data[t.calculateIndex(indices)]
		}
		t.data = contiguousData
		t.stride = make([]int, len(t.shape))
		for i := 0; i < len(t.shape); i++ {
			t.stride[i] = 1
		}
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

// NewTensorFromData creates a new tensor from raw data
func NewTensorFromData(data []int8) *Tensor {
	if len(data) == 0 {
		return nil
	}

	// Create tensor with 1D shape
	t := &Tensor{
		data:   make([]int8, len(data)),
		shape:  []int{len(data)},
		stride: []int{1},
	}

	// Copy data
	copy(t.data, data)

	return t
}

// Transpose returns a new tensor with dimensions permuted according to the given order.
// The order slice specifies the new order of dimensions.
// For example, if t has shape [2,3,4] and order is [0,2,1], the result will have shape [2,4,3].
func (t *Tensor) Transpose(order ...int) *Tensor {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		panic("tensor: Transpose called on closed tensor")
	}

	if len(order) != len(t.shape) {
		panic("tensor: order length must match tensor rank")
	}

	// Validate order
	used := make([]bool, len(order))
	for _, o := range order {
		if o < 0 || o >= len(order) {
			panic("tensor: invalid dimension in order")
		}
		if used[o] {
			panic("tensor: duplicate dimension in order")
		}
		used[o] = true
	}

	// Create new tensor with permuted shape
	newShape := make([]int, len(order))
	for i, o := range order {
		newShape[i] = t.shape[o]
	}

	// Create new tensor
	result := &Tensor{
		data:   make([]int8, len(t.data)),
		shape:  newShape,
		stride: make([]int, len(order)),
	}

	// Calculate new strides
	stride := 1
	for i := len(order) - 1; i >= 0; i-- {
		result.stride[i] = stride
		stride *= newShape[i]
	}

	// Copy data with permutation
	for i := 0; i < len(t.data); i++ {
		oldIndices := t.calculateIndices(i)
		newIndices := make([]int, len(order))
		for j, o := range order {
			newIndices[j] = oldIndices[o]
		}
		newIndex := 0
		for j, idx := range newIndices {
			newIndex += idx * result.stride[j]
		}
		result.data[newIndex] = t.data[i]
	}

	return result
}

// Repeat repeats the tensor along the specified dimension.
// The dimension must be valid (0 <= dim < len(t.shape)).
// The count must be positive.
func (t *Tensor) Repeat(dim int, count int) *Tensor {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		panic("tensor: Repeat called on closed tensor")
	}

	if dim < 0 || dim >= len(t.shape) {
		panic("tensor: invalid dimension for repeat")
	}
	if count <= 0 {
		panic("tensor: repeat count must be positive")
	}

	// Create new shape
	newShape := make([]int, len(t.shape))
	copy(newShape, t.shape)
	newShape[dim] *= count

	// Create new tensor
	result := &Tensor{
		data:   make([]int8, len(t.data)*count),
		shape:  newShape,
		stride: make([]int, len(t.shape)),
	}

	// Calculate new strides
	stride := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		result.stride[i] = stride
		stride *= newShape[i]
	}

	// Copy data with repetition
	for i := 0; i < len(t.data); i++ {
		oldIndices := t.calculateIndices(i)
		for c := 0; c < count; c++ {
			newIndices := make([]int, len(oldIndices))
			copy(newIndices, oldIndices)
			newIndices[dim] = oldIndices[dim] + c*t.shape[dim]
			newIndex := 0
			for j, idx := range newIndices {
				newIndex += idx * result.stride[j]
			}
			result.data[newIndex] = t.data[i]
		}
	}

	return result
}

// Add performs element-wise addition of two tensors.
// Both tensors must have the same shape.
func (t *Tensor) Add(other *Tensor) *Tensor {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		panic("tensor: Add called on closed tensor")
	}

	if other == nil {
		panic("tensor: cannot add nil tensor")
	}

	if other.closed {
		panic("tensor: cannot add closed tensor")
	}

	// Validate shapes match
	if len(t.shape) != len(other.shape) {
		panic("tensor: shapes must match for addition")
	}
	for i := range t.shape {
		if t.shape[i] != other.shape[i] {
			panic("tensor: shapes must match for addition")
		}
	}

	// Create result tensor
	result := &Tensor{
		data:   make([]int8, len(t.data)),
		shape:  t.shape,
		stride: t.stride,
	}

	// Add elements
	for i := 0; i < len(t.data); i++ {
		// Convert to int32 to handle overflow during addition
		sum := int32(t.data[i]) + int32(other.data[i])
		// Clamp to int8 range (-128 to 127)
		if sum > 127 {
			result.data[i] = 127
		} else if sum < -128 {
			result.data[i] = -128
		} else {
			result.data[i] = int8(sum)
		}
	}

	return result
}

// SetTernary assigns a value to the tensor, clamping to ternary range (-1, 0, 1)
func (t *Tensor) SetTernary(value int8, indices ...int) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		panic("tensor: SetTernary called on closed tensor")
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

// Verify interface implementation
var (
	_ TensorType        = (*Tensor)(nil)
	_ ParallelProcessor = (*Tensor)(nil)
)
