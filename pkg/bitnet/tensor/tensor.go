// Package tensor implements a multi-dimensional array data structure optimized
// for ternary values (-1, 0, +1). It provides efficient operations for tensor
// manipulation, including reshaping, transposition, and parallel processing.
// The package is designed for use in neural network computations with a focus
// on memory efficiency and thread safety.
package tensor

import (
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/loggers"
)

// DebugLog logs debug information to stderr using the configured logger.
func DebugLog(format string, args ...interface{}) {
	loggers.Printf(loggers.Debug, format, args...)
}

// TensorType defines the core tensor operations that must be implemented
// by any tensor-like data structure. It provides methods for accessing and
// modifying tensor elements, retrieving shape information, and managing
// tensor lifecycle.
type TensorType interface {
	Get(indices ...int) int8
	Set(value int8, indices ...int)
	Shape() []int
	Data() []int8
	Close()
}

// ParallelProcessor defines operations that can be executed in parallel
// across tensor elements. It provides a method for applying a function
// to each element of the tensor concurrently.
type ParallelProcessor interface {
	ParallelForEach(fn func(indices []int, value int8))
}

// Tensor represents a multi-dimensional array of ternary values (-1, 0, +1).
// It provides thread-safe operations for tensor manipulation and supports
// efficient parallel processing of tensor elements.
type Tensor struct {
	data   []int8       // Underlying data storage
	shape  []int        // Dimensions of the tensor
	stride []int        // Stride values for efficient indexing
	mu     sync.RWMutex // Mutex for thread safety
	closed bool         // Flag indicating if tensor is closed
}

// tensorOp represents a tensor operation to be performed.
// It is used internally for managing concurrent operations.
type tensorOp struct {
	opType   string        // "get" or "set"
	indices  []int         // Indices for the operation
	value    int8          // Value to set (for set operations)
	resultCh chan int8     // Channel for operation results
	doneCh   chan struct{} // Channel for operation completion
}

// NewTensor creates a new tensor with the given shape.
// The shape parameter defines the dimensions of the tensor.
// Returns nil if no shape is provided.
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

// Get retrieves a value from the tensor at the specified indices.
// Panics if the tensor is closed, indices are invalid, or out of range.
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

// Set assigns a value to the tensor at the specified indices.
// The value is clamped to the int8 range [-128, 127].
// Panics if the tensor is closed, indices are invalid, or out of range.
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

// setRaw assigns a value to the tensor without clamping (for internal use only).
// Panics if the tensor is closed, indices are invalid, or out of range.
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

// Shape returns a copy of the tensor's dimensions.
// Panics if the tensor is closed.
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

// Data returns a copy of the underlying data array.
// Panics if the tensor is closed.
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

// ParallelForEach processes each element in parallel using the provided function.
// The function is called with the indices and value for each element.
// Panics if the tensor is closed.
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

// Close marks the tensor as closed and releases any resources.
// After closing, all operations on the tensor will panic.
func (t *Tensor) Close() {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.closed {
		t.closed = true
		t.data = nil
	}
}

// calculateIndex converts multi-dimensional indices to a linear index.
// Returns -1 if the indices are invalid.
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

// calculateIndices converts a linear index to multi-dimensional indices.
// Returns nil if the index is invalid.
func (t *Tensor) calculateIndices(index int) []int {
	indices := make([]int, len(t.shape))
	stride := 1

	for i := len(t.shape) - 1; i >= 0; i-- {
		indices[i] = (index / stride) % t.shape[i]
		stride *= t.shape[i]
	}

	return indices
}

// Reshape creates a new tensor with the same data but different dimensions.
// The total number of elements must remain the same.
// Returns nil if the new shape is invalid.
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
			loggers.Printf(loggers.Debug, "Invalid shape dimension encountered: %v", shape)
			panic("tensor: invalid shape dimension")
		}
		newSize *= dim
	}

	// Verify total size matches
	if newSize != len(t.data) {
		panic("tensor: total size must match")
	}

	// Debug output for current shape, stride, and data length
	loggers.Printf(loggers.Debug, "Current shape: %v, stride: %v, data length: %d", t.shape, t.stride, len(t.data))
	loggers.Printf(loggers.Debug, "Target shape: %v, product: %d", shape, newSize)

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

// NewTensorFromData creates a new tensor from existing data.
// The shape is inferred from the data length.
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

// Transpose creates a new tensor with dimensions reordered according to the order parameter.
// The order parameter specifies the new order of dimensions.
// Returns nil if the order is invalid.
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

// Repeat creates a new tensor by repeating the tensor along the specified dimension.
// The count parameter specifies how many times to repeat.
// Returns nil if the dimension or count is invalid.
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
// The tensors must have the same shape.
// Returns nil if the shapes don't match.
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

// SetTernary sets a ternary value (-1, 0, +1) at the specified indices.
// The value is clamped to the ternary range.
// Panics if the tensor is closed, indices are invalid, or out of range.
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
