// Package tensor implements a multi-dimensional array data structure optimized
// for ternary values (-1, 0, +1) in BitNet inference.
//
// # Quantized Tensor Implementation for BitNet
//
// This file provides the core tensor data structure and operations for BitNet's
// quantized neural network computations. It is a critical component of the
// BitNet implementation (Issue #170) and supports the token decoding
// functionality (Issue #190).
//
// Key aspects:
//   - All tensors store ternary values (-1, 0, +1) as int8 for memory efficiency
//   - Thread-safe operations with mutex protection and atomic flags
//   - Optimized for CPU efficiency with parallel processing support
//   - Memory pooling for frequently used tensor shapes
//   - Not suitable for training or float32 inference
//
// Implementation Status:
//   - Core tensor operations with ternary value support
//   - Thread-safe operations with proper synchronization
//   - Parallel processing support for bulk operations
//   - Memory-efficient storage format
//   - Support for matrix multiplication and linear transformations
//   - Shape validation and error handling
//
// Usage:
//   - Used throughout BitNet for storing and manipulating quantized weights and activations
//   - Maintainers should not change the ternary value handling without full pipeline review
//   - Use ParallelForEach for bulk operations to maximize CPU utilization
//   - Use BitLinear for quantized linear transformations
//   - Validate tensor shapes using the provided validation functions
//
// Caveats:
//   - Values are automatically clamped to ternary range in Set operations
//   - Thread safety comes with performance overhead; use ParallelForEach for bulk operations
//   - Any change must be validated against end-to-end BitNet inference
//   - Memory usage scales with tensor dimensions
//   - Matrix operations require matching dimensions
//
// For more details, see:
//   - BitNet issue #170: Main feature implementation
//   - BitNet issue #190: Token decoding and inference loop
//   - Additional tasks: https://github.com/hyperifyio/gnd/issues?q=is%3Aissue+state%3Aopen+label%3Abitnet+label%3Atask
package tensor

import (
	"errors"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/hyperifyio/gnd/pkg/bitnet/logging"
)

var (
	ErrTensorInvalidShape       = errors.New("tensor: invalid shape dimension")
	ErrTensorInvalidIndices     = errors.New("tensor: invalid number of indices")
	ErrTensorIndexOutOfRange    = errors.New("tensor: index out of range")
	ErrTensorInvalidReshape     = errors.New("tensor: cannot reshape tensor with different total size")
	ErrTensorInvalidTranspose   = errors.New("tensor: invalid transpose order")
	ErrTensorInvalidDimension   = errors.New("tensor: invalid dimension in transpose order")
	ErrTensorDuplicateDimension = errors.New("tensor: duplicate dimension in transpose order")
	ErrTensorInvalidRepeat      = errors.New("tensor: invalid dimension for repeat")
	ErrTensorInvalidRepeatCount = errors.New("tensor: repeat count must be positive")
	ErrTensorShapeMismatch      = errors.New("tensor: cannot add tensors with different shapes")
)

// Tensor represents a multi-dimensional array of ternary values (-1, 0, +1).
// It provides thread-safe operations for tensor manipulation and supports
// efficient parallel processing of tensor elements.
type Tensor struct {
	data   []int8       // Underlying data storage
	shape  []int        // Dimensions of the tensor
	stride []int        // Stride values for efficient indexing
	mu     sync.RWMutex // Mutex for thread safety
	closed uint32       // Atomic flag: 0=open, 1=closed
}

// NewTensor creates a new tensor with the given shape.
// The shape parameter defines the dimensions of the tensor.
// Returns an error if no shape is provided.
func NewTensor(shape ...int) (*Tensor, error) {
	if len(shape) == 0 {
		return nil, ErrTensorInvalidShape
	}
	for _, dim := range shape {
		if dim <= 0 {
			logging.DebugLogf("Invalid shape dimension encountered: %v", shape)
			return nil, ErrTensorInvalidShape
		}
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

	return t, nil
}

// Get retrieves a value from the tensor at the specified indices.
func (t *Tensor) Get(indices ...int) (int8, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		logging.DebugLogf("tensor: operation on closed tensor (method: Get)")
		return 0, ErrTensorClosed
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	if len(indices) != len(t.shape) {
		return 0, ErrTensorInvalidIndices
	}

	index, err := t.calculateIndex(indices)
	if err != nil {
		return 0, err
	}
	if index < 0 || index >= len(t.data) {
		return 0, ErrTensorIndexOutOfRange
	}

	return t.data[index], nil
}

// Set assigns a value to the tensor at the specified indices.
// The value is clamped to the ternary range [-1, 0, 1].
func (t *Tensor) Set(value int8, indices ...int) error {
	if atomic.LoadUint32(&t.closed) == 1 {
		logging.DebugLogf("tensor: operation on closed tensor (method: Set)")
		return ErrTensorClosed
	}
	// Clamp to ternary range
	if value > 0 {
		value = 1
	} else if value < 0 {
		value = -1
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(indices) != len(t.shape) {
		return ErrTensorInvalidIndices
	}

	index, err := t.calculateIndex(indices)
	if err != nil {
		return err
	}
	if index < 0 || index >= len(t.data) {
		return ErrTensorIndexOutOfRange
	}

	t.data[index] = value
	return nil
}

// SetRaw assigns a value to the tensor without clamping (for internal use only).
func (t *Tensor) SetRaw(value int8, indices ...int) error {
	if atomic.LoadUint32(&t.closed) == 1 {
		logging.DebugLogf("tensor: operation on closed tensor (method: SetRaw)")
		return ErrTensorClosed
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(indices) != len(t.shape) {
		return ErrTensorInvalidIndices
	}

	index, err := t.calculateIndex(indices)
	if err != nil {
		return err
	}
	if index < 0 || index >= len(t.data) {
		return ErrTensorIndexOutOfRange
	}

	t.data[index] = value // No clamping
	return nil
}

// Data returns a reference to the underlying data array.
// The caller must not modify the returned slice.
func (t *Tensor) Data() ([]int8, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		logging.DebugLogf("tensor: operation on closed tensor (method: Data)")
		return nil, ErrTensorClosed
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.data, nil
}

// Shape returns a reference to the tensor's dimensions.
// The caller must not modify the returned slice.
func (t *Tensor) Shape() ([]int, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		logging.DebugLogf("tensor: operation on closed tensor (method: Shape)")
		return nil, ErrTensorClosed
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.shape, nil
}

// ParallelForEach processes each element in parallel using the provided function.
// The function is called with the indices and value for each element.
func (t *Tensor) ParallelForEach(fn func(indices []int, value int8)) error {
	if atomic.LoadUint32(&t.closed) == 1 {
		logging.DebugLogf("tensor: operation on closed tensor (method: ParallelForEach)")
		return ErrTensorClosed
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	// Create a copy of the data to avoid race conditions
	data := make([]int8, len(t.data))
	copy(data, t.data)

	// Get number of CPU cores
	numCPU := runtime.NumCPU()
	if numCPU < 1 {
		numCPU = 1
	}

	// Calculate chunk size
	chunkSize := len(data) / numCPU
	if chunkSize < 1 {
		chunkSize = 1
	}

	// Create wait group for synchronization
	var wg sync.WaitGroup
	wg.Add(numCPU)

	// Process chunks in parallel
	for i := 0; i < numCPU; i++ {
		go func(start int) {
			defer wg.Done()

			// Calculate end index
			end := start + chunkSize
			if end > len(data) {
				end = len(data)
			}

			// Process chunk
			for j := start; j < end; j++ {
				indices := t.calculateIndices(j)
				fn(indices, data[j])
			}
		}(i * chunkSize)
	}

	// Wait for all goroutines to complete
	wg.Wait()
	return nil
}

// Close releases all resources associated with the tensor.
// After calling Close, the tensor cannot be used anymore.
func (t *Tensor) Close() error {
	if t == nil {
		return ErrNilTensor
	}
	if atomic.CompareAndSwapUint32(&t.closed, 0, 1) {
		// Store shape for debug logging before clearing fields
		shape := make([]int, len(t.shape))
		copy(shape, t.shape)
		logging.DebugLogf("Closing tensor with shape: %v", shape)

		// Clear fields
		t.data = nil
		t.shape = nil
		t.stride = nil
		runtime.GC()
	}
	return nil
}

// calculateIndex converts multi-dimensional indices to a linear index.
// Returns an error if the indices are invalid.
func (t *Tensor) calculateIndex(indices []int) (int, error) {
	if len(indices) != len(t.shape) {
		return 0, ErrTensorInvalidIndices
	}
	index := 0
	for i, idx := range indices {
		if idx < 0 || idx >= t.shape[i] {
			return 0, ErrTensorIndexOutOfRange
		}
		index += idx * t.stride[i]
	}
	return index, nil
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

// equalShape checks if two shapes are equal
func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Reshape creates a new tensor with the same data but different shape.
// The total number of elements must remain the same.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		return nil, ErrTensorClosed
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	// Validate new shape
	for _, dim := range shape {
		if dim <= 0 {
			return nil, ErrTensorInvalidShape
		}
	}

	// Calculate total size of new shape
	newSize := 1
	for _, dim := range shape {
		newSize *= dim
	}

	// Verify total size matches
	oldSize := 1
	for _, dim := range t.shape {
		oldSize *= dim
	}

	if newSize != oldSize {
		return nil, ErrTensorInvalidReshape
	}

	// Create new tensor with same data but new shape
	result, err := NewTensor(shape...)
	if err != nil {
		return nil, err
	}

	// Copy data
	copy(result.data, t.data)

	return result, nil
}

// NewTensorFromData creates a new tensor from existing data.
// The shape is inferred from the data length.
// If rows > 0, creates a 2D tensor with the specified number of rows.
// Otherwise creates a 1D tensor.
func NewTensorFromData(data []int8, rows int) (*Tensor, error) {
	if len(data) == 0 {
		// Return a 1D tensor with zero length
		return &Tensor{
			data:   make([]int8, 0),
			shape:  []int{0},
			stride: []int{1},
		}, nil
	}

	if rows <= 0 {
		// Create 1D tensor
		t := &Tensor{
			data:   make([]int8, len(data)),
			shape:  []int{len(data)},
			stride: []int{1},
		}
		copy(t.data, data)
		return t, nil
	}

	// Create 2D tensor
	cols := len(data) / rows
	if cols*rows != len(data) {
		return nil, ErrTensorInvalidShape // Invalid dimensions
	}

	t := &Tensor{
		data:   make([]int8, len(data)),
		shape:  []int{rows, cols},
		stride: []int{cols, 1},
	}
	copy(t.data, data)
	return t, nil
}

// Transpose creates a new tensor with dimensions reordered according to the given order.
func (t *Tensor) Transpose(order ...int) (*Tensor, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		return nil, ErrTensorClosed
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	// Validate order
	if len(order) != len(t.shape) {
		return nil, ErrTensorInvalidTranspose
	}

	// Check for duplicate dimensions
	seen := make(map[int]bool)
	for _, dim := range order {
		if dim < 0 || dim >= len(t.shape) {
			return nil, ErrTensorInvalidDimension
		}
		if seen[dim] {
			return nil, ErrTensorDuplicateDimension
		}
		seen[dim] = true
	}

	// Calculate new shape and stride
	newShape := make([]int, len(t.shape))
	newStride := make([]int, len(t.shape))
	for i, dim := range order {
		newShape[i] = t.shape[dim]
		newStride[i] = t.stride[dim]
	}

	// Create new tensor
	result, err := NewTensor(newShape...)
	if err != nil {
		return nil, err
	}

	// Copy data with reordered indices
	for i := 0; i < len(t.data); i++ {
		oldIndices := t.calculateIndices(i)
		newIndices := make([]int, len(order))
		for j, dim := range order {
			newIndices[j] = oldIndices[dim]
		}
		newIndex, err := result.calculateIndex(newIndices)
		if err != nil {
			return nil, err
		}
		result.data[newIndex] = t.data[i]
	}

	return result, nil
}

// Repeat creates a new tensor by repeating the tensor along the specified dimension.
func (t *Tensor) Repeat(dim int, count int) (*Tensor, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		return nil, ErrTensorClosed
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	// Validate dimension
	if dim < 0 || dim >= len(t.shape) {
		return nil, ErrTensorInvalidRepeat
	}

	// Validate count
	if count <= 0 {
		return nil, ErrTensorInvalidRepeatCount
	}

	// Calculate new shape
	newShape := make([]int, len(t.shape))
	copy(newShape, t.shape)
	newShape[dim] *= count

	// Create new tensor
	result, err := NewTensor(newShape...)
	if err != nil {
		return nil, err
	}

	// Copy data with repetition
	for i := 0; i < len(t.data); i++ {
		oldIndices := t.calculateIndices(i)
		for c := 0; c < count; c++ {
			newIndices := make([]int, len(oldIndices))
			copy(newIndices, oldIndices)
			newIndices[dim] = oldIndices[dim] + c*t.shape[dim]
			newIndex, err := result.calculateIndex(newIndices)
			if err != nil {
				return nil, err
			}
			result.data[newIndex] = t.data[i]
		}
	}

	return result, nil
}

// Add performs element-wise addition of two tensors.
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	if t == nil || other == nil {
		return nil, ErrNilTensor
	}
	if atomic.LoadUint32(&t.closed) == 1 || atomic.LoadUint32(&other.closed) == 1 {
		return nil, ErrTensorClosed
	}

	// Lock both tensors for reading
	t.mu.RLock()
	other.mu.RLock()
	defer t.mu.RUnlock()
	defer other.mu.RUnlock()

	// Validate shapes
	if !equalShape(t.shape, other.shape) {
		return nil, ErrTensorShapeMismatch
	}

	// Create result tensor
	result, err := NewTensor(t.shape...)
	if err != nil {
		return nil, err
	}

	// Perform addition
	for i := 0; i < len(t.data); i++ {
		sum := int32(t.data[i]) + int32(other.data[i])
		if sum > 127 {
			sum = 127
		} else if sum < -128 {
			sum = -128
		}
		result.data[i] = int8(sum)
	}

	return result, nil
}

// SetTernary sets a value at the specified indices, clamping to ternary range (-1, 0, +1).
func (t *Tensor) SetTernary(value int8, indices ...int) error {
	if atomic.LoadUint32(&t.closed) == 1 {
		return ErrTensorClosed
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(indices) != len(t.shape) {
		return ErrTensorInvalidIndices
	}

	index, err := t.calculateIndex(indices)
	if err != nil {
		return err
	}
	if index < 0 || index >= len(t.data) {
		return ErrTensorIndexOutOfRange
	}

	// Clamp to ternary range
	if value > 0 {
		value = 1
	} else if value < 0 {
		value = -1
	}

	t.data[index] = value
	return nil
}

// MatMul performs matrix multiplication between two tensors.
// The operation is optimized for ternary values (-1, 0, +1).
func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		logging.DebugLogf("tensor: operation on closed tensor (method: MatMul)")
		return nil, ErrTensorClosed
	}
	if atomic.LoadUint32(&other.closed) == 1 {
		logging.DebugLogf("tensor: operation on closed tensor (method: MatMul)")
		return nil, ErrTensorClosed
	}

	t.mu.RLock()
	defer t.mu.RUnlock()
	other.mu.RLock()
	defer other.mu.RUnlock()

	// Get shapes
	tShape := t.shape
	otherShape := other.shape

	// Validate shapes for matrix multiplication
	if len(tShape) < 2 || len(otherShape) < 2 {
		return nil, ErrTensorInvalidShape
	}

	// Get dimensions
	m := tShape[0]
	n := tShape[1]
	p := otherShape[1]

	// Create output tensor
	result, err := NewTensor(m, p)
	if err != nil {
		return nil, err
	}

	// Perform matrix multiplication
	for i := 0; i < m; i++ {
		for j := 0; j < p; j++ {
			var sum int32
			for k := 0; k < n; k++ {
				a, err := t.Get(i, k)
				if err != nil {
					result.Close()
					return nil, err
				}
				b, err := other.Get(k, j)
				if err != nil {
					result.Close()
					return nil, err
				}
				sum += int32(a) * int32(b)
			}
			// Convert to ternary value
			var ternary int8
			if sum > 0 {
				ternary = 1
			} else if sum < 0 {
				ternary = -1
			}
			if err := result.Set(ternary, i, j); err != nil {
				result.Close()
				return nil, err
			}
		}
	}

	return result, nil
}

// Scale multiplies each element of the tensor by a scalar value.
// The result is converted to a ternary value (-1, 0, +1).
func (t *Tensor) Scale(scale float32) (*Tensor, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		logging.DebugLogf("tensor: operation on closed tensor (method: Scale)")
		return nil, ErrTensorClosed
	}

	t.mu.RLock()
	defer t.mu.RUnlock()

	// Create output tensor with same shape
	result, err := NewTensor(t.shape...)
	if err != nil {
		return nil, err
	}

	// Scale each element
	for i := 0; i < len(t.data); i++ {
		scaled := float32(t.data[i]) * scale
		// Convert to ternary value
		var ternary int8
		if scaled > 0.5 {
			ternary = 1
		} else if scaled < -0.5 {
			ternary = -1
		}
		result.data[i] = ternary
	}

	return result, nil
}

// Softmax applies the softmax function along the specified axis.
// The result is converted to ternary values (-1, 0, +1).
func (t *Tensor) Softmax(axis int) (*Tensor, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		logging.DebugLogf("tensor: operation on closed tensor (method: Softmax)")
		return nil, ErrTensorClosed
	}

	t.mu.RLock()
	defer t.mu.RUnlock()

	// Validate axis
	if axis < 0 || axis >= len(t.shape) {
		return nil, ErrTensorInvalidDimension
	}

	// Create output tensor with same shape
	result, err := NewTensor(t.shape...)
	if err != nil {
		return nil, err
	}

	// Calculate softmax along the specified axis
	axisSize := t.shape[axis]
	axisStride := t.stride[axis]

	// For each position along other dimensions
	for i := 0; i < len(t.data); i += axisStride * axisSize {
		// Find max value for numerical stability
		var maxVal float32
		for j := 0; j < axisSize; j++ {
			val := float32(t.data[i+j*axisStride])
			if val > maxVal {
				maxVal = val
			}
		}

		// Calculate exp and sum
		var sum float32
		exps := make([]float32, axisSize)
		for j := 0; j < axisSize; j++ {
			val := float32(t.data[i+j*axisStride])
			exp := float32(math.Exp(float64(val - maxVal)))
			exps[j] = exp
			sum += exp
		}

		// Normalize and convert to ternary
		for j := 0; j < axisSize; j++ {
			prob := exps[j] / sum
			var ternary int8
			if prob > 0.5 {
				ternary = 1
			} else if prob < 0.5 {
				ternary = -1
			}
			result.data[i+j*axisStride] = ternary
		}
	}

	return result, nil
}
