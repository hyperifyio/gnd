// Package tensor implements a multi-dimensional array data structure optimized
// for ternary values (-1, 0, +1). It provides efficient operations for tensor
// manipulation, including reshaping, transposition, and parallel processing.
// The package is designed for use in neural network computations with a focus
// on memory efficiency and thread safety.
package tensor

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/hyperifyio/gnd/pkg/bitnet/errors"
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
	Close() error
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
	closed uint32       // Atomic flag: 0=open, 1=closed
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
	for _, dim := range shape {
		if dim <= 0 {
			loggers.Printf(loggers.Debug, "Invalid shape dimension encountered: %v", shape)
			panic("tensor: invalid shape dimension")
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

	return t
}

// Get retrieves a value from the tensor at the specified indices.
// Panics if the tensor is closed, indices are invalid, or out of range.
func (t *Tensor) Get(indices ...int) int8 {
	if atomic.LoadUint32(&t.closed) == 1 {
		panic("tensor: Get called on closed tensor")
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

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
	if atomic.LoadUint32(&t.closed) == 1 {
		panic("tensor: Set called on closed tensor")
	}
	t.mu.Lock()
	defer t.mu.Unlock()

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
	if atomic.LoadUint32(&t.closed) == 1 {
		panic("tensor: Set called on closed tensor")
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(indices) != len(t.shape) {
		panic("tensor: invalid number of indices")
	}

	index := t.calculateIndex(indices)
	if index < 0 || index >= len(t.data) {
		panic("tensor: index out of range")
	}

	t.data[index] = value // No clamping
}

// Data returns a reference to the underlying data array.
// The caller must not modify the returned slice.
// Panics if the tensor is closed.
func (t *Tensor) Data() []int8 {
	if atomic.LoadUint32(&t.closed) == 1 {
		panic("tensor: Data called on closed tensor")
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.data
}

// Shape returns a reference to the tensor's dimensions.
// The caller must not modify the returned slice.
// Panics if the tensor is closed.
func (t *Tensor) Shape() []int {
	if atomic.LoadUint32(&t.closed) == 1 {
		panic("tensor: Shape called on closed tensor")
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.shape
}

// ParallelForEach processes each element in parallel using the provided function.
// The function is called with the indices and value for each element.
// Panics if the tensor is closed.
func (t *Tensor) ParallelForEach(fn func(indices []int, value int8)) {
	if atomic.LoadUint32(&t.closed) == 1 {
		panic("tensor: ParallelForEach called on closed tensor")
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
}

// Close releases all resources associated with the tensor.
// After calling Close, the tensor cannot be used anymore.
func (t *Tensor) Close() error {
	if t == nil {
		return errors.ErrNilTensor
	}
	if atomic.CompareAndSwapUint32(&t.closed, 0, 1) {
		// Store shape for debug logging before clearing fields
		shape := make([]int, len(t.shape))
		copy(shape, t.shape)
		fmt.Printf("[DEBUG] Closing tensor with shape: %v\n", shape)

		// Clear fields
		t.data = nil
		t.shape = nil
		t.stride = nil
		runtime.GC()
	}
	return nil
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
// Panics if the tensor is closed or if the new shape is invalid.
func (t *Tensor) Reshape(shape ...int) TensorOperations {
	if t == nil {
		return nil
	}
	if atomic.LoadUint32(&t.closed) == 1 {
		return nil
	}

	t.mu.RLock()
	defer t.mu.RUnlock()

	// Store current shape for debug logging
	currentShape := make([]int, len(t.shape))
	copy(currentShape, t.shape)
	fmt.Printf("[DEBUG] Reshape: t shape: %v, new dims: %v\n", currentShape, shape)

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
		panic(fmt.Sprintf("tensor: cannot reshape tensor of size %d to shape %v", len(t.data), shape))
	}

	// Calculate new stride
	newStride := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		newStride[i] = stride
		stride *= shape[i]
	}

	// Create new tensor with same data
	return &Tensor{
		data:   t.data,
		shape:  shape,
		stride: newStride,
	}
}

// NewTensorFromData creates a new tensor from existing data.
// The shape is inferred from the data length.
// If rows > 0, creates a 2D tensor with the specified number of rows.
// Otherwise creates a 1D tensor.
func NewTensorFromData(data []int8, rows int) *Tensor {
	if len(data) == 0 {
		// Return a 1D tensor with zero length
		return &Tensor{
			data:   make([]int8, 0),
			shape:  []int{0},
			stride: []int{1},
		}
	}

	if rows <= 0 {
		// Create 1D tensor
		t := &Tensor{
			data:   make([]int8, len(data)),
			shape:  []int{len(data)},
			stride: []int{1},
		}
		copy(t.data, data)
		return t
	}

	// Create 2D tensor
	cols := len(data) / rows
	if cols*rows != len(data) {
		return nil // Invalid dimensions
	}

	t := &Tensor{
		data:   make([]int8, len(data)),
		shape:  []int{rows, cols},
		stride: []int{cols, 1},
	}
	copy(t.data, data)
	return t
}

// Transpose creates a new tensor with transposed dimensions.
// The order parameter specifies the new order of dimensions.
// Panics if the tensor is closed or if the order is invalid.
func (t *Tensor) Transpose(order ...int) *Tensor {
	if t == nil {
		return nil
	}
	if atomic.LoadUint32(&t.closed) == 1 {
		return nil
	}
	fmt.Printf("[DEBUG] Transpose: t shape: %v, dims: %v\n", t.Shape(), order)

	// Validate order
	if len(order) != len(t.shape) {
		panic("tensor: invalid transpose order")
	}

	// Check for duplicate dimensions
	seen := make(map[int]bool)
	for _, dim := range order {
		if dim < 0 || dim >= len(t.shape) {
			panic("tensor: invalid dimension in transpose order")
		}
		if seen[dim] {
			panic("tensor: duplicate dimension in transpose order")
		}
		seen[dim] = true
	}

	// Create new shape and stride
	newShape := make([]int, len(order))
	newStride := make([]int, len(order))
	for i, dim := range order {
		newShape[i] = t.shape[dim]
		newStride[i] = t.stride[dim]
	}

	// Create new tensor with same data
	return &Tensor{
		data:   t.data,
		shape:  newShape,
		stride: newStride,
	}
}

// Repeat creates a new tensor by repeating the tensor along the specified dimension.
// The count parameter specifies how many times to repeat.
// Returns nil if the dimension or count is invalid.
func (t *Tensor) Repeat(dim int, count int) *Tensor {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed == 1 {
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
// Returns a new tensor with the result.
func (t *Tensor) Add(other *Tensor) *Tensor {
	if atomic.LoadUint32(&t.closed) == 1 {
		panic("tensor: Add called on closed tensor")
	}
	if atomic.LoadUint32(&other.closed) == 1 {
		panic("tensor: Add called with closed tensor")
	}

	t.mu.RLock()
	other.mu.RLock()
	defer t.mu.RUnlock()
	defer other.mu.RUnlock()

	// Verify shapes match
	if !equalShape(t.shape, other.shape) {
		panic("tensor: cannot add tensors with different shapes")
	}

	// Create result tensor
	result := NewTensor(t.shape...)

	// Add elements in parallel
	numWorkers := runtime.NumCPU()
	chunkSize := (len(t.data) + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for i := 0; i < len(t.data); i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > len(t.data) {
				end = len(t.data)
			}
			for j := start; j < end; j++ {
				sum := int32(t.data[j]) + int32(other.data[j])
				if sum > 127 {
					result.data[j] = 127
				} else if sum < -128 {
					result.data[j] = -128
				} else {
					result.data[j] = int8(sum)
				}
			}
		}(i)
	}
	wg.Wait()

	return result
}

// SetTernary sets a ternary value (-1, 0, +1) at the specified indices.
// The value is clamped to the ternary range.
// Panics if the tensor is closed, indices are invalid, or out of range.
func (t *Tensor) SetTernary(value int8, indices ...int) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed == 1 {
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

// MatMul performs matrix multiplication between two tensors.
// The last dimension of the first tensor must match the second-to-last
// dimension of the second tensor.
// Returns a new tensor with the result.
func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	if t == nil || other == nil {
		return nil, errors.ErrNilTensor
	}
	if atomic.LoadUint32(&t.closed) == 1 || atomic.LoadUint32(&other.closed) == 1 {
		return nil, errors.ErrTensorClosed
	}
	// Add debug output for shape and stride
	fmt.Printf("[DEBUG] MatMul: t shape: %v, t stride: %v, other shape: %v, other stride: %v\n", t.Shape(), t.stride, other.Shape(), other.stride)

	t.mu.RLock()
	other.mu.RLock()
	defer t.mu.RUnlock()
	defer other.mu.RUnlock()

	// Get shapes
	tShape := t.shape
	oShape := other.shape

	// Validate shapes
	if len(tShape) < 2 || len(oShape) < 2 {
		return nil, errors.ErrInvalidShape
	}
	if tShape[len(tShape)-1] != oShape[len(oShape)-2] {
		return nil, errors.ErrInvalidShape
	}

	// Compute broadcasted batch shape
	batchShape := broadcastShapes(tShape[:len(tShape)-2], oShape[:len(oShape)-2])
	if batchShape == nil {
		return nil, errors.ErrInvalidShape
	}

	// Output shape: batchShape + [tShape[-2], oShape[-1]]
	outShape := append(batchShape, tShape[len(tShape)-2], oShape[len(oShape)-1])

	// Create result tensor
	result := NewTensor(outShape...)

	// Calculate strides for broadcasting
	tBatchStride := make([]int, len(batchShape))
	oBatchStride := make([]int, len(batchShape))

	// Initialize strides for the last two dimensions
	tBatchStride[len(batchShape)-1] = tShape[len(tShape)-2] * tShape[len(tShape)-1]
	oBatchStride[len(batchShape)-1] = oShape[len(oShape)-2] * oShape[len(oShape)-1]

	// Calculate remaining strides
	for i := len(batchShape) - 2; i >= 0; i-- {
		tBatchStride[i] = tBatchStride[i+1] * batchShape[i+1]
		oBatchStride[i] = oBatchStride[i+1] * batchShape[i+1]
	}

	// Add debug output for result shape
	fmt.Printf("[DEBUG] MatMul: result shape: %v\n", result.Shape())

	// Perform matrix multiplication in parallel
	numWorkers := runtime.NumCPU()
	chunkSize := (len(result.data) + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for i := 0; i < len(result.data); i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > len(result.data) {
				end = len(result.data)
			}
			for j := start; j < end; j++ {
				indices := result.calculateIndices(j)
				var sum int32

				// Calculate batch offsets
				tBatchOffset := 0
				oBatchOffset := 0
				for k := 0; k < len(batchShape); k++ {
					idx := indices[k]
					if k < len(tShape)-2 {
						tBatchOffset += idx * tBatchStride[k]
					}
					if k < len(oShape)-2 {
						oBatchOffset += idx * oBatchStride[k]
					}
				}

				// Add debug output for indices and offsets
				if j == start {
					fmt.Printf("[DEBUG] MatMul: indices=%v, tBatchOffset=%d, oBatchOffset=%d\n", indices, tBatchOffset, oBatchOffset)
				}

				// Perform matrix multiplication for this batch
				for k := 0; k < tShape[len(tShape)-1]; k++ {
					tIdx := tBatchOffset + indices[len(batchShape)]*tShape[len(tShape)-1] + k
					oIdx := oBatchOffset + k*oShape[len(oShape)-1] + indices[len(batchShape)+1]
					if tIdx < 0 || tIdx >= len(t.data) || oIdx < 0 || oIdx >= len(other.data) {
						fmt.Printf("[ERROR] MatMul: tIdx=%d, oIdx=%d, t.data len=%d, other.data len=%d\n", tIdx, oIdx, len(t.data), len(other.data))
						continue
					}
					sum += int32(t.data[tIdx]) * int32(other.data[oIdx])
				}

				// Clamp result to int8 range
				if sum > 127 {
					result.data[j] = 127
				} else if sum < -128 {
					result.data[j] = -128
				} else {
					result.data[j] = int8(sum)
				}
			}
		}(i)
	}
	wg.Wait()

	return result, nil
}

// broadcastShapes computes the broadcasted shape for two input shapes, or returns nil if not broadcastable.
func broadcastShapes(a, b []int) []int {
	// Right-align shapes
	n := maxInt(len(a), len(b))
	out := make([]int, n)
	for i := 0; i < n; i++ {
		var ad, bd int
		if i < len(a) {
			ad = a[len(a)-1-i]
		} else {
			ad = 1
		}
		if i < len(b) {
			bd = b[len(b)-1-i]
		} else {
			bd = 1
		}
		if ad != bd && ad != 1 && bd != 1 {
			return nil
		}
		if ad > bd {
			out[n-1-i] = ad
		} else {
			out[n-1-i] = bd
		}
	}
	return out
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Softmax applies the softmax function along the specified axis.
// Returns a new tensor with the result.
func (t *Tensor) Softmax(axis int) (*Tensor, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		panic("tensor: Softmax called on closed tensor")
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	// Support negative axis (e.g., -1 means last axis)
	if axis < 0 {
		axis += len(t.shape)
	}
	// Validate axis
	if axis < 0 || axis >= len(t.shape) {
		return nil, errors.ErrInvalidAxis
	}

	// Create result tensor
	result := NewTensor(t.shape...)

	// Calculate softmax in parallel
	numWorkers := runtime.NumCPU()
	chunkSize := (len(t.data) + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
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
				var maxVal float32 = float32(t.data[j])
				var sum float32 = 0
				for k := 0; k < t.shape[axis]; k++ {
					indices[axis] = k
					val := float32(t.data[t.calculateIndex(indices)])
					if val > maxVal {
						maxVal = val
					}
				}
				for k := 0; k < t.shape[axis]; k++ {
					indices[axis] = k
					val := float32(t.data[t.calculateIndex(indices)])
					sum += float32(math.Exp(float64(val - maxVal)))
				}
				indices[axis] = j % t.shape[axis]
				val := float32(t.data[t.calculateIndex(indices)])
				result.data[j] = int8(float32(math.Exp(float64(val-maxVal))) / sum * 127)
			}
		}(i)
	}
	wg.Wait()

	return result, nil
}

// Scale multiplies all values in the tensor by the given scale factor.
// Returns a new tensor with the result.
func (t *Tensor) Scale(scale float32) *Tensor {
	if atomic.LoadUint32(&t.closed) == 1 {
		panic("tensor: Scale called on closed tensor")
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	// Create result tensor
	result := NewTensor(t.shape...)

	// Scale values in parallel
	numWorkers := runtime.NumCPU()
	chunkSize := (len(t.data) + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for i := 0; i < len(t.data); i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > len(t.data) {
				end = len(t.data)
			}
			for j := start; j < end; j++ {
				scaled := float32(t.data[j]) * scale
				if scaled > 127 {
					result.data[j] = 127
				} else if scaled < -128 {
					result.data[j] = -128
				} else {
					result.data[j] = int8(scaled)
				}
			}
		}(i)
	}
	wg.Wait()

	return result
}

// Verify interface implementations
var (
	_ TensorType        = (*Tensor)(nil)
	_ ParallelProcessor = (*Tensor)(nil)
)
