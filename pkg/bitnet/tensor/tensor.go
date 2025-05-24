// Package tensor implements a multi-dimensional array data structure optimized
// for ternary values (-1, 0, +1). It provides efficient operations for tensor
// manipulation, including reshaping, transposition, and parallel processing.
// The package is designed for use in neural network computations with a focus
// on memory efficiency and thread safety.
package tensor

import (
	"errors"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	bitneterrors "github.com/hyperifyio/gnd/pkg/bitnet/errors"
	"github.com/hyperifyio/gnd/pkg/bitnet/internal/math/utils"
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

// DebugLog logs debug information to stderr using the configured logger.
// Deprecated: Use logging.DebugLogf instead.
func DebugLog(format string, args ...interface{}) {
	logging.DebugLogf(format, args...)
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
		DebugLog("tensor: operation on closed tensor (method: Get)")
		return 0, bitneterrors.ErrTensorClosed
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
		DebugLog("tensor: operation on closed tensor (method: Set)")
		return bitneterrors.ErrTensorClosed
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

// setRaw assigns a value to the tensor without clamping (for internal use only).
func (t *Tensor) setRaw(value int8, indices ...int) error {
	if atomic.LoadUint32(&t.closed) == 1 {
		DebugLog("tensor: operation on closed tensor (method: setRaw)")
		return bitneterrors.ErrTensorClosed
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
		DebugLog("tensor: operation on closed tensor (method: Data)")
		return nil, bitneterrors.ErrTensorClosed
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.data, nil
}

// Shape returns a reference to the tensor's dimensions.
// The caller must not modify the returned slice.
func (t *Tensor) Shape() ([]int, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		DebugLog("tensor: operation on closed tensor (method: Shape)")
		return nil, bitneterrors.ErrTensorClosed
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.shape, nil
}

// ParallelForEach processes each element in parallel using the provided function.
// The function is called with the indices and value for each element.
func (t *Tensor) ParallelForEach(fn func(indices []int, value int8)) error {
	if atomic.LoadUint32(&t.closed) == 1 {
		DebugLog("tensor: operation on closed tensor (method: ParallelForEach)")
		return bitneterrors.ErrTensorClosed
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
		return bitneterrors.ErrNilTensor
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
		return nil, bitneterrors.ErrTensorClosed
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
		return nil, bitneterrors.ErrTensorClosed
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
		return nil, bitneterrors.ErrTensorClosed
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
		return nil, bitneterrors.ErrNilTensor
	}
	if atomic.LoadUint32(&t.closed) == 1 || atomic.LoadUint32(&other.closed) == 1 {
		return nil, bitneterrors.ErrTensorClosed
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
		return bitneterrors.ErrTensorClosed
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
// The last dimension of the first tensor must match the second-to-last
// dimension of the second tensor.
// Returns a new tensor with the result.
func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	if t == nil || other == nil {
		return nil, bitneterrors.ErrNilTensor
	}
	if atomic.LoadUint32(&t.closed) == 1 || atomic.LoadUint32(&other.closed) == 1 {
		return nil, bitneterrors.ErrTensorClosed
	}

	t.mu.RLock()
	other.mu.RLock()
	defer t.mu.RUnlock()
	defer other.mu.RUnlock()

	// Get shapes
	tShape, err := t.Shape()
	if err != nil {
		return nil, err
	}
	oShape, err := other.Shape()
	if err != nil {
		return nil, err
	}

	// Add debug output for shape and stride
	logging.DebugLogf("MatMul: t shape: %v, t stride: %v, other shape: %v, other stride: %v", tShape, t.stride, oShape, other.stride)

	// Validate shapes
	if len(tShape) < 2 || len(oShape) < 2 {
		return nil, bitneterrors.ErrInvalidShape
	}
	if tShape[len(tShape)-1] != oShape[len(oShape)-2] {
		return nil, bitneterrors.ErrInvalidShape
	}

	// Compute broadcasted batch shape
	batchShape := utils.BroadcastShapes(tShape[:len(tShape)-2], oShape[:len(oShape)-2])
	if batchShape == nil {
		return nil, bitneterrors.ErrInvalidShape
	}

	// Output shape: batchShape + [tShape[-2], oShape[-1]]
	outShape := append(batchShape, tShape[len(tShape)-2], oShape[len(oShape)-1])

	// Create result tensor
	result, err := NewTensor(outShape...)
	if err != nil {
		return nil, err
	}

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
	resultShape, err := result.Shape()
	if err != nil {
		return nil, err
	}
	logging.DebugLogf("MatMul: result shape: %v", resultShape)

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
					logging.DebugLogf("MatMul: indices=%v, tBatchOffset=%d, oBatchOffset=%d", indices, tBatchOffset, oBatchOffset)
				}

				// Perform matrix multiplication for this batch
				for k := 0; k < tShape[len(tShape)-1]; k++ {
					tIdx := tBatchOffset + indices[len(batchShape)]*tShape[len(tShape)-1] + k
					oIdx := oBatchOffset + k*oShape[len(oShape)-1] + indices[len(batchShape)+1]
					if tIdx < 0 || tIdx >= len(t.data) || oIdx < 0 || oIdx >= len(other.data) {
						logging.DebugLogf("MatMul: tIdx=%d, oIdx=%d, t.data len=%d, other.data len=%d", tIdx, oIdx, len(t.data), len(other.data))
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

// Softmax applies the softmax function along the specified axis.
// Returns a new tensor with the result.
func (t *Tensor) Softmax(axis int) (*Tensor, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		return nil, bitneterrors.ErrTensorClosed
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	// Support negative axis (e.g., -1 means last axis)
	if axis < 0 {
		axis += len(t.shape)
	}
	// Validate axis
	if axis < 0 || axis >= len(t.shape) {
		return nil, bitneterrors.ErrInvalidAxis
	}

	// Create result tensor
	result, err := NewTensor(t.shape...)
	if err != nil {
		return nil, err
	}

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
					idx, err := t.calculateIndex(indices)
					if err != nil {
						continue
					}
					val := float32(t.data[idx])
					if val > maxVal {
						maxVal = val
					}
				}
				for k := 0; k < t.shape[axis]; k++ {
					indices[axis] = k
					idx, err := t.calculateIndex(indices)
					if err != nil {
						continue
					}
					val := float32(t.data[idx])
					sum += float32(math.Exp(float64(val - maxVal)))
				}
				indices[axis] = j % t.shape[axis]
				idx, err := t.calculateIndex(indices)
				if err != nil {
					continue
				}
				val := float32(t.data[idx])
				result.data[j] = int8(float32(math.Exp(float64(val-maxVal))) / sum * 127)
			}
		}(i)
	}
	wg.Wait()

	return result, nil
}

// Scale multiplies all values in the tensor by the given scale factor.
// Returns a new tensor with the result.
func (t *Tensor) Scale(scale float32) (*Tensor, error) {
	if atomic.LoadUint32(&t.closed) == 1 {
		return nil, bitneterrors.ErrTensorClosed
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	// Create result tensor
	result, err := NewTensor(t.shape...)
	if err != nil {
		return nil, err
	}

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

	return result, nil
}
