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
	ops    chan tensorOp
	done   chan struct{}
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
		ops:    make(chan tensorOp, 100), // Buffer size of 100 for better performance
		done:   make(chan struct{}),
	}

	// Start operation handler goroutine
	go t.handleOps()

	return t
}

// Close closes the tensor and signals the handler to exit
func (t *Tensor) Close() {
	close(t.done)
}

// handleOps processes tensor operations in a single goroutine
func (t *Tensor) handleOps() {
	for {
		select {
		case <-t.done:
			// After done, keep draining ops and close result/done channels to unblock senders
			for op := range t.ops {
				if op.opType == "get" && op.resultCh != nil {
					close(op.resultCh)
				} else if op.opType == "set" && op.doneCh != nil {
					close(op.doneCh)
				}
			}
			return
		case op := <-t.ops:
			select {
			case <-t.done:
				if op.opType == "get" && op.resultCh != nil {
					close(op.resultCh)
				} else if op.opType == "set" && op.doneCh != nil {
					close(op.doneCh)
				}
				continue
			default:
				if op.opType == "get" {
					idx := t.calculateIndex(op.indices)
					op.resultCh <- t.data[idx]
					close(op.resultCh)
				} else if op.opType == "set" {
					idx := t.calculateIndex(op.indices)
					if op.value > 1 {
						op.value = 1
					} else if op.value < -1 {
						op.value = -1
					}
					t.data[idx] = op.value
					close(op.doneCh)
				}
			}
		}
	}
}

// calculateIndex calculates the linear index from multi-dimensional indices
func (t *Tensor) calculateIndex(indices []int) int {
	if len(indices) != len(t.shape) {
		panic("invalid number of indices")
	}

	idx := 0
	for i, v := range indices {
		if v < 0 || v >= t.shape[i] {
			panic("index out of range")
		}
		idx += v * t.stride[i]
	}
	return idx
}

// Get returns the value at the given indices
func (t *Tensor) Get(indices ...int) int8 {
	select {
	case <-t.done:
		panic("tensor is closed")
	default:
		resultCh := make(chan int8, 1)
		t.ops <- tensorOp{
			opType:   "get",
			indices:  indices,
			resultCh: resultCh,
		}
		return <-resultCh
	}
}

// Set sets the value at the given indices
func (t *Tensor) Set(value int8, indices ...int) {
	select {
	case <-t.done:
		panic("tensor is closed")
	default:
		doneCh := make(chan struct{})
		t.ops <- tensorOp{
			opType:  "set",
			indices: indices,
			value:   value,
			doneCh:  doneCh,
		}
		<-doneCh
	}
}

// Shape returns the shape of the tensor
func (t *Tensor) Shape() []int {
	return t.shape
}

// Data returns the underlying data array
func (t *Tensor) Data() []int8 {
	return t.data
}

// ParallelForEach applies the given function to each element in parallel
func (t *Tensor) ParallelForEach(fn func(indices []int, value int8)) {
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
		t.forEach(func(indices []int, _ int8) {
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
func (t *Tensor) forEach(fn func(indices []int, value int8)) {
	indices := make([]int, len(t.shape))
	t.forEachRecursive(0, indices, fn)
}

// forEachRecursive recursively traverses the tensor
func (t *Tensor) forEachRecursive(dim int, indices []int, fn func(indices []int, value int8)) {
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
