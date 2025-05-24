package tensor

// rawTensor represents a 2D matrix of int8 values without locking or clamping
type rawTensor struct {
	data []int8
	rows int
	cols int
}

// newRawTensor creates a new rawTensor with the given dimensions
func newRawTensor(rows, cols int) *rawTensor {
	if rows <= 0 || cols <= 0 {
		panic("rawTensor: dimensions must be positive")
	}
	return &rawTensor{
		data: make([]int8, rows*cols),
		rows: rows,
		cols: cols,
	}
}

// newRawTensorFrom creates a rawTensor from an existing Tensor
func newRawTensorFrom(t TensorReader) *rawTensor {
	shape := t.Shape()
	if len(shape) != 2 {
		panic("rawTensor: input must be 2D")
	}
	rows, cols := shape[0], shape[1]
	rt := newRawTensor(rows, cols)
	data := t.Data()
	for i := 0; i < len(data); i++ {
		rt.data[i] = data[i] // No clamping
	}
	return rt
}

// Get retrieves a value from the tensor at the specified indices
func (r *rawTensor) Get(indices ...int) int8 {
	if len(indices) != 2 {
		panic("rawTensor: Get requires exactly 2 indices")
	}
	return r.data[indices[0]*r.cols+indices[1]]
}

// Set assigns a value to the tensor at the specified indices
func (r *rawTensor) Set(value int8, indices ...int) {
	if len(indices) != 2 {
		panic("rawTensor: Set requires exactly 2 indices")
	}
	r.data[indices[0]*r.cols+indices[1]] = value // No clamping
}

// Data returns the underlying data slice
func (r *rawTensor) Data() []int8 {
	return r.data
}

// Shape returns the dimensions of the tensor
func (r *rawTensor) Shape() []int {
	return []int{r.rows, r.cols}
}

// Close is a no-op for rawTensor as it doesn't manage resources
func (r *rawTensor) Close() error { return nil }

// Reshape creates a new rawTensor with the given shape
func (r *rawTensor) Reshape(shape ...int) TensorOperations {
	if len(shape) != 2 {
		panic("rawTensor: Reshape requires exactly 2 dimensions")
	}
	rows, cols := shape[0], shape[1]
	if rows*cols != len(r.data) {
		panic("rawTensor: cannot reshape tensor with different total size")
	}
	return &rawTensor{
		data: r.data,
		rows: rows,
		cols: cols,
	}
}

// ParallelForEach processes each element in parallel
func (r *rawTensor) ParallelForEach(fn func(indices []int, value int8)) {
	for i := 0; i < r.rows; i++ {
		for j := 0; j < r.cols; j++ {
			fn([]int{i, j}, r.data[i*r.cols+j])
		}
	}
}

// Verify interface implementations
var (
	_ TensorReader     = (*rawTensor)(nil)
	_ TensorWriter     = (*rawTensor)(nil)
	_ TensorOperations = (*rawTensor)(nil)
)
