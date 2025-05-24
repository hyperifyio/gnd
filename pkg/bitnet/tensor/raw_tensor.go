package tensor

import "errors"

var (
	ErrRawTensorInvalidDimensions = errors.New("raw_tensor: dimensions must be positive")
	ErrRawTensorInvalidShape      = errors.New("raw_tensor: input must be 2D")
	ErrRawTensorInvalidIndices    = errors.New("raw_tensor: requires exactly 2 indices")
	ErrRawTensorInvalidReshape    = errors.New("raw_tensor: cannot reshape tensor with different total size")
	ErrRawTensorNilInput          = errors.New("raw_tensor: input tensor is nil")
)

// rawTensor represents a 2D matrix of int8 values without locking or clamping
type rawTensor struct {
	data []int8
	rows int
	cols int
}

// newRawTensor creates a new rawTensor with the given dimensions
func newRawTensor(rows, cols int) (*rawTensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, ErrRawTensorInvalidDimensions
	}
	return &rawTensor{
		data: make([]int8, rows*cols),
		rows: rows,
		cols: cols,
	}, nil
}

// newRawTensorFrom creates a rawTensor from an existing Tensor
func newRawTensorFrom(t *Tensor) (*rawTensor, error) {
	if t == nil {
		DebugLog("raw_tensor: input tensor is nil in newRawTensorFrom")
		return nil, ErrRawTensorNilInput
	}
	shape, err := t.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 {
		return nil, ErrRawTensorInvalidShape
	}
	rows, cols := shape[0], shape[1]
	rt, err := newRawTensor(rows, cols)
	if err != nil {
		return nil, err
	}
	data, err := t.Data()
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(data); i++ {
		rt.data[i] = data[i] // No clamping
	}
	return rt, nil
}

// Get retrieves a value from the tensor at the specified indices
func (r *rawTensor) Get(indices ...int) (int8, error) {
	if len(indices) != 2 {
		return 0, ErrRawTensorInvalidIndices
	}
	return r.data[indices[0]*r.cols+indices[1]], nil
}

// Set assigns a value to the tensor at the specified indices
func (r *rawTensor) Set(value int8, indices ...int) error {
	if len(indices) != 2 {
		return ErrRawTensorInvalidIndices
	}
	r.data[indices[0]*r.cols+indices[1]] = value // No clamping
	return nil
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
func (r *rawTensor) Reshape(shape ...int) (*rawTensor, error) {
	if len(shape) != 2 {
		return nil, ErrRawTensorInvalidIndices
	}
	rows, cols := shape[0], shape[1]
	if rows*cols != len(r.data) {
		return nil, ErrRawTensorInvalidReshape
	}
	return &rawTensor{
		data: r.data,
		rows: rows,
		cols: cols,
	}, nil
}

// ParallelForEach processes each element in parallel
func (r *rawTensor) ParallelForEach(fn func(indices []int, value int8)) {
	for i := 0; i < r.rows; i++ {
		for j := 0; j < r.cols; j++ {
			fn([]int{i, j}, r.data[i*r.cols+j])
		}
	}
}
