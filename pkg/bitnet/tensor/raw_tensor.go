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
func newRawTensorFrom(t *Tensor) *rawTensor {
	if len(t.Shape()) != 2 {
		panic("rawTensor: input must be 2D")
	}
	rows, cols := t.Shape()[0], t.Shape()[1]
	rt := newRawTensor(rows, cols)
	data := t.Data()
	for i := 0; i < len(data); i++ {
		rt.data[i] = data[i] // No clamping
	}
	return rt
}

// At returns the value at position (i,j)
func (r *rawTensor) At(i, j int) int8 {
	return r.data[i*r.cols+j]
}

// Set assigns value v to position (i,j)
func (r *rawTensor) Set(i, j int, v int8) {
	r.data[i*r.cols+j] = v // No clamping
}

// Data returns the underlying data slice
func (r *rawTensor) Data() []int8 {
	return r.data
}

// Shape returns the dimensions of the tensor
func (r *rawTensor) Shape() (rows, cols int) {
	return r.rows, r.cols
}
