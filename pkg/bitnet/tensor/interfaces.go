package tensor

// TensorOwner represents an entity that has full control over a tensor's lifecycle.
// Only owners can call Close() on tensors.
type TensorOwner interface {
	Close()
}

// TensorReader represents an entity that can only read tensor data.
// Readers cannot modify or close tensors.
type TensorReader interface {
	Get(indices ...int) int8
	Shape() []int
	Data() []int8
}

// TensorWriter represents an entity that can read and write tensor data.
// Writers cannot close tensors.
type TensorWriter interface {
	TensorReader
	Set(value int8, indices ...int)
}

// TensorParallelProcessor represents an entity that can perform parallel operations on tensors.
// Parallel processors cannot close tensors.
type TensorParallelProcessor interface {
	ParallelForEach(fn func(indices []int, value int8))
}

// TensorOperations defines the operations that can be performed on a tensor.
type TensorOperations interface {
	Get(indices ...int) int8
	Set(value int8, indices ...int)
	Shape() []int
	Data() []int8
	Close() error
	Reshape(shape ...int) TensorOperations
}

// Verify interface implementations
var (
	_ TensorOperations = (*Tensor)(nil)
)
