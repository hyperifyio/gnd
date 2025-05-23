package tensor

import "errors"

var (
	// ErrTensorClosed is returned when attempting to operate on a closed tensor
	ErrTensorClosed = errors.New("tensor: operation attempted on closed tensor")
	// ErrInvalidShape is returned when a tensor has an invalid shape
	ErrInvalidShape = errors.New("tensor: invalid shape")
	// ErrDimensionMismatch is returned when tensor dimensions don't match for an operation
	ErrDimensionMismatch = errors.New("tensor: dimension mismatch")
)
