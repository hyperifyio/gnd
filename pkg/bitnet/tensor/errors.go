// Package tensor defines error types for BitNet's quantized tensor operations.
//
// # Error Definitions for BitNet Tensor Operations
//
// This file provides standardized error types used throughout the tensor package
// for consistent error handling in BitNet's quantized operations.
//
// Key aspects:
//   - Standardized error messages for tensor operations.
//   - Clear error types for common tensor operation failures.
//   - Used throughout BitNet's tensor package.
//
// Usage:
//   - Used for error handling in tensor operations.
//   - Maintainers should use these error types for consistency.
//
// Caveats:
//   - Error messages should be kept consistent with BitNet's error handling.
//   - Any change must be validated against end-to-end BitNet inference.
//
// For more details, see BitNet issue #190 and the BitNet project documentation.
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
