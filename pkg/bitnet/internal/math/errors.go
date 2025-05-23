// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import "errors"

// Common error definitions for the math package.
//
// These errors are used throughout the math package to indicate
// invalid input shapes, dimension mismatches, and other issues
// encountered during tensor operations, attention mechanisms,
// and linear transformations.
var (
	// ErrInvalidInputShape is returned when a tensor has an invalid shape for the operation.
	ErrInvalidInputShape = errors.New("math: invalid input shape")
	// ErrInvalidDimensions is returned when tensor dimensions are not as expected.
	ErrInvalidDimensions = errors.New("math: invalid dimensions")
	// ErrNonSquareMatrix is returned when a matrix is expected to be square but is not.
	ErrNonSquareMatrix = errors.New("math: must be square matrix")
	// ErrDimensionMismatch is returned when tensor dimensions do not match for an operation.
	ErrDimensionMismatch = errors.New("math: dimension mismatch")
	// ErrInvalidHeadCount is returned when the number of attention heads is invalid.
	ErrInvalidHeadCount = errors.New("math: invalid number of heads")
	// ErrInvalidHeadDimension is returned when the head dimension is invalid for attention.
	ErrInvalidHeadDimension = errors.New("math: invalid head dimension")
	// ErrHiddenDimMismatch is returned when the hidden dimension does not match the expected value.
	ErrHiddenDimMismatch = errors.New("math: hidden dimension mismatch")
	// ErrInvalidGammaShape is returned when the gamma parameter for layer normalization is not 1D or does not match the hidden dimension.
	ErrInvalidGammaShape = errors.New("math: gamma must be 1D tensor with matching hidden dimension")

	// ErrLinearInputShape is returned when the input to a linear layer is not 2D or 3D.
	ErrLinearInputShape = errors.New("linear: input must be 2D or 3D tensor")
	// ErrLinearInputDimension is returned when the input dimension does not match the linear layer's expected input dimension.
	ErrLinearInputDimension = errors.New("linear: input dimension mismatch")
	// ErrLinearWeightsShape is returned when the weights for a linear layer have an invalid shape.
	ErrLinearWeightsShape = errors.New("linear: invalid weights shape")
)
