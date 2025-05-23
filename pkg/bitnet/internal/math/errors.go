// Package math implements mathematical operations for the BitNet model.
package math

import "errors"

// Common error definitions for the math package
var (
	// Tensor validation errors
	ErrInvalidInputShape    = errors.New("math: invalid input shape")
	ErrInvalidDimensions    = errors.New("math: invalid dimensions")
	ErrNonSquareMatrix      = errors.New("math: must be square matrix")
	ErrDimensionMismatch    = errors.New("math: dimension mismatch")
	ErrInvalidHeadCount     = errors.New("math: invalid number of heads")
	ErrInvalidHeadDimension = errors.New("math: invalid head dimension")
	ErrHiddenDimMismatch    = errors.New("math: hidden dimension mismatch")
	ErrInvalidGammaShape    = errors.New("math: gamma must be 1D tensor with matching hidden dimension")

	// Linear layer errors
	ErrLinearInputShape     = errors.New("linear: input must be 2D or 3D tensor")
	ErrLinearInputDimension = errors.New("linear: input dimension mismatch")
	ErrLinearWeightsShape   = errors.New("linear: invalid weights shape")
)
