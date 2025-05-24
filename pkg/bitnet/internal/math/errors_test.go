package math

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestErrorDefinitions verifies that all error definitions are properly set up
// and can be used for error checking.
func TestErrorDefinitions(t *testing.T) {
	tests := []struct {
		name    string
		err     error
		message string
	}{
		{
			name:    "ErrInvalidInputShape",
			err:     ErrInvalidInputShape,
			message: "math: invalid input shape",
		},
		{
			name:    "ErrInvalidDimensions",
			err:     ErrInvalidDimensions,
			message: "math: invalid dimensions",
		},
		{
			name:    "ErrNonSquareMatrix",
			err:     ErrNonSquareMatrix,
			message: "math: must be square matrix",
		},
		{
			name:    "ErrDimensionMismatch",
			err:     ErrDimensionMismatch,
			message: "math: dimension mismatch",
		},
		{
			name:    "ErrInvalidHeadCount",
			err:     ErrInvalidHeadCount,
			message: "math: invalid number of heads",
		},
		{
			name:    "ErrInvalidHeadDimension",
			err:     ErrInvalidHeadDimension,
			message: "math: invalid head dimension",
		},
		{
			name:    "ErrHiddenDimMismatch",
			err:     ErrHiddenDimMismatch,
			message: "math: hidden dimension mismatch",
		},
		{
			name:    "ErrInvalidGammaShape",
			err:     ErrInvalidGammaShape,
			message: "math: gamma must be 1D tensor with matching hidden dimension",
		},
		{
			name:    "ErrLinearInputShape",
			err:     ErrLinearInputShape,
			message: "linear: input must be 2D or 3D tensor",
		},
		{
			name:    "ErrLinearInputDimension",
			err:     ErrLinearInputDimension,
			message: "linear: input dimension mismatch",
		},
		{
			name:    "ErrLinearWeightsShape",
			err:     ErrLinearWeightsShape,
			message: "linear: invalid weights shape",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test error message
			assert.Equal(t, tt.message, tt.err.Error())

			// Test error type
			assert.True(t, errors.Is(tt.err, tt.err))

			// Test error wrapping
			wrappedErr := errors.New("wrapped: " + tt.err.Error())
			assert.False(t, errors.Is(wrappedErr, tt.err))
		})
	}
}

// TestErrorUniqueness verifies that all error definitions are unique
// and not aliases of each other.
func TestErrorUniqueness(t *testing.T) {
	allErrors := []error{
		ErrInvalidInputShape,
		ErrInvalidDimensions,
		ErrNonSquareMatrix,
		ErrDimensionMismatch,
		ErrInvalidHeadCount,
		ErrInvalidHeadDimension,
		ErrHiddenDimMismatch,
		ErrInvalidGammaShape,
		ErrLinearInputShape,
		ErrLinearInputDimension,
		ErrLinearWeightsShape,
	}

	// Check that each error is unique
	for i, err1 := range allErrors {
		for j, err2 := range allErrors {
			if i != j {
				assert.False(t, errors.Is(err1, err2),
					"Error %v should not be an alias of %v", err1, err2)
			}
		}
	}
}

// TestErrorUsage demonstrates how to use these errors in practice
// and verifies that error checking works as expected.
func TestErrorUsage(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		checkErr error
		wantIs   bool
	}{
		{
			name:     "exact match",
			err:      ErrInvalidInputShape,
			checkErr: ErrInvalidInputShape,
			wantIs:   true,
		},
		{
			name:     "different errors",
			err:      ErrInvalidInputShape,
			checkErr: ErrInvalidDimensions,
			wantIs:   false,
		},
		{
			name:     "wrapped error",
			err:      errors.New("wrapped: " + ErrInvalidInputShape.Error()),
			checkErr: ErrInvalidInputShape,
			wantIs:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.wantIs, errors.Is(tt.err, tt.checkErr))
		})
	}
}

// TestErrorMessages verifies that error messages are properly formatted
// and contain the expected information.
func TestErrorMessages(t *testing.T) {
	tests := []struct {
		name    string
		err     error
		prefix  string
		message string
	}{
		{
			name:    "math package error",
			err:     ErrInvalidInputShape,
			prefix:  "math:",
			message: "invalid input shape",
		},
		{
			name:    "linear package error",
			err:     ErrLinearInputShape,
			prefix:  "linear:",
			message: "input must be 2D or 3D tensor",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errMsg := tt.err.Error()
			assert.Contains(t, errMsg, tt.prefix)
			assert.Contains(t, errMsg, tt.message)
		})
	}
}
