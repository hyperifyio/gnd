// Package shape provides shape validation functions for BitNet math operations.
//
// # Shape Validation for BitNet
//
// This package provides shape validation functions used across all math packages in the BitNet implementation.
// It enforces the specific tensor shapes required by the BitNet b1.58-2B 4T model architecture.
//
// Key aspects:
//   - Enforces correct tensor shapes for all quantized operations in BitNet
//   - Validates shapes for attention heads (20 heads with 5 unique K/V sets)
//   - Ensures compatibility with the model's hidden dimension (2560)
//   - Provides clear error messages for shape mismatches
//   - Optimized for inference-only pipeline
//
// Implementation details:
//   - Validates tensor shapes for attention, FFN, and transformer layers
//   - Enforces BitNet-specific constraints (e.g., head dimensions)
//   - Centralizes error handling for shape mismatches
//   - Supports the model's 4096-token context length
//
// Related tasks and dependencies:
//   - #176: Set Model Constants (Architecture Hyperparameters)
//   - #182: Compute Scaled Dot-Product Attention
//   - #185: Feed-Forward Network (FFN) Sublayer
//   - #186: Integrate Attention Sublayer (Pre-Norm & Residual)
//   - #187: Integrate Feed-Forward Sublayer (Pre-Norm & Residual)
//
// Usage:
//   - Used throughout BitNet math packages to validate tensor shapes
//   - Critical for maintaining correct quantized inference
//   - Maintainers should not change shape conventions without full pipeline review
//
// Caveats:
//   - Shape validation is critical for correct quantized inference
//   - Any change must be validated against end-to-end BitNet inference
//   - Performance critical - changes should be benchmarked against existing implementation
//   - Must maintain compatibility with BitNet's binary-weight quantization
//
// For more details, see BitNet issue #170 and the BitNet project documentation.
package shape

import (
	"errors"

	"github.com/hyperifyio/gnd/pkg/bitnet/logging"
)

var (
	// ErrInvalidDimensions is returned when a tensor's shape has the wrong number of dimensions.
	ErrInvalidDimensions = errors.New("invalid number of dimensions")
	// ErrInvalidInputShape is returned when a tensor's shape is invalid for the operation.
	ErrInvalidInputShape = errors.New("invalid input shape")
	// ErrNonSquareMatrix is returned when a matrix is not square.
	ErrNonSquareMatrix = errors.New("matrix must be square")
	// ErrDimensionMismatch is returned when two tensors have mismatched dimensions.
	ErrDimensionMismatch = errors.New("dimension mismatch")
	// ErrInvalidHeadCount is returned when the number of attention heads is invalid.
	ErrInvalidHeadCount = errors.New("invalid number of attention heads")
	// ErrInvalidHeadDimension is returned when the head dimension is invalid.
	ErrInvalidHeadDimension = errors.New("invalid head dimension")
	// ErrHiddenDimMismatch is returned when the hidden dimension does not match the number of heads.
	ErrHiddenDimMismatch = errors.New("hidden dimension must equal num_heads * head_dim")
)

// Common tensor shape dimension constants for attention and transformer layers.
const (
	// MinHeadDim is the minimum allowed head dimension for attention heads.
	MinHeadDim = 8
	// MaxHeadDim is the maximum allowed head dimension for attention heads.
	MaxHeadDim = 256
	// MinNumHeads is the minimum allowed number of attention heads.
	MinNumHeads = 1
	// MaxNumHeads is the maximum allowed number of attention heads.
	MaxNumHeads = 32
)

// Shape represents a tensor's dimensions as a slice of integers.
type Shape []int

// Common shape types for semantic clarity in function signatures.
type (
	// BatchSeqHidden represents a shape of [batch_size, seq_len, hidden_dim].
	BatchSeqHidden Shape
	// BatchHeadsSeqHead represents a shape of [batch_size, num_heads, seq_len, head_dim].
	BatchHeadsSeqHead Shape
	// HiddenHidden represents a shape of [hidden_dim, hidden_dim].
	HiddenHidden Shape
)

// ValidateShape checks if a shape matches any of the expected dimensions.
// If multiple dimensions are provided, the shape must match one of them.
// Returns ErrInvalidDimensions if the shape does not match.
func ValidateShape(shape Shape, expectedDims ...int) error {
	if shape == nil {
		logging.DebugLogf("shape is nil, expected dimensions %v", expectedDims)
		return ErrInvalidDimensions
	}
	for _, dim := range expectedDims {
		if len(shape) == dim {
			return nil
		}
	}
	logging.DebugLogf("shape must have one of dimensions %v, got %dD", expectedDims, len(shape))
	return ErrInvalidDimensions
}

// ValidateBatchSeqHiddenShape checks if a shape has form [batch_size, seq_len, hidden_dim].
// Returns ErrInvalidInputShape if the shape does not match.
func ValidateBatchSeqHiddenShape(shape Shape, name string) error {
	if err := ValidateShape(shape, 3); err != nil {
		logging.DebugLogf("%s: %v", name, err)
		return err
	}
	if shape[0] <= 0 {
		logging.DebugLogf("%s: batch size must be positive, got %d", name, shape[0])
		return ErrInvalidInputShape
	}
	if shape[1] <= 0 {
		logging.DebugLogf("%s: sequence length must be positive, got %d", name, shape[1])
		return ErrInvalidInputShape
	}
	if shape[2] <= 0 {
		logging.DebugLogf("%s: hidden dimension must be positive, got %d", name, shape[2])
		return ErrInvalidInputShape
	}
	return nil
}

// ValidateBatchHeadsSeqHeadShape checks if a shape has form [batch_size, num_heads, seq_len, head_dim]
func ValidateBatchHeadsSeqHeadShape(shape Shape, name string) error {
	if err := ValidateShape(shape, 4); err != nil {
		logging.DebugLogf("%s: %v", name, err)
		return err
	}
	if shape[0] <= 0 {
		logging.DebugLogf("%s: batch size must be positive, got %d", name, shape[0])
		return ErrInvalidInputShape
	}
	if shape[1] < MinNumHeads || shape[1] > MaxNumHeads {
		logging.DebugLogf("%s: number of heads must be between %d and %d, got %d", name, MinNumHeads, MaxNumHeads, shape[1])
		return ErrInvalidHeadCount
	}
	if shape[2] <= 0 {
		logging.DebugLogf("%s: sequence length must be positive, got %d", name, shape[2])
		return ErrInvalidInputShape
	}
	if shape[3] < MinHeadDim || shape[3] > MaxHeadDim {
		logging.DebugLogf("%s: head dimension must be between %d and %d, got %d", name, MinHeadDim, MaxHeadDim, shape[3])
		return ErrInvalidHeadDimension
	}
	return nil
}

// ValidateHiddenHiddenShape checks if a shape has form [hidden_dim, hidden_dim]
func ValidateHiddenHiddenShape(shape Shape, name string) error {
	if err := ValidateShape(shape, 2); err != nil {
		logging.DebugLogf("%s: %v", name, err)
		return err
	}
	if shape[0] <= 0 || shape[1] <= 0 {
		logging.DebugLogf("%s: dimensions must be positive, got %v", name, shape)
		return ErrInvalidInputShape
	}
	if shape[0] != shape[1] {
		logging.DebugLogf("%s must be square matrix, got shape %v", name, shape)
		return ErrNonSquareMatrix
	}
	return nil
}

// ValidateMatchingShapes checks if two shapes match
func ValidateMatchingShapes(shape1, shape2 Shape, name1, name2 string) error {
	if len(shape1) != len(shape2) {
		logging.DebugLogf("%s and %s must have same number of dimensions, got %d and %d",
			name1, name2, len(shape1), len(shape2))
		return ErrDimensionMismatch
	}
	for i := range shape1 {
		if shape1[i] != shape2[i] {
			logging.DebugLogf("%s and %s must have matching dimensions, got %v and %v",
				name1, name2, shape1, shape2)
			return ErrDimensionMismatch
		}
	}
	return nil
}

// ValidateHeadDimensions checks if head dimensions are valid
func ValidateHeadDimensions(hiddenDim, numHeads, headDim int) error {
	if numHeads < MinNumHeads || numHeads > MaxNumHeads {
		logging.DebugLogf("number of heads must be between %d and %d, got %d",
			MinNumHeads, MaxNumHeads, numHeads)
		return ErrInvalidHeadCount
	}
	if headDim < MinHeadDim || headDim > MaxHeadDim {
		logging.DebugLogf("head dimension must be between %d and %d, got %d",
			MinHeadDim, MaxHeadDim, headDim)
		return ErrInvalidHeadDimension
	}
	if hiddenDim != numHeads*headDim {
		logging.DebugLogf("hidden dimension must equal num_heads * head_dim, got %d != %d * %d",
			hiddenDim, numHeads, headDim)
		return ErrHiddenDimMismatch
	}
	return nil
}
