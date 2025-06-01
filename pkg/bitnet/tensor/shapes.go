package tensor

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/logging"
	"github.com/hyperifyio/gnd/pkg/bitnet/math/shape"
)

// Package tensor provides shape validation functions for BitNet tensors.
//
// # Shape Validation for BitNet
//
// This file provides functions to validate tensor shapes for various BitNet operations.
// These functions ensure that tensors have the correct dimensions for their intended use.
//
// Key aspects:
//   - Validates tensor shapes for common BitNet operations
//   - Provides specific validation for attention and linear layers
//   - Ensures consistent tensor dimensions across operations
//   - Helps prevent runtime errors from shape mismatches
//
// Implementation Details:
//   - Validates common shapes like [batch_size, seq_len, hidden_dim]
//   - Validates attention shapes like [batch_size, num_heads, seq_len, head_dim]
//   - Validates linear layer shapes like [hidden_dim, hidden_dim]
//   - Provides detailed error messages for debugging
//
// Usage:
//   - Use before performing tensor operations
//   - Validate input shapes for each layer
//   - Check tensor compatibility before operations
//   - Debug shape-related issues
//
// Caveats:
//   - Validation adds runtime overhead
//   - Some operations may require additional shape checks
//   - Error messages may not cover all edge cases
//   - Shape validation does not guarantee correct values
//
// For more details, see BitNet issue #190 and the BitNet project documentation.

// ValidateTensorShape checks if a tensor's shape matches any of the expected dimensions.
// If multiple dimensions are provided, the tensor's shape must match one of them.
// Returns ErrInvalidDimensions if the shape does not match.
func ValidateTensorShape(t *Tensor, expectedDims ...int) error {
	if t == nil {
		logging.DebugLogf("tensor is nil, expected dimensions %v", expectedDims)
		return shape.ErrInvalidDimensions
	}
	tensorShape, err := t.Shape()
	if err != nil {
		return err
	}
	return shape.ValidateShape(tensorShape, expectedDims...)
}

// ValidateTensorShapeBatchSeqHidden checks if a tensor has shape [batch_size, seq_len, hidden_dim].
// Returns ErrInvalidInputShape if the shape does not match.
func ValidateTensorShapeBatchSeqHidden(t *Tensor, name string) error {
	if t == nil {
		logging.DebugLogf("%s: tensor is nil", name)
		return shape.ErrInvalidInputShape
	}
	tensorShape, err := t.Shape()
	if err != nil {
		return err
	}
	return shape.ValidateBatchSeqHiddenShape(tensorShape, name)
}

// ValidateTensorShapeBatchHeadsSeqHead checks if a tensor has shape [batch_size, num_heads, seq_len, head_dim]
func ValidateTensorShapeBatchHeadsSeqHead(t *Tensor, name string) error {
	if t == nil {
		logging.DebugLogf("%s: tensor is nil", name)
		return shape.ErrInvalidInputShape
	}
	tensorShape, err := t.Shape()
	if err != nil {
		return err
	}
	return shape.ValidateBatchHeadsSeqHeadShape(tensorShape, name)
}

// ValidateTensorShapeHiddenHidden checks if a tensor has shape [hidden_dim, hidden_dim]
func ValidateTensorShapeHiddenHidden(t *Tensor, name string) error {
	if t == nil {
		logging.DebugLogf("%s: tensor is nil", name)
		return shape.ErrInvalidInputShape
	}
	tensorShape, err := t.Shape()
	if err != nil {
		return err
	}
	return shape.ValidateHiddenHiddenShape(tensorShape, name)
}

// ValidateMatchingTensorShapes checks if two tensors have matching shapes
func ValidateMatchingTensorShapes(t1, t2 *Tensor, name1, name2 string) error {
	if t1 == nil || t2 == nil {
		logging.DebugLogf("tensors must not be nil: %s=%v, %s=%v", name1, t1 == nil, name2, t2 == nil)
		return shape.ErrInvalidInputShape
	}
	shape1, err := t1.Shape()
	if err != nil {
		return err
	}
	shape2, err := t2.Shape()
	if err != nil {
		return err
	}
	return shape.ValidateMatchingShapes(shape1, shape2, name1, name2)
}
