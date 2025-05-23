// Package math implements mathematical operations for the BitNet model.
package math

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// Common tensor shape dimensions
const (
	// MinHeadDim is the minimum allowed head dimension
	MinHeadDim = 8
	// MaxHeadDim is the maximum allowed head dimension
	MaxHeadDim = 256
	// MinNumHeads is the minimum allowed number of attention heads
	MinNumHeads = 1
	// MaxNumHeads is the maximum allowed number of attention heads
	MaxNumHeads = 32
)

// Shape represents a tensor's dimensions
type Shape []int

// Common shape types
type (
	// BatchSeqHidden represents [batch_size, seq_len, hidden_dim]
	BatchSeqHidden Shape
	// BatchHeadsSeqHead represents [batch_size, num_heads, seq_len, head_dim]
	BatchHeadsSeqHead Shape
	// HiddenHidden represents [hidden_dim, hidden_dim]
	HiddenHidden Shape
)

// ValidateShape checks if a tensor's shape matches any of the expected dimensions.
// If multiple dimensions are provided, the tensor's shape must match one of them.
func ValidateShape(t *tensor.Tensor, expectedDims ...int) error {
	shape := t.Shape()
	for _, dim := range expectedDims {
		if len(shape) == dim {
			return nil
		}
	}
	tensor.DebugLog("tensor must have one of dimensions %v, got %dD", expectedDims, len(shape))
	return ErrInvalidDimensions
}

// ValidateBatchSeqHidden checks if a tensor has shape [batch_size, seq_len, hidden_dim]
func ValidateBatchSeqHidden(t *tensor.Tensor, name string) error {
	if err := ValidateShape(t, 3); err != nil {
		tensor.DebugLog("%s: %v", name, err)
		return ErrInvalidInputShape
	}
	return nil
}

// ValidateBatchHeadsSeqHead checks if a tensor has shape [batch_size, num_heads, seq_len, head_dim]
func ValidateBatchHeadsSeqHead(t *tensor.Tensor, name string) error {
	if err := ValidateShape(t, 4); err != nil {
		tensor.DebugLog("%s: %v", name, err)
		return ErrInvalidInputShape
	}
	return nil
}

// ValidateHiddenHidden checks if a tensor has shape [hidden_dim, hidden_dim]
func ValidateHiddenHidden(t *tensor.Tensor, name string) error {
	if err := ValidateShape(t, 2); err != nil {
		tensor.DebugLog("%s: %v", name, err)
		return ErrInvalidInputShape
	}
	if t.Shape()[0] != t.Shape()[1] {
		tensor.DebugLog("%s must be square matrix, got shape %v", name, t.Shape())
		return ErrNonSquareMatrix
	}
	return nil
}

// ValidateMatchingShapes checks if two tensors have matching shapes
func ValidateMatchingShapes(t1, t2 *tensor.Tensor, name1, name2 string) error {
	shape1 := t1.Shape()
	shape2 := t2.Shape()
	if len(shape1) != len(shape2) {
		tensor.DebugLog("%s and %s must have same number of dimensions, got %d and %d",
			name1, name2, len(shape1), len(shape2))
		return ErrDimensionMismatch
	}
	for i := range shape1 {
		if shape1[i] != shape2[i] {
			tensor.DebugLog("%s and %s must have matching dimensions, got %v and %v",
				name1, name2, shape1, shape2)
			return ErrDimensionMismatch
		}
	}
	return nil
}

// ValidateHeadDimensions checks if head dimensions are valid
func ValidateHeadDimensions(hiddenDim, numHeads, headDim int) error {
	if numHeads < MinNumHeads || numHeads > MaxNumHeads {
		tensor.DebugLog("number of heads must be between %d and %d, got %d",
			MinNumHeads, MaxNumHeads, numHeads)
		return ErrInvalidHeadCount
	}
	if headDim < MinHeadDim || headDim > MaxHeadDim {
		tensor.DebugLog("head dimension must be between %d and %d, got %d",
			MinHeadDim, MaxHeadDim, headDim)
		return ErrInvalidHeadDimension
	}
	if hiddenDim != numHeads*headDim {
		tensor.DebugLog("hidden dimension must equal num_heads * head_dim, got %d != %d * %d",
			hiddenDim, numHeads, headDim)
		return ErrHiddenDimMismatch
	}
	return nil
}
