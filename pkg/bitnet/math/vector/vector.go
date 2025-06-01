// Package vector provides core tensor operations for BitNet math operations.
//
// # Quantized Vector Operations for BitNet
//
// This package implements core vector operations using ternary (int8: -1, 0, +1) values,
// as required by the BitNet model's quantized architecture.
//
// Key aspects:
//   - All matrices and vectors use int8 storage for memory and performance
//   - Addition and multiplication are clamped to the ternary range [-1, 0, +1]
//   - Designed for CPU efficiency and low memory use in BitNet inference
//   - Not suitable for high-precision or training use
//
// Implementation details:
//   - Dot product and vector creation with ternary clamping
//   - Efficient memory management for vector operations
//
// Related tasks and dependencies:
//   - #174: Implement Vector Operations (Core implementation)
//   - #182: Compute Scaled Dot-Product Attention (Depends on #174)
//   - #185: Feed-Forward Network (FFN) Sublayer (Depends on #174)
//   - #186: Integrate Attention Sublayer (Pre-Norm & Residual) (Depends on #174)
//   - #187: Integrate Feed-Forward Sublayer (Pre-Norm & Residual) (Depends on #174)
//
// Usage:
//   - Used for quantized weight and activation operations in BitNet transformer blocks
//   - Maintainers should not change the quantization logic without updating the entire pipeline
//
// Caveats:
//   - Floating-point properties (e.g., exact sums/products) do not hold due to clamping
//   - Tests should check for correct clamping and quantized behavior, not float math
//   - Any change must be validated against end-to-end BitNet inference
//   - Performance critical - changes should be benchmarked against existing implementation
//
// For more details, see BitNet issue #190 and the BitNet project documentation.
package vector

import "errors"

// Vector represents a 1D vector of ternary values (-1, 0, +1)
type Vector struct {
	Data []int8
}

// NewVector creates a new vector with the given length
func NewVector(length int) *Vector {
	return &Vector{
		Data: make([]int8, length),
	}
}

// DotProduct computes the dot product of two vectors with ternary values
func DotProduct(a, b *Vector) (int8, error) {
	if len(a.Data) != len(b.Data) {
		return 0, ErrVectorLengthMismatch
	}

	var sum int32
	for i := 0; i < len(a.Data); i++ {
		sum += int32(a.Data[i]) * int32(b.Data[i])
	}
	// Clamp to ternary values
	if sum > 1 {
		sum = 1
	} else if sum < -1 {
		sum = -1
	}
	return int8(sum), nil
}

var (
	ErrVectorLengthMismatch = errors.New("vector: lengths must match")
)
