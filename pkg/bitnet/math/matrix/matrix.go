// Package matrix provides core tensor operations for BitNet math operations.
//
// # Quantized Matrix Operations for BitNet
//
// This package implements core matrix operations using ternary (int8: -1, 0, +1) values,
// as required by the BitNet model's quantized architecture.
//
// Key aspects:
//   - All matrices use int8 storage for memory and performance
//   - Addition and multiplication are clamped to the ternary range [-1, 0, +1]
//   - Designed for CPU efficiency and low memory use in BitNet inference
//   - Not suitable for high-precision or training use
//
// Implementation details:
//   - Matrix addition and multiplication with ternary clamping
//   - Efficient memory management for matrix operations
//   - Support for 2D matrix operations
//
// Related tasks and dependencies:
//   - #173: Implement Matrix Operations (Core implementation)
//   - #182: Compute Scaled Dot-Product Attention (Depends on #173)
//   - #185: Feed-Forward Network (FFN) Sublayer (Depends on #173)
//   - #186: Integrate Attention Sublayer (Pre-Norm & Residual) (Depends on #173)
//   - #187: Integrate Feed-Forward Sublayer (Pre-Norm & Residual) (Depends on #173)
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
package matrix

import "errors"

// Matrix represents a 2D matrix of ternary values (-1, 0, +1)
type Matrix struct {
	Data   []int8
	Rows   int
	Cols   int
	Stride int
}

// NewMatrix creates a new matrix with the given dimensions
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		Data:   make([]int8, rows*cols),
		Rows:   rows,
		Cols:   cols,
		Stride: cols,
	}
}

// Get returns the value at the specified position
func (m *Matrix) Get(row, col int) int8 {
	return m.Data[row*m.Stride+col]
}

// Set sets the value at the specified position
func (m *Matrix) Set(row, col int, value int8) {
	m.Data[row*m.Stride+col] = value
}

// Add performs matrix addition with ternary values
func Add(a, b *Matrix) (*Matrix, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, ErrMatrixDimensionMismatch
	}

	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < len(a.Data); i++ {
		sum := a.Data[i] + b.Data[i]
		// Clamp to ternary values
		if sum > 1 {
			sum = 1
		} else if sum < -1 {
			sum = -1
		}
		result.Data[i] = sum
	}
	return result, nil
}

// Mul performs matrix multiplication with ternary values
func Mul(a, b *Matrix) (*Matrix, error) {
	if a.Cols != b.Rows {
		return nil, ErrMatrixIncompatibleDimensions
	}

	result := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			var sum int32
			for k := 0; k < a.Cols; k++ {
				sum += int32(a.Get(i, k)) * int32(b.Get(k, j))
			}
			// Clamp to ternary values
			if sum > 1 {
				sum = 1
			} else if sum < -1 {
				sum = -1
			}
			result.Set(i, j, int8(sum))
		}
	}
	return result, nil
}

var ErrMatrixDimensionMismatch = errors.New("matrix: dimensions must match")

var ErrMatrixIncompatibleDimensions = errors.New("matrix: dimensions incompatible for multiplication")
