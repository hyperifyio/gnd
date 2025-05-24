package math

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
func Add(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("matrix dimensions must match")
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
	return result
}

// Mul performs matrix multiplication with ternary values
func Mul(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic("matrix dimensions incompatible for multiplication")
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
	return result
}

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
func DotProduct(a, b *Vector) int8 {
	if len(a.Data) != len(b.Data) {
		panic("vector lengths must match")
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
	return int8(sum)
}
