package math

// Matrix represents a 2D matrix of float32 values
type Matrix struct {
	Data   []float32
	Rows   int
	Cols   int
	Stride int
}

// NewMatrix creates a new matrix with the given dimensions
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		Data:   make([]float32, rows*cols),
		Rows:   rows,
		Cols:   cols,
		Stride: cols,
	}
}

// Get returns the value at the specified position
func (m *Matrix) Get(row, col int) float32 {
	return m.Data[row*m.Stride+col]
}

// Set sets the value at the specified position
func (m *Matrix) Set(row, col int, value float32) {
	m.Data[row*m.Stride+col] = value
}

// Add performs matrix addition
func Add(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("matrix dimensions must match")
	}

	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] + b.Data[i]
	}
	return result
}

// Mul performs matrix multiplication
func Mul(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic("matrix dimensions incompatible for multiplication")
	}

	result := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			var sum float32
			for k := 0; k < a.Cols; k++ {
				sum += a.Get(i, k) * b.Get(k, j)
			}
			result.Set(i, j, sum)
		}
	}
	return result
}

// Vector represents a 1D vector of float32 values
type Vector struct {
	Data []float32
}

// NewVector creates a new vector with the given length
func NewVector(length int) *Vector {
	return &Vector{
		Data: make([]float32, length),
	}
}

// DotProduct computes the dot product of two vectors
func DotProduct(a, b *Vector) float32 {
	if len(a.Data) != len(b.Data) {
		panic("vector lengths must match")
	}

	var sum float32
	for i := 0; i < len(a.Data); i++ {
		sum += a.Data[i] * b.Data[i]
	}
	return sum
}
