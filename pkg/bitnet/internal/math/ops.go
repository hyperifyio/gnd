package math

import (
	"runtime"
	"sync"
)

// Matrix represents a 2D matrix of float32 values
type Matrix struct {
	Data   []float32
	Rows   int
	Cols   int
	Stride int
}

// matrixPool manages a pool of matrices for reuse
var matrixPool = sync.Pool{
	New: func() interface{} {
		return &Matrix{}
	},
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

// GetMatrixFromPool gets a matrix from the pool or creates a new one
func GetMatrixFromPool(rows, cols int) *Matrix {
	m := matrixPool.Get().(*Matrix)
	if cap(m.Data) < rows*cols {
		m.Data = make([]float32, rows*cols)
	}
	m.Data = m.Data[:rows*cols]
	m.Rows = rows
	m.Cols = cols
	m.Stride = cols
	return m
}

// PutMatrixToPool returns a matrix to the pool
func PutMatrixToPool(m *Matrix) {
	matrixPool.Put(m)
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

	result := GetMatrixFromPool(a.Rows, a.Cols)
	defer PutMatrixToPool(result)

	// Process in batches for better cache utilization
	const batchSize = 32
	for i := 0; i < len(a.Data); i += batchSize {
		end := i + batchSize
		if end > len(a.Data) {
			end = len(a.Data)
		}
		for j := i; j < end; j++ {
			result.Data[j] = a.Data[j] + b.Data[j]
		}
	}
	return result
}

// Mul performs matrix multiplication
func Mul(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic("matrix dimensions incompatible for multiplication")
	}

	result := GetMatrixFromPool(a.Rows, b.Cols)
	defer PutMatrixToPool(result)

	// Get number of CPU cores
	numCPU := runtime.NumCPU()
	if numCPU < 2 {
		// Sequential processing for single CPU
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

	// Parallel processing for multiple CPUs
	var wg sync.WaitGroup
	rowsPerCPU := (a.Rows + numCPU - 1) / numCPU

	for cpu := 0; cpu < numCPU; cpu++ {
		wg.Add(1)
		startRow := cpu * rowsPerCPU
		endRow := startRow + rowsPerCPU
		if endRow > a.Rows {
			endRow = a.Rows
		}

		go func(start, end int) {
			defer wg.Done()
			// Process rows in batches for better cache utilization
			const batchSize = 32
			for i := start; i < end; i += batchSize {
				rowEnd := i + batchSize
				if rowEnd > end {
					rowEnd = end
				}
				for j := i; j < rowEnd; j++ {
					for k := 0; k < b.Cols; k++ {
						var sum float32
						for l := 0; l < a.Cols; l++ {
							sum += a.Get(j, l) * b.Get(l, k)
						}
						result.Set(j, k, sum)
					}
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()
	return result
}

// Vector represents a 1D vector of float32 values
type Vector struct {
	Data []float32
}

// vectorPool manages a pool of vectors for reuse
var vectorPool = sync.Pool{
	New: func() interface{} {
		return &Vector{}
	},
}

// NewVector creates a new vector with the given length
func NewVector(length int) *Vector {
	return &Vector{
		Data: make([]float32, length),
	}
}

// GetVectorFromPool gets a vector from the pool or creates a new one
func GetVectorFromPool(length int) *Vector {
	v := vectorPool.Get().(*Vector)
	if cap(v.Data) < length {
		v.Data = make([]float32, length)
	}
	v.Data = v.Data[:length]
	return v
}

// PutVectorToPool returns a vector to the pool
func PutVectorToPool(v *Vector) {
	vectorPool.Put(v)
}

// DotProduct computes the dot product of two vectors
func DotProduct(a, b *Vector) float32 {
	if len(a.Data) != len(b.Data) {
		panic("vector lengths must match")
	}

	var sum float32
	// Process in batches for better cache utilization
	const batchSize = 32
	for i := 0; i < len(a.Data); i += batchSize {
		end := i + batchSize
		if end > len(a.Data) {
			end = len(a.Data)
		}
		for j := i; j < end; j++ {
			sum += a.Data[j] * b.Data[j]
		}
	}
	return sum
}
