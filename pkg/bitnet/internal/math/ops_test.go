package math

import (
	"fmt"
	"testing"
)

func BenchmarkMatrix_Add(b *testing.B) {
	sizes := []struct {
		rows, cols int
	}{
		{100, 100},
		{1000, 1000},
		{100, 1000},
		{1000, 100},
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d", size.rows, size.cols), func(b *testing.B) {
			matA := NewMatrix(size.rows, size.cols)
			matB := NewMatrix(size.rows, size.cols)

			// Initialize matrices with test data
			for i := 0; i < size.rows; i++ {
				for j := 0; j < size.cols; j++ {
					matA.Set(i, j, float32(i+j))
					matB.Set(i, j, float32(i-j))
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result := Add(matA, matB)
				_ = result
			}
		})
	}
}

func BenchmarkMatrix_Mul(b *testing.B) {
	sizes := []struct {
		rows, cols, inner int
	}{
		{100, 100, 100},
		{1000, 100, 1000},
		{100, 1000, 100},
		{1000, 1000, 1000},
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%dx%d", size.rows, size.inner, size.cols), func(b *testing.B) {
			matA := NewMatrix(size.rows, size.inner)
			matB := NewMatrix(size.inner, size.cols)

			// Initialize matrices with test data
			for i := 0; i < size.rows; i++ {
				for j := 0; j < size.inner; j++ {
					matA.Set(i, j, float32(i+j))
				}
			}
			for i := 0; i < size.inner; i++ {
				for j := 0; j < size.cols; j++ {
					matB.Set(i, j, float32(i-j))
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result := Mul(matA, matB)
				_ = result
			}
		})
	}
}

func BenchmarkVector_DotProduct(b *testing.B) {
	sizes := []int{100, 1000, 10000, 100000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			vecA := NewVector(size)
			vecB := NewVector(size)

			// Initialize vectors with test data
			for i := 0; i < size; i++ {
				vecA.Data[i] = float32(i)
				vecB.Data[i] = float32(size - i)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result := DotProduct(vecA, vecB)
				_ = result
			}
		})
	}
}

func BenchmarkMatrix_Pool(b *testing.B) {
	sizes := []struct {
		rows, cols int
	}{
		{100, 100},
		{1000, 1000},
		{100, 1000},
		{1000, 100},
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d", size.rows, size.cols), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m := GetMatrixFromPool(size.rows, size.cols)
				PutMatrixToPool(m)
			}
		})
	}
}

func BenchmarkVector_Pool(b *testing.B) {
	sizes := []int{100, 1000, 10000, 100000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				v := GetVectorFromPool(size)
				PutVectorToPool(v)
			}
		})
	}
}
