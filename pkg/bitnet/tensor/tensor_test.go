package tensor

import (
	"fmt"
	"math"
	"testing"
)

// TestNewTensor tests tensor creation with various shapes
func TestNewTensor(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		wantSize int
	}{
		{
			name:     "1D tensor",
			shape:    []int{10},
			wantSize: 10,
		},
		{
			name:     "2D tensor",
			shape:    []int{3, 4},
			wantSize: 12,
		},
		{
			name:     "3D tensor",
			shape:    []int{2, 3, 4},
			wantSize: 24,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(tt.shape...)
			if tensor == nil {
				t.Fatal("NewTensor returned nil")
			}
			if len(tensor.data) != tt.wantSize {
				t.Errorf("NewTensor() size = %v, want %v", len(tensor.data), tt.wantSize)
			}
			if len(tensor.shape) != len(tt.shape) {
				t.Errorf("NewTensor() shape length = %v, want %v", len(tensor.shape), len(tt.shape))
			}
			for i, s := range tt.shape {
				if tensor.shape[i] != s {
					t.Errorf("NewTensor() shape[%d] = %v, want %v", i, tensor.shape[i], s)
				}
			}
		})
	}
}

// TestTensor_Get tests tensor value retrieval
func TestTensor_Get(t *testing.T) {
	tensor := NewTensor(2, 3)
	// Initialize with test values
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			tensor.Set(float64(i*3+j), i, j)
		}
	}

	tests := []struct {
		name    string
		indices []int
		want    float64
		wantErr bool
	}{
		{
			name:    "valid indices",
			indices: []int{1, 2},
			want:    5.0,
			wantErr: false,
		},
		{
			name:    "out of bounds",
			indices: []int{2, 0},
			want:    0.0,
			wantErr: true,
		},
		{
			name:    "wrong dimensions",
			indices: []int{1},
			want:    0.0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil && !tt.wantErr {
					t.Errorf("Get() panic = %v, wantErr %v", r, tt.wantErr)
				}
			}()

			got := tensor.Get(tt.indices...)
			if !tt.wantErr && got != tt.want {
				t.Errorf("Get() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestTensor_Set tests tensor value assignment
func TestTensor_Set(t *testing.T) {
	tensor := NewTensor(2, 3)

	tests := []struct {
		name    string
		value   float64
		indices []int
		wantErr bool
	}{
		{
			name:    "valid indices",
			value:   42.0,
			indices: []int{1, 2},
			wantErr: false,
		},
		{
			name:    "out of bounds",
			value:   42.0,
			indices: []int{2, 0},
			wantErr: true,
		},
		{
			name:    "wrong dimensions",
			value:   42.0,
			indices: []int{1},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil && !tt.wantErr {
					t.Errorf("Set() panic = %v, wantErr %v", r, tt.wantErr)
				}
			}()

			tensor.Set(tt.value, tt.indices...)
			if !tt.wantErr {
				got := tensor.Get(tt.indices...)
				if got != tt.value {
					t.Errorf("Set() value = %v, want %v", got, tt.value)
				}
			}
		})
	}
}

// TestTensor_Shape tests tensor shape retrieval
func TestTensor_Shape(t *testing.T) {
	tensor := NewTensor(2, 3, 4)
	shape := tensor.Shape()
	if len(shape) != 3 {
		t.Errorf("Tensor.Shape() length = %v, want %v", len(shape), 3)
	}
	if shape[0] != 2 || shape[1] != 3 || shape[2] != 4 {
		t.Errorf("Tensor.Shape() = %v, want %v", shape, []int{2, 3, 4})
	}
}

// TestTensor_Data tests tensor data retrieval
func TestTensor_Data(t *testing.T) {
	tensor := NewTensor(2, 2)
	tensor.Set(1.0, 0, 0)
	tensor.Set(2.0, 0, 1)
	tensor.Set(3.0, 1, 0)
	tensor.Set(4.0, 1, 1)

	data := tensor.Data()
	if len(data) != 4 {
		t.Errorf("Tensor.Data() length = %v, want %v", len(data), 4)
	}
	if data[0] != 1.0 || data[1] != 2.0 || data[2] != 3.0 || data[3] != 4.0 {
		t.Errorf("Tensor.Data() = %v, want %v", data, []float64{1.0, 2.0, 3.0, 4.0})
	}
}

// TestTensor_ParallelForEach tests parallel processing
func TestTensor_ParallelForEach(t *testing.T) {
	tensor := NewTensor(3, 3)
	sum := 0.0
	count := 0

	tensor.ParallelForEach(func(indices []int, value float64) {
		sum += value
		count++
	})

	if count != 9 {
		t.Errorf("ParallelForEach() count = %v, want %v", count, 9)
	}
	if sum != 0.0 {
		t.Errorf("ParallelForEach() sum = %v, want %v", sum, 0.0)
	}
}

// floatEquals compares two float64 values with a small epsilon
func floatEquals(a, b float64) bool {
	epsilon := 1e-6
	return math.Abs(a-b) < epsilon
}

// TestTensor_InterfaceCompliance tests interface implementation
func TestTensor_InterfaceCompliance(t *testing.T) {
	var _ TensorType = &Tensor{}
	var _ ParallelProcessor = &Tensor{}
}

// BenchmarkNewTensor tests tensor creation performance
func BenchmarkNewTensor(b *testing.B) {
	shapes := [][]int{
		{100},
		{100, 100},
		{50, 50, 50},
		{20, 20, 20, 20},
	}

	for _, shape := range shapes {
		b.Run(fmt.Sprintf("shape_%v", shape), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				NewTensor(shape...)
			}
		})
	}
}

// BenchmarkTensor_Get tests value retrieval performance
func BenchmarkTensor_Get(b *testing.B) {
	tensor := NewTensor(100, 100)
	b.Run("2D_access", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			tensor.Get(50, 50)
		}
	})

	b.Run("2D_access_sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for j := 0; j < 100; j++ {
				tensor.Get(i%100, j)
			}
		}
	})
}

// BenchmarkTensor_Set tests value assignment performance
func BenchmarkTensor_Set(b *testing.B) {
	tensor := NewTensor(100, 100)
	b.Run("2D_assignment", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			tensor.Set(float64(i), 50, 50)
		}
	})

	b.Run("2D_assignment_sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for j := 0; j < 100; j++ {
				tensor.Set(float64(i), i%100, j)
			}
		}
	})
}

// BenchmarkTensor_ParallelForEach tests parallel processing performance
func BenchmarkTensor_ParallelForEach(b *testing.B) {
	sizes := [][]int{
		{100, 100},
		{1000, 1000},
		{100, 100, 100},
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d", size[0], size[1]), func(b *testing.B) {
			tensor := NewTensor(size...)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tensor.ParallelForEach(func(indices []int, value float64) {
					// Do nothing, just measure overhead
				})
			}
		})
	}
}

// BenchmarkTensor_Data tests data array access performance
func BenchmarkTensor_Data(b *testing.B) {
	tensor := NewTensor(100, 100)
	b.Run("data_access", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = tensor.Data()
		}
	})

	b.Run("data_iteration", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			data := tensor.Data()
			for j := range data {
				data[j] = float64(j)
			}
		}
	})
}

// BenchmarkTensor_Shape tests shape retrieval performance
func BenchmarkTensor_Shape(b *testing.B) {
	shapes := [][]int{
		{100},
		{100, 100},
		{50, 50, 50},
		{20, 20, 20, 20},
	}

	for _, shape := range shapes {
		b.Run(fmt.Sprintf("shape_%v", shape), func(b *testing.B) {
			tensor := NewTensor(shape...)
			for i := 0; i < b.N; i++ {
				_ = tensor.Shape()
			}
		})
	}
}

// BenchmarkTensor_Operations tests common tensor operations
func BenchmarkTensor_Operations(b *testing.B) {
	tensor := NewTensor(100, 100)
	b.Run("get_set_cycle", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			val := tensor.Get(50, 50)
			tensor.Set(val+1, 50, 50)
		}
	})

	b.Run("sequential_access", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for j := 0; j < 100; j++ {
				for k := 0; k < 100; k++ {
					val := tensor.Get(j, k)
					tensor.Set(val+1, j, k)
				}
			}
		}
	})
}
