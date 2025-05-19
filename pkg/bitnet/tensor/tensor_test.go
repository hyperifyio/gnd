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
		wantErr  bool
	}{
		{
			name:     "1D tensor",
			shape:    []int{3},
			wantSize: 3,
			wantErr:  false,
		},
		{
			name:     "2D tensor",
			shape:    []int{2, 3},
			wantSize: 6,
			wantErr:  false,
		},
		{
			name:     "3D tensor",
			shape:    []int{2, 3, 4},
			wantSize: 24,
			wantErr:  false,
		},
		{
			name:     "empty shape",
			shape:    []int{},
			wantSize: 0,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil && !tt.wantErr {
					t.Errorf("NewTensor() panic = %v, wantErr %v", r, tt.wantErr)
				}
			}()

			got := NewTensor(tt.shape...)
			if tt.wantErr {
				t.Error("NewTensor() expected panic")
				return
			}

			if len(got.data) != tt.wantSize {
				t.Errorf("NewTensor() size = %v, want %v", len(got.data), tt.wantSize)
			}

			if len(got.shape) != len(tt.shape) {
				t.Errorf("NewTensor() shape length = %v, want %v", len(got.shape), len(tt.shape))
			}

			for i, s := range tt.shape {
				if got.shape[i] != s {
					t.Errorf("NewTensor() shape[%d] = %v, want %v", i, got.shape[i], s)
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
			tensor.Set(float32(i*3+j), i, j)
		}
	}

	tests := []struct {
		name    string
		indices []int
		want    float32
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
		value   float32
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
	tests := []struct {
		name  string
		shape []int
	}{
		{
			name:  "1D tensor",
			shape: []int{3},
		},
		{
			name:  "2D tensor",
			shape: []int{2, 3},
		},
		{
			name:  "3D tensor",
			shape: []int{2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(tt.shape...)
			got := tensor.Shape()

			if len(got) != len(tt.shape) {
				t.Errorf("Shape() length = %v, want %v", len(got), len(tt.shape))
			}

			for i, s := range tt.shape {
				if got[i] != s {
					t.Errorf("Shape()[%d] = %v, want %v", i, got[i], s)
				}
			}
		})
	}
}

// TestTensor_Data tests tensor data retrieval
func TestTensor_Data(t *testing.T) {
	tensor := NewTensor(2, 3)
	// Initialize with test values
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			tensor.Set(float32(i*3+j), i, j)
		}
	}

	got := tensor.Data()
	if len(got) != 6 {
		t.Errorf("Data() length = %v, want %v", len(got), 6)
	}

	// Verify values
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			idx := i*3 + j
			if got[idx] != float32(idx) {
				t.Errorf("Data()[%d] = %v, want %v", idx, got[idx], float32(idx))
			}
		}
	}
}

// TestTensor_ParallelForEach tests parallel processing
func TestTensor_ParallelForEach(t *testing.T) {
	tensor := NewTensor(4, 4)
	// Initialize with test values
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			tensor.Set(float32(i*4+j), i, j)
		}
	}

	// Test doubling all values
	tensor.ParallelForEach(func(value float32, indices ...int) float32 {
		return value * 2
	})

	// Verify results
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			got := tensor.Get(i, j)
			want := float32(i*4+j) * 2
			if !floatEquals(got, want) {
				t.Errorf("ParallelForEach() value at (%d,%d) = %v, want %v", i, j, got, want)
			}
		}
	}
}

// floatEquals compares two float32 values with a small epsilon
func floatEquals(a, b float32) bool {
	epsilon := float32(1e-6)
	return math.Abs(float64(a-b)) < float64(epsilon)
}

// TestTensor_InterfaceCompliance tests interface implementation
func TestTensor_InterfaceCompliance(t *testing.T) {
	var _ TensorType = &Tensor{}
	var _ ParallelProcessor = &Tensor{}
}

// BenchmarkNewTensor tests tensor creation performance
func BenchmarkNewTensor(b *testing.B) {
	shapes := [][]int{
		{100},            // 1D
		{100, 100},       // 2D
		{50, 50, 50},     // 3D
		{20, 20, 20, 20}, // 4D
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
	// Initialize with test values
	for i := 0; i < 100; i++ {
		for j := 0; j < 100; j++ {
			tensor.Set(float32(i*100+j), i, j)
		}
	}

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
			tensor.Set(float32(i), 50, 50)
		}
	})

	b.Run("2D_assignment_sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for j := 0; j < 100; j++ {
				tensor.Set(float32(i*100+j), i%100, j)
			}
		}
	})
}

// BenchmarkTensor_ParallelForEach tests parallel processing performance
func BenchmarkTensor_ParallelForEach(b *testing.B) {
	sizes := []struct {
		name  string
		shape []int
	}{
		{"100x100", []int{100, 100}},
		{"1000x1000", []int{1000, 1000}},
		{"100x100x100", []int{100, 100, 100}},
	}

	for _, size := range sizes {
		tensor := NewTensor(size.shape...)
		// Initialize with test values
		for i := 0; i < size.shape[0]; i++ {
			for j := 0; j < size.shape[1]; j++ {
				tensor.Set(float32(i*size.shape[1]+j), i, j)
			}
		}

		b.Run(size.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				tensor.ParallelForEach(func(value float32, indices ...int) float32 {
					return value * 2
				})
			}
		})
	}
}

// BenchmarkTensor_Data tests data array access performance
func BenchmarkTensor_Data(b *testing.B) {
	tensor := NewTensor(100, 100)
	// Initialize with test values
	for i := 0; i < 100; i++ {
		for j := 0; j < 100; j++ {
			tensor.Set(float32(i*100+j), i, j)
		}
	}

	b.Run("data_access", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = tensor.Data()
		}
	})

	b.Run("data_iteration", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			data := tensor.Data()
			sum := float32(0)
			for _, v := range data {
				sum += v
			}
		}
	})
}

// BenchmarkTensor_Shape tests shape retrieval performance
func BenchmarkTensor_Shape(b *testing.B) {
	shapes := [][]int{
		{100},            // 1D
		{100, 100},       // 2D
		{50, 50, 50},     // 3D
		{20, 20, 20, 20}, // 4D
	}

	for _, shape := range shapes {
		tensor := NewTensor(shape...)
		b.Run(fmt.Sprintf("shape_%v", shape), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = tensor.Shape()
			}
		})
	}
}

// BenchmarkTensor_Operations tests common tensor operations
func BenchmarkTensor_Operations(b *testing.B) {
	tensor := NewTensor(100, 100)
	// Initialize with test values
	for i := 0; i < 100; i++ {
		for j := 0; j < 100; j++ {
			tensor.Set(float32(i*100+j), i, j)
		}
	}

	b.Run("get_set_cycle", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			val := tensor.Get(50, 50)
			tensor.Set(val+1, 50, 50)
		}
	})

	b.Run("sequential_access", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for j := 0; j < 100; j++ {
				val := tensor.Get(i%100, j)
				tensor.Set(val+1, i%100, j)
			}
		}
	})
}
