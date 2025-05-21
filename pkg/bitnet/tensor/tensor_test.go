package tensor

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"
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
			// Use ternary values (-1, 0, +1)
			val := int8((i*3+j)%3 - 1)
			tensor.Set(val, i, j)
		}
	}

	tests := []struct {
		name    string
		indices []int
		want    int8
		wantErr bool
	}{
		{
			name:    "valid indices",
			indices: []int{1, 2},
			want:    1, // (1*3+2) % 3 - 1 = 5 % 3 - 1 = 2 - 1 = 1
			wantErr: false,
		},
		{
			name:    "out of bounds",
			indices: []int{2, 0},
			want:    0,
			wantErr: true,
		},
		{
			name:    "wrong dimensions",
			indices: []int{1},
			want:    0,
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
		value   int8
		indices []int
		wantErr bool
	}{
		{
			name:    "valid indices",
			value:   1,
			indices: []int{1, 2},
			wantErr: false,
		},
		{
			name:    "out of bounds",
			value:   1,
			indices: []int{2, 0},
			wantErr: true,
		},
		{
			name:    "wrong dimensions",
			value:   1,
			indices: []int{1},
			wantErr: true,
		},
		{
			name:    "clamp to ternary",
			value:   2,
			indices: []int{0, 0},
			wantErr: false,
		},
		{
			name:    "clamp to ternary negative",
			value:   -2,
			indices: []int{0, 0},
			wantErr: false,
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
				expected := tt.value
				if expected > 1 {
					expected = 1
				} else if expected < -1 {
					expected = -1
				}
				if got != expected {
					t.Errorf("Set() value = %v, want %v", got, expected)
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
	tensor.Set(1, 0, 0)
	tensor.Set(-1, 0, 1)
	tensor.Set(0, 1, 0)
	tensor.Set(1, 1, 1)

	data := tensor.Data()
	if len(data) != 4 {
		t.Errorf("Tensor.Data() length = %v, want %v", len(data), 4)
	}
	if data[0] != 1 || data[1] != -1 || data[2] != 0 || data[3] != 1 {
		t.Errorf("Tensor.Data() = %v, want %v", data, []int8{1, -1, 0, 1})
	}
}

// TestTensor_Close tests tensor cleanup
func TestTensor_Close(t *testing.T) {
	tensor := NewTensor(2, 2)
	defer tensor.Close()

	// Set initial values
	tensor.Set(1, 0, 0)
	tensor.Set(-1, 0, 1)
	tensor.Set(0, 1, 0)
	tensor.Set(1, 1, 1)

	// Verify tensor is working before close
	if tensor.Get(0, 0) != 1 {
		t.Errorf("Get(0, 0) = %v, want %v", tensor.Get(0, 0), 1)
	}

	// Close tensor
	tensor.Close()

	// Add a delay to ensure handler has exited and ops channel is drained
	time.Sleep(100 * time.Millisecond)

	// Verify operations panic after close
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Get() did not panic after Close()")
			}
		}()
		tensor.Get(0, 0)
	}()

	// Verify no concurrent access after close
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r == nil {
				t.Error("Get() did not panic in goroutine after Close()")
			}
		}()
		tensor.Get(0, 0)
	}()
	wg.Wait()
}

// TestTensor_ParallelForEach tests parallel processing
func TestTensor_ParallelForEach(t *testing.T) {
	tensor := NewTensor(3, 3)
	defer tensor.Close()
	var sum atomic.Int32
	var count atomic.Int32

	tensor.ParallelForEach(func(indices []int, value int8) {
		sum.Add(int32(value))
		count.Add(1)
	})

	if count.Load() != 9 {
		t.Errorf("ParallelForEach() count = %v, want %v", count.Load(), 9)
	}
	if sum.Load() != 0 {
		t.Errorf("ParallelForEach() sum = %v, want %v", sum.Load(), 0)
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
			tensor.Set(1, 50, 50)
		}
	})

	b.Run("2D_assignment_sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for j := 0; j < 100; j++ {
				tensor.Set(1, i%100, j)
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
				tensor.ParallelForEach(func(indices []int, value int8) {
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
				data[j] = 1
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
			tensor.Set(val, 50, 50)
		}
	})

	b.Run("sequential_access", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for j := 0; j < 100; j++ {
				for k := 0; k < 100; k++ {
					val := tensor.Get(j, k)
					tensor.Set(val, j, k)
				}
			}
		}
	})
}
