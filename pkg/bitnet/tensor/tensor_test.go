package tensor

import (
	"fmt"
	"math"
	"sync"
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
	tensor := NewTensor(2, 3)
	if tensor == nil {
		t.Fatal("NewTensor returned nil")
	}

	// Fill with some data
	for i := 0; i < 6; i++ {
		tensor.Set(int8(i%3-1), tensor.calculateIndices(i)...)
	}

	// Close the tensor
	tensor.Close()

	// Verify that operations panic after close
	operations := []struct {
		name string
		fn   func()
	}{
		{
			name: "Get",
			fn:   func() { tensor.Get(0, 0) },
		},
		{
			name: "Set",
			fn:   func() { tensor.Set(1, 0, 0) },
		},
		{
			name: "Shape",
			fn:   func() { tensor.Shape() },
		},
		{
			name: "Data",
			fn:   func() { tensor.Data() },
		},
		{
			name: "ParallelForEach",
			fn:   func() { tensor.ParallelForEach(func(indices []int, value int8) {}) },
		},
		{
			name: "Reshape",
			fn:   func() { tensor.Reshape(3, 2) },
		},
	}

	for _, op := range operations {
		t.Run(op.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("%s did not panic after Close", op.name)
				}
			}()
			op.fn()
		})
	}
}

// TestTensor_ParallelForEach tests parallel processing
func TestTensor_ParallelForEach(t *testing.T) {
	tensor := NewTensor(2, 3)
	if tensor == nil {
		t.Fatal("NewTensor returned nil")
	}

	// Fill with test data
	for i := 0; i < 6; i++ {
		tensor.Set(int8(i%3-1), tensor.calculateIndices(i)...)
	}

	// Create a map to track visited elements
	visited := make(map[string]int8)
	var mu sync.Mutex

	// Process each element
	tensor.ParallelForEach(func(indices []int, value int8) {
		mu.Lock()
		defer mu.Unlock()
		key := fmt.Sprintf("%v", indices)
		visited[key] = value
	})

	// Verify all elements were processed
	if len(visited) != 6 {
		t.Errorf("Processed %d elements, want 6", len(visited))
	}

	// Verify values
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			key := fmt.Sprintf("[%d %d]", i, j)
			got := visited[key]
			want := int8((i*3+j)%3 - 1)
			if got != want {
				t.Errorf("visited[%s] = %v, want %v", key, got, want)
			}
		}
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

func TestTensor_Reshape(t *testing.T) {
	tests := []struct {
		name         string
		initialShape []int
		newShape     []int
		wantErr      bool
	}{
		{
			name:         "valid reshape 2x3 to 3x2",
			initialShape: []int{2, 3},
			newShape:     []int{3, 2},
			wantErr:      false,
		},
		{
			name:         "valid reshape 2x2x2 to 4x2",
			initialShape: []int{2, 2, 2},
			newShape:     []int{4, 2},
			wantErr:      false,
		},
		{
			name:         "invalid reshape - different total size",
			initialShape: []int{2, 3},
			newShape:     []int{4, 2},
			wantErr:      true,
		},
		{
			name:         "invalid reshape - zero dimension",
			initialShape: []int{2, 3},
			newShape:     []int{0, 6},
			wantErr:      true,
		},
		{
			name:         "invalid reshape - negative dimension",
			initialShape: []int{2, 3},
			newShape:     []int{-1, 6},
			wantErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create initial tensor
			tensor := NewTensor(tt.initialShape...)
			if tensor == nil {
				t.Fatal("NewTensor returned nil")
			}

			// Fill with some test data
			for i := 0; i < len(tensor.Data()); i++ {
				tensor.Set(int8(i%3-1), tensor.calculateIndices(i)...)
			}

			// Test reshape
			if tt.wantErr {
				defer func() {
					if r := recover(); r == nil {
						t.Error("Reshape did not panic as expected")
					}
				}()
			}

			reshaped := tensor.Reshape(tt.newShape...)
			if !tt.wantErr {
				if reshaped == nil {
					t.Fatal("Reshape returned nil")
				}

				// Verify shape
				gotShape := reshaped.Shape()
				if len(gotShape) != len(tt.newShape) {
					t.Errorf("Shape length = %v, want %v", len(gotShape), len(tt.newShape))
				}
				for i := range gotShape {
					if gotShape[i] != tt.newShape[i] {
						t.Errorf("Shape[%d] = %v, want %v", i, gotShape[i], tt.newShape[i])
					}
				}

				// Verify data is preserved
				originalData := tensor.Data()
				reshapedData := reshaped.Data()
				if len(originalData) != len(reshapedData) {
					t.Errorf("Data length = %v, want %v", len(reshapedData), len(originalData))
				}
				for i := range originalData {
					if originalData[i] != reshapedData[i] {
						t.Errorf("Data[%d] = %v, want %v", i, reshapedData[i], originalData[i])
					}
				}
			}
		})
	}
}

func TestTensor_CalculateIndices(t *testing.T) {
	tensor := NewTensor(2, 3, 4)
	if tensor == nil {
		t.Fatal("NewTensor returned nil")
	}

	tests := []struct {
		flatIndex int
		want      []int
	}{
		{0, []int{0, 0, 0}},
		{1, []int{0, 0, 1}},
		{3, []int{0, 0, 3}},
		{4, []int{0, 1, 0}},
		{11, []int{0, 2, 3}},
		{12, []int{1, 0, 0}},
		{23, []int{1, 2, 3}},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("index_%d", tt.flatIndex), func(t *testing.T) {
			got := tensor.calculateIndices(tt.flatIndex)
			if len(got) != len(tt.want) {
				t.Errorf("len(got) = %v, want %v", len(got), len(tt.want))
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("got[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestTensor_CalculateIndex(t *testing.T) {
	tensor := NewTensor(2, 3, 4)
	if tensor == nil {
		t.Fatal("NewTensor returned nil")
	}

	tests := []struct {
		indices []int
		want    int
	}{
		{[]int{0, 0, 0}, 0},
		{[]int{0, 0, 1}, 1},
		{[]int{0, 0, 3}, 3},
		{[]int{0, 1, 0}, 4},
		{[]int{0, 2, 3}, 11},
		{[]int{1, 0, 0}, 12},
		{[]int{1, 2, 3}, 23},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("indices_%v", tt.indices), func(t *testing.T) {
			got := tensor.calculateIndex(tt.indices)
			if got != tt.want {
				t.Errorf("calculateIndex(%v) = %v, want %v", tt.indices, got, tt.want)
			}
		})
	}

	// Test panics for invalid index count
	panicTests := []struct {
		name    string
		indices []int
	}{
		{"too few indices", []int{0, 0}},
		{"too many indices", []int{0, 0, 0, 0}},
	}

	for _, tt := range panicTests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("calculateIndex(%v) did not panic as expected", tt.indices)
				}
			}()
			_ = tensor.calculateIndex(tt.indices)
		})
	}

	// Test -1 for out-of-bounds/negative indices
	invalidValueTests := []struct {
		name    string
		indices []int
	}{
		{"negative index", []int{0, -1, 0}},
		{"index out of range", []int{0, 0, 4}},
	}

	for _, tt := range invalidValueTests {
		t.Run(tt.name, func(t *testing.T) {
			got := tensor.calculateIndex(tt.indices)
			if got != -1 {
				t.Errorf("calculateIndex(%v) = %v, want -1", tt.indices, got)
			}
		})
	}
}

func BenchmarkTensor_CalculateIndex(b *testing.B) {
	tensor := NewTensor(100, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tensor.calculateIndex([]int{50, 50})
	}
}
