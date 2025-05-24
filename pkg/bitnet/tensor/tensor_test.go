package tensor

import (
	"fmt"
	"math"
	"reflect"
	"sync"
	"testing"
)

// TestNewTensor tests tensor creation with various shapes
func TestNewTensor(t *testing.T) {
	tests := []struct {
		name    string
		dims    []int
		wantErr bool
	}{
		{
			name:    "valid 2D",
			dims:    []int{2, 3},
			wantErr: false,
		},
		{
			name:    "valid 3D",
			dims:    []int{2, 3, 4},
			wantErr: false,
		},
		{
			name:    "invalid negative",
			dims:    []int{-1, 2},
			wantErr: true,
		},
		{
			name:    "invalid zero",
			dims:    []int{0, 2},
			wantErr: true,
		},
		{
			name:    "invalid empty",
			dims:    []int{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewTensor(tt.dims...)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewTensor() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got == nil {
				t.Error("NewTensor() returned nil tensor for valid input")
			}
		})
	}
}

// TestTensor_Get tests tensor value retrieval
func TestTensor_Get(t *testing.T) {
	tensor, err := NewTensor(2, 3)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	// Test valid indices
	val, err := tensor.Get(0, 0)
	if err != nil {
		t.Errorf("Get(0, 0) error = %v, want nil", err)
	}
	if val != 0 {
		t.Errorf("Get(0, 0) = %v, want 0", val)
	}

	// Test invalid indices
	_, err = tensor.Get(-1, 0)
	if err == nil {
		t.Error("Get(-1, 0) error = nil, want error")
	}

	_, err = tensor.Get(2, 0)
	if err == nil {
		t.Error("Get(2, 0) error = nil, want error")
	}

	_, err = tensor.Get(0, 3)
	if err == nil {
		t.Error("Get(0, 3) error = nil, want error")
	}
}

// TestTensor_Set tests tensor value assignment
func TestTensor_Set(t *testing.T) {
	tensor, err := NewTensor(2, 3)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	// Test valid indices
	err = tensor.Set(1, 0, 0)
	if err != nil {
		t.Errorf("Set(1, 0, 0) error = %v, want nil", err)
	}

	val, err := tensor.Get(0, 0)
	if err != nil {
		t.Errorf("Get(0, 0) error = %v, want nil", err)
	}
	if val != 1 {
		t.Errorf("Get(0, 0) = %v, want 1", val)
	}

	// Test invalid indices
	err = tensor.Set(1, -1, 0)
	if err == nil {
		t.Error("Set(1, -1, 0) error = nil, want error")
	}

	err = tensor.Set(1, 2, 0)
	if err == nil {
		t.Error("Set(1, 2, 0) error = nil, want error")
	}

	err = tensor.Set(1, 0, 3)
	if err == nil {
		t.Error("Set(1, 0, 3) error = nil, want error")
	}

	// Test clamping to ternary
	err = tensor.Set(2, 0, 0)
	if err != nil {
		t.Errorf("Set(2, 0, 0) error = %v, want nil", err)
	}

	val, err = tensor.Get(0, 0)
	if err != nil {
		t.Errorf("Get(0, 0) error = %v, want nil", err)
	}
	if val != 1 {
		t.Errorf("Get(0, 0) = %v, want 1", val)
	}

	err = tensor.Set(-2, 0, 0)
	if err != nil {
		t.Errorf("Set(-2, 0, 0) error = %v, want nil", err)
	}

	val, err = tensor.Get(0, 0)
	if err != nil {
		t.Errorf("Get(0, 0) error = %v, want nil", err)
	}
	if val != -1 {
		t.Errorf("Get(0, 0) = %v, want -1", val)
	}
}

// TestTensor_Shape tests tensor shape retrieval
func TestTensor_Shape(t *testing.T) {
	tensor, err := NewTensor(2, 3)
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	shape, err := tensor.Shape()
	if err != nil {
		t.Fatalf("Tensor.Shape() failed: %v", err)
	}
	if len(shape) != 2 {
		t.Errorf("Shape() length = %d, want %d", len(shape), 2)
	}
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("Shape() = %v, want [2 3]", shape)
	}
}

// TestTensor_Data tests tensor data retrieval
func TestTensor_Data(t *testing.T) {
	tensor, err := NewTensor(2, 3)
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	data, err := tensor.Data()
	if err != nil {
		t.Fatalf("Tensor.Data() failed: %v", err)
	}
	if len(data) != 6 {
		t.Errorf("Data() length = %d, want %d", len(data), 6)
	}
}

// TestTensor_Close tests tensor cleanup
func TestTensor_Close(t *testing.T) {
	tensor, err := NewTensor(2, 3)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	// Test operations after closing
	tensor.Close()

	// Get should return error
	_, err = tensor.Get(0, 0)
	if err == nil {
		t.Error("Get after Close() error = nil, want error")
	}

	// Set should return error
	err = tensor.Set(1, 0, 0)
	if err == nil {
		t.Error("Set after Close() error = nil, want error")
	}

	// Multiple Close() calls should not panic
	tensor.Close()
}

// TestTensor_ParallelForEach tests parallel processing
func TestTensor_ParallelForEach(t *testing.T) {
	tensor, err := NewTensor(2, 3)
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
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
			got, err := tensor.Get(i, j)
			if err != nil {
				t.Fatalf("Get() failed: %v", err)
			}
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
				_, err := NewTensor(shape...)
				if err != nil {
					b.Fatalf("NewTensor failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkTensor_Get tests value retrieval performance
func BenchmarkTensor_Get(b *testing.B) {
	tensor, err := NewTensor(100, 100)
	if err != nil {
		b.Fatalf("NewTensor failed: %v", err)
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
	tensor, err := NewTensor(100, 100)
	if err != nil {
		b.Fatalf("NewTensor failed: %v", err)
	}
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
			tensor, err := NewTensor(size...)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
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
	tensor, err := NewTensor(100, 100)
	if err != nil {
		b.Fatalf("NewTensor failed: %v", err)
	}
	b.Run("data_access", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = tensor.Data()
		}
	})

	b.Run("data_iteration", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			data, err := tensor.Data()
			if err != nil {
				b.Fatalf("Tensor.Data() failed: %v", err)
			}
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
			tensor, err := NewTensor(shape...)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
			for i := 0; i < b.N; i++ {
				_, _ = tensor.Shape()
			}
		})
	}
}

// BenchmarkTensor_Operations tests common tensor operations
func BenchmarkTensor_Operations(b *testing.B) {
	tensor, err := NewTensor(100, 100)
	if err != nil {
		b.Fatalf("NewTensor failed: %v", err)
	}
	b.Run("get_set_cycle", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			val, err := tensor.Get(50, 50)
			if err != nil {
				b.Fatalf("Get() failed: %v", err)
			}
			tensor.Set(val, 50, 50)
		}
	})

	b.Run("sequential_access", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for j := 0; j < 100; j++ {
				for k := 0; k < 100; k++ {
					val, err := tensor.Get(j, k)
					if err != nil {
						b.Fatalf("Get() failed: %v", err)
					}
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
			tensor, err := NewTensor(tt.initialShape...)
			if err != nil {
				t.Fatalf("NewTensor failed: %v", err)
			}

			// Fill with some test data
			data, err := tensor.Data()
			if err != nil {
				t.Fatalf("Tensor.Data() failed: %v", err)
			}
			for i := 0; i < len(data); i++ {
				err = tensor.Set(int8(i%3-1), tensor.calculateIndices(i)...)
				if err != nil {
					t.Fatalf("Set failed: %v", err)
				}
			}

			// Test reshape
			reshaped, err := tensor.Reshape(tt.newShape...)
			if (err != nil) != tt.wantErr {
				t.Errorf("Reshape() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			if reshaped == nil {
				t.Fatal("Reshape returned nil")
			}

			// Verify shape
			gotShape, err := reshaped.Shape()
			if err != nil {
				t.Fatalf("Tensor.Shape() failed: %v", err)
			}
			if len(gotShape) != len(tt.newShape) {
				t.Errorf("Shape length = %v, want %v", len(gotShape), len(tt.newShape))
			}
			for i := range gotShape {
				if gotShape[i] != tt.newShape[i] {
					t.Errorf("Shape[%d] = %v, want %v", i, gotShape[i], tt.newShape[i])
				}
			}

			// Verify data is preserved
			originalData, err := tensor.Data()
			if err != nil {
				t.Fatalf("Tensor.Data() failed: %v", err)
			}
			reshapedData, err := reshaped.Data()
			if err != nil {
				t.Fatalf("Tensor.Data() failed: %v", err)
			}
			if len(originalData) != len(reshapedData) {
				t.Errorf("Data length = %v, want %v", len(reshapedData), len(originalData))
			}
			for i := range originalData {
				if originalData[i] != reshapedData[i] {
					t.Errorf("Data[%d] = %v, want %v", i, reshapedData[i], originalData[i])
				}
			}
		})
	}
}

func TestTensor_CalculateIndices(t *testing.T) {
	tensor, err := NewTensor(2, 3, 4)
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
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

// TestTensor_CalculateIndex tests index calculation
func TestTensor_CalculateIndex(t *testing.T) {
	tensor, err := NewTensor(2, 3)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	tests := []struct {
		name    string
		indices []int
		wantErr bool
	}{
		{
			name:    "valid indices",
			indices: []int{1, 2},
			wantErr: false,
		},
		{
			name:    "too few indices",
			indices: []int{0},
			wantErr: true,
		},
		{
			name:    "too many indices",
			indices: []int{0, 0, 0},
			wantErr: true,
		},
		{
			name:    "negative index",
			indices: []int{-1, 0},
			wantErr: true,
		},
		{
			name:    "index out of range",
			indices: []int{2, 0},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tensor.calculateIndex(tt.indices)
			if (err != nil) != tt.wantErr {
				t.Errorf("calculateIndex(%v) error = %v, wantErr %v", tt.indices, err, tt.wantErr)
			}
		})
	}
}

func BenchmarkTensor_CalculateIndex(b *testing.B) {
	tensor, err := NewTensor(100, 100)
	if err != nil {
		b.Fatalf("NewTensor failed: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tensor.calculateIndex([]int{50, 50})
	}
}

func TestTensorReshapeEdgeCase(t *testing.T) {
	tensor, err := NewTensor(1, 4)
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	// Fill with valid ternary values (-1, 0, 1)
	for i := 0; i < 4; i++ {
		tensor.SetTernary(int8(i%3-1), 0, i)
	}
	// Attempt to reshape to [1,1,4]
	reshaped, err := tensor.Reshape(1, 1, 4)
	if err != nil {
		t.Fatalf("Reshape failed: %v", err)
	}
	shape, err := reshaped.Shape()
	if err != nil {
		t.Fatalf("Tensor.Shape() failed: %v", err)
	}
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 4 {
		t.Errorf("Reshaped tensor shape = %v, want [1 1 4]", shape)
	}
	// Debug output
	data, err := reshaped.Data()
	if err != nil {
		t.Fatalf("Tensor.Data() failed: %v", err)
	}
	fmt.Printf("Reshaped tensor data: %v\n", data)
	fmt.Printf("Reshaped tensor shape: %v\n", shape)
	// Check data integrity
	for i := 0; i < 4; i++ {
		got, err := reshaped.Get(0, 0, i)
		if err != nil {
			t.Fatalf("Get() failed: %v", err)
		}
		if got != int8(i%3-1) {
			t.Errorf("Reshaped tensor data mismatch at %d: got %v, want %v", i, got, int8(i%3-1))
		}
	}
}

func TestTensor_Transpose(t *testing.T) {
	tests := []struct {
		name      string
		shape     []int
		order     []int
		wantErr   bool
		wantShape []int
	}{
		{
			name:      "valid 2D transpose",
			shape:     []int{2, 3},
			order:     []int{1, 0},
			wantErr:   false,
			wantShape: []int{3, 2},
		},
		{
			name:      "valid 3D transpose",
			shape:     []int{2, 3, 4},
			order:     []int{0, 2, 1},
			wantErr:   false,
			wantShape: []int{2, 4, 3},
		},
		{
			name:      "invalid order length",
			shape:     []int{2, 3},
			order:     []int{0},
			wantErr:   true,
			wantShape: nil,
		},
		{
			name:      "invalid dimension",
			shape:     []int{2, 3},
			order:     []int{0, 2},
			wantErr:   true,
			wantShape: nil,
		},
		{
			name:      "duplicate dimension",
			shape:     []int{2, 3},
			order:     []int{0, 0},
			wantErr:   true,
			wantShape: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create tensor
			tensor, err := NewTensor(tt.shape...)
			if err != nil {
				t.Fatalf("NewTensor failed: %v", err)
			}

			// Fill with test data
			data, err := tensor.Data()
			if err != nil {
				t.Fatalf("Tensor.Data() failed: %v", err)
			}
			for i := 0; i < len(data); i++ {
				err = tensor.Set(int8(i%3-1), tensor.calculateIndices(i)...)
				if err != nil {
					t.Fatalf("Set failed: %v", err)
				}
			}

			// Test transpose
			transposed, err := tensor.Transpose(tt.order...)
			if (err != nil) != tt.wantErr {
				t.Errorf("Transpose() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			if transposed == nil {
				t.Fatal("Transpose returned nil")
			}

			// Verify shape
			gotShape, err := transposed.Shape()
			if err != nil {
				t.Fatalf("Tensor.Shape() failed: %v", err)
			}
			if len(gotShape) != len(tt.wantShape) {
				t.Errorf("Shape length = %v, want %v", len(gotShape), len(tt.wantShape))
			}
			for i := range gotShape {
				if gotShape[i] != tt.wantShape[i] {
					t.Errorf("Shape[%d] = %v, want %v", i, gotShape[i], tt.wantShape[i])
				}
			}

			// Verify data integrity
			data, err = tensor.Data()
			if err != nil {
				t.Fatalf("Tensor.Data() failed: %v", err)
			}
			for i := 0; i < len(data); i++ {
				oldIndices := tensor.calculateIndices(i)
				newIndices := make([]int, len(tt.order))
				for j, o := range tt.order {
					newIndices[j] = oldIndices[o]
				}
				got, err := transposed.Get(newIndices...)
				if err != nil {
					t.Fatalf("Get() failed: %v", err)
				}
				want, err := tensor.Get(oldIndices...)
				if err != nil {
					t.Fatalf("Get() failed: %v", err)
				}
				if got != want {
					t.Errorf("Data mismatch at indices %v: got %v, want %v", newIndices, got, want)
				}
			}
		})
	}
}

func TestTensor_Repeat(t *testing.T) {
	tests := []struct {
		name      string
		shape     []int
		dim       int
		count     int
		wantErr   bool
		wantShape []int
	}{
		{
			name:      "valid 2D repeat",
			shape:     []int{2, 3},
			dim:       0,
			count:     2,
			wantErr:   false,
			wantShape: []int{4, 3},
		},
		{
			name:      "valid 3D repeat",
			shape:     []int{2, 3, 4},
			dim:       1,
			count:     3,
			wantErr:   false,
			wantShape: []int{2, 9, 4},
		},
		{
			name:      "invalid dimension",
			shape:     []int{2, 3},
			dim:       2,
			count:     2,
			wantErr:   true,
			wantShape: nil,
		},
		{
			name:      "invalid count",
			shape:     []int{2, 3},
			dim:       0,
			count:     0,
			wantErr:   true,
			wantShape: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create tensor
			tensor, err := NewTensor(tt.shape...)
			if err != nil {
				t.Fatalf("NewTensor failed: %v", err)
			}

			// Fill with test data
			data, err := tensor.Data()
			if err != nil {
				t.Fatalf("Tensor.Data() failed: %v", err)
			}
			for i := 0; i < len(data); i++ {
				err = tensor.Set(int8(i%3-1), tensor.calculateIndices(i)...)
				if err != nil {
					t.Fatalf("Set failed: %v", err)
				}
			}

			// Test repeat
			repeated, err := tensor.Repeat(tt.dim, tt.count)
			if (err != nil) != tt.wantErr {
				t.Errorf("Repeat() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			if repeated == nil {
				t.Fatal("Repeat returned nil")
			}

			// Verify shape
			gotShape, err := repeated.Shape()
			if err != nil {
				t.Fatalf("Tensor.Shape() failed: %v", err)
			}
			if len(gotShape) != len(tt.wantShape) {
				t.Errorf("Shape length = %v, want %v", len(gotShape), len(tt.wantShape))
			}
			for i := range gotShape {
				if gotShape[i] != tt.wantShape[i] {
					t.Errorf("Shape[%d] = %v, want %v", i, gotShape[i], tt.wantShape[i])
				}
			}

			// Verify data integrity
			data, err = tensor.Data()
			if err != nil {
				t.Fatalf("Tensor.Data() failed: %v", err)
			}
			for i := 0; i < len(data); i++ {
				oldIndices := tensor.calculateIndices(i)
				for c := 0; c < tt.count; c++ {
					newIndices := make([]int, len(oldIndices))
					copy(newIndices, oldIndices)
					shape, err := tensor.Shape()
					if err != nil {
						t.Fatalf("Tensor.Shape() failed: %v", err)
					}
					newIndices[tt.dim] = oldIndices[tt.dim] + c*shape[tt.dim]
					got, err := repeated.Get(newIndices...)
					if err != nil {
						t.Fatalf("Get() failed: %v", err)
					}
					want, err := tensor.Get(oldIndices...)
					if err != nil {
						t.Fatalf("Get() failed: %v", err)
					}
					if got != want {
						t.Errorf("Data mismatch at indices %v: got %v, want %v", newIndices, got, want)
					}
				}
			}
		})
	}
}

func TestTensor_Add(t *testing.T) {
	tests := []struct {
		name    string
		shape   []int
		values1 []int8
		values2 []int8
		want    []int8
		wantErr bool
	}{
		{
			name:    "valid addition",
			shape:   []int{2, 2},
			values1: []int8{1, 2, 3, 4},
			values2: []int8{5, 6, 7, 8},
			want:    []int8{6, 8, 10, 12},
			wantErr: false,
		},
		{
			name:    "shape mismatch",
			shape:   []int{2, 2},
			values1: []int8{1, 2, 3, 4},
			values2: []int8{5, 6},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create tensors
			t1, err := NewTensor(tt.shape...)
			if err != nil {
				t.Fatalf("NewTensor failed: %v", err)
			}
			var t2 *Tensor
			if tt.wantErr && tt.name == "shape mismatch" {
				t2, err = NewTensor(1, 2) // Truly mismatched shape
				if err != nil {
					t.Fatalf("NewTensor failed: %v", err)
				}
			} else {
				t2, err = NewTensor(tt.shape...)
				if err != nil {
					t.Fatalf("NewTensor failed: %v", err)
				}
			}
			if t1 == nil || t2 == nil {
				t.Fatal("NewTensor returned nil")
			}

			// Fill with test data
			for i := 0; i < len(tt.values1); i++ {
				indices := t1.calculateIndices(i)
				err = t1.setRaw(tt.values1[i], indices...)
				if err != nil {
					t.Fatalf("setRaw failed: %v", err)
				}
			}
			for i := 0; i < len(tt.values2); i++ {
				indices := t2.calculateIndices(i)
				err = t2.setRaw(tt.values2[i], indices...)
				if err != nil {
					t.Fatalf("setRaw failed: %v", err)
				}
			}

			// Test addition
			result, err := t1.Add(t2)
			if (err != nil) != tt.wantErr {
				t.Errorf("Add() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			if result == nil {
				t.Fatal("Add returned nil")
			}

			// Verify shape
			gotShape, err := result.Shape()
			if err != nil {
				t.Fatalf("Tensor.Shape() failed: %v", err)
			}
			if len(gotShape) != len(tt.shape) {
				t.Errorf("Shape length = %v, want %v", len(gotShape), len(tt.shape))
			}
			for i := range gotShape {
				if gotShape[i] != tt.shape[i] {
					t.Errorf("Shape[%d] = %v, want %v", i, gotShape[i], tt.shape[i])
				}
			}

			// Verify values
			data, err := result.Data()
			if err != nil {
				t.Fatalf("Tensor.Data() failed: %v", err)
			}
			if len(data) != len(tt.want) {
				t.Errorf("Data length = %v, want %v", len(data), len(tt.want))
			}
			for i := range data {
				if data[i] != tt.want[i] {
					t.Errorf("Data[%d] = %v, want %v", i, data[i], tt.want[i])
				}
			}
		})
	}
}

func TestTensor_SetTernary(t *testing.T) {
	tests := []struct {
		name    string
		value   int8
		indices []int
		want    int8
	}{
		{
			name:    "set valid ternary value",
			value:   1,
			indices: []int{0, 0},
			want:    1,
		},
		{
			name:    "set invalid ternary value",
			value:   2,
			indices: []int{0, 0},
			want:    1,
		},
		{
			name:    "set negative ternary value",
			value:   -2,
			indices: []int{0, 0},
			want:    -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := NewTensor(2, 3)
			if err != nil {
				t.Fatalf("NewTensor failed: %v", err)
			}
			tensor.SetTernary(tt.value, tt.indices...)
			got, err := tensor.Get(tt.indices...)
			if err != nil {
				t.Fatalf("Get() failed: %v", err)
			}
			if got != tt.want {
				t.Errorf("Get() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewTensorFromData(t *testing.T) {
	tests := []struct {
		name  string
		data  []int8
		rows  int
		want  []int8
		shape []int
	}{
		{
			name:  "valid 2D data",
			data:  []int8{1, -1, 0, 1},
			rows:  2,
			want:  []int8{1, -1, 0, 1},
			shape: []int{2, 2},
		},
		{
			name:  "valid 1D data",
			data:  []int8{1, -1, 0, 1},
			rows:  0,
			want:  []int8{1, -1, 0, 1},
			shape: []int{4},
		},
		{
			name:  "empty data",
			data:  []int8{},
			rows:  0,
			want:  []int8{},
			shape: []int{0},
		},
		{
			name:  "invalid dimensions",
			data:  []int8{1, 2, 3},
			rows:  2,
			want:  nil,
			shape: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewTensorFromData(tt.data, tt.rows)
			if tt.want == nil {
				if err == nil {
					t.Error("NewTensorFromData() error = nil, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("NewTensorFromData() failed: %v", err)
			}
			if got == nil {
				t.Fatal("NewTensorFromData() returned nil")
			}
			shape, err := got.Shape()
			if err != nil {
				t.Fatalf("Tensor.Shape() failed: %v", err)
			}
			if len(shape) != len(tt.shape) {
				t.Errorf("Shape() length = %d, want %d", len(shape), len(tt.shape))
			}
			for i := range tt.shape {
				if shape[i] != tt.shape[i] {
					t.Errorf("Shape()[%d] = %d, want %d", i, shape[i], tt.shape[i])
				}
			}
			data, err := got.Data()
			if err != nil {
				t.Fatalf("Tensor.Data() failed: %v", err)
			}
			if len(data) != len(tt.want) {
				t.Errorf("Data() length = %d, want %d", len(data), len(tt.want))
			}
			for i := range data {
				if data[i] != tt.want[i] {
					t.Errorf("Data()[%d] = %v, want %v", i, data[i], tt.want[i])
				}
			}
		})
	}
}

func TestDebugLog(t *testing.T) {
	// Test that DebugLog doesn't panic
	DebugLog("Test debug message")
	DebugLog("Test debug message with args: %d, %s", 42, "test")
}

func TestTensor_setRaw(t *testing.T) {
	tests := []struct {
		name    string
		value   int8
		indices []int
		want    int8
		wantErr bool
	}{
		{
			name:    "set raw value within range",
			value:   42,
			indices: []int{0, 0},
			want:    42,
			wantErr: false,
		},
		{
			name:    "set raw value at max int8",
			value:   127,
			indices: []int{0, 1},
			want:    127,
			wantErr: false,
		},
		{
			name:    "set raw value at min int8",
			value:   -128,
			indices: []int{1, 0},
			want:    -128,
			wantErr: false,
		},
		{
			name:    "invalid indices",
			value:   1,
			indices: []int{1},
			want:    0,
			wantErr: true,
		},
		{
			name:    "out of bounds",
			value:   1,
			indices: []int{2, 0},
			want:    0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := NewTensor(2, 2)
			if err != nil {
				t.Fatalf("NewTensor failed: %v", err)
			}

			err = tensor.setRaw(tt.value, tt.indices...)
			if (err != nil) != tt.wantErr {
				t.Errorf("setRaw() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				got, err := tensor.Get(tt.indices...)
				if err != nil {
					t.Fatalf("Get() failed: %v", err)
				}
				if got != tt.want {
					t.Errorf("setRaw() value = %v, want %v", got, tt.want)
				}
			}
		})
	}

	// Test setRaw after Close
	t.Run("setRaw after Close", func(t *testing.T) {
		tensor, err := NewTensor(2, 2)
		if err != nil {
			t.Fatalf("NewTensor failed: %v", err)
		}
		err = tensor.Close()
		if err != nil {
			t.Fatalf("Close failed: %v", err)
		}
		err = tensor.setRaw(1, 0, 0)
		if err == nil {
			t.Error("setRaw did not return error after Close")
		}
	})
}

func TestTensor_Reshape_EdgeCases(t *testing.T) {
	tests := []struct {
		name         string
		initialShape []int
		newShape     []int
		setup        func(*Tensor)
		wantErr      bool
	}{
		{
			name:         "reshape with non-contiguous data",
			initialShape: []int{2, 3},
			newShape:     []int{3, 2},
			setup: func(t *Tensor) {
				// Set values in non-sequential order
				t.Set(1, 0, 0)
				t.Set(2, 1, 2)
				t.Set(3, 0, 1)
			},
			wantErr: false,
		},
		{
			name:         "reshape with zero values",
			initialShape: []int{2, 2},
			newShape:     []int{4, 1},
			setup: func(t *Tensor) {
				// Set all values to zero
				for i := 0; i < 2; i++ {
					for j := 0; j < 2; j++ {
						t.Set(0, i, j)
					}
				}
			},
			wantErr: false,
		},
		{
			name:         "reshape with negative values",
			initialShape: []int{2, 2},
			newShape:     []int{4, 1},
			setup: func(t *Tensor) {
				// Set negative values
				t.Set(-1, 0, 0)
				t.Set(-2, 0, 1)
				t.Set(-3, 1, 0)
				t.Set(-4, 1, 1)
			},
			wantErr: false,
		},
		{
			name:         "reshape with large dimensions",
			initialShape: []int{100, 100},
			newShape:     []int{1000, 10},
			setup: func(t *Tensor) {
				// Set pattern of values
				for i := 0; i < 100; i++ {
					for j := 0; j < 100; j++ {
						t.Set(int8((i+j)%3-1), i, j)
					}
				}
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := NewTensor(tt.initialShape...)
			if err != nil {
				t.Fatalf("NewTensor failed: %v", err)
			}

			tt.setup(tensor)

			if tt.wantErr {
				defer func() {
					if r := recover(); r == nil {
						t.Error("Reshape did not panic as expected")
					}
				}()
			}

			reshaped, err := tensor.Reshape(tt.newShape...)
			if !tt.wantErr {
				if reshaped == nil {
					t.Fatal("Reshape returned nil")
				}

				// Verify shape
				gotShape, err := reshaped.Shape()
				if err != nil {
					t.Fatalf("Tensor.Shape() failed: %v", err)
				}
				if len(gotShape) != len(tt.newShape) {
					t.Errorf("Shape length = %v, want %v", len(gotShape), len(tt.newShape))
				}
				for i := range gotShape {
					if gotShape[i] != tt.newShape[i] {
						t.Errorf("Shape[%d] = %v, want %v", i, gotShape[i], tt.newShape[i])
					}
				}

				// Verify data is preserved
				originalData, err := tensor.Data()
				if err != nil {
					t.Fatalf("Tensor.Data() failed: %v", err)
				}
				reshapedData, err := reshaped.Data()
				if err != nil {
					t.Fatalf("Tensor.Data() failed: %v", err)
				}
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

func TestTensor_SetTernary_EdgeCases(t *testing.T) {
	tests := []struct {
		name    string
		value   int8
		indices []int
		want    int8
		wantErr bool
	}{
		{
			name:    "set ternary value at boundary",
			value:   1,
			indices: []int{0, 0},
			want:    1,
			wantErr: false,
		},
		{
			name:    "set ternary value above boundary",
			value:   2,
			indices: []int{0, 0},
			want:    1,
			wantErr: false,
		},
		{
			name:    "set ternary value below boundary",
			value:   -2,
			indices: []int{0, 0},
			want:    -1,
			wantErr: false,
		},
		{
			name:    "set ternary value at max int8",
			value:   127,
			indices: []int{0, 0},
			want:    1,
			wantErr: false,
		},
		{
			name:    "set ternary value at min int8",
			value:   -128,
			indices: []int{0, 0},
			want:    -1,
			wantErr: false,
		},
		{
			name:    "set ternary value with invalid indices",
			value:   1,
			indices: []int{1},
			want:    0,
			wantErr: true,
		},
		{
			name:    "set ternary value out of bounds",
			value:   1,
			indices: []int{2, 0},
			want:    0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := NewTensor(2, 2)
			if err != nil {
				t.Fatalf("NewTensor failed: %v", err)
			}

			err = tensor.SetTernary(tt.value, tt.indices...)
			if (err != nil) != tt.wantErr {
				t.Errorf("SetTernary() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				got, err := tensor.Get(tt.indices...)
				if err != nil {
					t.Fatalf("Get() failed: %v", err)
				}
				if got != tt.want {
					t.Errorf("SetTernary() value = %v, want %v", got, tt.want)
				}
			}
		})
	}

	// Test SetTernary after Close
	t.Run("SetTernary after Close", func(t *testing.T) {
		tensor, err := NewTensor(2, 2)
		if err != nil {
			t.Fatalf("NewTensor failed: %v", err)
		}
		err = tensor.Close()
		if err != nil {
			t.Fatalf("Close failed: %v", err)
		}
		err = tensor.SetTernary(1, 0, 0)
		if err == nil {
			t.Error("SetTernary did not return error after Close")
		}
	})
}

func TestTensorLifecycle(t *testing.T) {
	// Create a new tensor
	tensor, err := NewTensor(2, 3)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	// Fill with data
	data, err := tensor.Data()
	if err != nil {
		t.Fatalf("Failed to get tensor data: %v", err)
	}
	for i := range data {
		data[i] = int8(i)
	}

	// Verify data
	data, err = tensor.Data()
	if err != nil {
		t.Fatalf("Failed to get tensor data: %v", err)
	}
	if len(data) != 6 {
		t.Errorf("Expected data length 6, got %d", len(data))
	}

	// Verify shape
	shape, err := tensor.Shape()
	if err != nil {
		t.Fatalf("Failed to get tensor shape: %v", err)
	}
	if !reflect.DeepEqual(shape, []int{2, 3}) {
		t.Errorf("Expected shape [2 3], got %v", shape)
	}

	// Close tensor
	err = tensor.Close()
	if err != nil {
		t.Errorf("Failed to close tensor: %v", err)
	}

	// Verify operations return errors after close
	_, err = tensor.Data()
	if err == nil {
		t.Error("Expected error after tensor close")
	}

	_, err = tensor.Shape()
	if err == nil {
		t.Error("Expected error after tensor close")
	}

	err = tensor.Set(0, 0, 0)
	if err == nil {
		t.Error("Expected error after tensor close")
	}

	_, err = tensor.Get(0, 0)
	if err == nil {
		t.Error("Expected error after tensor close")
	}
}
