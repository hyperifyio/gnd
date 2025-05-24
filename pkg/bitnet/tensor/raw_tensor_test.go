package tensor

import (
	"testing"
)

func TestRawTensor(t *testing.T) {
	tests := []struct {
		name      string
		rows      int
		cols      int
		setup     func(*rawTensor)
		expected  [][]int8
		wantPanic bool
	}{
		{
			name: "basic 2x2 operations",
			rows: 2,
			cols: 2,
			setup: func(rt *rawTensor) {
				rt.Set(0, 0, 1)
				rt.Set(0, 1, 2)
				rt.Set(1, 0, 3)
				rt.Set(1, 1, 4)
			},
			expected: [][]int8{
				{1, 2},
				{3, 4},
			},
			wantPanic: false,
		},
		{
			name: "full int8 range",
			rows: 2,
			cols: 2,
			setup: func(rt *rawTensor) {
				rt.Set(0, 0, -128)
				rt.Set(0, 1, 127)
				rt.Set(1, 0, 0)
				rt.Set(1, 1, 42)
			},
			expected: [][]int8{
				{-128, 127},
				{0, 42},
			},
			wantPanic: false,
		},
		{
			name: "large matrix",
			rows: 100,
			cols: 100,
			setup: func(rt *rawTensor) {
				for i := 0; i < 100; i++ {
					for j := 0; j < 100; j++ {
						rt.Set(i, j, int8((i+j)%256-128))
					}
				}
			},
			expected:  nil, // Will verify pattern instead of exact values
			wantPanic: false,
		},
		{
			name: "zero dimensions",
			rows: 0,
			cols: 0,
			setup: func(rt *rawTensor) {
				// No setup needed for zero dimensions
			},
			expected:  [][]int8{},
			wantPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.wantPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Error("expected panic")
					}
				}()
			}

			// Create raw tensor
			rt := newRawTensor(tt.rows, tt.cols)

			// Setup values
			tt.setup(rt)

			// Verify values
			if tt.expected != nil {
				for i := 0; i < tt.rows; i++ {
					for j := 0; j < tt.cols; j++ {
						got := rt.At(i, j)
						want := tt.expected[i][j]
						if got != want {
							t.Errorf("At(%d, %d) = %d, want %d", i, j, got, want)
						}
					}
				}
			} else if tt.name == "large matrix" {
				// Verify pattern for large matrix
				for i := 0; i < tt.rows; i++ {
					for j := 0; j < tt.cols; j++ {
						got := rt.At(i, j)
						want := int8((i+j)%256 - 128)
						if got != want {
							t.Errorf("At(%d, %d) = %d, want %d", i, j, got, want)
						}
					}
				}
			}

			// Verify Shape
			rows, cols := rt.Shape()
			if rows != tt.rows || cols != tt.cols {
				t.Errorf("Shape() = (%d, %d), want (%d, %d)", rows, cols, tt.rows, tt.cols)
			}

			// Verify Data
			data := rt.Data()
			if len(data) != tt.rows*tt.cols {
				t.Errorf("Data() length = %d, want %d", len(data), tt.rows*tt.cols)
			}
		})
	}
}

func TestNewRawTensorFrom(t *testing.T) {
	tests := []struct {
		name     string
		input    [][]int8
		expected [][]int8
	}{
		{
			name: "2x2 tensor",
			input: [][]int8{
				{1, 2},
				{3, 4},
			},
			expected: [][]int8{
				{1, 2},
				{3, 4},
			},
		},
		{
			name: "full int8 range",
			input: [][]int8{
				{-128, 127},
				{0, 42},
			},
			expected: [][]int8{
				{-128, 127},
				{0, 42},
			},
		},
		{
			name: "large tensor",
			input: [][]int8{
				{1, 2, 3, 4, 5},
				{6, 7, 8, 9, 10},
				{11, 12, 13, 14, 15},
			},
			expected: [][]int8{
				{1, 2, 3, 4, 5},
				{6, 7, 8, 9, 10},
				{11, 12, 13, 14, 15},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input tensor
			input := NewTensor(len(tt.input), len(tt.input[0]))
			for i := range tt.input {
				for j := range tt.input[i] {
					input.setRaw(tt.input[i][j], i, j)
				}
			}

			// Convert to raw tensor
			rt := newRawTensorFrom(input)

			// Verify values
			for i := 0; i < len(tt.expected); i++ {
				for j := 0; j < len(tt.expected[i]); j++ {
					got := rt.At(i, j)
					want := tt.expected[i][j]
					if got != want {
						t.Errorf("At(%d, %d) = %d, want %d", i, j, got, want)
					}
				}
			}

			// Verify shape
			rows, cols := rt.Shape()
			if rows != len(tt.expected) || cols != len(tt.expected[0]) {
				t.Errorf("Shape() = (%d, %d), want (%d, %d)", rows, cols, len(tt.expected), len(tt.expected[0]))
			}
		})
	}
}

func TestRawTensorPanics(t *testing.T) {
	tests := []struct {
		name string
		fn   func()
	}{
		{
			name: "1D tensor",
			fn: func() {
				t := NewTensor(2)
				newRawTensorFrom(t)
			},
		},
		{
			name: "3D tensor",
			fn: func() {
				t := NewTensor(2, 2, 2)
				newRawTensorFrom(t)
			},
		},
		{
			name: "nil tensor",
			fn: func() {
				newRawTensorFrom(nil)
			},
		},
		{
			name: "negative dimensions",
			fn: func() {
				newRawTensor(-1, 2)
			},
		},
		{
			name: "zero dimensions",
			fn: func() {
				newRawTensor(0, 0)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic")
				}
			}()
			tt.fn()
		})
	}
}

// BenchmarkRawTensor tests raw tensor operations performance
func BenchmarkRawTensor(b *testing.B) {
	sizes := []struct {
		rows int
		cols int
	}{
		{10, 10},
		{100, 100},
		{1000, 1000},
	}

	for _, size := range sizes {
		b.Run("", func(b *testing.B) {
			rt := newRawTensor(size.rows, size.cols)
			b.ResetTimer()

			// Benchmark Set operations
			b.Run("Set", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					rt.Set(i%size.rows, i%size.cols, int8(i%256-128))
				}
			})

			// Benchmark Get operations
			b.Run("Get", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					_ = rt.At(i%size.rows, i%size.cols)
				}
			})

			// Benchmark Data access
			b.Run("Data", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					_ = rt.Data()
				}
			})

			// Benchmark Shape access
			b.Run("Shape", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					_, _ = rt.Shape()
				}
			})
		})
	}
}

// BenchmarkRawTensorCreation tests raw tensor creation performance
func BenchmarkRawTensorCreation(b *testing.B) {
	sizes := []struct {
		rows int
		cols int
	}{
		{10, 10},
		{100, 100},
		{1000, 1000},
	}

	for _, size := range sizes {
		b.Run("", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = newRawTensor(size.rows, size.cols)
			}
		})
	}
}

// BenchmarkRawTensorFrom tests conversion from Tensor to rawTensor
func BenchmarkRawTensorFrom(b *testing.B) {
	sizes := []struct {
		rows int
		cols int
	}{
		{10, 10},
		{100, 100},
		{1000, 1000},
	}

	for _, size := range sizes {
		b.Run("", func(b *testing.B) {
			// Create input tensor
			input := NewTensor(size.rows, size.cols)
			for i := 0; i < size.rows; i++ {
				for j := 0; j < size.cols; j++ {
					input.Set(int8((i+j)%256-128), i, j)
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = newRawTensorFrom(input)
			}
		})
	}
}
