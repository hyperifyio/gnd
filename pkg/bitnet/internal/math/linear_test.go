package math

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewLinear(t *testing.T) {
	tests := []struct {
		name      string
		inDim     int
		outDim    int
		wantPanic bool
	}{
		{
			name:      "valid dimensions",
			inDim:     10,
			outDim:    20,
			wantPanic: false,
		},
		{
			name:      "zero input dimension",
			inDim:     0,
			outDim:    20,
			wantPanic: true,
		},
		{
			name:      "zero output dimension",
			inDim:     10,
			outDim:    0,
			wantPanic: true,
		},
		{
			name:      "negative input dimension",
			inDim:     -1,
			outDim:    20,
			wantPanic: true,
		},
		{
			name:      "negative output dimension",
			inDim:     10,
			outDim:    -1,
			wantPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					if !tt.wantPanic {
						t.Errorf("NewLinear() panic = %v, want no panic", r)
					}
				} else if tt.wantPanic {
					t.Error("NewLinear() did not panic, want panic")
				}
			}()

			layer := NewLinear(tt.inDim, tt.outDim)
			if !tt.wantPanic {
				require.NotNil(t, layer)
				assert.Equal(t, tt.inDim, layer.inDim)
				assert.Equal(t, tt.outDim, layer.outDim)
				assert.NotNil(t, layer.weights)
				assert.Equal(t, []int{tt.outDim, tt.inDim}, layer.weights.Shape())
			}
		})
	}
}

func TestLinear_Forward(t *testing.T) {
	tests := []struct {
		name      string
		inDim     int
		outDim    int
		input     *tensor.Tensor
		weights   *tensor.Tensor
		wantShape []int
		wantErr   bool
	}{
		{
			name:   "2D input valid shape",
			inDim:  3,
			outDim: 2,
			input: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 3)
				for i := 0; i < 2; i++ {
					for j := 0; j < 3; j++ {
						t.Set(1, i, j)
					}
				}
				return t
			}(),
			weights: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 3)
				for i := 0; i < 2; i++ {
					for j := 0; j < 3; j++ {
						t.Set(1, i, j)
					}
				}
				return t
			}(),
			wantShape: []int{2, 2},
			wantErr:   false,
		},
		{
			name:   "3D input valid shape",
			inDim:  3,
			outDim: 2,
			input: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 2, 3)
				for i := 0; i < 2; i++ {
					for j := 0; j < 2; j++ {
						for k := 0; k < 3; k++ {
							t.Set(1, i, j, k)
						}
					}
				}
				return t
			}(),
			weights: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 3)
				for i := 0; i < 2; i++ {
					for j := 0; j < 3; j++ {
						t.Set(1, i, j)
					}
				}
				return t
			}(),
			wantShape: []int{2, 2, 2},
			wantErr:   false,
		},
		{
			name:   "invalid input shape",
			inDim:  3,
			outDim: 2,
			input: func() *tensor.Tensor {
				return tensor.NewTensor(2, 3, 4, 5)
			}(),
			wantErr: true,
		},
		{
			name:   "mismatched input dimension",
			inDim:  3,
			outDim: 2,
			input: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 4)
				for i := 0; i < 2; i++ {
					for j := 0; j < 4; j++ {
						t.Set(1, i, j)
					}
				}
				return t
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer := NewLinear(tt.inDim, tt.outDim)
			require.NotNil(t, layer)

			if tt.weights != nil {
				err := layer.SetWeights(tt.weights)
				require.NoError(t, err)
			}

			output, err := layer.Forward(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, output)
			} else {
				require.NoError(t, err)
				require.NotNil(t, output)
				assert.Equal(t, tt.wantShape, output.Shape())
			}
		})
	}
}

func TestLinear_SetWeights(t *testing.T) {
	tests := []struct {
		name    string
		inDim   int
		outDim  int
		weights *tensor.Tensor
		wantErr bool
	}{
		{
			name:   "valid weights",
			inDim:  3,
			outDim: 2,
			weights: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 3)
				for i := 0; i < 2; i++ {
					for j := 0; j < 3; j++ {
						t.Set(1, i, j)
					}
				}
				return t
			}(),
			wantErr: false,
		},
		{
			name:    "nil weights",
			inDim:   3,
			outDim:  2,
			weights: nil,
			wantErr: true,
		},
		{
			name:   "invalid shape",
			inDim:  3,
			outDim: 2,
			weights: func() *tensor.Tensor {
				return tensor.NewTensor(3, 2)
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer := NewLinear(tt.inDim, tt.outDim)
			require.NotNil(t, layer)

			err := layer.SetWeights(tt.weights)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.weights, layer.weights)
			}
		})
	}
}

func TestLinear_GetWeights(t *testing.T) {
	layer := NewLinear(3, 2)
	require.NotNil(t, layer)

	weights := layer.GetWeights()
	assert.NotNil(t, weights)
	assert.Equal(t, []int{2, 3}, weights.Shape())
}

func TestLinear_Close(t *testing.T) {
	layer := NewLinear(3, 2)
	require.NotNil(t, layer)

	// Set some weights
	weights := tensor.NewTensor(2, 3)
	require.NoError(t, layer.SetWeights(weights))

	// Close the layer
	layer.Close()

	// Verify operations panic after close
	operations := []struct {
		name string
		fn   func()
	}{
		{
			name: "GetWeights",
			fn:   func() { layer.GetWeights() },
		},
		{
			name: "SetWeights",
			fn:   func() { layer.SetWeights(weights) },
		},
		{
			name: "Forward",
			fn:   func() { layer.Forward(weights) },
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

// Benchmarks

func BenchmarkLinear_Forward_2D(b *testing.B) {
	layer := NewLinear(512, 256)
	require.NotNil(b, layer)

	// Create input tensor
	input := tensor.NewTensor(32, 512)
	for i := 0; i < 32; i++ {
		for j := 0; j < 512; j++ {
			input.Set(1, i, j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		output, err := layer.Forward(input)
		require.NoError(b, err)
		require.NotNil(b, output)
		output.Close()
	}
}

func BenchmarkLinear_Forward_3D(b *testing.B) {
	layer := NewLinear(512, 256)
	require.NotNil(b, layer)

	// Create input tensor
	input := tensor.NewTensor(32, 16, 512)
	for i := 0; i < 32; i++ {
		for j := 0; j < 16; j++ {
			for k := 0; k < 512; k++ {
				input.Set(1, i, j, k)
			}
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		output, err := layer.Forward(input)
		require.NoError(b, err)
		require.NotNil(b, output)
		output.Close()
	}
}

func BenchmarkLinear_Forward_Profiled(b *testing.B) {
	inDim := 1024
	outDim := 2048
	batchSize := 32
	seqLen := 16

	layer := NewLinear(inDim, outDim)
	defer layer.Close()

	// Fill weights with some values
	weights := tensor.NewTensor(outDim, inDim)
	for i := 0; i < outDim; i++ {
		for j := 0; j < inDim; j++ {
			weights.Set(int8((i+j)%3-1), i, j)
		}
	}
	_ = layer.SetWeights(weights)

	// Create a 3D input tensor
	input := tensor.NewTensor(batchSize, seqLen, inDim)
	for bIdx := 0; bIdx < batchSize; bIdx++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < inDim; d++ {
				input.Set(int8((bIdx+s+d)%3-1), bIdx, s, d)
			}
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		output, err := layer.Forward(input)
		if err != nil {
			b.Fatal(err)
		}
		output.Close()
	}
}
