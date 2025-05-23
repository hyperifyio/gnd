package math

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewLMHead(t *testing.T) {
	tests := []struct {
		name      string
		hiddenDim int
		vocabSize int
		wantPanic bool
	}{
		{
			name:      "valid dimensions",
			hiddenDim: 2560,
			vocabSize: 128000,
			wantPanic: false,
		},
		{
			name:      "zero hidden dimension",
			hiddenDim: 0,
			vocabSize: 128000,
			wantPanic: true,
		},
		{
			name:      "zero vocabulary size",
			hiddenDim: 2560,
			vocabSize: 0,
			wantPanic: true,
		},
		{
			name:      "negative hidden dimension",
			hiddenDim: -1,
			vocabSize: 128000,
			wantPanic: true,
		},
		{
			name:      "negative vocabulary size",
			hiddenDim: 2560,
			vocabSize: -1,
			wantPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					if !tt.wantPanic {
						t.Errorf("NewLMHead() panic = %v, want no panic", r)
					}
				} else if tt.wantPanic {
					t.Error("NewLMHead() did not panic, want panic")
				}
			}()

			layer := NewLMHead(tt.hiddenDim, tt.vocabSize)
			if !tt.wantPanic {
				require.NotNil(t, layer)
				assert.Equal(t, tt.hiddenDim, layer.hiddenDim)
				assert.Equal(t, tt.vocabSize, layer.vocabSize)
				assert.Nil(t, layer.weights)
			}
		})
	}
}

func TestLMHead_Forward(t *testing.T) {
	tests := []struct {
		name      string
		hiddenDim int
		vocabSize int
		input     *tensor.Tensor
		weights   *tensor.Tensor
		wantShape []int
		wantErr   bool
	}{
		{
			name:      "valid input and weights",
			hiddenDim: 512,
			vocabSize: 32000,
			input: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 3, 512)
				for i := 0; i < 2; i++ {
					for j := 0; j < 3; j++ {
						for k := 0; k < 512; k++ {
							t.Set(1, i, j, k)
						}
					}
				}
				return t
			}(),
			weights: func() *tensor.Tensor {
				t := tensor.NewTensor(32000, 512)
				for i := 0; i < 32000; i++ {
					for j := 0; j < 512; j++ {
						t.Set(1, i, j)
					}
				}
				return t
			}(),
			wantShape: []int{2, 3, 32000},
			wantErr:   false,
		},
		{
			name:      "nil weights",
			hiddenDim: 512,
			vocabSize: 32000,
			input: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 3, 512)
				for i := 0; i < 2; i++ {
					for j := 0; j < 3; j++ {
						for k := 0; k < 512; k++ {
							t.Set(1, i, j, k)
						}
					}
				}
				return t
			}(),
			weights:   nil,
			wantShape: nil,
			wantErr:   true,
		},
		{
			name:      "invalid input shape",
			hiddenDim: 512,
			vocabSize: 32000,
			input: func() *tensor.Tensor {
				return tensor.NewTensor(2, 3, 4, 5)
			}(),
			weights: func() *tensor.Tensor {
				t := tensor.NewTensor(32000, 512)
				for i := 0; i < 32000; i++ {
					for j := 0; j < 512; j++ {
						t.Set(1, i, j)
					}
				}
				return t
			}(),
			wantShape: nil,
			wantErr:   true,
		},
		{
			name:      "mismatched input dimension",
			hiddenDim: 512,
			vocabSize: 32000,
			input: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 3, 256)
				for i := 0; i < 2; i++ {
					for j := 0; j < 3; j++ {
						for k := 0; k < 256; k++ {
							t.Set(1, i, j, k)
						}
					}
				}
				return t
			}(),
			weights: func() *tensor.Tensor {
				t := tensor.NewTensor(32000, 512)
				for i := 0; i < 32000; i++ {
					for j := 0; j < 512; j++ {
						t.Set(1, i, j)
					}
				}
				return t
			}(),
			wantShape: nil,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer := NewLMHead(tt.hiddenDim, tt.vocabSize)
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

func TestLMHead_SetWeights(t *testing.T) {
	tests := []struct {
		name      string
		hiddenDim int
		vocabSize int
		weights   *tensor.Tensor
		wantErr   bool
	}{
		{
			name:      "valid weights",
			hiddenDim: 2560,
			vocabSize: 128000,
			weights: func() *tensor.Tensor {
				t := tensor.NewTensor(128000, 2560)
				for i := 0; i < 128000; i++ {
					for j := 0; j < 2560; j++ {
						t.Set(1, i, j)
					}
				}
				return t
			}(),
			wantErr: false,
		},
		{
			name:      "nil weights",
			hiddenDim: 2560,
			vocabSize: 128000,
			weights:   nil,
			wantErr:   true,
		},
		{
			name:      "invalid shape",
			hiddenDim: 2560,
			vocabSize: 128000,
			weights: func() *tensor.Tensor {
				return tensor.NewTensor(2560, 128000)
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer := NewLMHead(tt.hiddenDim, tt.vocabSize)
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

func TestLMHead_GetWeights(t *testing.T) {
	layer := NewLMHead(2560, 128000)
	require.NotNil(t, layer)

	weights := layer.GetWeights()
	assert.Nil(t, weights)

	// Set weights
	weights = tensor.NewTensor(128000, 2560)
	for i := 0; i < 128000; i++ {
		for j := 0; j < 2560; j++ {
			weights.Set(1, i, j)
		}
	}
	err := layer.SetWeights(weights)
	require.NoError(t, err)

	// Get weights
	got := layer.GetWeights()
	assert.Equal(t, weights, got)
}

func TestLMHead_Close(t *testing.T) {
	layer := NewLMHead(2560, 128000)
	require.NotNil(t, layer)

	// Set some weights
	weights := tensor.NewTensor(128000, 2560)
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

func BenchmarkLMHead_Forward(b *testing.B) {
	layer := NewLMHead(2560, 128000)
	require.NotNil(b, layer)

	// Create input tensor
	input := tensor.NewTensor(32, 16, 2560)
	for i := 0; i < 32; i++ {
		for j := 0; j < 16; j++ {
			for k := 0; k < 2560; k++ {
				input.Set(1, i, j, k)
			}
		}
	}

	// Create weights tensor
	weights := tensor.NewTensor(128000, 2560)
	for i := 0; i < 128000; i++ {
		for j := 0; j < 2560; j++ {
			weights.Set(1, i, j)
		}
	}
	require.NoError(b, layer.SetWeights(weights))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		output, err := layer.Forward(input)
		require.NoError(b, err)
		require.NotNil(b, output)
		output.Close()
	}
}

func BenchmarkLMHead_Forward_Profiled(b *testing.B) {
	layer := NewLMHead(2560, 128000)
	require.NotNil(b, layer)

	// Create input tensor
	input := tensor.NewTensor(32, 16, 2560)
	for i := 0; i < 32; i++ {
		for j := 0; j < 16; j++ {
			for k := 0; k < 2560; k++ {
				input.Set(int8((i+j+k)%3-1), i, j, k)
			}
		}
	}

	// Create weights tensor
	weights := tensor.NewTensor(128000, 2560)
	for i := 0; i < 128000; i++ {
		for j := 0; j < 2560; j++ {
			weights.Set(int8((i+j)%3-1), i, j)
		}
	}
	require.NoError(b, layer.SetWeights(weights))

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
