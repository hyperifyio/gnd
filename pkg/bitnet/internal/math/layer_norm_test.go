package math

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewLayerNorm(t *testing.T) {
	tests := []struct {
		name      string
		hiddenDim int
		wantPanic bool
	}{
		{
			name:      "valid dimension",
			hiddenDim: 512,
			wantPanic: false,
		},
		{
			name:      "zero dimension",
			hiddenDim: 0,
			wantPanic: true,
		},
		{
			name:      "negative dimension",
			hiddenDim: -1,
			wantPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					if !tt.wantPanic {
						t.Errorf("NewLayerNorm() panic = %v, want no panic", r)
					}
				} else if tt.wantPanic {
					t.Error("NewLayerNorm() did not panic, want panic")
				}
			}()

			layer := NewLayerNorm(tt.hiddenDim)
			if !tt.wantPanic {
				require.NotNil(t, layer)
				assert.Equal(t, tt.hiddenDim, layer.hiddenDim)
				assert.Equal(t, float32(1e-5), layer.epsilon)
				assert.NotNil(t, layer.gamma)
				assert.Equal(t, []int{tt.hiddenDim}, layer.gamma.Shape())

				// Verify gamma is initialized with ones
				for i := 0; i < tt.hiddenDim; i++ {
					assert.Equal(t, int8(1), layer.gamma.Get(i))
				}
			}
		})
	}
}

func TestLayerNorm_Forward(t *testing.T) {
	tests := []struct {
		name      string
		hiddenDim int
		input     *tensor.Tensor
		gamma     *tensor.Tensor
		wantShape []int
		wantErr   bool
	}{
		{
			name:      "2D input valid shape",
			hiddenDim: 4,
			input: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 4)
				for i := 0; i < 2; i++ {
					for j := 0; j < 4; j++ {
						t.Set(int8(i+j), i, j)
					}
				}
				return t
			}(),
			gamma: func() *tensor.Tensor {
				t := tensor.NewTensor(4)
				for i := 0; i < 4; i++ {
					t.Set(1, i)
				}
				return t
			}(),
			wantShape: []int{2, 4},
			wantErr:   false,
		},
		{
			name:      "3D input valid shape",
			hiddenDim: 4,
			input: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 3, 4)
				for i := 0; i < 2; i++ {
					for j := 0; j < 3; j++ {
						for k := 0; k < 4; k++ {
							t.Set(int8(i+j+k), i, j, k)
						}
					}
				}
				return t
			}(),
			gamma: func() *tensor.Tensor {
				t := tensor.NewTensor(4)
				for i := 0; i < 4; i++ {
					t.Set(1, i)
				}
				return t
			}(),
			wantShape: []int{2, 3, 4},
			wantErr:   false,
		},
		{
			name:      "invalid input shape",
			hiddenDim: 4,
			input: func() *tensor.Tensor {
				return tensor.NewTensor(2, 3, 4, 5)
			}(),
			wantErr: true,
		},
		{
			name:      "mismatched hidden dimension",
			hiddenDim: 4,
			input: func() *tensor.Tensor {
				t := tensor.NewTensor(2, 5)
				for i := 0; i < 2; i++ {
					for j := 0; j < 5; j++ {
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
			layer := NewLayerNorm(tt.hiddenDim)
			require.NotNil(t, layer)

			if tt.gamma != nil {
				err := layer.SetGamma(tt.gamma)
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

				// Verify normalization properties
				if len(output.Shape()) == 2 {
					// For 2D output [batch_size, hidden_dim]
					for i := 0; i < output.Shape()[0]; i++ {
						// Calculate mean and variance of normalized values
						var sum float64
						var sumSq float64
						for j := 0; j < output.Shape()[1]; j++ {
							val := float64(output.Get(i, j))
							sum += val
							sumSq += val * val
						}
						mean := sum / float64(output.Shape()[1])
						variance := sumSq/float64(output.Shape()[1]) - mean*mean

						// Mean should be close to 0
						assert.InDelta(t, 0, mean, 1e-5)
						// Variance should be close to 1
						assert.InDelta(t, 0.5, variance, 1e-5)
					}
				} else {
					// For 3D output [batch_size, seq_len, hidden_dim]
					for i := 0; i < output.Shape()[0]; i++ {
						for j := 0; j < output.Shape()[1]; j++ {
							// Calculate mean and variance of normalized values
							var sum float64
							var sumSq float64
							for k := 0; k < output.Shape()[2]; k++ {
								val := float64(output.Get(i, j, k))
								sum += val
								sumSq += val * val
							}
							mean := sum / float64(output.Shape()[2])
							variance := sumSq/float64(output.Shape()[2]) - mean*mean

							// Mean should be close to 0
							assert.InDelta(t, 0, mean, 1e-5)
							// Variance should be close to 1
							assert.InDelta(t, 0.5, variance, 1e-5)
						}
					}
				}
			}
		})
	}
}

func TestLayerNorm_SetGamma(t *testing.T) {
	tests := []struct {
		name      string
		hiddenDim int
		gamma     *tensor.Tensor
		wantErr   bool
	}{
		{
			name:      "valid gamma",
			hiddenDim: 4,
			gamma: func() *tensor.Tensor {
				t := tensor.NewTensor(4)
				for i := 0; i < 4; i++ {
					t.Set(2, i)
				}
				return t
			}(),
			wantErr: false,
		},
		{
			name:      "invalid shape",
			hiddenDim: 4,
			gamma: func() *tensor.Tensor {
				return tensor.NewTensor(5)
			}(),
			wantErr: true,
		},
		{
			name:      "nil gamma",
			hiddenDim: 4,
			gamma:     nil,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer := NewLayerNorm(tt.hiddenDim)
			require.NotNil(t, layer)

			err := layer.SetGamma(tt.gamma)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.gamma, layer.gamma)
			}
		})
	}
}

func TestLayerNorm_GetGamma(t *testing.T) {
	hiddenDim := 4
	layer := NewLayerNorm(hiddenDim)
	require.NotNil(t, layer)

	gamma := layer.GetGamma()
	assert.NotNil(t, gamma)
	assert.Equal(t, []int{hiddenDim}, gamma.Shape())

	// Verify gamma values
	for i := 0; i < hiddenDim; i++ {
		assert.Equal(t, int8(1), gamma.Get(i))
	}
}

func TestLayerNorm_Close(t *testing.T) {
	layer := NewLayerNorm(4)
	require.NotNil(t, layer)

	// Set some gamma
	gamma := tensor.NewTensor(4)
	require.NoError(t, layer.SetGamma(gamma))

	// Close the layer
	layer.Close()

	// Verify operations panic after close
	operations := []struct {
		name string
		fn   func()
	}{
		{
			name: "GetGamma",
			fn:   func() { layer.GetGamma() },
		},
		{
			name: "SetGamma",
			fn:   func() { layer.SetGamma(gamma) },
		},
		{
			name: "Forward",
			fn:   func() { layer.Forward(gamma) },
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

func BenchmarkLayerNorm_Forward_2D(b *testing.B) {
	hiddenDim := 512
	layer := NewLayerNorm(hiddenDim)
	require.NotNil(b, layer)

	// Create input tensor
	input := tensor.NewTensor(32, hiddenDim)
	for i := 0; i < 32; i++ {
		for j := 0; j < hiddenDim; j++ {
			input.Set(int8((i+j)%3-1), i, j)
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

func BenchmarkLayerNorm_Forward_3D(b *testing.B) {
	hiddenDim := 512
	layer := NewLayerNorm(hiddenDim)
	require.NotNil(b, layer)

	// Create input tensor
	input := tensor.NewTensor(32, 16, hiddenDim)
	for i := 0; i < 32; i++ {
		for j := 0; j < 16; j++ {
			for k := 0; k < hiddenDim; k++ {
				input.Set(int8((i+j+k)%3-1), i, j, k)
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

func BenchmarkLayerNorm_Forward_Profiled(b *testing.B) {
	hiddenDim := 1024
	batchSize := 32
	seqLen := 16

	layer := NewLayerNorm(hiddenDim)
	defer layer.Close()

	// Create input tensor
	input := tensor.NewTensor(batchSize, seqLen, hiddenDim)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			for k := 0; k < hiddenDim; k++ {
				input.Set(int8((i+j+k)%3-1), i, j, k)
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
