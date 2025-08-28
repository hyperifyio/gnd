package layer_norm

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
			layer, err := NewLayerNorm(tt.hiddenDim)
			if tt.wantPanic {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, layer)
			assert.Equal(t, tt.hiddenDim, layer.hiddenDim)
			assert.Equal(t, float32(1e-5), layer.epsilon)
			assert.NotNil(t, layer.gamma)
			shape, err := layer.gamma.Shape()
			if err != nil {
				t.Fatalf("Failed to get gamma shape: %v", err)
			}
			assert.Equal(t, []int{tt.hiddenDim}, shape)

			// Verify gamma is initialized with ones
			for i := 0; i < tt.hiddenDim; i++ {
				val, err := layer.gamma.Get(i)
				if err != nil {
					t.Fatalf("Failed to get gamma value: %v", err)
				}
				assert.Equal(t, int8(1), val)
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
				t, err := tensor.NewTensor(2, 4)
				if err != nil {
					panic(err) // This is a test setup, so we can panic
				}
				for i := 0; i < 2; i++ {
					for j := 0; j < 4; j++ {
						t.Set(int8(i+j), i, j)
					}
				}
				return t
			}(),
			gamma: func() *tensor.Tensor {
				t, err := tensor.NewTensor(4)
				if err != nil {
					panic(err) // This is a test setup, so we can panic
				}
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
				t, err := tensor.NewTensor(2, 3, 4)
				if err != nil {
					panic(err) // This is a test setup, so we can panic
				}
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
				t, err := tensor.NewTensor(4)
				if err != nil {
					panic(err) // This is a test setup, so we can panic
				}
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
				t, err := tensor.NewTensor(2, 3, 4, 5)
				if err != nil {
					panic(err) // This is a test setup, so we can panic
				}
				return t
			}(),
			wantErr: true,
		},
		{
			name:      "mismatched hidden dimension",
			hiddenDim: 4,
			input: func() *tensor.Tensor {
				t, err := tensor.NewTensor(2, 5)
				if err != nil {
					panic(err) // This is a test setup, so we can panic
				}
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
			layer, err := NewLayerNorm(tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create layer norm: %v", err)
			}
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
				shape, err := output.Shape()
				if err != nil {
					t.Fatalf("Failed to get output shape: %v", err)
				}
				assert.Equal(t, tt.wantShape, shape)

				// Verify normalization properties
				if len(shape) == 2 {
					// For 2D output [batch_size, hidden_dim]
					for i := 0; i < shape[0]; i++ {
						// Calculate mean and variance of normalized values
						var sum float64
						var sumSq float64
						for j := 0; j < shape[1]; j++ {
							val, err := output.Get(i, j)
							if err != nil {
								t.Fatalf("Failed to get output value at (%d,%d): %v", i, j, err)
							}
							sum += float64(val)
							sumSq += float64(val) * float64(val)
						}
						mean := sum / float64(shape[1])
						variance := (sumSq / float64(shape[1])) - (mean * mean)

						// Mean should be close to 0 after normalization
						assert.InDelta(t, 0.0, mean, 1e-5, "Mean should be close to 0")
						// Variance should be close to 0.25 after scaling
						assert.InDelta(t, 0.25, variance, 1e-5, "Variance should be close to 0.25 after scaling")
					}
				} else {
					// For 3D output [batch_size, seq_len, hidden_dim]
					for i := 0; i < shape[0]; i++ {
						for j := 0; j < shape[1]; j++ {
							// Calculate mean and variance of normalized values
							var sum float64
							var sumSq float64
							for k := 0; k < shape[2]; k++ {
								val, err := output.Get(i, j, k)
								if err != nil {
									t.Fatalf("Failed to get output value at (%d,%d,%d): %v", i, j, k, err)
								}
								sum += float64(val)
								sumSq += float64(val) * float64(val)
							}
							mean := sum / float64(shape[2])
							variance := (sumSq / float64(shape[2])) - (mean * mean)

							// Mean should be close to 0 after normalization
							assert.InDelta(t, 0.0, mean, 1e-5, "Mean should be close to 0")
							// Variance should be close to 0.25 after scaling
							assert.InDelta(t, 0.25, variance, 1e-5, "Variance should be close to 0.25 after scaling")
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
				t, err := tensor.NewTensor(4)
				if err != nil {
					panic(err) // This is a test setup, so we can panic
				}
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
				t, err := tensor.NewTensor(5)
				if err != nil {
					panic(err) // This is a test setup, so we can panic
				}
				return t
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
			layer, err := NewLayerNorm(tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create layer norm: %v", err)
			}
			require.NotNil(t, layer)

			err = layer.SetGamma(tt.gamma)
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
	layer, err := NewLayerNorm(hiddenDim)
	if err != nil {
		t.Fatalf("Failed to create layer norm: %v", err)
	}
	require.NotNil(t, layer)

	gamma, err := layer.GetGamma()
	if err != nil {
		t.Fatalf("Failed to get gamma: %v", err)
	}
	assert.NotNil(t, gamma)
	shape, err := gamma.Shape()
	if err != nil {
		t.Fatalf("Failed to get gamma shape: %v", err)
	}
	assert.Equal(t, []int{hiddenDim}, shape)

	// Verify gamma values
	for i := 0; i < hiddenDim; i++ {
		val, err := gamma.Get(i)
		if err != nil {
			t.Fatalf("Failed to get gamma value at index %d: %v", i, err)
		}
		assert.Equal(t, int8(1), val)
	}
}

func TestLayerNorm_Close(t *testing.T) {
	layer, err := NewLayerNorm(4)
	if err != nil {
		t.Fatalf("Failed to create layer norm: %v", err)
	}
	require.NotNil(t, layer)

	// Set some gamma
	gamma, err := tensor.NewTensor(4)
	if err != nil {
		t.Fatalf("Failed to create gamma tensor: %v", err)
	}
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

func TestLayerNormGammaClosedPanic(t *testing.T) {
	norm, err := NewLayerNorm(4)
	if err != nil {
		t.Fatalf("Failed to create layer norm: %v", err)
	}
	gamma, err := tensor.NewTensor(4)
	if err != nil {
		t.Fatalf("Failed to create gamma tensor: %v", err)
	}
	for i := 0; i < 4; i++ {
		gamma.Set(1, i)
	}
	norm.SetGamma(gamma)
	gamma.Close() // Close gamma before Forward
	input, err := tensor.NewTensor(1, 4)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer input.Close()
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic when gamma tensor is closed, but did not panic")
		}
	}()
	_, _ = norm.Forward(input)
}

// Benchmarks

func BenchmarkLayerNorm_Forward_2D(b *testing.B) {
	hiddenDim := 512
	layer, err := NewLayerNorm(hiddenDim)
	if err != nil {
		b.Fatalf("Failed to create layer norm: %v", err)
	}
	require.NotNil(b, layer)

	// Create input tensor
	input, err := tensor.NewTensor(32, hiddenDim)
	if err != nil {
		b.Fatalf("Failed to create input tensor: %v", err)
	}
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
	layer, err := NewLayerNorm(hiddenDim)
	if err != nil {
		b.Fatalf("Failed to create layer norm: %v", err)
	}
	require.NotNil(b, layer)

	// Create input tensor
	input, err := tensor.NewTensor(32, 16, hiddenDim)
	if err != nil {
		b.Fatalf("Failed to create input tensor: %v", err)
	}
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

	layer, err := NewLayerNorm(hiddenDim)
	if err != nil {
		b.Fatalf("Failed to create layer norm: %v", err)
	}
	defer layer.Close()

	// Create input tensor
	input, err := tensor.NewTensor(batchSize, seqLen, hiddenDim)
	if err != nil {
		b.Fatalf("Failed to create input tensor: %v", err)
	}
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
