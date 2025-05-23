package math

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/stretchr/testify/require"
)

func TestFFNSublayer(t *testing.T) {
	tests := []struct {
		name            string
		hiddenDim       int
		intermediateDim int
		input           [][][]int8
		upWeights       [][]int8
		downWeights     [][]int8
		gamma           []float32
	}{
		{
			name:            "standard FFN",
			hiddenDim:       8,
			intermediateDim: 16,
			input: [][][]int8{
				{
					{1, 0, -1, 1, 0, -1, 1, 0},
					{-1, 1, 0, -1, 1, 0, -1, 1},
				},
			},
			upWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
			},
			downWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
			},
			gamma: []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create FFN sublayer
			ffn := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)

			// Create input tensor
			input := tensor.NewTensor(len(tt.input), len(tt.input[0]), len(tt.input[0][0]))
			for i := range tt.input {
				for j := range tt.input[i] {
					for k := range tt.input[i][j] {
						input.Set(tt.input[i][j][k], i, j, k)
					}
				}
			}

			// Create weight tensors
			upWeights := tensor.NewTensor(len(tt.upWeights), len(tt.upWeights[0]))
			for i := range tt.upWeights {
				for j := range tt.upWeights[i] {
					upWeights.Set(tt.upWeights[i][j], i, j)
				}
			}

			downWeights := tensor.NewTensor(len(tt.downWeights), len(tt.downWeights[0]))
			for i := range tt.downWeights {
				for j := range tt.downWeights[i] {
					downWeights.Set(tt.downWeights[i][j], i, j)
				}
			}

			// Set weights and gamma
			ffn.SetWeights(upWeights, downWeights)
			ffn.SetGamma(tt.gamma)

			// Forward pass
			output, err := ffn.Forward(input)
			if err != nil {
				t.Errorf("FFN Forward failed: %v", err)
				return
			}

			// Verify output shape
			if len(output.Shape()) != 3 {
				t.Errorf("output shape = %v, want 3 dimensions", output.Shape())
			}
			if output.Shape()[0] != len(tt.input) {
				t.Errorf("output batch size = %d, want %d", output.Shape()[0], len(tt.input))
			}
			if output.Shape()[1] != len(tt.input[0]) {
				t.Errorf("output seq len = %d, want %d", output.Shape()[1], len(tt.input[0]))
			}
			if output.Shape()[2] != len(tt.input[0][0]) {
				t.Errorf("output hidden dim = %d, want %d", output.Shape()[2], len(tt.input[0][0]))
			}

			// Check that output is not all zeros and has some variance
			allZero := true
			var minVal, maxVal int8
			for i := 0; i < output.Shape()[0]; i++ {
				for j := 0; j < output.Shape()[1]; j++ {
					for k := 0; k < output.Shape()[2]; k++ {
						val := output.Get(i, j, k)
						if val != 0 {
							allZero = false
						}
						if i == 0 && j == 0 && k == 0 {
							minVal, maxVal = val, val
						} else {
							if val < minVal {
								minVal = val
							}
							if val > maxVal {
								maxVal = val
							}
						}
					}
				}
			}
			if allZero {
				t.Errorf("output is all zeros, want nonzero values")
			}
			if minVal == maxVal {
				t.Errorf("output has no variance, want a range of values")
			}
		})
	}
}

func TestFFNSublayerPanics(t *testing.T) {
	tests := []struct {
		name            string
		hiddenDim       int
		intermediateDim int
		input           *tensor.Tensor
	}{
		{
			name:            "invalid input shape",
			hiddenDim:       8,
			intermediateDim: 16,
			input:           tensor.NewTensor(2, 2),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ffn := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			_, err := ffn.Forward(tt.input)
			if err == nil {
				t.Error("expected error for invalid input shape")
			}
		})
	}
}

func BenchmarkFFNSublayer(b *testing.B) {
	benchmarks := []struct {
		name            string
		hiddenDim       int
		intermediateDim int
		seqLen          int
	}{
		{
			name:            "small",
			hiddenDim:       64,
			intermediateDim: 128,
			seqLen:          32,
		},
		{
			name:            "medium",
			hiddenDim:       256,
			intermediateDim: 512,
			seqLen:          128,
		},
		{
			name:            "large",
			hiddenDim:       512,
			intermediateDim: 1024,
			seqLen:          512,
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			// Create FFN sublayer
			ffn := NewFFNSublayer(bm.hiddenDim, bm.intermediateDim)

			// Create input tensor
			input := tensor.NewTensor(1, bm.seqLen, bm.hiddenDim)
			for i := 0; i < bm.seqLen; i++ {
				for j := 0; j < bm.hiddenDim; j++ {
					input.Set(int8((i+j)%8-4), 0, i, j)
				}
			}

			// Create weight tensors
			upWeights := tensor.NewTensor(bm.intermediateDim, bm.hiddenDim)
			downWeights := tensor.NewTensor(bm.hiddenDim, bm.intermediateDim)

			// Fill weights with pseudo-random but deterministic data
			for i := 0; i < bm.intermediateDim; i++ {
				for j := 0; j < bm.hiddenDim; j++ {
					upWeights.Set(int8((i+j)%8-4), i, j)
				}
			}
			for i := 0; i < bm.hiddenDim; i++ {
				for j := 0; j < bm.intermediateDim; j++ {
					downWeights.Set(int8((i-j)%8-4), i, j)
				}
			}

			// Set weights and gamma
			ffn.SetWeights(upWeights, downWeights)
			gamma := make([]float32, bm.hiddenDim)
			for i := range gamma {
				gamma[i] = 1.0
			}
			ffn.SetGamma(gamma)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := ffn.Forward(input)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestFFNSublayer_SingleTokenShape(t *testing.T) {
	hiddenDim := 4
	intermediateDim := 8
	batchSize := 1
	seqLen := 1

	// Create FFNSublayer
	ffnSublayer := NewFFNSublayer(hiddenDim, intermediateDim)

	// Set dummy weights and gamma
	upWeights := tensor.NewTensor(intermediateDim, hiddenDim)
	downWeights := tensor.NewTensor(hiddenDim, intermediateDim)
	for i := 0; i < intermediateDim; i++ {
		for j := 0; j < hiddenDim; j++ {
			upWeights.Set(1, i, j)
		}
	}
	for i := 0; i < hiddenDim; i++ {
		for j := 0; j < intermediateDim; j++ {
			downWeights.Set(1, i, j)
		}
	}
	ffnSublayer.SetWeights(upWeights, downWeights)
	ffnSublayer.SetGamma([]float32{1, 1, 1, 1})

	// Create input tensor [1, 1, 4]
	input := tensor.NewTensor(batchSize, seqLen, hiddenDim)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			for k := 0; k < hiddenDim; k++ {
				input.Set(int8(k+1), i, j, k)
			}
		}
	}

	// Print input shape and data
	t.Logf("Input shape: %v", input.Shape())
	t.Logf("Input data: %v", input.Data())

	// Run forward pass and catch panics
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("FFNSublayer.Forward panicked: %v", r)
		}
	}()
	output, err := ffnSublayer.Forward(input)
	if err != nil {
		t.Errorf("FFN Forward failed: %v", err)
		return
	}

	// Print output shape and data
	t.Logf("Output shape: %v", output.Shape())
	t.Logf("Output data: %v", output.Data())

	// Check output shape
	if len(output.Shape()) != 3 || output.Shape()[0] != batchSize || output.Shape()[1] != seqLen || output.Shape()[2] != hiddenDim {
		t.Errorf("Output shape = %v, want [%d %d %d]", output.Shape(), batchSize, seqLen, hiddenDim)
	}
}

func TestFFNSublayer_CloseResources(t *testing.T) {
	tests := []struct {
		name            string
		hiddenDim       int
		intermediateDim int
	}{
		{
			name:            "standard",
			hiddenDim:       4,
			intermediateDim: 8,
		},
		{
			name:            "large",
			hiddenDim:       512,
			intermediateDim: 2048,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ffn := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)

			// Create and set weights
			upWeights := tensor.NewTensor(tt.intermediateDim, tt.hiddenDim)
			downWeights := tensor.NewTensor(tt.hiddenDim, tt.intermediateDim)
			ffn.SetWeights(upWeights, downWeights)
			defer upWeights.Close()
			defer downWeights.Close()

			// Set gamma
			gamma := make([]float32, tt.hiddenDim)
			for i := range gamma {
				gamma[i] = 1.0
			}
			ffn.SetGamma(gamma)

			// Close the FFN
			ffn.Close()

			// Verify resources are released by checking if we can create a new FFN
			// with the same dimensions without memory issues
			newFFN := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			require.NotNil(t, newFFN)
			newFFN.Close()
		})
	}
}

func TestFFNSublayer_SetWeights(t *testing.T) {
	tests := []struct {
		name            string
		hiddenDim       int
		intermediateDim int
		upWeights       [][]int8
		downWeights     [][]int8
	}{
		{
			name:            "standard_weights",
			hiddenDim:       4,
			intermediateDim: 8,
			upWeights: [][]int8{
				{1, 0, -1, 1},
				{-1, 1, 0, -1},
				{1, 0, -1, 1},
				{-1, 1, 0, -1},
				{1, 0, -1, 1},
				{-1, 1, 0, -1},
				{1, 0, -1, 1},
				{-1, 1, 0, -1},
			},
			downWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
			},
		},
		{
			name:            "all_zeros",
			hiddenDim:       4,
			intermediateDim: 8,
			upWeights:       make([][]int8, 8),
			downWeights:     make([][]int8, 4),
		},
	}

	// Fill all_zeros test data
	for i := range tests[1].upWeights {
		tests[1].upWeights[i] = make([]int8, 4)
	}
	for i := range tests[1].downWeights {
		tests[1].downWeights[i] = make([]int8, 8)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ffn := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			defer ffn.Close()

			// Create weight tensors
			upWeights := tensor.NewTensor(tt.intermediateDim, tt.hiddenDim)
			for i := range tt.upWeights {
				for j := range tt.upWeights[i] {
					upWeights.Set(tt.upWeights[i][j], i, j)
				}
			}
			defer upWeights.Close()
			// Debug print
			t.Logf("upWeights shape: %v", upWeights.Shape())

			downWeights := tensor.NewTensor(tt.hiddenDim, tt.intermediateDim)
			for i := range tt.downWeights {
				for j := range tt.downWeights[i] {
					downWeights.Set(tt.downWeights[i][j], i, j)
				}
			}
			defer downWeights.Close()
			// Debug print
			t.Logf("downWeights shape: %v", downWeights.Shape())

			// Set weights
			ffn.SetWeights(upWeights, downWeights)

			// Set gamma
			gamma := make([]float32, tt.hiddenDim)
			for i := range gamma {
				gamma[i] = 1.0
			}
			ffn.SetGamma(gamma)

			// Verify weights were set by running forward pass
			input := tensor.NewTensor(1, 1, tt.hiddenDim)
			for i := 0; i < tt.hiddenDim; i++ {
				input.Set(1.0, 0, 0, i)
			}
			defer input.Close()

			output, err := ffn.Forward(input)
			require.NoError(t, err)
			require.NotNil(t, output)
			defer output.Close()

			// Verify output shape
			require.Equal(t, []int{1, 1, tt.hiddenDim}, output.Shape())
		})
	}
}

func TestFFNSublayer_SetGamma(t *testing.T) {
	tests := []struct {
		name            string
		hiddenDim       int
		intermediateDim int
		gamma           []float32
	}{
		{
			name:            "ones",
			hiddenDim:       4,
			intermediateDim: 8,
			gamma:           []float32{1.0, 1.0, 1.0, 1.0},
		},
		{
			name:            "scaled",
			hiddenDim:       4,
			intermediateDim: 8,
			gamma:           []float32{0.5, 1.0, 2.0, 0.25},
		},
		{
			name:            "zeros",
			hiddenDim:       4,
			intermediateDim: 8,
			gamma:           []float32{0.0, 0.0, 0.0, 0.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ffn := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			defer ffn.Close()

			// Set up weights with valid shapes
			upWeights := tensor.NewTensor(tt.intermediateDim, tt.hiddenDim)
			downWeights := tensor.NewTensor(tt.hiddenDim, tt.intermediateDim)
			for i := 0; i < tt.intermediateDim; i++ {
				for j := 0; j < tt.hiddenDim; j++ {
					upWeights.Set(1, i, j)
				}
			}
			for i := 0; i < tt.hiddenDim; i++ {
				for j := 0; j < tt.intermediateDim; j++ {
					downWeights.Set(1, i, j)
				}
			}
			ffn.SetWeights(upWeights, downWeights)
			defer upWeights.Close()
			defer downWeights.Close()
			// Debug print
			t.Logf("upWeights shape: %v", upWeights.Shape())
			t.Logf("downWeights shape: %v", downWeights.Shape())

			// Set gamma
			ffn.SetGamma(tt.gamma)

			// Verify gamma was set by running forward pass
			input := tensor.NewTensor(1, 1, tt.hiddenDim)
			for i := 0; i < tt.hiddenDim; i++ {
				input.Set(1.0, 0, 0, i)
			}
			defer input.Close()

			output, err := ffn.Forward(input)
			require.NoError(t, err)
			require.NotNil(t, output)
			defer output.Close()

			// Verify output shape
			require.Equal(t, []int{1, 1, tt.hiddenDim}, output.Shape())
		})
	}
}

func TestFFNSublayer_ForwardEdgeCases(t *testing.T) {
	tests := []struct {
		name            string
		hiddenDim       int
		intermediateDim int
		input           *tensor.Tensor
		wantErr         bool
	}{
		{
			name:            "nil input",
			hiddenDim:       4,
			intermediateDim: 8,
			input:           nil,
			wantErr:         true,
		},
		{
			name:            "invalid shape",
			hiddenDim:       4,
			intermediateDim: 8,
			input:           tensor.NewTensor(2, 3), // 2D tensor with wrong dimensions (should be 2,4)
			wantErr:         true,
		},
		{
			name:            "dimension mismatch",
			hiddenDim:       4,
			intermediateDim: 8,
			input:           tensor.NewTensor(1, 3), // hiddenDim=3, expected=4
			wantErr:         true,
		},
		{
			name:            "empty tensor",
			hiddenDim:       4,
			intermediateDim: 8,
			wantErr:         false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ffn := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			defer ffn.Close()

			// Set up weights and gamma
			upWeights := tensor.NewTensor(tt.intermediateDim, tt.hiddenDim)
			downWeights := tensor.NewTensor(tt.hiddenDim, tt.intermediateDim)
			for i := 0; i < tt.intermediateDim; i++ {
				for j := 0; j < tt.hiddenDim; j++ {
					upWeights.Set(1, i, j)
				}
			}
			for i := 0; i < tt.hiddenDim; i++ {
				for j := 0; j < tt.intermediateDim; j++ {
					downWeights.Set(1, i, j)
				}
			}
			ffn.SetWeights(upWeights, downWeights)
			defer upWeights.Close()
			defer downWeights.Close()

			gamma := make([]float32, tt.hiddenDim)
			for i := range gamma {
				gamma[i] = 1.0
			}
			ffn.SetGamma(gamma)

			if tt.input == nil {
				require.Panics(t, func() {
					ffn.Forward(tt.input)
				}, "Expected panic for nil input")
				return
			}

			if tt.name == "empty tensor" {
				require.Panics(t, func() {
					_ = tensor.NewTensor(1, 0, 4)
				}, "Expected panic for empty tensor with zero dimension")
				return
			}

			// Run forward pass
			output, err := ffn.Forward(tt.input)
			if tt.wantErr {
				require.Error(t, err)
				require.Nil(t, output)
			} else {
				require.NoError(t, err)
				require.NotNil(t, output)
				defer output.Close()
			}
		})
	}
}
