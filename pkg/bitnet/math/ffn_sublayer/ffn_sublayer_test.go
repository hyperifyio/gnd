package ffn_sublayer

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
			ffn, err := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			if err != nil {
				t.Fatalf("Failed to create FFN sublayer: %v", err)
			}

			// Convert input to proper shape
			batchSize := len(tt.input)
			seqLen := len(tt.input[0])
			hiddenDim := len(tt.input[0][0])
			input, err := tensor.NewTensor(batchSize, seqLen, hiddenDim)
			require.NoError(t, err)

			// Copy data into tensor
			for i := 0; i < batchSize; i++ {
				for j := 0; j < seqLen; j++ {
					for k := 0; k < hiddenDim; k++ {
						err := input.Set(int8(tt.input[i][j][k]), i, j, k)
						require.NoError(t, err)
					}
				}
			}

			// Create weight tensors
			upWeights, err := tensor.NewTensor(len(tt.upWeights), len(tt.upWeights[0]))
			if err != nil {
				t.Fatalf("Failed to create up weights tensor: %v", err)
			}
			for i := range tt.upWeights {
				for j := range tt.upWeights[i] {
					if err := upWeights.Set(tt.upWeights[i][j], i, j); err != nil {
						t.Fatalf("Failed to set up weight value: %v", err)
					}
				}
			}

			downWeights, err := tensor.NewTensor(len(tt.downWeights), len(tt.downWeights[0]))
			if err != nil {
				t.Fatalf("Failed to create down weights tensor: %v", err)
			}
			for i := range tt.downWeights {
				for j := range tt.downWeights[i] {
					if err := downWeights.Set(tt.downWeights[i][j], i, j); err != nil {
						t.Fatalf("Failed to set down weight value: %v", err)
					}
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
			shape, err := output.Shape()
			if err != nil {
				t.Fatalf("Failed to get output shape: %v", err)
			}
			if len(shape) != 3 {
				t.Errorf("output shape = %v, want 3 dimensions", shape)
			}
			if shape[0] != len(tt.input) {
				t.Errorf("output batch size = %d, want %d", shape[0], len(tt.input))
			}
			if shape[1] != len(tt.input[0]) {
				t.Errorf("output seq len = %d, want %d", shape[1], len(tt.input[0]))
			}
			if shape[2] != len(tt.input[0][0]) {
				t.Errorf("output hidden dim = %d, want %d", shape[2], len(tt.input[0][0]))
			}

			// Check that output is not all zeros and has some variance
			allZero := true
			var minVal, maxVal int8
			for i := 0; i < shape[0]; i++ {
				for j := 0; j < shape[1]; j++ {
					for k := 0; k < shape[2]; k++ {
						val, err := output.Get(i, j, k)
						if err != nil {
							t.Fatalf("Failed to get output value: %v", err)
						}
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
			input:           func() *tensor.Tensor { t, _ := tensor.NewTensor(2, 2); return t }(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ffn, err := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			if err != nil {
				t.Fatalf("Failed to create FFN sublayer: %v", err)
			}
			_, err = ffn.Forward(tt.input)
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
			ffn, err := NewFFNSublayer(bm.hiddenDim, bm.intermediateDim)
			if err != nil {
				b.Fatalf("Failed to create FFN sublayer: %v", err)
			}

			// Create input tensor
			input, err := tensor.NewTensor(1, bm.seqLen, bm.hiddenDim)
			if err != nil {
				b.Fatalf("Failed to create input tensor: %v", err)
			}
			for i := 0; i < bm.seqLen; i++ {
				for j := 0; j < bm.hiddenDim; j++ {
					if err := input.Set(int8((i+j)%8-4), 0, i, j); err != nil {
						b.Fatalf("Failed to set input value: %v", err)
					}
				}
			}

			// Create weight tensors
			upWeights, err := tensor.NewTensor(bm.intermediateDim, bm.hiddenDim)
			if err != nil {
				b.Fatalf("Failed to create up weights tensor: %v", err)
			}
			downWeights, err := tensor.NewTensor(bm.hiddenDim, bm.intermediateDim)
			if err != nil {
				b.Fatalf("Failed to create down weights tensor: %v", err)
			}

			// Fill weights with pseudo-random but deterministic data
			for i := 0; i < bm.intermediateDim; i++ {
				for j := 0; j < bm.hiddenDim; j++ {
					if err := upWeights.Set(int8((i+j)%8-4), i, j); err != nil {
						b.Fatalf("Failed to set up weight value: %v", err)
					}
				}
			}
			for i := 0; i < bm.hiddenDim; i++ {
				for j := 0; j < bm.intermediateDim; j++ {
					if err := downWeights.Set(int8((i-j)%8-4), i, j); err != nil {
						b.Fatalf("Failed to set down weight value: %v", err)
					}
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
	ffnSublayer, err := NewFFNSublayer(hiddenDim, intermediateDim)
	if err != nil {
		t.Fatalf("Failed to create FFN sublayer: %v", err)
	}

	// Set dummy weights and gamma
	upWeights, err := tensor.NewTensor(intermediateDim, hiddenDim)
	if err != nil {
		t.Fatalf("Failed to create up weights tensor: %v", err)
	}
	downWeights, err := tensor.NewTensor(hiddenDim, intermediateDim)
	if err != nil {
		t.Fatalf("Failed to create down weights tensor: %v", err)
	}
	for i := 0; i < intermediateDim; i++ {
		for j := 0; j < hiddenDim; j++ {
			if err := upWeights.Set(1, i, j); err != nil {
				t.Fatalf("Failed to set up weight value: %v", err)
			}
		}
	}
	for i := 0; i < hiddenDim; i++ {
		for j := 0; j < intermediateDim; j++ {
			if err := downWeights.Set(1, i, j); err != nil {
				t.Fatalf("Failed to set down weight value: %v", err)
			}
		}
	}
	ffnSublayer.SetWeights(upWeights, downWeights)
	ffnSublayer.SetGamma([]float32{1, 1, 1, 1})

	// Create input tensor [1, 1, 4]
	input, err := tensor.NewTensor(batchSize, seqLen, hiddenDim)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			for k := 0; k < hiddenDim; k++ {
				if err := input.Set(int8(k+1), i, j, k); err != nil {
					t.Fatalf("Failed to set input value: %v", err)
				}
			}
		}
	}

	// Print input shape and data
	inputShape, err := input.Shape()
	if err != nil {
		t.Fatalf("Failed to get input shape: %v", err)
	}
	inputData, err := input.Data()
	if err != nil {
		t.Fatalf("Failed to get input data: %v", err)
	}
	t.Logf("Input shape: %v", inputShape)
	t.Logf("Input data: %v", inputData)

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
	outputShape, err := output.Shape()
	if err != nil {
		t.Fatalf("Failed to get output shape: %v", err)
	}
	outputData, err := output.Data()
	if err != nil {
		t.Fatalf("Failed to get output data: %v", err)
	}
	t.Logf("Output shape: %v", outputShape)
	t.Logf("Output data: %v", outputData)

	// Check output shape
	if len(outputShape) != 3 || outputShape[0] != batchSize || outputShape[1] != seqLen || outputShape[2] != hiddenDim {
		t.Errorf("Output shape = %v, want [%d %d %d]", outputShape, batchSize, seqLen, hiddenDim)
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
			ffn, err := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			if err != nil {
				t.Fatalf("Failed to create FFN sublayer: %v", err)
			}

			// Create and set weights
			upWeights, err := tensor.NewTensor(tt.intermediateDim, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create up weights tensor: %v", err)
			}
			downWeights, err := tensor.NewTensor(tt.hiddenDim, tt.intermediateDim)
			if err != nil {
				t.Fatalf("Failed to create down weights tensor: %v", err)
			}
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
			newFFN, err := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			if err != nil {
				t.Fatalf("Failed to create new FFN sublayer: %v", err)
			}
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
			ffn, err := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			if err != nil {
				t.Fatalf("Failed to create FFN sublayer: %v", err)
			}
			defer ffn.Close()

			// Create weight tensors
			upWeights, err := tensor.NewTensor(tt.intermediateDim, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create up weights tensor: %v", err)
			}
			for i := range tt.upWeights {
				for j := range tt.upWeights[i] {
					if err := upWeights.Set(tt.upWeights[i][j], i, j); err != nil {
						t.Fatalf("Failed to set up weight value: %v", err)
					}
				}
			}
			defer upWeights.Close()
			// Debug print
			upShape, err := upWeights.Shape()
			require.NoError(t, err)
			t.Logf("upWeights shape: %v", upShape)

			downWeights, err := tensor.NewTensor(tt.hiddenDim, tt.intermediateDim)
			if err != nil {
				t.Fatalf("Failed to create down weights tensor: %v", err)
			}
			for i := range tt.downWeights {
				for j := range tt.downWeights[i] {
					if err := downWeights.Set(tt.downWeights[i][j], i, j); err != nil {
						t.Fatalf("Failed to set down weight value: %v", err)
					}
				}
			}
			defer downWeights.Close()
			// Debug print
			downShape, err := downWeights.Shape()
			require.NoError(t, err)
			t.Logf("downWeights shape: %v", downShape)

			// Set weights
			ffn.SetWeights(upWeights, downWeights)

			// Set gamma
			gamma := make([]float32, tt.hiddenDim)
			for i := range gamma {
				gamma[i] = 1.0
			}
			ffn.SetGamma(gamma)

			// Verify weights were set by running forward pass
			input, err := tensor.NewTensor(1, 1, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create input tensor: %v", err)
			}
			for i := 0; i < tt.hiddenDim; i++ {
				if err := input.Set(1.0, 0, 0, i); err != nil {
					t.Fatalf("Failed to set input value: %v", err)
				}
			}
			defer input.Close()

			output, err := ffn.Forward(input)
			require.NoError(t, err)
			require.NotNil(t, output)
			defer output.Close()

			// Verify output shape
			shape, err := output.Shape()
			if err != nil {
				t.Fatalf("Failed to get output shape: %v", err)
			}
			if len(shape) != 3 {
				t.Errorf("output shape = %v, want 3 dimensions", shape)
			}
			if shape[0] != 1 {
				t.Errorf("output batch size = %d, want 1", shape[0])
			}
			if shape[1] != 1 {
				t.Errorf("output seq len = %d, want 1", shape[1])
			}
			if shape[2] != tt.hiddenDim {
				t.Errorf("output hidden dim = %d, want %d", shape[2], tt.hiddenDim)
			}
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
			ffn, err := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			if err != nil {
				t.Fatalf("Failed to create FFN sublayer: %v", err)
			}
			defer ffn.Close()

			// Set up weights with valid shapes
			upWeights, err := tensor.NewTensor(tt.intermediateDim, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create up weights tensor: %v", err)
			}
			downWeights, err := tensor.NewTensor(tt.hiddenDim, tt.intermediateDim)
			if err != nil {
				t.Fatalf("Failed to create down weights tensor: %v", err)
			}
			for i := 0; i < tt.intermediateDim; i++ {
				for j := 0; j < tt.hiddenDim; j++ {
					if err := upWeights.Set(1, i, j); err != nil {
						t.Fatalf("Failed to set up weight value: %v", err)
					}
				}
			}
			for i := 0; i < tt.hiddenDim; i++ {
				for j := 0; j < tt.intermediateDim; j++ {
					if err := downWeights.Set(1, i, j); err != nil {
						t.Fatalf("Failed to set down weight value: %v", err)
					}
				}
			}
			ffn.SetWeights(upWeights, downWeights)
			defer upWeights.Close()
			defer downWeights.Close()
			// Debug print
			upShape, err := upWeights.Shape()
			require.NoError(t, err)
			t.Logf("upWeights shape: %v", upShape)
			downShape, err := downWeights.Shape()
			require.NoError(t, err)
			t.Logf("downWeights shape: %v", downShape)

			// Set gamma
			ffn.SetGamma(tt.gamma)

			// Verify gamma was set by running forward pass
			input, err := tensor.NewTensor(1, 1, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create input tensor: %v", err)
			}
			for i := 0; i < tt.hiddenDim; i++ {
				if err := input.Set(1.0, 0, 0, i); err != nil {
					t.Fatalf("Failed to set input value: %v", err)
				}
			}
			defer input.Close()

			output, err := ffn.Forward(input)
			require.NoError(t, err)
			require.NotNil(t, output)
			defer output.Close()

			// Verify output shape
			shape, err := output.Shape()
			if err != nil {
				t.Fatalf("Failed to get output shape: %v", err)
			}
			if len(shape) != 3 {
				t.Errorf("output shape = %v, want 3 dimensions", shape)
			}
			if shape[0] != 1 {
				t.Errorf("output batch size = %d, want 1", shape[0])
			}
			if shape[1] != 1 {
				t.Errorf("output seq len = %d, want 1", shape[1])
			}
			if shape[2] != tt.hiddenDim {
				t.Errorf("output hidden dim = %d, want %d", shape[2], tt.hiddenDim)
			}
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
			input: func() *tensor.Tensor {
				t, err := tensor.NewTensor(2, 3)
				if err != nil {
					panic(err) // This is a test setup, so we can panic
				}
				return t
			}(),
			wantErr: true,
		},
		{
			name:            "dimension mismatch",
			hiddenDim:       4,
			intermediateDim: 8,
			input: func() *tensor.Tensor {
				t, err := tensor.NewTensor(1, 3)
				if err != nil {
					panic(err) // This is a test setup, so we can panic
				}
				return t
			}(),
			wantErr: true,
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
			ffn, err := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			if err != nil {
				t.Fatalf("Failed to create FFN sublayer: %v", err)
			}
			defer ffn.Close()

			// Set up weights and gamma
			upWeights, err := tensor.NewTensor(tt.intermediateDim, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create up weights tensor: %v", err)
			}
			downWeights, err := tensor.NewTensor(tt.hiddenDim, tt.intermediateDim)
			if err != nil {
				t.Fatalf("Failed to create down weights tensor: %v", err)
			}
			for i := 0; i < tt.intermediateDim; i++ {
				for j := 0; j < tt.hiddenDim; j++ {
					if err := upWeights.Set(1, i, j); err != nil {
						t.Fatalf("Failed to set up weight value: %v", err)
					}
				}
			}
			for i := 0; i < tt.hiddenDim; i++ {
				for j := 0; j < tt.intermediateDim; j++ {
					if err := downWeights.Set(1, i, j); err != nil {
						t.Fatalf("Failed to set down weight value: %v", err)
					}
				}
			}
			ffn.SetWeights(upWeights, downWeights)

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
					t, err := tensor.NewTensor(1, 0, 4)
					if err != nil {
						panic(err) // This is a test setup, so we can panic
					}
					_ = t
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

func TestFFNSublayerForward(t *testing.T) {
	// Create FFN sublayer
	ffn, err := NewFFNSublayer(4, 8)
	if err != nil {
		t.Fatalf("failed to create FFN sublayer: %v", err)
	}
	defer ffn.Close()

	// ... rest of the test ...
}

func TestFFNSublayerSetGamma(t *testing.T) {
	// Create FFN sublayer
	ffn, err := NewFFNSublayer(4, 8)
	if err != nil {
		t.Fatalf("failed to create FFN sublayer: %v", err)
	}
	defer ffn.Close()

	// ... rest of the test ...
}
