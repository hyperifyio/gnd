package math

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
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
			output := ffn.Forward(input)

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
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic")
				}
			}()

			ffn := NewFFNSublayer(tt.hiddenDim, tt.intermediateDim)
			ffn.Forward(tt.input)
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
				_ = ffn.Forward(input)
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
	output := ffnSublayer.Forward(input)

	// Print output shape and data
	t.Logf("Output shape: %v", output.Shape())
	t.Logf("Output data: %v", output.Data())

	// Check output shape
	if len(output.Shape()) != 3 || output.Shape()[0] != batchSize || output.Shape()[1] != seqLen || output.Shape()[2] != hiddenDim {
		t.Errorf("Output shape = %v, want [%d %d %d]", output.Shape(), batchSize, seqLen, hiddenDim)
	}
}
