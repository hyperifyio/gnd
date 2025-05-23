package math

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

func TestAttentionSublayer(t *testing.T) {
	tests := []struct {
		name       string
		hiddenDim  int
		numHeads   int
		numKVHeads int
		input      [][][]int8
		qWeights   [][]int8
		kWeights   [][]int8
		vWeights   [][]int8
		outWeights [][]int8
		gamma      []float32
	}{
		{
			name:       "standard attention",
			hiddenDim:  32,
			numHeads:   4,
			numKVHeads: 4,
			input: [][][]int8{
				{
					{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
					{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				},
			},
			qWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
			},
			kWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
			},
			vWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
			},
			outWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
			},
			gamma: []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
		},
		{
			name:       "grouped-query attention",
			hiddenDim:  32,
			numHeads:   8,
			numKVHeads: 4,
			input: [][][]int8{
				{
					{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
					{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				},
			},
			qWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
			},
			kWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
			},
			vWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
			},
			outWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
			},
			gamma: []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create attention sublayer
			attn, err := NewAttentionSublayer(tt.hiddenDim, tt.numHeads, tt.numKVHeads)
			if err != nil {
				t.Fatalf("Failed to create attention sublayer: %v", err)
			}

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
			qWeights := tensor.NewTensor(len(tt.qWeights), len(tt.qWeights[0]))
			for i := range tt.qWeights {
				for j := range tt.qWeights[i] {
					qWeights.Set(tt.qWeights[i][j], i, j)
				}
			}

			kWeights := tensor.NewTensor(len(tt.kWeights), len(tt.kWeights[0]))
			for i := range tt.kWeights {
				for j := range tt.kWeights[i] {
					kWeights.Set(tt.kWeights[i][j], i, j)
				}
			}

			vWeights := tensor.NewTensor(len(tt.vWeights), len(tt.vWeights[0]))
			for i := range tt.vWeights {
				for j := range tt.vWeights[i] {
					vWeights.Set(tt.vWeights[i][j], i, j)
				}
			}

			outWeights := tensor.NewTensor(len(tt.outWeights), len(tt.outWeights[0]))
			for i := range tt.outWeights {
				for j := range tt.outWeights[i] {
					outWeights.Set(tt.outWeights[i][j], i, j)
				}
			}

			// Set weights
			attn.SetWeights(qWeights, kWeights, vWeights, outWeights)

			// Convert gamma to tensor
			gammaTensor := tensor.NewTensor(tt.hiddenDim)
			for i, v := range tt.gamma {
				gammaTensor.Set(int8(v), i)
			}

			// Set gamma
			if err := attn.SetGamma(gammaTensor); err != nil {
				t.Fatalf("Failed to set gamma: %v", err)
			}

			// Forward pass
			output, err := attn.Forward(input)
			if err != nil {
				t.Fatalf("Forward pass failed: %v", err)
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

func TestAttentionSublayerPanics(t *testing.T) {
	tests := []struct {
		name       string
		hiddenDim  int
		numHeads   int
		numKVHeads int
		input      *tensor.Tensor
	}{
		{
			name:       "invalid input shape",
			hiddenDim:  8,
			numHeads:   2,
			numKVHeads: 2,
			input:      tensor.NewTensor(2, 2),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic")
				}
			}()

			attn, _ := NewAttentionSublayer(tt.hiddenDim, tt.numHeads, tt.numKVHeads)
			attn.Forward(tt.input)
		})
	}
}

func BenchmarkAttentionSublayer(b *testing.B) {
	benchmarks := []struct {
		name       string
		hiddenDim  int
		numHeads   int
		numKVHeads int
		seqLen     int
	}{
		{
			name:       "small",
			hiddenDim:  64,
			numHeads:   4,
			numKVHeads: 4,
			seqLen:     32,
		},
		{
			name:       "medium",
			hiddenDim:  256,
			numHeads:   8,
			numKVHeads: 8,
			seqLen:     128,
		},
		{
			name:       "large",
			hiddenDim:  512,
			numHeads:   16,
			numKVHeads: 16,
			seqLen:     512,
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			// Create attention sublayer
			attn, err := NewAttentionSublayer(bm.hiddenDim, bm.numHeads, bm.numKVHeads)
			if err != nil {
				b.Fatalf("Failed to create attention sublayer: %v", err)
			}

			// Create input tensor
			input := tensor.NewTensor(1, bm.seqLen, bm.hiddenDim)
			for i := 0; i < bm.seqLen; i++ {
				for j := 0; j < bm.hiddenDim; j++ {
					input.Set(int8((i+j)%8-4), 0, i, j)
				}
			}

			// Create weight tensors
			qWeights := tensor.NewTensor(bm.hiddenDim, bm.hiddenDim)
			kWeights := tensor.NewTensor(bm.hiddenDim, bm.hiddenDim)
			vWeights := tensor.NewTensor(bm.hiddenDim, bm.hiddenDim)
			outWeights := tensor.NewTensor(bm.hiddenDim, bm.hiddenDim)

			// Fill weights with pseudo-random but deterministic data
			for i := 0; i < bm.hiddenDim; i++ {
				for j := 0; j < bm.hiddenDim; j++ {
					qWeights.Set(int8((i+j)%8-4), i, j)
					kWeights.Set(int8((i-j)%8-4), i, j)
					vWeights.Set(int8((i*j)%8-4), i, j)
					outWeights.Set(int8((i+j)%8-4), i, j)
				}
			}

			// Set weights and gamma
			attn.SetWeights(qWeights, kWeights, vWeights, outWeights)
			gamma := make([]float32, bm.hiddenDim)
			for i := range gamma {
				gamma[i] = 1.0
			}

			// Convert gamma to tensor
			gammaTensor := tensor.NewTensor(bm.hiddenDim)
			for i, v := range gamma {
				gammaTensor.Set(int8(v), i)
			}

			// Set gamma
			if err := attn.SetGamma(gammaTensor); err != nil {
				b.Fatalf("Failed to set gamma: %v", err)
			}

			// Forward pass
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := attn.Forward(input)
				if err != nil {
					b.Fatalf("Forward pass failed: %v", err)
				}
			}
		})
	}
}
