package math

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/stretchr/testify/require"
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
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
			},
			vWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
			},
			outWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
			},
			gamma: []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
		},
		{
			name:       "grouped-query attention",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 4,
			input: [][][]int8{
				{
					{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
					{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				},
			},
			qWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
			},
			kWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
			},
			vWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
			},
			outWeights: [][]int8{
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

func TestNewAttentionSublayer(t *testing.T) {
	tests := []struct {
		name       string
		hiddenSize int
		numHeads   int
		numKVHeads int
		wantErr    bool
	}{
		{
			name:       "valid dimensions",
			hiddenSize: 64,
			numHeads:   8,
			numKVHeads: 8,
			wantErr:    false,
		},
		{
			name:       "invalid head count",
			hiddenSize: 64,
			numHeads:   33,
			numKVHeads: 8,
			wantErr:    true,
		},
		{
			name:       "invalid KV heads",
			hiddenSize: 64,
			numHeads:   8,
			numKVHeads: 9,
			wantErr:    true,
		},
		{
			name:       "non-divisible heads",
			hiddenSize: 64,
			numHeads:   8,
			numKVHeads: 3,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewAttentionSublayer(tt.hiddenSize, tt.numHeads, tt.numKVHeads)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewAttentionSublayer() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestAttentionSublayer_SetWeights(t *testing.T) {
	hiddenSize := 64
	numHeads := 8
	numKVHeads := 8

	tests := []struct {
		name       string
		qWeights   *tensor.Tensor
		kWeights   *tensor.Tensor
		vWeights   *tensor.Tensor
		outWeights *tensor.Tensor
		wantErr    bool
	}{
		{
			name:       "valid weights",
			qWeights:   tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads),
			kWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			vWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			outWeights: tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize),
			wantErr:    false,
		},
		{
			name:       "invalid query weights shape",
			qWeights:   tensor.NewTensor(hiddenSize-1, numHeads*hiddenSize/numHeads),
			kWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			vWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			outWeights: tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize),
			wantErr:    true,
		},
		{
			name:       "invalid key weights shape",
			qWeights:   tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads),
			kWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads-1),
			vWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			outWeights: tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize),
			wantErr:    true,
		},
		{
			name:       "invalid value weights shape",
			qWeights:   tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads),
			kWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			vWeights:   tensor.NewTensor(hiddenSize-1, numKVHeads*hiddenSize/numKVHeads),
			outWeights: tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize),
			wantErr:    true,
		},
		{
			name:       "invalid output weights shape",
			qWeights:   tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads),
			kWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			vWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			outWeights: tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize+1),
			wantErr:    true,
		},
		{
			name:       "nil query weights",
			qWeights:   nil,
			kWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			vWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			outWeights: tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize),
			wantErr:    true,
		},
		{
			name:       "nil key weights",
			qWeights:   tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads),
			kWeights:   nil,
			vWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			outWeights: tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize),
			wantErr:    true,
		},
		{
			name:       "nil value weights",
			qWeights:   tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads),
			kWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			vWeights:   nil,
			outWeights: tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize),
			wantErr:    true,
		},
		{
			name:       "nil output weights",
			qWeights:   tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads),
			kWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			vWeights:   tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads),
			outWeights: nil,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attn, err := NewAttentionSublayer(hiddenSize, numHeads, numKVHeads)
			if err != nil {
				t.Fatalf("Failed to create attention sublayer: %v", err)
			}
			err = attn.SetWeights(tt.qWeights, tt.kWeights, tt.vWeights, tt.outWeights)
			if (err != nil) != tt.wantErr {
				t.Errorf("SetWeights() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestAttentionSublayer_SetGamma(t *testing.T) {
	// Create a valid attention sublayer
	hiddenSize := 64
	numHeads := 8
	numKVHeads := 8
	attn, err := NewAttentionSublayer(hiddenSize, numHeads, numKVHeads)
	if err != nil {
		t.Fatalf("Failed to create attention sublayer: %v", err)
	}

	tests := []struct {
		name    string
		gamma   *tensor.Tensor
		wantErr bool
	}{
		{
			name:    "valid gamma",
			gamma:   tensor.NewTensor(hiddenSize),
			wantErr: false,
		},
		{
			name:    "invalid gamma shape",
			gamma:   tensor.NewTensor(hiddenSize + 1),
			wantErr: true,
		},
		{
			name:    "nil gamma",
			gamma:   nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := attn.SetGamma(tt.gamma)
			if (err != nil) != tt.wantErr {
				t.Errorf("SetGamma() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestAttentionSublayer_Forward(t *testing.T) {
	// Create a valid attention sublayer
	hiddenSize := 64
	numHeads := 8
	numKVHeads := 8
	attn, err := NewAttentionSublayer(hiddenSize, numHeads, numKVHeads)
	if err != nil {
		t.Fatalf("Failed to create attention sublayer: %v", err)
	}

	// Set up valid weights
	qWeights := tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads)
	kWeights := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
	vWeights := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
	outWeights := tensor.NewTensor(hiddenSize, hiddenSize)
	gamma := tensor.NewTensor(hiddenSize)

	err = attn.SetWeights(qWeights, kWeights, vWeights, outWeights)
	if err != nil {
		t.Fatalf("Failed to set weights: %v", err)
	}
	err = attn.SetGamma(gamma)
	if err != nil {
		t.Fatalf("Failed to set gamma: %v", err)
	}

	tests := []struct {
		name    string
		input   *tensor.Tensor
		wantErr bool
	}{
		{
			name:    "valid 2D input",
			input:   tensor.NewTensor(1, hiddenSize),
			wantErr: false,
		},
		{
			name:    "valid 3D input",
			input:   tensor.NewTensor(1, 1, hiddenSize),
			wantErr: false,
		},
		{
			name:    "invalid input shape",
			input:   tensor.NewTensor(1, hiddenSize+1),
			wantErr: true,
		},
		{
			name:    "nil input",
			input:   nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := attn.Forward(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Forward() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestEqualShape(t *testing.T) {
	tests := []struct {
		name   string
		shape1 []int
		shape2 []int
		want   bool
	}{
		{
			name:   "equal shapes",
			shape1: []int{2, 3, 4},
			shape2: []int{2, 3, 4},
			want:   true,
		},
		{
			name:   "different lengths",
			shape1: []int{2, 3, 4},
			shape2: []int{2, 3},
			want:   false,
		},
		{
			name:   "different values",
			shape1: []int{2, 3, 4},
			shape2: []int{2, 3, 5},
			want:   false,
		},
		{
			name:   "empty shapes",
			shape1: []int{},
			shape2: []int{},
			want:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := equalShape(tt.shape1, tt.shape2)
			if got != tt.want {
				t.Errorf("equalShape() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAttentionSublayer_Close(t *testing.T) {
	// Create a new attention sublayer
	sublayer, err := NewAttentionSublayer(512, 8, 8) // 512 hidden dim, 8 heads, 8 kv heads
	require.NoError(t, err)
	require.NotNil(t, sublayer)

	// Set some weights
	qWeights := tensor.NewTensor(512, 512)
	kWeights := tensor.NewTensor(512, 512)
	vWeights := tensor.NewTensor(512, 512)
	outWeights := tensor.NewTensor(512, 512)
	err = sublayer.SetWeights(qWeights, kWeights, vWeights, outWeights)
	require.NoError(t, err)

	// Set gamma
	gamma := tensor.NewTensor(512)
	err = sublayer.SetGamma(gamma)
	require.NoError(t, err)

	// Close the sublayer
	sublayer.Close()

	// Verify that operations panic after close
	operations := []struct {
		name string
		fn   func()
	}{
		{
			name: "Forward",
			fn: func() {
				input := tensor.NewTensor(32, 16, 512)
				sublayer.Forward(input)
			},
		},
		{
			name: "SetWeights",
			fn: func() {
				qWeights := tensor.NewTensor(512, 512)
				kWeights := tensor.NewTensor(512, 512)
				vWeights := tensor.NewTensor(512, 512)
				outWeights := tensor.NewTensor(512, 512)
				sublayer.SetWeights(qWeights, kWeights, vWeights, outWeights)
			},
		},
		{
			name: "SetGamma",
			fn: func() {
				gamma := tensor.NewTensor(512)
				sublayer.SetGamma(gamma)
			},
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
