package math

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

func TestQKVProjection(t *testing.T) {
	tests := []struct {
		name       string
		hiddenDim  int
		numHeads   int
		numKVHeads int
		input      [][]int8
		qWeights   [][]int8
		kWeights   [][]int8
		vWeights   [][]int8
	}{
		{
			name:       "standard attention",
			hiddenDim:  32,
			numHeads:   4,
			numKVHeads: 4,
			input: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
			},
			qWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
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
		},
		{
			name:       "grouped-query attention",
			hiddenDim:  32,
			numHeads:   8,
			numKVHeads: 4,
			input: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
			},
			qWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
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
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create QKV projection
			proj := NewQKVProjection(tt.hiddenDim, tt.numHeads, tt.numKVHeads)

			// Create input tensor
			input, err := tensor.NewTensor(len(tt.input), len(tt.input[0]))
			if err != nil {
				t.Fatalf("failed to create input tensor: %v", err)
			}
			for i := range tt.input {
				for j := range tt.input[i] {
					if err := input.Set(tt.input[i][j], i, j); err != nil {
						t.Fatalf("failed to set input tensor value: %v", err)
					}
				}
			}

			// Create weight tensors
			qWeights, err := tensor.NewTensor(tt.hiddenDim, tt.numHeads*(tt.hiddenDim/tt.numHeads))
			if err != nil {
				t.Fatalf("failed to create q weights tensor: %v", err)
			}
			for i := range tt.qWeights {
				for j := range tt.qWeights[i] {
					if i < tt.hiddenDim && j < tt.numHeads*(tt.hiddenDim/tt.numHeads) {
						if err := qWeights.Set(tt.qWeights[i][j], i, j); err != nil {
							t.Fatalf("failed to set q weights tensor value: %v", err)
						}
					}
				}
			}

			kWeights, err := tensor.NewTensor(tt.hiddenDim, tt.numKVHeads*(tt.hiddenDim/tt.numKVHeads))
			if err != nil {
				t.Fatalf("failed to create k weights tensor: %v", err)
			}
			for i := range tt.kWeights {
				for j := range tt.kWeights[i] {
					if i < tt.hiddenDim && j < tt.numKVHeads*(tt.hiddenDim/tt.numKVHeads) {
						if err := kWeights.Set(tt.kWeights[i][j], i, j); err != nil {
							t.Fatalf("failed to set k weights tensor value: %v", err)
						}
					}
				}
			}

			vWeights, err := tensor.NewTensor(tt.hiddenDim, tt.numKVHeads*(tt.hiddenDim/tt.numKVHeads))
			if err != nil {
				t.Fatalf("failed to create v weights tensor: %v", err)
			}
			for i := range tt.vWeights {
				for j := range tt.vWeights[i] {
					if i < tt.hiddenDim && j < tt.numKVHeads*(tt.hiddenDim/tt.numKVHeads) {
						if err := vWeights.Set(tt.vWeights[i][j], i, j); err != nil {
							t.Fatalf("failed to set v weights tensor value: %v", err)
						}
					}
				}
			}

			// Debug output for weight shapes
			loggers.Printf(loggers.Debug, "Test case: %s", tt.name)
			loggers.Printf(loggers.Debug, "Hidden dim: %d", tt.hiddenDim)
			loggers.Printf(loggers.Debug, "Num heads: %d", tt.numHeads)
			loggers.Printf(loggers.Debug, "Num KV heads: %d", tt.numKVHeads)

			qShape, err := qWeights.Shape()
			if err != nil {
				t.Fatalf("failed to get q weights shape: %v", err)
			}
			loggers.Printf(loggers.Debug, "Q weights shape: %v", qShape)

			kShape, err := kWeights.Shape()
			if err != nil {
				t.Fatalf("failed to get k weights shape: %v", err)
			}
			loggers.Printf(loggers.Debug, "K weights shape: %v", kShape)

			vShape, err := vWeights.Shape()
			if err != nil {
				t.Fatalf("failed to get v weights shape: %v", err)
			}
			loggers.Printf(loggers.Debug, "V weights shape: %v", vShape)

			// Set weights
			if err := proj.SetWeights(qWeights, kWeights, vWeights); err != nil {
				t.Fatalf("failed to set weights: %v", err)
			}

			// Project input
			q, k, v, err := proj.Project(input)
			if err != nil {
				t.Fatalf("QKVProjection.Project failed: %v", err)
			}

			// Verify output shapes
			qShape, err = q.Shape()
			if err != nil {
				t.Fatalf("failed to get q shape: %v", err)
			}
			if len(qShape) != 4 {
				t.Errorf("q shape = %v, want 4 dimensions", qShape)
			}

			kShape, err = k.Shape()
			if err != nil {
				t.Fatalf("failed to get k shape: %v", err)
			}
			if len(kShape) != 4 {
				t.Errorf("k shape = %v, want 4 dimensions", kShape)
			}

			vShape, err = v.Shape()
			if err != nil {
				t.Fatalf("failed to get v shape: %v", err)
			}
			if len(vShape) != 4 {
				t.Errorf("v shape = %v, want 4 dimensions", vShape)
			}

			// Verify batch size
			if qShape[0] != len(tt.input) {
				t.Errorf("q batch size = %d, want %d", qShape[0], len(tt.input))
			}
			if kShape[0] != len(tt.input) {
				t.Errorf("k batch size = %d, want %d", kShape[0], len(tt.input))
			}
			if vShape[0] != len(tt.input) {
				t.Errorf("v batch size = %d, want %d", vShape[0], len(tt.input))
			}

			// Verify number of heads
			if qShape[1] != tt.numHeads {
				t.Errorf("q num heads = %d, want %d", qShape[1], tt.numHeads)
			}
			if kShape[1] != tt.numHeads {
				t.Errorf("k num heads = %d, want %d", kShape[1], tt.numHeads)
			}
			if vShape[1] != tt.numHeads {
				t.Errorf("v num heads = %d, want %d", vShape[1], tt.numHeads)
			}

			// Verify sequence length
			if qShape[2] != 1 {
				t.Errorf("q seq len = %d, want 1", qShape[2])
			}
			if kShape[2] != 1 {
				t.Errorf("k seq len = %d, want 1", kShape[2])
			}
			if vShape[2] != 1 {
				t.Errorf("v seq len = %d, want 1", vShape[2])
			}

			// Verify head dimension
			if qShape[3] != tt.hiddenDim/tt.numHeads {
				t.Errorf("q head dim = %d, want %d", qShape[3], tt.hiddenDim/tt.numHeads)
			}
			if kShape[3] != tt.hiddenDim/tt.numHeads {
				t.Errorf("k head dim = %d, want %d", kShape[3], tt.hiddenDim/tt.numHeads)
			}
			if vShape[3] != tt.hiddenDim/tt.numHeads {
				t.Errorf("v head dim = %d, want %d", vShape[3], tt.hiddenDim/tt.numHeads)
			}
		})
	}
}

func equalShapes(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
