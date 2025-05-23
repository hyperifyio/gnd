package math

import (
	"fmt"
	"os"
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
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
			input := tensor.NewTensor(len(tt.input), len(tt.input[0]))
			for i := range tt.input {
				for j := range tt.input[i] {
					input.Set(tt.input[i][j], i, j)
				}
			}

			// Create weight tensors
			qWeights := tensor.NewTensor(tt.hiddenDim, tt.numHeads*(tt.hiddenDim/tt.numHeads))
			for i := range tt.qWeights {
				for j := range tt.qWeights[i] {
					if i < tt.hiddenDim && j < tt.numHeads*(tt.hiddenDim/tt.numHeads) {
						qWeights.Set(tt.qWeights[i][j], i, j)
					}
				}
			}

			kWeights := tensor.NewTensor(tt.hiddenDim, tt.numKVHeads*(tt.hiddenDim/tt.numKVHeads))
			for i := range tt.kWeights {
				for j := range tt.kWeights[i] {
					if i < tt.hiddenDim && j < tt.numKVHeads*(tt.hiddenDim/tt.numKVHeads) {
						kWeights.Set(tt.kWeights[i][j], i, j)
					}
				}
			}

			vWeights := tensor.NewTensor(tt.hiddenDim, tt.numKVHeads*(tt.hiddenDim/tt.numKVHeads))
			for i := range tt.vWeights {
				for j := range tt.vWeights[i] {
					if i < tt.hiddenDim && j < tt.numKVHeads*(tt.hiddenDim/tt.numKVHeads) {
						vWeights.Set(tt.vWeights[i][j], i, j)
					}
				}
			}

			// Debug output for weight shapes
			fmt.Fprintf(os.Stderr, "[DEBUG] Test case: %s\n", tt.name)
			fmt.Fprintf(os.Stderr, "[DEBUG] Hidden dim: %d\n", tt.hiddenDim)
			fmt.Fprintf(os.Stderr, "[DEBUG] Num heads: %d\n", tt.numHeads)
			fmt.Fprintf(os.Stderr, "[DEBUG] Num KV heads: %d\n", tt.numKVHeads)
			fmt.Fprintf(os.Stderr, "[DEBUG] Q weights shape: %v\n", qWeights.Shape())
			fmt.Fprintf(os.Stderr, "[DEBUG] K weights shape: %v\n", kWeights.Shape())
			fmt.Fprintf(os.Stderr, "[DEBUG] V weights shape: %v\n", vWeights.Shape())

			// Set weights
			proj.SetWeights(qWeights, kWeights, vWeights)

			// Project input
			q, k, v, err := proj.Project(input)
			if err != nil {
				t.Fatalf("QKVProjection.Project failed: %v", err)
			}

			// Verify output shapes
			if len(q.Shape()) != 4 {
				t.Errorf("q shape = %v, want 4 dimensions", q.Shape())
			}
			if len(k.Shape()) != 4 {
				t.Errorf("k shape = %v, want 4 dimensions", k.Shape())
			}
			if len(v.Shape()) != 4 {
				t.Errorf("v shape = %v, want 4 dimensions", v.Shape())
			}

			// Verify batch size
			if q.Shape()[0] != len(tt.input) {
				t.Errorf("q batch size = %d, want %d", q.Shape()[0], len(tt.input))
			}
			if k.Shape()[0] != len(tt.input) {
				t.Errorf("k batch size = %d, want %d", k.Shape()[0], len(tt.input))
			}
			if v.Shape()[0] != len(tt.input) {
				t.Errorf("v batch size = %d, want %d", v.Shape()[0], len(tt.input))
			}

			// Verify number of heads
			if q.Shape()[1] != tt.numHeads {
				t.Errorf("q num heads = %d, want %d", q.Shape()[1], tt.numHeads)
			}
			if k.Shape()[1] != tt.numHeads {
				t.Errorf("k num heads = %d, want %d", k.Shape()[1], tt.numHeads)
			}
			if v.Shape()[1] != tt.numHeads {
				t.Errorf("v num heads = %d, want %d", v.Shape()[1], tt.numHeads)
			}

			// Verify sequence length
			if q.Shape()[2] != 1 {
				t.Errorf("q seq len = %d, want 1", q.Shape()[2])
			}
			if k.Shape()[2] != 1 {
				t.Errorf("k seq len = %d, want 1", k.Shape()[2])
			}
			if v.Shape()[2] != 1 {
				t.Errorf("v seq len = %d, want 1", v.Shape()[2])
			}

			// Verify head dimension
			if q.Shape()[3] != tt.hiddenDim/tt.numHeads {
				t.Errorf("q head dim = %d, want %d", q.Shape()[3], tt.hiddenDim/tt.numHeads)
			}
			if k.Shape()[3] != tt.hiddenDim/tt.numHeads {
				t.Errorf("k head dim = %d, want %d", k.Shape()[3], tt.hiddenDim/tt.numHeads)
			}
			if v.Shape()[3] != tt.hiddenDim/tt.numHeads {
				t.Errorf("v head dim = %d, want %d", v.Shape()[3], tt.hiddenDim/tt.numHeads)
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
