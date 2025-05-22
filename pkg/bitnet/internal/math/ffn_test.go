package math

import (
	"fmt"
	"strings"
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

func TestFFN(t *testing.T) {
	tests := []struct {
		name            string
		hiddenDim       int
		intermediateDim int
		input           [][][]int8
		upWeights       [][]int8
		downWeights     [][]int8
		expected        [][][]int8
	}{
		{
			name:            "simple FFN with all zeros",
			hiddenDim:       4,
			intermediateDim: 8,
			input: [][][]int8{
				{
					{0, 0, 0, 0},
					{0, 0, 0, 0},
				},
			},
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
			expected: [][][]int8{
				{
					{0, 0, 0, 0},
					{0, 0, 0, 0},
				},
			},
		},
		{
			name:            "FFN with positive values",
			hiddenDim:       4,
			intermediateDim: 8,
			input: [][][]int8{
				{
					{1, 1, 1, 1},
					{1, 1, 1, 1},
				},
			},
			upWeights: [][]int8{
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
			},
			downWeights: [][]int8{
				{1, 1, 1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1, 1, 1},
			},
			expected: [][][]int8{
				{
					{8, 8, 8, 8}, // 8 = 4 (input) * 1 (up weight) * 2 (down weight)
					{8, 8, 8, 8}, // 8 = 4 (input) * 1 (up weight) * 2 (down weight)
				},
			},
		},
		{
			name:            "FFN with negative values",
			hiddenDim:       4,
			intermediateDim: 8,
			input: [][][]int8{
				{
					{-1, -1, -1, -1},
					{-1, -1, -1, -1},
				},
			},
			upWeights: [][]int8{
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
			},
			downWeights: [][]int8{
				{1, 1, 1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1, 1, 1},
			},
			expected: [][][]int8{
				{
					{0, 0, 0, 0}, // ReLU² of negative values is 0
					{0, 0, 0, 0}, // ReLU² of negative values is 0
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create FFN
			ffn := NewFFN(tt.hiddenDim, tt.intermediateDim)

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

			// Set weights
			ffn.SetWeights(upWeights, downWeights)

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
			if output.Shape()[2] != tt.hiddenDim {
				t.Errorf("output hidden dim = %d, want %d", output.Shape()[2], tt.hiddenDim)
			}

			// Verify output values
			for i := range tt.expected {
				for j := range tt.expected[i] {
					for k := range tt.expected[i][j] {
						got := output.Get(i, j, k)
						want := tt.expected[i][j][k]
						if got != want {
							t.Errorf("output[%d][%d][%d] = %d, want %d", i, j, k, got, want)
						}
					}
				}
			}
		})
	}
}

func TestFFNPanics(t *testing.T) {
	tests := []struct {
		name            string
		hiddenDim       int
		intermediateDim int
		input           [][][]int8
		upWeights       [][]int8
		downWeights     [][]int8
		expectedPanic   string
		panicIn         string // "forward" or "setweights"
	}{
		{
			name:            "invalid input shape",
			hiddenDim:       4,
			intermediateDim: 8,
			input: [][][]int8{
				{
					{1, 2}, // Wrong dimension
				},
			},
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
			expectedPanic: "tensor: total size must match",
			panicIn:       "forward",
		},
		{
			name:            "invalid up weights shape",
			hiddenDim:       4,
			intermediateDim: 8,
			input: [][][]int8{
				{
					{1, 0, -1, 1},
				},
			},
			upWeights: [][]int8{
				{1, 0, -1}, // Wrong dimension
				{-1, 1, 0},
			},
			downWeights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
			},
			expectedPanic: "invalid up-projection weights shape",
			panicIn:       "setweights",
		},
		{
			name:            "invalid down weights shape",
			hiddenDim:       4,
			intermediateDim: 8,
			input: [][][]int8{
				{
					{1, 0, -1, 1},
				},
			},
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
				{1, 0, -1}, // Wrong dimension
				{-1, 1, 0},
			},
			expectedPanic: "invalid down-projection weights shape",
			panicIn:       "setweights",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ffn := NewFFN(tt.hiddenDim, tt.intermediateDim)

			if tt.panicIn == "setweights" {
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
				defer func() {
					if r := recover(); r == nil {
						t.Errorf("SetWeights() did not panic")
					} else if r != tt.expectedPanic {
						t.Errorf("SetWeights() panicked with %v, want %v", r, tt.expectedPanic)
					}
				}()
				ffn.SetWeights(upWeights, downWeights)
				return
			}

			// For "forward" panic
			input := tensor.NewTensor(len(tt.input), len(tt.input[0]), len(tt.input[0][0]))
			for i := range tt.input {
				for j := range tt.input[i] {
					for k := range tt.input[i][j] {
						input.Set(tt.input[i][j][k], i, j, k)
					}
				}
			}
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
			ffn.SetWeights(upWeights, downWeights)
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("Forward() did not panic")
				} else if tt.panicIn == "forward" && tt.name == "invalid input shape" {
					var msg string
					switch v := r.(type) {
					case string:
						msg = v
					case error:
						msg = v.Error()
					default:
						msg = fmt.Sprintf("%v", v)
					}
					if !strings.Contains(msg, tt.expectedPanic) {
						t.Errorf("Forward() panicked with %T: %q, want substring %q", r, msg, tt.expectedPanic)
					}
				} else if r != tt.expectedPanic {
					t.Errorf("Forward() panicked with %v, want %v", r, tt.expectedPanic)
				}
			}()
			ffn.Forward(input)
		})
	}
}
