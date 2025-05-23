package math

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/stretchr/testify/require"
)

func TestAttentionOutputProjection(t *testing.T) {
	tests := []struct {
		name      string
		hiddenDim int
		numHeads  int
		input     [][][]int8
		weights   [][]int8
		expected  [][][]int8
	}{
		{
			name:      "simple projection",
			hiddenDim: 8,
			numHeads:  2,
			input: [][][]int8{
				{
					{1, 0, -1, 1, 0, -1, 1, 0},
					{-1, 1, 0, -1, 1, 0, -1, 1},
				},
			},
			weights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1},
			},
			expected: [][][]int8{
				{
					{5, -3, 5, -3, 5, -3, 5, -3},
					{-3, 6, -3, 6, -3, 6, -3, 6},
				},
			},
		},
		{
			name:      "larger projection",
			hiddenDim: 16,
			numHeads:  4,
			input: [][][]int8{
				{
					{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
					{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				},
			},
			weights: [][]int8{
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
				{1, 0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, -1, 1, 0},
				{-1, 1, 0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 1},
			},
			expected: [][][]int8{
				{
					{10, -6, 10, -6, 10, -6, 10, -6, 10, -6, 10, -6, 10, -6, 10, -6},
					{-6, 12, -6, 12, -6, 12, -6, 12, -6, 12, -6, 12, -6, 12, -6, 12},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create attention output projection
			out := NewAttentionOutputProjection(tt.hiddenDim, tt.numHeads)

			// Create input tensor
			input := tensor.NewTensor(len(tt.input), len(tt.input[0]), len(tt.input[0][0]))
			for i := range tt.input {
				for j := range tt.input[i] {
					for k := range tt.input[i][j] {
						input.Set(tt.input[i][j][k], i, j, k)
					}
				}
			}

			// Create weight tensor
			weights := tensor.NewTensor(len(tt.weights), len(tt.weights[0]))
			for i := range tt.weights {
				for j := range tt.weights[i] {
					weights.Set(tt.weights[i][j], i, j)
				}
			}

			// Set weights
			out.SetWeights(weights)

			// Project input
			output, err := out.Project(input)
			if err != nil {
				t.Errorf("Project failed: %v", err)
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

func TestAttentionOutputProjectionPanics(t *testing.T) {
	tests := []struct {
		name        string
		hiddenDim   int
		numHeads    int
		input       *tensor.Tensor
		weights     *tensor.Tensor
		shouldPanic bool
	}{
		{
			name:        "invalid input shape",
			hiddenDim:   8,
			numHeads:    2,
			input:       tensor.NewTensor(2, 2),
			weights:     tensor.NewTensor(8, 8),
			shouldPanic: false,
		},
		{
			name:        "invalid weights shape",
			hiddenDim:   8,
			numHeads:    2,
			input:       tensor.NewTensor(1, 2, 8),
			weights:     tensor.NewTensor(8, 4),
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := NewAttentionOutputProjection(tt.hiddenDim, tt.numHeads)
			if tt.weights != nil {
				if tt.shouldPanic {
					defer func() {
						if r := recover(); r == nil {
							t.Error("expected panic for invalid weights shape")
						}
					}()
				}
				out.SetWeights(tt.weights)
			}
			if tt.input != nil {
				_, err := out.Project(tt.input)
				if err == nil && !tt.shouldPanic {
					t.Error("expected error for invalid input shape")
				}
			}
		})
	}
}

func TestAttentionOutputProjection_Close(t *testing.T) {
	// Create a new attention output projection
	proj := NewAttentionOutputProjection(512, 8)
	require.NotNil(t, proj)

	// Set some weights
	weights := tensor.NewTensor(512, 512)
	require.NoError(t, proj.SetWeights(weights))

	// Close the projection
	proj.Close()

	// Verify that operations panic after close
	operations := []struct {
		name string
		fn   func()
	}{
		{
			name: "Project",
			fn: func() {
				input := tensor.NewTensor(32, 16, 512)
				proj.Project(input)
			},
		},
		{
			name: "SetWeights",
			fn: func() {
				weights := tensor.NewTensor(512, 512)
				proj.SetWeights(weights)
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

	// Verify that the weights are closed
	require.Nil(t, proj.outProj, "outProj should be nil after Close")
}
