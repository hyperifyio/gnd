package attention_output

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
			out, err := NewAttentionOutputProjection(tt.hiddenDim, tt.numHeads)
			if err != nil {
				t.Fatalf("Failed to create attention output projection: %v", err)
			}

			// Create input tensor
			input, err := tensor.NewTensor(len(tt.input), len(tt.input[0]), len(tt.input[0][0]))
			if err != nil {
				t.Fatalf("Failed to create input tensor: %v", err)
			}
			for i := range tt.input {
				for j := range tt.input[i] {
					for k := range tt.input[i][j] {
						if err := input.Set(tt.input[i][j][k], i, j, k); err != nil {
							t.Fatalf("Failed to set input tensor value: %v", err)
						}
					}
				}
			}

			// Create weight tensor
			weights, err := tensor.NewTensor(len(tt.weights), len(tt.weights[0]))
			if err != nil {
				t.Fatalf("Failed to create weight tensor: %v", err)
			}
			for i := range tt.weights {
				for j := range tt.weights[i] {
					if err := weights.Set(tt.weights[i][j], i, j); err != nil {
						t.Fatalf("Failed to set weight tensor value: %v", err)
					}
				}
			}

			// Set weights
			if err := out.SetWeights(weights); err != nil {
				t.Fatalf("Failed to set weights: %v", err)
			}

			// Project input
			output, err := out.Project(input)
			if err != nil {
				t.Fatalf("Project failed: %v", err)
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
			if shape[2] != tt.hiddenDim {
				t.Errorf("output hidden dim = %d, want %d", shape[2], tt.hiddenDim)
			}

			// Verify output values
			for i := range tt.expected {
				for j := range tt.expected[i] {
					for k := range tt.expected[i][j] {
						got, err := output.Get(i, j, k)
						if err != nil {
							t.Fatalf("Failed to get output value: %v", err)
						}
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
		wantErr     bool
	}{
		{
			name:        "invalid input shape",
			hiddenDim:   8,
			numHeads:    2,
			input:       func() *tensor.Tensor { t, _ := tensor.NewTensor(2, 2); return t }(),
			weights:     func() *tensor.Tensor { t, _ := tensor.NewTensor(8, 8); return t }(),
			shouldPanic: false,
			wantErr:     true,
		},
		{
			name:        "invalid weights shape",
			hiddenDim:   8,
			numHeads:    2,
			input:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 2, 8); return t }(),
			weights:     func() *tensor.Tensor { t, _ := tensor.NewTensor(8, 4); return t }(),
			shouldPanic: false,
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := NewAttentionOutputProjection(tt.hiddenDim, tt.numHeads)
			if err != nil {
				t.Fatalf("Failed to create attention output projection: %v", err)
			}
			if tt.weights != nil {
				err := out.SetWeights(tt.weights)
				if (err != nil) != tt.wantErr {
					t.Errorf("SetWeights() error = %v, wantErr %v", err, tt.wantErr)
				}
			}
			if tt.input != nil {
				_, err := out.Project(tt.input)
				if (err != nil) != tt.wantErr {
					t.Errorf("Project() error = %v, wantErr %v", err, tt.wantErr)
				}
			}
		})
	}
}

func TestAttentionOutputProjection_Close(t *testing.T) {
	// Create a new attention output projection
	proj, err := NewAttentionOutputProjection(512, 8)
	if err != nil {
		t.Fatalf("Failed to create attention output projection: %v", err)
	}
	require.NotNil(t, proj)

	// Set some weights
	weights, err := tensor.NewTensor(512, 512)
	if err != nil {
		t.Fatalf("Failed to create weight tensor: %v", err)
	}
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
				input, _ := tensor.NewTensor(32, 16, 512)
				proj.Project(input)
			},
		},
		{
			name: "SetWeights",
			fn: func() {
				weights, _ := tensor.NewTensor(512, 512)
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
