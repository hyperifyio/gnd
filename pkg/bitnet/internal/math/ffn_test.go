package math

import (
	"fmt"
	"strings"
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/stretchr/testify/require"
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
			input, err := tensor.NewTensor(len(tt.input), len(tt.input[0]), len(tt.input[0][0]))
			if err != nil {
				t.Fatalf("failed to create input tensor: %v", err)
			}
			for i := range tt.input {
				for j := range tt.input[i] {
					for k := range tt.input[i][j] {
						if err := input.Set(tt.input[i][j][k], i, j, k); err != nil {
							t.Fatalf("failed to set input tensor value: %v", err)
						}
					}
				}
			}

			// Create weight tensors
			upWeights, err := tensor.NewTensor(len(tt.upWeights), len(tt.upWeights[0]))
			if err != nil {
				t.Fatalf("failed to create up weights tensor: %v", err)
			}
			for i := range tt.upWeights {
				for j := range tt.upWeights[i] {
					if err := upWeights.Set(tt.upWeights[i][j], i, j); err != nil {
						t.Fatalf("failed to set up weights tensor value: %v", err)
					}
				}
			}

			downWeights, err := tensor.NewTensor(len(tt.downWeights), len(tt.downWeights[0]))
			if err != nil {
				t.Fatalf("failed to create down weights tensor: %v", err)
			}
			for i := range tt.downWeights {
				for j := range tt.downWeights[i] {
					if err := downWeights.Set(tt.downWeights[i][j], i, j); err != nil {
						t.Fatalf("failed to set down weights tensor value: %v", err)
					}
				}
			}

			// Set weights
			if err := ffn.SetWeights(upWeights, downWeights); err != nil {
				t.Fatalf("failed to set weights: %v", err)
			}

			// Forward pass
			output, err := ffn.Forward(input)
			if err != nil {
				t.Errorf("FFN Forward failed: %v", err)
				return
			}

			// Verify output shape
			shape, err := output.Shape()
			if err != nil {
				t.Fatalf("failed to get output shape: %v", err)
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
							t.Fatalf("failed to get output value: %v", err)
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
				upWeights, err := tensor.NewTensor(len(tt.upWeights), len(tt.upWeights[0]))
				if err != nil {
					t.Fatalf("failed to create up weights tensor: %v", err)
				}
				for i := range tt.upWeights {
					for j := range tt.upWeights[i] {
						if err := upWeights.Set(tt.upWeights[i][j], i, j); err != nil {
							t.Fatalf("failed to set up weights tensor value: %v", err)
						}
					}
				}
				downWeights, err := tensor.NewTensor(len(tt.downWeights), len(tt.downWeights[0]))
				if err != nil {
					t.Fatalf("failed to create down weights tensor: %v", err)
				}
				for i := range tt.downWeights {
					for j := range tt.downWeights[i] {
						if err := downWeights.Set(tt.downWeights[i][j], i, j); err != nil {
							t.Fatalf("failed to set down weights tensor value: %v", err)
						}
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
			input, err := tensor.NewTensor(len(tt.input), len(tt.input[0]), len(tt.input[0][0]))
			if err != nil {
				t.Fatalf("failed to create input tensor: %v", err)
			}
			for i := range tt.input {
				for j := range tt.input[i] {
					for k := range tt.input[i][j] {
						if err := input.Set(tt.input[i][j][k], i, j, k); err != nil {
							t.Fatalf("failed to set input tensor value: %v", err)
						}
					}
				}
			}
			upWeights, err := tensor.NewTensor(len(tt.upWeights), len(tt.upWeights[0]))
			if err != nil {
				t.Fatalf("failed to create up weights tensor: %v", err)
			}
			for i := range tt.upWeights {
				for j := range tt.upWeights[i] {
					if err := upWeights.Set(tt.upWeights[i][j], i, j); err != nil {
						t.Fatalf("failed to set up weights tensor value: %v", err)
					}
				}
			}
			downWeights, err := tensor.NewTensor(len(tt.downWeights), len(tt.downWeights[0]))
			if err != nil {
				t.Fatalf("failed to create down weights tensor: %v", err)
			}
			for i := range tt.downWeights {
				for j := range tt.downWeights[i] {
					if err := downWeights.Set(tt.downWeights[i][j], i, j); err != nil {
						t.Fatalf("failed to set down weights tensor value: %v", err)
					}
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
						msg = strings.TrimSpace(fmt.Sprintf("%v", v))
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

func TestFFN_Close(t *testing.T) {
	// Create a new FFN
	ffn := NewFFN(512, 2048) // 512 hidden dim, 2048 intermediate dim
	require.NotNil(t, ffn)

	// Set some weights
	upWeights, err := tensor.NewTensor(2048, 512)
	if err != nil {
		t.Fatalf("failed to create up weights tensor: %v", err)
	}
	downWeights, err := tensor.NewTensor(512, 2048)
	if err != nil {
		t.Fatalf("failed to create down weights tensor: %v", err)
	}
	if err := ffn.SetWeights(upWeights, downWeights); err != nil {
		t.Fatalf("failed to set weights: %v", err)
	}

	// Close the FFN
	ffn.Close()

	// Verify that operations panic after close
	operations := []struct {
		name string
		fn   func()
	}{
		{
			name: "Forward",
			fn: func() {
				input, err := tensor.NewTensor(32, 16, 512)
				if err != nil {
					t.Fatalf("failed to create input tensor: %v", err)
				}
				ffn.Forward(input)
			},
		},
		{
			name: "SetWeights",
			fn: func() {
				upWeights, err := tensor.NewTensor(2048, 512)
				if err != nil {
					t.Fatalf("failed to create up weights tensor: %v", err)
				}
				downWeights, err := tensor.NewTensor(512, 2048)
				if err != nil {
					t.Fatalf("failed to create down weights tensor: %v", err)
				}
				ffn.SetWeights(upWeights, downWeights)
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

func TestFFN_applyReLU2(t *testing.T) {
	tests := []struct {
		name        string
		inputShape  []int
		inputValues [][]int8
		wantErr     bool
		wantValues  [][]int8
	}{
		{
			name:       "valid 2D input with positive values",
			inputShape: []int{2, 3},
			inputValues: [][]int8{
				{1, 2, 3},
				{4, 5, 6},
			},
			wantErr: false,
			wantValues: [][]int8{
				{0, 0, 0}, // Values divided by 16 and clamped
				{1, 1, 2},
			},
		},
		{
			name:       "valid 2D input with negative values",
			inputShape: []int{2, 3},
			inputValues: [][]int8{
				{-1, -2, -3},
				{-4, -5, -6},
			},
			wantErr: false,
			wantValues: [][]int8{
				{0, 0, 0}, // ReLU² of negative values is 0
				{0, 0, 0},
			},
		},
		{
			name:       "valid 2D input with mixed values",
			inputShape: []int{2, 3},
			inputValues: [][]int8{
				{-1, 0, 1},
				{-2, 2, -3},
			},
			wantErr: false,
			wantValues: [][]int8{
				{0, 0, 0},
				{0, 0, 0},
			},
		},
		{
			name:       "invalid 1D input",
			inputShape: []int{3},
			inputValues: [][]int8{
				{1, 2, 3},
			},
			wantErr: true,
		},
		{
			name:       "invalid 3D input",
			inputShape: []int{2, 2, 2},
			inputValues: [][]int8{
				{1, 2, 3, 4}, // Flattened 2x2 matrix
				{5, 6, 7, 8}, // Flattened 2x2 matrix
			},
			wantErr: true,
		},
		{
			name:        "empty input",
			inputShape:  []int{0, 0},
			inputValues: [][]int8{},
			wantErr:     false,
			wantValues:  [][]int8{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.name == "empty input" {
				defer func() {
					if r := recover(); r == nil {
						t.Error("expected panic for empty input shape, but did not panic")
					}
				}()
			}
			input, err := tensor.NewTensor(tt.inputShape...)
			if err != nil {
				t.Fatalf("failed to create input tensor: %v", err)
			}
			if input != nil {
				for i := range tt.inputValues {
					for j := range tt.inputValues[i] {
						if len(tt.inputShape) == 1 {
							if err := input.Set(tt.inputValues[i][j], j); err != nil {
								t.Fatalf("failed to set input tensor value: %v", err)
							}
						} else if len(tt.inputShape) == 2 {
							if err := input.Set(tt.inputValues[i][j], i, j); err != nil {
								t.Fatalf("failed to set input tensor value: %v", err)
							}
						}
					}
				}
			}

			// Create FFN with arbitrary dimensions
			ffn := NewFFN(4, 8)
			defer ffn.Close()

			// Call applyReLU2
			output, err := ffn.applyReLU2(input)

			// Check error
			if tt.wantErr {
				if err == nil {
					t.Error("applyReLU2() error = nil, want error")
				}
				if output != nil {
					t.Error("applyReLU2() output = non-nil, want nil")
				}
				return
			}

			if err != nil {
				t.Errorf("applyReLU2() error = %v, want nil", err)
				return
			}

			if output == nil {
				t.Error("applyReLU2() output = nil, want non-nil")
				return
			}

			// Verify output shape
			shape, err := output.Shape()
			if err != nil {
				t.Fatalf("failed to get output shape: %v", err)
			}
			if len(shape) != 2 {
				t.Errorf("output shape = %v, want 2 dimensions", shape)
				return
			}

			// Verify output values
			for i := range tt.wantValues {
				for j := range tt.wantValues[i] {
					got, err := output.Get(i, j)
					if err != nil {
						t.Fatalf("failed to get output value: %v", err)
					}
					want := tt.wantValues[i][j]
					if got != want {
						t.Errorf("output[%d][%d] = %d, want %d", i, j, got, want)
					}
				}
			}

			// Clean up
			output.Close()
		})
	}
}
