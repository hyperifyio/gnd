package ffn

import (
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
			ffn, err := NewFFN(tt.hiddenDim, tt.intermediateDim)
			require.NoError(t, err)
			defer ffn.Close()

			// Create input tensor
			input, err := tensor.NewTensor(len(tt.input), len(tt.input[0]), len(tt.input[0][0]))
			require.NoError(t, err)
			defer input.Close()

			// Copy data into tensor
			for i := range tt.input {
				for j := range tt.input[i] {
					for k := range tt.input[i][j] {
						err := input.Set(tt.input[i][j][k], i, j, k)
						require.NoError(t, err)
					}
				}
			}

			// Create weight tensors
			upWeights, err := tensor.NewTensor(len(tt.upWeights), len(tt.upWeights[0]))
			require.NoError(t, err)
			defer upWeights.Close()

			downWeights, err := tensor.NewTensor(len(tt.downWeights), len(tt.downWeights[0]))
			require.NoError(t, err)
			defer downWeights.Close()

			// Copy weights into tensors
			for i := range tt.upWeights {
				for j := range tt.upWeights[i] {
					err := upWeights.Set(tt.upWeights[i][j], i, j)
					require.NoError(t, err)
				}
			}
			for i := range tt.downWeights {
				for j := range tt.downWeights[i] {
					err := downWeights.Set(tt.downWeights[i][j], i, j)
					require.NoError(t, err)
				}
			}

			// Set weights
			err = ffn.SetWeights(upWeights, downWeights)
			require.NoError(t, err)

			// Forward pass
			output, err := ffn.Forward(input)
			require.NoError(t, err)
			defer output.Close()

			// Verify output shape
			shape, err := output.Shape()
			require.NoError(t, err)
			require.Equal(t, 3, len(shape))
			require.Equal(t, len(tt.input), shape[0])
			require.Equal(t, len(tt.input[0]), shape[1])
			require.Equal(t, tt.hiddenDim, shape[2])

			// Verify output values
			for i := range tt.expected {
				for j := range tt.expected[i] {
					for k := range tt.expected[i][j] {
						got, err := output.Get(i, j, k)
						require.NoError(t, err)
						want := tt.expected[i][j][k]
						require.Equal(t, want, got)
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
		expectedErr     error
		errorIn         string // "forward" or "setweights"
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
			expectedErr: ErrInvalidWeightsShape,
			errorIn:     "forward",
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
			expectedErr: ErrInvalidWeightsShape,
			errorIn:     "setweights",
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
			expectedErr: ErrInvalidWeightsShape,
			errorIn:     "setweights",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ffn, err := NewFFN(tt.hiddenDim, tt.intermediateDim)
			require.NoError(t, err)

			if tt.errorIn == "setweights" {
				upWeights, err := tensor.NewTensor(len(tt.upWeights), len(tt.upWeights[0]))
				require.NoError(t, err)
				defer upWeights.Close()
				for i := range tt.upWeights {
					for j := range tt.upWeights[i] {
						err := upWeights.Set(tt.upWeights[i][j], i, j)
						require.NoError(t, err)
					}
				}
				downWeights, err := tensor.NewTensor(len(tt.downWeights), len(tt.downWeights[0]))
				require.NoError(t, err)
				defer downWeights.Close()
				for i := range tt.downWeights {
					for j := range tt.downWeights[i] {
						err := downWeights.Set(tt.downWeights[i][j], i, j)
						require.NoError(t, err)
					}
				}
				err = ffn.SetWeights(upWeights, downWeights)
				require.Error(t, err)
				require.Equal(t, tt.expectedErr, err)
				return
			}

			// For "forward" error
			input, err := tensor.NewTensor(len(tt.input), len(tt.input[0]), len(tt.input[0][0]))
			require.NoError(t, err)
			defer input.Close()
			for i := range tt.input {
				for j := range tt.input[i] {
					for k := range tt.input[i][j] {
						err := input.Set(tt.input[i][j][k], i, j, k)
						require.NoError(t, err)
					}
				}
			}
			upWeights, err := tensor.NewTensor(len(tt.upWeights), len(tt.upWeights[0]))
			require.NoError(t, err)
			defer upWeights.Close()
			for i := range tt.upWeights {
				for j := range tt.upWeights[i] {
					err := upWeights.Set(tt.upWeights[i][j], i, j)
					require.NoError(t, err)
				}
			}
			downWeights, err := tensor.NewTensor(len(tt.downWeights), len(tt.downWeights[0]))
			require.NoError(t, err)
			defer downWeights.Close()
			for i := range tt.downWeights {
				for j := range tt.downWeights[i] {
					err := downWeights.Set(tt.downWeights[i][j], i, j)
					require.NoError(t, err)
				}
			}
			ffn.SetWeights(upWeights, downWeights)
			_, err = ffn.Forward(input)
			require.Error(t, err)
			require.Equal(t, tt.expectedErr, err)
		})
	}
}

func TestFFN_Close(t *testing.T) {
	// Create a new FFN
	ffn, err := NewFFN(512, 2048) // 512 hidden dim, 2048 intermediate dim
	require.NoError(t, err)
	require.NotNil(t, ffn)

	// Set some weights
	upWeights, err := tensor.NewTensor(2048, 512)
	require.NoError(t, err)
	downWeights, err := tensor.NewTensor(512, 2048)
	require.NoError(t, err)
	err = ffn.SetWeights(upWeights, downWeights)
	require.NoError(t, err)

	// Close the FFN
	err = ffn.Close()
	require.NoError(t, err)

	// Verify that operations return error after close
	operations := []struct {
		name string
		fn   func() error
	}{
		{
			name: "Forward",
			fn: func() error {
				input, err := tensor.NewTensor(32, 16, 512)
				require.NoError(t, err)
				_, err = ffn.Forward(input)
				return err
			},
		},
		{
			name: "SetWeights",
			fn: func() error {
				upWeights, err := tensor.NewTensor(2048, 512)
				require.NoError(t, err)
				downWeights, err := tensor.NewTensor(512, 2048)
				require.NoError(t, err)
				return ffn.SetWeights(upWeights, downWeights)
			},
		},
	}

	for _, op := range operations {
		t.Run(op.name, func(t *testing.T) {
			err := op.fn()
			require.Error(t, err)
			require.Equal(t, ErrFFNClosed, err)
		})
	}
}

func TestFFN_applyReLU2(t *testing.T) {
	tests := []struct {
		name        string
		inputShape  []int
		inputValues [][]int8
		wantErr     error
		wantValues  [][]int8
	}{
		{
			name:       "valid 2D input with positive values",
			inputShape: []int{2, 3},
			inputValues: [][]int8{
				{1, 2, 3},
				{4, 5, 6},
			},
			wantErr: nil,
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
			wantErr: nil,
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
			wantErr: nil,
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
			wantErr: ErrInvalidInputShape,
		},
		{
			name:       "invalid 3D input",
			inputShape: []int{2, 2, 2},
			inputValues: [][]int8{
				{5, 6, 7, 8}, // Flattened 2x2 matrix
			},
			wantErr: ErrInvalidInputShape,
		},
		{
			name:        "empty input",
			inputShape:  []int{0, 0},
			inputValues: [][]int8{},
			wantErr:     ErrInvalidInputShape,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input, err := tensor.NewTensor(tt.inputShape...)
			require.NoError(t, err)
			defer input.Close()
			if input != nil {
				for i := range tt.inputValues {
					for j := range tt.inputValues[i] {
						if len(tt.inputShape) == 1 {
							err := input.Set(tt.inputValues[i][j], j)
							require.NoError(t, err)
						} else if len(tt.inputShape) == 2 {
							err := input.Set(tt.inputValues[i][j], i, j)
							require.NoError(t, err)
						}
					}
				}
			}

			// Create FFN with arbitrary dimensions
			ffn, err := NewFFN(4, 8)
			require.NoError(t, err)
			defer ffn.Close()

			// Call applyReLU2
			output, err := ffn.applyReLU2(input)

			// Check error
			if tt.wantErr != nil {
				require.Error(t, err)
				require.Equal(t, tt.wantErr, err)
				if output != nil {
					t.Error("applyReLU2() output = non-nil, want nil")
				}
				return
			}

			require.NoError(t, err)
			require.NotNil(t, output)

			// Verify output shape
			shape, err := output.Shape()
			require.NoError(t, err)
			require.Equal(t, 2, len(shape))

			// Verify output values
			for i := range tt.wantValues {
				for j := range tt.wantValues[i] {
					got, err := output.Get(i, j)
					require.NoError(t, err)
					want := tt.wantValues[i][j]
					require.Equal(t, want, got)
				}
			}

			// Clean up
			output.Close()
		})
	}
}

func TestFFNForward(t *testing.T) {
	tests := []struct {
		name        string
		hiddenDim   int
		interDim    int
		input       [][]int8
		upWeights   [][]int8
		downWeights [][]int8
		want        [][]int8
		wantErr     error
	}{
		{
			name:      "basic forward pass",
			hiddenDim: 4,
			interDim:  8,
			input: [][]int8{
				{1, 2, 3, 4},
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
			want: [][]int8{
				{0, 0, 0, 0}, // Updated expected values based on actual computation
			},
			wantErr: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create FFN
			ffn, err := NewFFN(tt.hiddenDim, tt.interDim)
			require.NoError(t, err)
			defer ffn.Close()

			// Create input tensor
			input, err := tensor.NewTensor(len(tt.input), len(tt.input[0]))
			require.NoError(t, err)
			defer input.Close()

			// Copy input data
			for i := range tt.input {
				for j := range tt.input[i] {
					err := input.Set(tt.input[i][j], i, j)
					require.NoError(t, err)
				}
			}

			// Create weight tensors
			upWeights, err := tensor.NewTensor(len(tt.upWeights), len(tt.upWeights[0]))
			require.NoError(t, err)
			defer upWeights.Close()

			downWeights, err := tensor.NewTensor(len(tt.downWeights), len(tt.downWeights[0]))
			require.NoError(t, err)
			defer downWeights.Close()

			// Copy weights into tensors
			for i := range tt.upWeights {
				for j := range tt.upWeights[i] {
					err := upWeights.Set(tt.upWeights[i][j], i, j)
					require.NoError(t, err)
				}
			}
			for i := range tt.downWeights {
				for j := range tt.downWeights[i] {
					err := downWeights.Set(tt.downWeights[i][j], i, j)
					require.NoError(t, err)
				}
			}

			// Set weights
			err = ffn.SetWeights(upWeights, downWeights)
			require.NoError(t, err)

			// Forward pass
			output, err := ffn.Forward(input)
			require.NoError(t, err)
			defer output.Close()

			// Verify output shape
			shape, err := output.Shape()
			require.NoError(t, err)
			require.Equal(t, []int{len(tt.input), len(tt.input[0])}, shape)

			// Verify output values
			for i := range tt.want {
				for j := range tt.want[i] {
					got, err := output.Get(i, j)
					require.NoError(t, err)
					require.Equal(t, tt.want[i][j], got)
				}
			}
		})
	}
}

func TestFFNInitialization(t *testing.T) {
	tests := []struct {
		name            string
		hiddenDim       int
		intermediateDim int
		wantErr         error
	}{
		{
			name:            "valid dimensions",
			hiddenDim:       1024,
			intermediateDim: 4096,
			wantErr:         nil,
		},
		{
			name:            "zero hidden dim",
			hiddenDim:       0,
			intermediateDim: 4096,
			wantErr:         ErrInvalidWeightsShape,
		},
		{
			name:            "zero intermediate dim",
			hiddenDim:       1024,
			intermediateDim: 0,
			wantErr:         ErrInvalidWeightsShape,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ffn, err := NewFFN(tt.hiddenDim, tt.intermediateDim)
			if tt.wantErr != nil {
				require.Error(t, err)
				require.Equal(t, tt.wantErr, err)
				require.Nil(t, ffn)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, ffn)
			defer ffn.Close()

			// Verify the FFN was created with correct dimensions
			require.Equal(t, tt.hiddenDim, ffn.hiddenDim)
			require.Equal(t, tt.intermediateDim, ffn.intermediateDim)
		})
	}
}

func TestFFNSetWeights(t *testing.T) {
	tests := []struct {
		name        string
		hiddenDim   int
		interDim    int
		upWeights   [][]int8
		downWeights [][]int8
		wantErr     error
	}{
		{
			name:      "valid weights",
			hiddenDim: 4,
			interDim:  8,
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
			wantErr: nil,
		},
		{
			name:      "invalid up weights shape",
			hiddenDim: 4,
			interDim:  8,
			upWeights: [][]int8{
				{1, 2, 3}, // Wrong shape
				{4, 5, 6},
			},
			downWeights: [][]int8{
				{1, 2, 3, 4, 5, 6, 7, 8},
				{9, 10, 11, 12, 13, 14, 15, 16},
			},
			wantErr: ErrInvalidWeightsShape,
		},
		{
			name:      "invalid down weights shape",
			hiddenDim: 4,
			interDim:  8,
			upWeights: [][]int8{
				{1, 2, 3, 4},
				{5, 6, 7, 8},
				{9, 10, 11, 12},
				{13, 14, 15, 16},
				{17, 18, 19, 20},
				{21, 22, 23, 24},
				{25, 26, 27, 28},
				{29, 30, 31, 32},
			},
			downWeights: [][]int8{
				{1, 2}, // Wrong shape
				{3, 4},
			},
			wantErr: ErrInvalidWeightsShape,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create FFN
			ffn, err := NewFFN(tt.hiddenDim, tt.interDim)
			require.NoError(t, err)
			defer ffn.Close()

			// Create weight tensors
			upWeights, err := tensor.NewTensor(len(tt.upWeights), len(tt.upWeights[0]))
			require.NoError(t, err)
			defer upWeights.Close()

			downWeights, err := tensor.NewTensor(len(tt.downWeights), len(tt.downWeights[0]))
			require.NoError(t, err)
			defer downWeights.Close()

			// Copy weights into tensors
			for i := range tt.upWeights {
				for j := range tt.upWeights[i] {
					err := upWeights.Set(tt.upWeights[i][j], i, j)
					require.NoError(t, err)
				}
			}
			for i := range tt.downWeights {
				for j := range tt.downWeights[i] {
					err := downWeights.Set(tt.downWeights[i][j], i, j)
					require.NoError(t, err)
				}
			}

			// Set weights
			err = ffn.SetWeights(upWeights, downWeights)
			if tt.wantErr != nil {
				require.Error(t, err)
				require.Equal(t, tt.wantErr, err)
				return
			}
			require.NoError(t, err)
		})
	}
}

func TestFFNClose(t *testing.T) {
	// Create FFN
	ffn, err := NewFFN(4, 2)
	require.NoError(t, err)

	// Close FFN
	err = ffn.Close()
	require.NoError(t, err)

	// Try to use closed FFN
	_, err = ffn.Forward(nil)
	require.Error(t, err)
	require.Equal(t, ErrFFNClosed, err)
}

func TestFFNForwardWithInvalidInput(t *testing.T) {
	ffn, err := NewFFN(10, 20)
	require.NoError(t, err)
	_, err = ffn.Forward(nil)
	require.ErrorIs(t, err, ErrInvalidInputShape)
}

func TestFFNForwardWithInvalidShape(t *testing.T) {
	ffn, err := NewFFN(10, 20)
	require.NoError(t, err)
	input, _ := tensor.NewTensor(5)
	_, err = ffn.Forward(input)
	require.ErrorIs(t, err, ErrInvalidInputShape)
}

func TestFFNForwardWithInvalidBatchSize(t *testing.T) {
	ffn, err := NewFFN(10, 20)
	require.NoError(t, err)
	input, _ := tensor.NewTensor(0, 10)
	_, err = ffn.Forward(input)
	require.ErrorIs(t, err, ErrInvalidInputShape)
}
