package attention_sublayer

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/stretchr/testify/require"
)

func equalShape(a, b []int) bool {
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

func TestAttentionSublayer(t *testing.T) {
	tests := []struct {
		name       string
		hiddenDim  int
		numHeads   int
		numKVHeads int
		input      func() (*tensor.Tensor, error)
	}{
		{
			name:       "standard attention",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 8,
			input: func() (*tensor.Tensor, error) {
				return tensor.NewTensor(1, 32, 64)
			},
		},
		{
			name:       "grouped-query attention",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 2,
			input: func() (*tensor.Tensor, error) {
				return tensor.NewTensor(1, 32, 64)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input tensor
			input, err := tt.input()
			if err != nil {
				t.Fatalf("Failed to create input tensor: %v", err)
			}
			defer input.Close()

			// Create attention sublayer
			attn, err := NewAttentionSublayer(tt.hiddenDim, tt.numHeads, tt.numKVHeads)
			if err != nil {
				t.Fatalf("Failed to create attention sublayer: %v", err)
			}
			defer attn.Close()

			// Calculate dimensions for weights
			headDim := tt.hiddenDim / tt.numHeads

			// Initialize weights with correct shapes
			qWeights, err := tensor.NewTensor(tt.hiddenDim, tt.numHeads*headDim)
			if err != nil {
				t.Fatalf("Failed to create Q weights tensor: %v", err)
			}
			kWeights, err := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create K weights tensor: %v", err)
			}
			vWeights, err := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create V weights tensor: %v", err)
			}
			outWeights, err := tensor.NewTensor(tt.numHeads*headDim, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create output weights tensor: %v", err)
			}

			// Fill weights with pseudo-random but deterministic data
			for i := 0; i < tt.hiddenDim; i++ {
				for j := 0; j < tt.numHeads*headDim; j++ {
					if err := qWeights.Set(int8((i+j)%8-4), i, j); err != nil {
						t.Fatalf("Failed to set Q weight value: %v", err)
					}
				}
				for j := 0; j < tt.hiddenDim; j++ {
					if err := kWeights.Set(int8((i-j)%8-4), i, j); err != nil {
						t.Fatalf("Failed to set K weight value: %v", err)
					}
					if err := vWeights.Set(int8((i*j)%8-4), i, j); err != nil {
						t.Fatalf("Failed to set V weight value: %v", err)
					}
				}
			}
			for i := 0; i < tt.numHeads*headDim; i++ {
				for j := 0; j < tt.hiddenDim; j++ {
					if err := outWeights.Set(int8((i+j)%8-4), i, j); err != nil {
						t.Fatalf("Failed to set output weight value: %v", err)
					}
				}
			}

			// Set weights
			if err := attn.SetWeights(qWeights, kWeights, vWeights, outWeights); err != nil {
				t.Fatalf("Failed to set weights: %v", err)
			}

			// Initialize input with non-zero values
			inputShape, err := input.Shape()
			if err != nil {
				t.Fatalf("Failed to get input shape: %v", err)
			}
			for i := 0; i < inputShape[0]; i++ {
				for j := 0; j < inputShape[1]; j++ {
					for k := 0; k < inputShape[2]; k++ {
						if err := input.Set(int8((i+j+k)%8-4), i, j, k); err != nil {
							t.Fatalf("Failed to set input value: %v", err)
						}
					}
				}
			}

			// Forward pass
			output, err := attn.Forward(input)
			if err != nil {
				t.Fatalf("Forward pass failed: %v", err)
			}
			defer output.Close()

			// Verify output shape
			outputShape, err := output.Shape()
			if err != nil {
				t.Fatalf("Failed to get output shape: %v", err)
			}
			if len(outputShape) != 3 {
				t.Errorf("output shape = %v, want 3 dimensions", outputShape)
			}
			if outputShape[0] != 1 {
				t.Errorf("output batch size = %d, want 1", outputShape[0])
			}
			if outputShape[1] != 32 {
				t.Errorf("output seq len = %d, want 32", outputShape[1])
			}
			if outputShape[2] != 64 {
				t.Errorf("output hidden dim = %d, want 64", outputShape[2])
			}

			// Verify output is not all zeros
			outputData, err := output.Data()
			if err != nil {
				t.Fatalf("Failed to get output data: %v", err)
			}
			allZero := true
			for _, v := range outputData {
				if v != 0 {
					allZero = false
					break
				}
			}
			if allZero {
				t.Error("Output is all zeros, want nonzero values")
			}

			// Verify output has variance
			minVal := outputData[0]
			maxVal := outputData[0]
			for _, v := range outputData {
				if v < minVal {
					minVal = v
				}
				if v > maxVal {
					maxVal = v
				}
			}
			if minVal == maxVal {
				t.Error("Output has no variance, want a range of values")
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
		input      func() (*tensor.Tensor, error)
	}{
		{
			name:       "invalid input shape",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 8,
			input: func() (*tensor.Tensor, error) {
				return tensor.NewTensor(2, 2)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic")
				} else if s, ok := r.(string); !ok || s != "tensor: invalid hidden dimension" {
					t.Errorf("unexpected panic message: %v", r)
				}
			}()

			// Create input tensor
			input, err := tt.input()
			if err != nil {
				t.Fatalf("Failed to create input tensor: %v", err)
			}
			defer input.Close()

			attn, err := NewAttentionSublayer(tt.hiddenDim, tt.numHeads, tt.numKVHeads)
			if err != nil {
				t.Fatalf("Failed to create attention sublayer: %v", err)
			}
			defer attn.Close()

			// Initialize weights
			headDim := tt.hiddenDim / tt.numHeads

			qWeights, err := tensor.NewTensor(tt.hiddenDim, tt.numHeads*headDim)
			if err != nil {
				t.Fatalf("Failed to create Q weights tensor: %v", err)
			}
			kWeights, err := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create K weights tensor: %v", err)
			}
			vWeights, err := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create V weights tensor: %v", err)
			}
			outWeights, err := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			if err != nil {
				t.Fatalf("Failed to create output weights tensor: %v", err)
			}

			// Set weights
			if err := attn.SetWeights(qWeights, kWeights, vWeights, outWeights); err != nil {
				t.Fatalf("Failed to set weights: %v", err)
			}

			attn.Forward(input)
		})
	}
}

// Helper function to create a tensor and handle errors
func createTensor(b *testing.B, shape ...int) *tensor.Tensor {
	t, err := tensor.NewTensor(shape...)
	if err != nil {
		b.Fatalf("Failed to create tensor: %v", err)
	}
	return t
}

func BenchmarkAttentionSublayer(b *testing.B) {
	// Create attention sublayer
	hiddenSize := 512
	numHeads := 8
	numKVHeads := 8
	attn, err := NewAttentionSublayer(hiddenSize, numHeads, numKVHeads)
	if err != nil {
		b.Fatalf("Failed to create attention sublayer: %v", err)
	}
	defer attn.Close()

	// Create input tensor
	input := createTensor(b, 1, 32, hiddenSize)
	defer input.Close()

	// Initialize weights
	qWeights := createTensor(b, hiddenSize, numHeads*hiddenSize/numHeads)
	defer qWeights.Close()

	kWeights := createTensor(b, hiddenSize, numKVHeads*hiddenSize/numKVHeads)
	defer kWeights.Close()

	vWeights := createTensor(b, hiddenSize, numKVHeads*hiddenSize/numKVHeads)
	defer vWeights.Close()

	outWeights := createTensor(b, numHeads*hiddenSize/numHeads, hiddenSize)
	defer outWeights.Close()

	// Set weights
	if err := attn.SetWeights(qWeights, kWeights, vWeights, outWeights); err != nil {
		b.Fatalf("Failed to set weights: %v", err)
	}

	// Benchmark forward pass
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		output, err := attn.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
		output.Close()
	}
}

func BenchmarkAttentionSublayerWithInvalidWeights(b *testing.B) {
	// Create attention sublayer
	hiddenSize := 512
	numHeads := 8
	numKVHeads := 8
	attn, err := NewAttentionSublayer(hiddenSize, numHeads, numKVHeads)
	if err != nil {
		b.Fatalf("Failed to create attention sublayer: %v", err)
	}
	defer attn.Close()

	// Create input tensor
	input := createTensor(b, 1, 32, hiddenSize)
	defer input.Close()

	// Initialize weights with invalid shapes
	qWeights := createTensor(b, hiddenSize-1, numHeads*hiddenSize/numHeads)
	defer qWeights.Close()

	kWeights := createTensor(b, hiddenSize-1, numKVHeads*hiddenSize/numKVHeads)
	defer kWeights.Close()

	vWeights := createTensor(b, hiddenSize-1, numKVHeads*hiddenSize/numKVHeads)
	defer vWeights.Close()

	outWeights := createTensor(b, numHeads*hiddenSize/numHeads, hiddenSize)
	defer outWeights.Close()

	// Set weights
	if err := attn.SetWeights(qWeights, kWeights, vWeights, outWeights); err != nil {
		b.Fatalf("Failed to set weights: %v", err)
	}

	// Benchmark forward pass
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		output, err := attn.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
		output.Close()
	}
}

func BenchmarkAttentionSublayerWithDifferentShapes(b *testing.B) {
	// Create attention sublayer
	hiddenSize := 512
	numHeads := 8
	numKVHeads := 8
	attn, err := NewAttentionSublayer(hiddenSize, numHeads, numKVHeads)
	if err != nil {
		b.Fatalf("Failed to create attention sublayer: %v", err)
	}
	defer attn.Close()

	// Create input tensor
	input := createTensor(b, 1, 32, hiddenSize)
	defer input.Close()

	// Initialize weights with different shapes
	qWeights := createTensor(b, hiddenSize, numHeads*hiddenSize/numHeads)
	defer qWeights.Close()

	kWeights := createTensor(b, hiddenSize, numKVHeads*hiddenSize/numKVHeads-1)
	defer kWeights.Close()

	vWeights := createTensor(b, hiddenSize, numKVHeads*hiddenSize/numKVHeads-1)
	defer vWeights.Close()

	outWeights := createTensor(b, numHeads*hiddenSize/numHeads, hiddenSize)
	defer outWeights.Close()

	// Set weights
	if err := attn.SetWeights(qWeights, kWeights, vWeights, outWeights); err != nil {
		b.Fatalf("Failed to set weights: %v", err)
	}

	// Benchmark forward pass
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		output, err := attn.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
		output.Close()
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
			numHeads:   7,
			numKVHeads: 7,
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
			name:     "valid weights",
			qWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads); return t }(),
			kWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			vWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			outWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize); return t }(),
			wantErr:    false,
		},
		{
			name:     "invalid query weights shape",
			qWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(hiddenSize-1, numHeads*hiddenSize/numHeads); return t }(),
			kWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			vWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			outWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize); return t }(),
			wantErr:    true,
		},
		{
			name:     "invalid key weights shape",
			qWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads); return t }(),
			kWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads-1)
				return t
			}(),
			vWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			outWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize); return t }(),
			wantErr:    true,
		},
		{
			name:     "invalid value weights shape",
			qWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads); return t }(),
			kWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			vWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize-1, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			outWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize); return t }(),
			wantErr:    true,
		},
		{
			name:     "invalid output weights shape",
			qWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads); return t }(),
			kWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			vWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			outWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize+1); return t }(),
			wantErr:    true,
		},
		{
			name:     "nil query weights",
			qWeights: nil,
			kWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			vWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			outWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize); return t }(),
			wantErr:    true,
		},
		{
			name:     "nil key weights",
			qWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads); return t }(),
			kWeights: nil,
			vWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			outWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize); return t }(),
			wantErr:    true,
		},
		{
			name:     "nil value weights",
			qWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads); return t }(),
			kWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			vWeights:   nil,
			outWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(numHeads*hiddenSize/numHeads, hiddenSize); return t }(),
			wantErr:    true,
		},
		{
			name:     "nil output weights",
			qWeights: func() *tensor.Tensor { t, _ := tensor.NewTensor(hiddenSize, numHeads*hiddenSize/numHeads); return t }(),
			kWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
			vWeights: func() *tensor.Tensor {
				t, _ := tensor.NewTensor(hiddenSize, numKVHeads*hiddenSize/numKVHeads)
				return t
			}(),
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
			gamma:   func() *tensor.Tensor { t, _ := tensor.NewTensor(hiddenSize); return t }(),
			wantErr: false,
		},
		{
			name:    "invalid gamma shape",
			gamma:   func() *tensor.Tensor { t, _ := tensor.NewTensor(hiddenSize + 1); return t }(),
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
	tests := []struct {
		name       string
		hiddenDim  int
		numHeads   int
		numKVHeads int
		input      *tensor.Tensor
		wantErr    bool
	}{
		{
			name:       "valid 2D input",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 8,
			input:      func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 64); return t }(),
			wantErr:    false,
		},
		{
			name:       "valid 3D input",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 8,
			input:      func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 32, 64); return t }(),
			wantErr:    false,
		},
		{
			name:       "invalid input shape",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 8,
			input:      func() *tensor.Tensor { t, _ := tensor.NewTensor(2, 2); return t }(),
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attn, err := NewAttentionSublayer(tt.hiddenDim, tt.numHeads, tt.numKVHeads)
			if err != nil {
				t.Fatalf("Failed to create attention sublayer: %v", err)
			}
			defer attn.Close()

			// Initialize weights
			headDim := tt.hiddenDim / tt.numHeads

			qWeights, _ := tensor.NewTensor(tt.hiddenDim, tt.numHeads*headDim)
			kWeights, _ := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			vWeights, _ := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			outWeights, _ := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)

			// Fill weights with non-zero values
			for i := 0; i < tt.hiddenDim; i++ {
				for j := 0; j < tt.numHeads*headDim; j++ {
					qWeights.Set(int8((i+j)%8-4), i, j)
				}
				for j := 0; j < tt.hiddenDim; j++ {
					kWeights.Set(int8((i-j)%8-4), i, j)
					vWeights.Set(int8((i*j)%8-4), i, j)
				}
				for j := 0; j < tt.hiddenDim; j++ {
					outWeights.Set(int8((i+j)%8-4), i, j)
				}
			}

			// Set weights
			if err := attn.SetWeights(qWeights, kWeights, vWeights, outWeights); err != nil {
				t.Fatalf("Failed to set weights: %v", err)
			}

			// Initialize input with non-zero values
			shape, _ := tt.input.Shape()
			for i := 0; i < shape[0]; i++ {
				for j := 0; j < shape[1]; j++ {
					if len(shape) == 2 {
						tt.input.Set(int8((i+j)%8-4), i, j)
					} else {
						for k := 0; k < shape[2]; k++ {
							tt.input.Set(int8((i+j+k)%8-4), i, j, k)
						}
					}
				}
			}

			// Forward pass
			if tt.wantErr {
				defer func() {
					if r := recover(); r == nil {
						t.Errorf("expected panic for invalid input shape")
					} else if s, ok := r.(string); !ok || s != "tensor: invalid hidden dimension" {
						t.Errorf("unexpected panic message: %v", r)
					}
				}()
				attn.Forward(tt.input)
				return
			}

			output, err := attn.Forward(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Forward() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil {
				return
			}
			defer output.Close()

			// Verify output shape
			outputShape, err := output.Shape()
			if err != nil {
				t.Fatalf("Failed to get output shape: %v", err)
			}
			if len(outputShape) != 3 {
				t.Errorf("output shape = %v, want 3 dimensions", outputShape)
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
	qWeights, _ := tensor.NewTensor(512, 512)
	kWeights, _ := tensor.NewTensor(512, 512)
	vWeights, _ := tensor.NewTensor(512, 512)
	outWeights, _ := tensor.NewTensor(512, 512)
	err = sublayer.SetWeights(qWeights, kWeights, vWeights, outWeights)
	require.NoError(t, err)

	// Set gamma
	gamma, _ := tensor.NewTensor(512)
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
				input, _ := tensor.NewTensor(32, 16, 512)
				sublayer.Forward(input)
			},
		},
		{
			name: "SetWeights",
			fn: func() {
				qWeights, _ := tensor.NewTensor(512, 512)
				kWeights, _ := tensor.NewTensor(512, 512)
				vWeights, _ := tensor.NewTensor(512, 512)
				outWeights, _ := tensor.NewTensor(512, 512)
				sublayer.SetWeights(qWeights, kWeights, vWeights, outWeights)
			},
		},
		{
			name: "SetGamma",
			fn: func() {
				gamma, _ := tensor.NewTensor(512)
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
