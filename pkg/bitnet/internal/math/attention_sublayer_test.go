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
		input      *tensor.Tensor
	}{
		{
			name:       "standard attention",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 8,
			input:      tensor.NewTensor(1, 32, 64),
		},
		{
			name:       "grouped-query attention",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 2,
			input:      tensor.NewTensor(1, 32, 64),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create attention sublayer
			attn, err := NewAttentionSublayer(tt.hiddenDim, tt.numHeads, tt.numKVHeads)
			if err != nil {
				t.Fatalf("Failed to create attention sublayer: %v", err)
			}
			defer attn.Close()

			// Calculate dimensions for weights
			headDim := tt.hiddenDim / tt.numHeads

			// Initialize weights with correct shapes
			qWeights := tensor.NewTensor(tt.hiddenDim, tt.numHeads*headDim)
			kWeights := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			vWeights := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			outWeights := tensor.NewTensor(tt.numHeads*headDim, tt.hiddenDim)

			// Fill weights with pseudo-random but deterministic data
			for i := 0; i < tt.hiddenDim; i++ {
				for j := 0; j < tt.numHeads*headDim; j++ {
					qWeights.Set(int8((i+j)%8-4), i, j)
				}
				for j := 0; j < tt.hiddenDim; j++ {
					kWeights.Set(int8((i-j)%8-4), i, j)
					vWeights.Set(int8((i*j)%8-4), i, j)
				}
			}
			for i := 0; i < tt.numHeads*headDim; i++ {
				for j := 0; j < tt.hiddenDim; j++ {
					outWeights.Set(int8((i+j)%8-4), i, j)
				}
			}

			// Set weights
			if err := attn.SetWeights(qWeights, kWeights, vWeights, outWeights); err != nil {
				t.Fatalf("Failed to set weights: %v", err)
			}

			// Initialize input with non-zero values
			for i := 0; i < tt.input.Shape()[0]; i++ {
				for j := 0; j < tt.input.Shape()[1]; j++ {
					for k := 0; k < tt.input.Shape()[2]; k++ {
						tt.input.Set(int8((i+j+k)%8-4), i, j, k)
					}
				}
			}

			// Forward pass
			output, err := attn.Forward(tt.input)
			if err != nil {
				t.Fatalf("Forward pass failed: %v", err)
			}
			defer output.Close()

			// Verify output shape
			if len(output.Shape()) != len(tt.input.Shape()) {
				t.Errorf("Output shape = %v, want %v", output.Shape(), tt.input.Shape())
			}

			// Verify output is not all zeros
			data := output.Data()
			allZero := true
			for _, v := range data {
				if v != 0 {
					allZero = false
					break
				}
			}
			if allZero {
				t.Error("Output is all zeros, want nonzero values")
			}

			// Verify output has variance
			minVal := data[0]
			maxVal := data[0]
			for _, v := range data {
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
		input      *tensor.Tensor
	}{
		{
			name:       "invalid input shape",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 8,
			input:      tensor.NewTensor(2, 2),
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

			attn, _ := NewAttentionSublayer(tt.hiddenDim, tt.numHeads, tt.numKVHeads)
			defer attn.Close()

			// Initialize weights
			headDim := tt.hiddenDim / tt.numHeads

			qWeights := tensor.NewTensor(tt.hiddenDim, tt.numHeads*headDim)
			kWeights := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			vWeights := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			outWeights := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)

			// Set weights
			if err := attn.SetWeights(qWeights, kWeights, vWeights, outWeights); err != nil {
				t.Fatalf("Failed to set weights: %v", err)
			}

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
			input:      tensor.NewTensor(1, 64),
			wantErr:    false,
		},
		{
			name:       "valid 3D input",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 8,
			input:      tensor.NewTensor(1, 32, 64),
			wantErr:    false,
		},
		{
			name:       "invalid input shape",
			hiddenDim:  64,
			numHeads:   8,
			numKVHeads: 8,
			input:      tensor.NewTensor(2, 2),
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

			qWeights := tensor.NewTensor(tt.hiddenDim, tt.numHeads*headDim)
			kWeights := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			vWeights := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)
			outWeights := tensor.NewTensor(tt.hiddenDim, tt.hiddenDim)

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
			for i := 0; i < tt.input.Shape()[0]; i++ {
				for j := 0; j < tt.input.Shape()[1]; j++ {
					if len(tt.input.Shape()) == 2 {
						tt.input.Set(int8((i+j)%8-4), i, j)
					} else {
						for k := 0; k < tt.input.Shape()[2]; k++ {
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

			// Verify output shape matches input shape
			if !equalShape(output.Shape(), tt.input.Shape()) {
				t.Errorf("Output shape = %v, want %v", output.Shape(), tt.input.Shape())
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
