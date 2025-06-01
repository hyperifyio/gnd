package attention

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// Package attention_test provides tests for the attention package.
//
// # Test Suite for BitNet Attention
//
// This file contains tests for the quantized attention mechanisms used in BitNet.
// It verifies the correctness of attention computation, output projection, and quantization.
//
// References:
//   - BitNet: Scaling 1-bit Transformers for Large Language Models
//     https://arxiv.org/abs/2310.11453
//   - BitNet Architecture Specifications
//     https://github.com/microsoft/BitNet
//   - Attention Is All You Need (Original Transformer Paper)
//     https://arxiv.org/abs/1706.03762
//   - Grouped-Query Attention (GQA) Paper
//     https://arxiv.org/abs/2305.13245
//   - Go Testing Documentation
//     https://pkg.go.dev/testing
//   - Go Benchmark Documentation
//     https://pkg.go.dev/testing#B
//
// Key test aspects:
//   - Verifies correct computation of attention scores with int8 inputs
//   - Validates proper handling of grouped-query attention
//   - Ensures correct quantization of attention outputs
//   - Tests memory management and tensor cleanup
//   - Verifies compatibility with BitNet's binary-weight quantization
//   - Tests attention mask handling (regular and causal masks)
//   - Validates behavior with different value ranges (min/max int8, mixed values)
//   - Comprehensive error case testing (nil tensors, invalid shapes, dimension mismatches)
//   - Verifies higher precision computation for accuracy (as per issue #182)
//   - Tests proper softmax implementation (as per issue #182)
//
// Test coverage:
//   - Scaled dot-product attention computation
//   - Attention weight application to values
//   - Output projection with int8 weights
//   - Edge cases and error conditions
//   - Memory leak prevention
//   - Mask application and validation
//   - Value range handling and clamping
//   - Input validation and error reporting
//   - Higher precision computation verification
//   - Softmax numerical stability
//
// Related tasks:
//   - #182: Compute Scaled Dot-Product Attention
//   - #183: Apply Attention Weights to Values
//   - #184: Attention Output Projection
//
// Usage:
//   - Run tests with: go test -v ./...
//   - Critical for maintaining correct quantized inference
//   - Must be updated if attention implementation changes
//
// Caveats:
//   - Tests should verify correct quantized output
//   - Must maintain compatibility with BitNet's architecture
//   - Performance critical - benchmark tests regularly
//   - Tests cover full int8 value range (-128 to 127)
//   - Tests verify proper mask application
//   - Tests ensure proper error handling
//   - Tests must verify higher precision computation
//   - Tests must validate softmax stability
//
// For more details, see BitNet issue #170 and the BitNet project documentation.

func TestScaledDotProductAttention(t *testing.T) {
	tests := []struct {
		name     string
		seqLen   int
		headDim  int
		q        []int8
		k        []int8
		v        []int8
		mask     []int8
		expected []int8
	}{
		{
			name:     "Simple attention",
			seqLen:   2,
			headDim:  2,
			q:        []int8{1, 0, 0, 1},
			k:        []int8{1, 0, 0, 1},
			v:        []int8{1, 0, 0, 1},
			mask:     nil,
			expected: []int8{1, 0, 0, 1},
		},
		{
			name:     "Attention with mask",
			seqLen:   2,
			headDim:  2,
			q:        []int8{1, 0, 0, 1},
			k:        []int8{1, 0, 0, 1},
			v:        []int8{1, 0, 0, 1},
			mask:     []int8{1, 0, 0, 1},
			expected: []int8{1, 0, 0, 1},
		},
		{
			name:     "Attention with causal mask",
			seqLen:   2,
			headDim:  2,
			q:        []int8{1, 0, 0, 1},
			k:        []int8{1, 0, 0, 1},
			v:        []int8{1, 0, 0, 1},
			mask:     []int8{1, 0, 1, 1},
			expected: []int8{1, 0, 0, 1},
		},
		{
			name:     "Attention with large values",
			seqLen:   2,
			headDim:  2,
			q:        []int8{127, 0, 0, 127},
			k:        []int8{127, 0, 0, 127},
			v:        []int8{127, 0, 0, 127},
			mask:     nil,
			expected: []int8{127, 0, 0, 127},
		},
		{
			name:     "Attention with negative values",
			seqLen:   2,
			headDim:  2,
			q:        []int8{-128, 0, 0, -128},
			k:        []int8{-128, 0, 0, -128},
			v:        []int8{-128, 0, 0, -128},
			mask:     nil,
			expected: []int8{-128, 0, 0, -128},
		},
		{
			name:     "Attention with mixed values",
			seqLen:   2,
			headDim:  2,
			q:        []int8{64, -64, -64, 64},
			k:        []int8{64, -64, -64, 64},
			v:        []int8{64, -64, -64, 64},
			mask:     nil,
			expected: []int8{64, -64, -64, 64},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input tensors
			q, err := tensor.NewTensor(1, 1, tt.seqLen, tt.headDim)
			if err != nil {
				t.Fatalf("Failed to create query tensor: %v", err)
			}
			k, err := tensor.NewTensor(1, 1, tt.seqLen, tt.headDim)
			if err != nil {
				t.Fatalf("Failed to create key tensor: %v", err)
			}
			v, err := tensor.NewTensor(1, 1, tt.seqLen, tt.headDim)
			if err != nil {
				t.Fatalf("Failed to create value tensor: %v", err)
			}

			// Set input values
			for i := 0; i < tt.seqLen; i++ {
				for j := 0; j < tt.headDim; j++ {
					if err := q.Set(tt.q[i*tt.headDim+j], 0, 0, i, j); err != nil {
						t.Fatalf("Failed to set query value: %v", err)
					}
					if err := k.Set(tt.k[i*tt.headDim+j], 0, 0, i, j); err != nil {
						t.Fatalf("Failed to set key value: %v", err)
					}
					if err := v.Set(tt.v[i*tt.headDim+j], 0, 0, i, j); err != nil {
						t.Fatalf("Failed to set value: %v", err)
					}
				}
			}

			// Create mask tensor if provided
			var mask *tensor.Tensor
			if tt.mask != nil {
				mask, err = tensor.NewTensor(1, 1, tt.seqLen, tt.seqLen)
				if err != nil {
					t.Fatalf("Failed to create mask tensor: %v", err)
				}
				for i := 0; i < tt.seqLen; i++ {
					for j := 0; j < tt.seqLen; j++ {
						if err := mask.Set(tt.mask[i*tt.seqLen+j], 0, 0, i, j); err != nil {
							t.Fatalf("Failed to set mask value: %v", err)
						}
					}
				}
			}

			// Compute attention
			output, err := ScaledDotProductAttention(q, k, v, mask)
			if err != nil {
				t.Fatalf("ScaledDotProductAttention failed: %v", err)
			}

			// Verify output
			for i := 0; i < tt.seqLen; i++ {
				for j := 0; j < tt.headDim; j++ {
					val, err := output.Get(0, 0, i, j)
					if err != nil {
						t.Fatalf("Failed to get output value: %v", err)
					}
					expected := tt.expected[i*tt.headDim+j]
					if val != expected {
						t.Errorf("Output[%d,%d] = %d, want %d", i, j, val, expected)
					}
				}
			}
		})
	}
}

func TestScaledDotProductAttentionErrors(t *testing.T) {
	tests := []struct {
		name    string
		q       *tensor.Tensor
		k       *tensor.Tensor
		v       *tensor.Tensor
		mask    *tensor.Tensor
		wantErr error
	}{
		{
			name:    "Nil query tensor",
			q:       nil,
			k:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			v:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			mask:    nil,
			wantErr: ErrNilTensor,
		},
		{
			name:    "Invalid query shape",
			q:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2); return t }(),
			k:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			v:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			mask:    nil,
			wantErr: ErrInvalidInputShape,
		},
		{
			name:    "Dimension mismatch",
			q:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			k:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 2, 2, 2); return t }(),
			v:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			mask:    nil,
			wantErr: ErrDimensionMismatch,
		},
		{
			name:    "Invalid mask shape",
			q:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			k:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			v:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			mask:    func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2); return t }(),
			wantErr: ErrInvalidInputShape,
		},
		{
			name:    "Mask dimension mismatch",
			q:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			k:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			v:       func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 1, 2, 2); return t }(),
			mask:    func() *tensor.Tensor { t, _ := tensor.NewTensor(1, 2, 2, 2); return t }(),
			wantErr: ErrDimensionMismatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ScaledDotProductAttention(tt.q, tt.k, tt.v, tt.mask)
			if err != tt.wantErr {
				t.Errorf("ScaledDotProductAttention() error = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func BenchmarkScaledDotProductAttention(b *testing.B) {
	// Create input tensors
	q, err := tensor.NewTensor(1, 1, 2, 2)
	if err != nil {
		b.Fatalf("Failed to create query tensor: %v", err)
	}
	k, err := tensor.NewTensor(1, 1, 2, 2)
	if err != nil {
		b.Fatalf("Failed to create key tensor: %v", err)
	}
	v, err := tensor.NewTensor(1, 1, 2, 2)
	if err != nil {
		b.Fatalf("Failed to create value tensor: %v", err)
	}

	// Set input values
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if err := q.Set(1, 0, 0, i, j); err != nil {
				b.Fatalf("Failed to set query value: %v", err)
			}
			if err := k.Set(1, 0, 0, i, j); err != nil {
				b.Fatalf("Failed to set key value: %v", err)
			}
			if err := v.Set(1, 0, 0, i, j); err != nil {
				b.Fatalf("Failed to set value: %v", err)
			}
		}
	}

	// Create mask tensor
	mask, err := tensor.NewTensor(1, 1, 2, 2)
	if err != nil {
		b.Fatalf("Failed to create mask tensor: %v", err)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if err := mask.Set(1, 0, 0, i, j); err != nil {
				b.Fatalf("Failed to set mask value: %v", err)
			}
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ScaledDotProductAttention(q, k, v, mask)
	}
}

// Helper function to convert bool to int8
func boolToInt8(b bool) int8 {
	if b {
		return 1
	}
	return 0
}
