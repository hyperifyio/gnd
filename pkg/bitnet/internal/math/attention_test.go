package math

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

func TestScaledDotProductAttention(t *testing.T) {
	tests := []struct {
		name     string
		seqLen   int
		headDim  int
		q        [][]int8
		k        [][]int8
		v        [][]int8
		expected [][]int8
	}{
		{
			name:    "simple attention",
			seqLen:  2,
			headDim: 8,
			q: [][]int8{
				{1, 1, 1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1, 1, 1},
			},
			k: [][]int8{
				{1, 1, 1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1, 1, 1},
			},
			v: [][]int8{
				{1, 1, 1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1, 1, 1},
			},
			expected: [][]int8{
				{1, 1, 1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1, 1, 1},
			},
		},
		{
			name:    "attention with scaling",
			seqLen:  2,
			headDim: 8,
			q: [][]int8{
				{2, 2, 2, 2, 2, 2, 2, 2},
				{2, 2, 2, 2, 2, 2, 2, 2},
			},
			k: [][]int8{
				{2, 2, 2, 2, 2, 2, 2, 2},
				{2, 2, 2, 2, 2, 2, 2, 2},
			},
			v: [][]int8{
				{2, 2, 2, 2, 2, 2, 2, 2},
				{2, 2, 2, 2, 2, 2, 2, 2},
			},
			expected: [][]int8{
				{2, 2, 2, 2, 2, 2, 2, 2},
				{2, 2, 2, 2, 2, 2, 2, 2},
			},
		},
		{
			name:    "attention with large values",
			seqLen:  2,
			headDim: 8,
			q: [][]int8{
				{100, 100, 100, 100, 100, 100, 100, 100},
				{100, 100, 100, 100, 100, 100, 100, 100},
			},
			k: [][]int8{
				{100, 100, 100, 100, 100, 100, 100, 100},
				{100, 100, 100, 100, 100, 100, 100, 100},
			},
			v: [][]int8{
				{100, 100, 100, 100, 100, 100, 100, 100},
				{100, 100, 100, 100, 100, 100, 100, 100},
			},
			expected: [][]int8{
				{100, 100, 100, 100, 100, 100, 100, 100},
				{100, 100, 100, 100, 100, 100, 100, 100},
			},
		},
		{
			name:    "attention with negative values",
			seqLen:  2,
			headDim: 8,
			q: [][]int8{
				{-100, -100, -100, -100, -100, -100, -100, -100},
				{-100, -100, -100, -100, -100, -100, -100, -100},
			},
			k: [][]int8{
				{-100, -100, -100, -100, -100, -100, -100, -100},
				{-100, -100, -100, -100, -100, -100, -100, -100},
			},
			v: [][]int8{
				{-100, -100, -100, -100, -100, -100, -100, -100},
				{-100, -100, -100, -100, -100, -100, -100, -100},
			},
			expected: [][]int8{
				{-100, -100, -100, -100, -100, -100, -100, -100},
				{-100, -100, -100, -100, -100, -100, -100, -100},
			},
		},
		{
			name:    "attention with mixed values",
			seqLen:  2,
			headDim: 8,
			q: [][]int8{
				{50, -50, 25, -25, 50, -50, 25, -25},
				{-25, 25, -50, 50, -25, 25, -50, 50},
			},
			k: [][]int8{
				{50, -50, 25, -25, 50, -50, 25, -25},
				{-25, 25, -50, 50, -25, 25, -50, 50},
			},
			v: [][]int8{
				{50, -50, 25, -25, 50, -50, 25, -25},
				{-25, 25, -50, 50, -25, 25, -50, 50},
			},
			expected: [][]int8{
				{50, -50, 25, -25, 50, -50, 25, -25},
				{-25, 25, -50, 50, -25, 25, -50, 50},
			},
		},
		{
			name:    "attention with non-multiple of 4 head_dim",
			seqLen:  2,
			headDim: 8,
			q: [][]int8{
				{1, 2, 3, 4, 5, 6, 7, 8},
				{8, 7, 6, 5, 4, 3, 2, 1},
			},
			k: [][]int8{
				{1, 2, 3, 4, 5, 6, 7, 8},
				{8, 7, 6, 5, 4, 3, 2, 1},
			},
			v: [][]int8{
				{1, 2, 3, 4, 5, 6, 7, 8},
				{8, 7, 6, 5, 4, 3, 2, 1},
			},
			expected: [][]int8{
				{1, 2, 3, 4, 5, 6, 7, 8},
				{8, 7, 6, 5, 4, 3, 2, 1},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input tensors as 4D: [1, 1, seqLen, headDim]
			q := tensor.NewTensor(1, 1, tt.seqLen, tt.headDim)
			k := tensor.NewTensor(1, 1, tt.seqLen, tt.headDim)
			v := tensor.NewTensor(1, 1, tt.seqLen, tt.headDim)

			// Fill tensors with test data
			for i := 0; i < tt.seqLen; i++ {
				for j := 0; j < tt.headDim; j++ {
					q.Set(tt.q[i][j], 0, 0, i, j)
					k.Set(tt.k[i][j], 0, 0, i, j)
					v.Set(tt.v[i][j], 0, 0, i, j)
				}
			}

			// Compute attention
			output, err := ScaledDotProductAttention(q, k, v)
			if err != nil {
				t.Fatalf("ScaledDotProductAttention failed: %v", err)
			}

			// Verify output shape
			if len(output.Shape()) != 4 {
				t.Errorf("output shape = %v, want 4 dimensions", output.Shape())
			}
			if output.Shape()[0] != 1 || output.Shape()[1] != 1 || output.Shape()[2] != tt.seqLen || output.Shape()[3] != tt.headDim {
				t.Errorf("output shape = %v, want [1 1 %d %d]", output.Shape(), tt.seqLen, tt.headDim)
			}

			// Verify output values
			for i := 0; i < tt.seqLen; i++ {
				for j := 0; j < tt.headDim; j++ {
					got := output.Get(0, 0, i, j)
					want := tt.expected[i][j]
					if got != want {
						t.Errorf("output[0][0][%d][%d] = %d, want %d", i, j, got, want)
					}
				}
			}
		})
	}
}

func TestScaledDotProductAttentionErrors(t *testing.T) {
	tests := []struct {
		name string
		q    *tensor.Tensor
		k    *tensor.Tensor
		v    *tensor.Tensor
	}{
		{
			name: "mismatched head dimensions",
			q:    tensor.NewTensor(2, 3),
			k:    tensor.NewTensor(2, 4),
			v:    tensor.NewTensor(2, 3),
		},
		{
			name: "mismatched sequence lengths",
			q:    tensor.NewTensor(2, 3),
			k:    tensor.NewTensor(3, 3),
			v:    tensor.NewTensor(2, 3),
		},
		{
			name: "non-2D tensors",
			q:    tensor.NewTensor(2, 3, 4),
			k:    tensor.NewTensor(2, 3),
			v:    tensor.NewTensor(2, 3),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ScaledDotProductAttention(tt.q, tt.k, tt.v)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

func BenchmarkScaledDotProductAttention(b *testing.B) {
	benchmarks := []struct {
		name    string
		seqLen  int
		headDim int
	}{
		{
			name:    "small",
			seqLen:  32,
			headDim: 32,
		},
		{
			name:    "medium",
			seqLen:  128,
			headDim: 64,
		},
		{
			name:    "large",
			seqLen:  512,
			headDim: 128,
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			q := tensor.NewTensor(bm.seqLen, bm.headDim)
			k := tensor.NewTensor(bm.seqLen, bm.headDim)
			v := tensor.NewTensor(bm.seqLen, bm.headDim)

			// Fill with pseudo-random but deterministic data
			for i := 0; i < bm.seqLen; i++ {
				for j := 0; j < bm.headDim; j++ {
					q.Set(int8((i+j)%8-4), i, j)
					k.Set(int8((i-j)%8-4), i, j)
					v.Set(int8((i*j)%8-4), i, j)
				}
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = ScaledDotProductAttention(q, k, v)
			}
		})
	}
}
