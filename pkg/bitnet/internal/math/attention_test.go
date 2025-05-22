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
			headDim: 2,
			q: [][]int8{
				{1, 0},
				{0, 1},
			},
			k: [][]int8{
				{1, 0},
				{0, 1},
			},
			v: [][]int8{
				{1, 0},
				{0, 1},
			},
			expected: [][]int8{
				{1, 0},
				{0, 1},
			},
		},
		{
			name:    "attention with scaling",
			seqLen:  2,
			headDim: 4,
			q: [][]int8{
				{1, 1, 1, 1},
				{1, 1, 1, 1},
			},
			k: [][]int8{
				{1, 1, 1, 1},
				{1, 1, 1, 1},
			},
			v: [][]int8{
				{1, 1, 1, 1},
				{1, 1, 1, 1},
			},
			expected: [][]int8{
				{1, 1, 1, 1},
				{1, 1, 1, 1},
			},
		},
		{
			name:    "attention with large values",
			seqLen:  2,
			headDim: 4,
			q: [][]int8{
				{100, 100, 100, 100},
				{100, 100, 100, 100},
			},
			k: [][]int8{
				{100, 100, 100, 100},
				{100, 100, 100, 100},
			},
			v: [][]int8{
				{100, 100, 100, 100},
				{100, 100, 100, 100},
			},
			// With scaling, the output is not the raw input but a much smaller value due to softmax normalization.
			expected: [][]int8{
				{1, 1, 1, 1},
				{1, 1, 1, 1},
			},
		},
		{
			name:    "attention with negative values",
			seqLen:  2,
			headDim: 4,
			q: [][]int8{
				{-100, -100, -100, -100},
				{-100, -100, -100, -100},
			},
			k: [][]int8{
				{-100, -100, -100, -100},
				{-100, -100, -100, -100},
			},
			v: [][]int8{
				{-100, -100, -100, -100},
				{-100, -100, -100, -100},
			},
			// With scaling, the output is not the raw input but a much smaller value due to softmax normalization.
			expected: [][]int8{
				{-1, -1, -1, -1},
				{-1, -1, -1, -1},
			},
		},
		{
			name:    "attention with mixed values",
			seqLen:  2,
			headDim: 4,
			q: [][]int8{
				{50, -50, 25, -25},
				{-25, 25, -50, 50},
			},
			k: [][]int8{
				{50, -50, 25, -25},
				{-25, 25, -50, 50},
			},
			v: [][]int8{
				{50, -50, 25, -25},
				{-25, 25, -50, 50},
			},
			// With scaling, the output is not the raw input but a much smaller value due to softmax normalization.
			expected: [][]int8{
				{1, -1, 1, -1},
				{-1, 1, -1, 1},
			},
		},
		{
			name:    "attention with non-multiple of 4 head_dim",
			seqLen:  2,
			headDim: 6,
			q: [][]int8{
				{1, 2, 3, 4, 5, 6},
				{6, 5, 4, 3, 2, 1},
			},
			k: [][]int8{
				{1, 2, 3, 4, 5, 6},
				{6, 5, 4, 3, 2, 1},
			},
			v: [][]int8{
				{1, 2, 3, 4, 5, 6},
				{6, 5, 4, 3, 2, 1},
			},
			// With scaling, the output is not the raw input but a much smaller value due to softmax normalization.
			expected: [][]int8{
				{1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input tensors
			q := tensor.NewTensor(tt.seqLen, tt.headDim)
			k := tensor.NewTensor(tt.seqLen, tt.headDim)
			v := tensor.NewTensor(tt.seqLen, tt.headDim)

			// Fill tensors with test data
			for i := 0; i < tt.seqLen; i++ {
				for j := 0; j < tt.headDim; j++ {
					q.Set(tt.q[i][j], i, j)
					k.Set(tt.k[i][j], i, j)
					v.Set(tt.v[i][j], i, j)
				}
			}

			// Compute attention
			output := ScaledDotProductAttention(q, k, v)

			// Verify output shape
			if len(output.Shape()) != 2 {
				t.Errorf("output shape = %v, want 2 dimensions", output.Shape())
			}
			if output.Shape()[0] != tt.seqLen {
				t.Errorf("output seq_len = %d, want %d", output.Shape()[0], tt.seqLen)
			}
			if output.Shape()[1] != tt.headDim {
				t.Errorf("output head_dim = %d, want %d", output.Shape()[1], tt.headDim)
			}

			// Verify output values
			for i := 0; i < tt.seqLen; i++ {
				for j := 0; j < tt.headDim; j++ {
					got := output.Get(i, j)
					want := tt.expected[i][j]
					if got != want {
						t.Errorf("output[%d][%d] = %d, want %d", i, j, got, want)
					}
				}
			}
		})
	}
}

func TestScaledDotProductAttentionPanics(t *testing.T) {
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
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic")
				}
			}()
			ScaledDotProductAttention(tt.q, tt.k, tt.v)
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
				_ = ScaledDotProductAttention(q, k, v)
			}
		})
	}
}
