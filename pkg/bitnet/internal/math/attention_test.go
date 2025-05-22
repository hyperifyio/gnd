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
	seqLen := 128
	headDim := 64

	q := tensor.NewTensor(seqLen, headDim)
	k := tensor.NewTensor(seqLen, headDim)
	v := tensor.NewTensor(seqLen, headDim)

	// Fill with pseudo-random but deterministic data
	for i := 0; i < seqLen; i++ {
		for j := 0; j < headDim; j++ {
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
}
