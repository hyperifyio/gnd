package math

import (
	"math"
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// ScaledDotProductAttention computes the attention weights and output
// for a single attention head using scaled dot-product attention.
// q: [seq_len, head_dim] - Query matrix
// k: [seq_len, head_dim] - Key matrix
// v: [seq_len, head_dim] - Value matrix
// Returns: [seq_len, head_dim] - Attention output
func ScaledDotProductAttention(q, k, v *tensor.Tensor) *tensor.Tensor {
	if len(q.Shape()) != 2 || len(k.Shape()) != 2 || len(v.Shape()) != 2 {
		panic("q, k, v must be 2D tensors")
	}
	if q.Shape()[1] != k.Shape()[1] || k.Shape()[1] != v.Shape()[1] {
		panic("head dimensions must match")
	}
	if q.Shape()[0] != k.Shape()[0] || k.Shape()[0] != v.Shape()[0] {
		panic("sequence lengths must match")
	}

	seqLen := q.Shape()[0]
	headDim := q.Shape()[1]

	// Pre-allocate slices for scores and weights to avoid repeated allocations
	scores := make([][]float32, seqLen)
	for i := range scores {
		scores[i] = make([]float32, seqLen)
	}
	weights := make([][]float32, seqLen)
	for i := range weights {
		weights[i] = make([]float32, seqLen)
	}

	// Process in parallel chunks
	var wg sync.WaitGroup
	chunkSize := seqLen / runtime.NumCPU()
	if chunkSize < 1 {
		chunkSize = 1
	}

	// Compute dot products
	for i := 0; i < seqLen; i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > seqLen {
				end = seqLen
			}

			for i := start; i < end; i++ {
				for j := 0; j < seqLen; j++ {
					var sum float32
					// Compute dot product between q[i] and k[j]
					for d := 0; d < headDim; d++ {
						sum += float32(q.Get(i, d)) * float32(k.Get(j, d))
					}
					// Scale by 1/sqrt(head_dim)
					scores[i][j] = sum / float32(math.Sqrt(float64(headDim)))
				}
			}
		}(i)
	}
	wg.Wait()

	// Apply softmax to get attention weights
	for i := 0; i < seqLen; i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > seqLen {
				end = seqLen
			}

			for i := start; i < end; i++ {
				// Find max for numerical stability
				maxScore := scores[i][0]
				for j := 1; j < seqLen; j++ {
					if scores[i][j] > maxScore {
						maxScore = scores[i][j]
					}
				}

				// Compute exp and sum
				var sum float32
				for j := 0; j < seqLen; j++ {
					weights[i][j] = float32(math.Exp(float64(scores[i][j] - maxScore)))
					sum += weights[i][j]
				}

				// Normalize
				for j := 0; j < seqLen; j++ {
					weights[i][j] /= sum
				}
			}
		}(i)
	}
	wg.Wait()

	// Compute weighted sum of values
	output := tensor.NewTensor(seqLen, headDim)
	for i := 0; i < seqLen; i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > seqLen {
				end = seqLen
			}

			for i := start; i < end; i++ {
				for d := 0; d < headDim; d++ {
					var sum float32
					for j := 0; j < seqLen; j++ {
						sum += weights[i][j] * float32(v.Get(j, d))
					}
					// Convert back to int8
					output.Set(int8(math.Round(float64(sum))), i, d)
				}
			}
		}(i)
	}
	wg.Wait()

	return output
}
