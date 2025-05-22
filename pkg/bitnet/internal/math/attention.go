package math

import (
	"math"
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// ScaledDotProductAttention computes the attention weights and output
// for a single attention head using scaled dot-product attention.
// q: [batch_size, num_heads, seq_len, head_dim] - Query matrix
// k: [batch_size, num_heads, seq_len, head_dim] - Key matrix
// v: [batch_size, num_heads, seq_len, head_dim] - Value matrix
// Returns: [batch_size, num_heads, seq_len, head_dim] - Attention output
func ScaledDotProductAttention(q, k, v *tensor.Tensor) *tensor.Tensor {
	if len(q.Shape()) != 4 || len(k.Shape()) != 4 || len(v.Shape()) != 4 {
		panic("q, k, v must be 4D tensors [batch_size, num_heads, seq_len, head_dim]")
	}
	if q.Shape()[0] != k.Shape()[0] || k.Shape()[0] != v.Shape()[0] {
		panic("batch sizes must match")
	}
	if q.Shape()[1] != k.Shape()[1] || k.Shape()[1] != v.Shape()[1] {
		panic("number of heads must match")
	}
	if q.Shape()[2] != k.Shape()[2] || k.Shape()[2] != v.Shape()[2] {
		panic("sequence lengths must match")
	}
	if q.Shape()[3] != k.Shape()[3] || k.Shape()[3] != v.Shape()[3] {
		panic("head dimensions must match")
	}

	batchSize := q.Shape()[0]
	numHeads := q.Shape()[1]
	seqLen := q.Shape()[2]
	headDim := q.Shape()[3]

	// Create output tensor
	output := tensor.NewTensor(batchSize, numHeads, seqLen, headDim)

	// Process in parallel chunks
	var wg sync.WaitGroup
	chunkSize := batchSize / runtime.NumCPU()
	if chunkSize < 1 {
		chunkSize = 1
	}

	for b := 0; b < batchSize; b += chunkSize {
		wg.Add(1)
		go func(startBatch int) {
			defer wg.Done()
			endBatch := startBatch + chunkSize
			if endBatch > batchSize {
				endBatch = batchSize
			}

			// For each batch element
			for b := startBatch; b < endBatch; b++ {
				// For each attention head
				for h := 0; h < numHeads; h++ {
					// Pre-allocate slices for scores and weights
					scores := make([][]float32, seqLen)
					for i := range scores {
						scores[i] = make([]float32, seqLen)
					}
					weights := make([][]float32, seqLen)
					for i := range weights {
						weights[i] = make([]float32, seqLen)
					}

					// Compute dot products
					for i := 0; i < seqLen; i++ {
						for j := 0; j < seqLen; j++ {
							var sum float32
							// Compute dot product between q[b,h,i] and k[b,h,j]
							for d := 0; d < headDim; d++ {
								sum += float32(q.Get(b, h, i, d)) * float32(k.Get(b, h, j, d))
							}
							// Scale by 1/sqrt(head_dim)
							scores[i][j] = sum / float32(math.Sqrt(float64(headDim)))
						}
					}

					// Apply softmax to get attention weights
					for i := 0; i < seqLen; i++ {
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

					// Compute weighted sum of values
					for i := 0; i < seqLen; i++ {
						for d := 0; d < headDim; d++ {
							var sum float32
							for j := 0; j < seqLen; j++ {
								sum += weights[i][j] * float32(v.Get(b, h, j, d))
							}
							// Clamp to int8 range and convert back to int8
							output.Set(int8(min(max(int32(math.Round(float64(sum))), -128), 127)), b, h, i, d)
						}
					}
				}
			}
		}(b)
	}
	wg.Wait()

	return output
}

// min returns the minimum of two int32 values
func min(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two int32 values
func max(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}
