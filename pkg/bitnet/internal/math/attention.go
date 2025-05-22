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
					// Process 4 elements at a time for better cache utilization
					d := 0
					for ; d+3 < headDim; d += 4 {
						q0 := float32(q.Get(i, d))
						q1 := float32(q.Get(i, d+1))
						q2 := float32(q.Get(i, d+2))
						q3 := float32(q.Get(i, d+3))
						k0 := float32(k.Get(j, d))
						k1 := float32(k.Get(j, d+1))
						k2 := float32(k.Get(j, d+2))
						k3 := float32(k.Get(j, d+3))
						sum += q0*k0 + q1*k1 + q2*k2 + q3*k3
					}
					// Process remaining elements
					for ; d < headDim; d++ {
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

	// Compute weighted sum of values using higher precision for accumulation
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
				// Process 4 dimensions at a time for better cache utilization
				d := 0
				for ; d+3 < headDim; d += 4 {
					var sum0, sum1, sum2, sum3 float32
					// Accumulate in higher precision (float32)
					for j := 0; j < seqLen; j++ {
						w := weights[i][j]
						v0 := float32(v.Get(j, d))
						v1 := float32(v.Get(j, d+1))
						v2 := float32(v.Get(j, d+2))
						v3 := float32(v.Get(j, d+3))
						sum0 += w * v0
						sum1 += w * v1
						sum2 += w * v2
						sum3 += w * v3
					}
					// Clamp to int8 range and convert back to int8
					output.Set(int8(min(max(int32(math.Round(float64(sum0))), -128), 127)), i, d)
					output.Set(int8(min(max(int32(math.Round(float64(sum1))), -128), 127)), i, d+1)
					output.Set(int8(min(max(int32(math.Round(float64(sum2))), -128), 127)), i, d+2)
					output.Set(int8(min(max(int32(math.Round(float64(sum3))), -128), 127)), i, d+3)
				}
				// Process remaining dimensions
				for ; d < headDim; d++ {
					var sum float32
					for j := 0; j < seqLen; j++ {
						sum += weights[i][j] * float32(v.Get(j, d))
					}
					output.Set(int8(min(max(int32(math.Round(float64(sum))), -128), 127)), i, d)
				}
			}
		}(i)
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
