package math

import (
	"errors"
	"math"
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.

var (
	ErrInputTensorsMustBe4D = errors.New("attention: input tensors must be 4D")
	ErrMismatchedSeqLengths = errors.New("attention: mismatched sequence lengths")
)

// ScaledDotProductAttention implements the scaled dot-product attention mechanism
// as described in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).
//
// The function computes attention weights using the formula:
//
//	Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
//
// Input tensors must be 4D with shape [batch_size, num_heads, seq_len, head_dim]:
//   - q: Query matrix
//   - k: Key matrix
//   - v: Value matrix
//
// All input tensors must have matching dimensions:
//   - Same batch_size
//   - Same num_heads
//   - Same seq_len
//   - Same head_dim
//
// Returns a 4D tensor with shape [batch_size, num_heads, seq_len, head_dim]
// containing the attention-weighted values.
//
// The function performs the following steps:
//  1. Computes dot products between queries and keys
//  2. Scales the dot products by 1/sqrt(head_dim)
//  3. Applies softmax to get attention weights
//  4. Computes weighted sum of values
//
// The computation is parallelized across batch elements for better performance.
// All intermediate computations use float32 for numerical stability,
// with final results clamped to int8 range [-128, 127].
func ScaledDotProductAttention(q, k, v *tensor.Tensor) (*tensor.Tensor, error) {
	// Validate input shapes
	if len(q.Shape()) != 4 || len(k.Shape()) != 4 || len(v.Shape()) != 4 {
		return nil, ErrInputTensorsMustBe4D
	}

	batchSize := q.Shape()[0]
	numHeads := q.Shape()[1]
	seqLen := q.Shape()[2]
	headDim := q.Shape()[3]

	// Validate head dimension
	if headDim < 8 || headDim > 256 {
		tensor.DebugLog("invalid head dimensions: head dimension must be between 8 and 256, got %d", headDim)
		return nil, ErrInvalidHeadDimension
	}

	// Validate sequence lengths
	if k.Shape()[2] != seqLen || v.Shape()[2] != seqLen {
		tensor.DebugLog("mismatched sequence lengths: q=%d, k=%d, v=%d", seqLen, k.Shape()[2], v.Shape()[2])
		return nil, ErrMismatchedSeqLengths
	}

	// Create output tensor
	output := tensor.NewTensor(batchSize, numHeads, seqLen, headDim)

	// Process in parallel chunks with a reasonable chunk size
	var wg sync.WaitGroup
	numCPU := runtime.NumCPU()
	chunkSize := (batchSize + numCPU - 1) / numCPU
	if chunkSize < 1 {
		chunkSize = 1
	}

	// Create a channel to collect errors
	errChan := make(chan error, numCPU)

	for i := 0; i < batchSize; i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > batchSize {
				end = batchSize
			}

			// Process each batch element
			for b := start; b < end; b++ {
				for h := 0; h < numHeads; h++ {
					// Compute attention scores for all positions at once
					scores := make([]float32, seqLen*seqLen)
					for s1 := 0; s1 < seqLen; s1++ {
						for s2 := 0; s2 < seqLen; s2++ {
							score := float32(0)
							for d := 0; d < headDim; d++ {
								qVal := float32(q.Get(b, h, s1, d))
								kVal := float32(k.Get(b, h, s2, d))
								score += qVal * kVal
							}
							// Scale by 1/sqrt(head_dim)
							score /= float32(math.Sqrt(float64(headDim)))
							scores[s1*seqLen+s2] = score
						}
					}

					// Compute softmax with numerical stability
					for s1 := 0; s1 < seqLen; s1++ {
						// Find max score for numerical stability
						maxScore := scores[s1*seqLen]
						for s2 := 1; s2 < seqLen; s2++ {
							if scores[s1*seqLen+s2] > maxScore {
								maxScore = scores[s1*seqLen+s2]
							}
						}

						// Compute exp and sum
						var sumExp float32
						for s2 := 0; s2 < seqLen; s2++ {
							scores[s1*seqLen+s2] = float32(math.Exp(float64(scores[s1*seqLen+s2] - maxScore)))
							sumExp += scores[s1*seqLen+s2]
						}

						// Normalize
						for s2 := 0; s2 < seqLen; s2++ {
							scores[s1*seqLen+s2] /= sumExp
						}
					}

					// Apply attention to values
					for s1 := 0; s1 < seqLen; s1++ {
						for d := 0; d < headDim; d++ {
							var val float32
							for s2 := 0; s2 < seqLen; s2++ {
								val += scores[s1*seqLen+s2] * float32(v.Get(b, h, s2, d))
							}
							// Clamp to int8 range, saturating for large values
							if val >= 127 {
								val = 127
							} else if val <= -128 {
								val = -128
							}
							output.Set(int8(val), b, h, s1, d)
						}
					}
				}
			}
		}(i)
	}

	// Wait for all goroutines to complete
	wg.Wait()

	// Check for errors
	select {
	case err := <-errChan:
		output.Close()
		return nil, err
	default:
		return output, nil
	}
}
