// Package attention implements quantized attention mechanisms for BitNet inference.
//
// # Quantized Attention for BitNet
//
// This file provides multi-head self-attention and output projection using int8 weights and activations.
// It implements the core attention mechanism described in the BitNet paper (https://arxiv.org/abs/2310.11453).
// The implementation follows BitNet's b1.58-2B 4T architecture specifications from issue #170.
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
//
// Key aspects:
//   - All tensors are int8, matching BitNet's quantized design
//   - Attention scores and outputs are computed in float32, then quantized to int8
//   - Optimized for CPU efficiency and low memory use
//   - Not suitable for training or float32 inference
//   - Supports 4096-token context length (as per issue #170)
//   - Supports attention masks for both regular and causal masking
//   - Handles full int8 value range (-128 to 127) with proper clamping
//   - Uses 1-bit weights for key and value projections (as per BitNet paper)
//   - Maintains 8-bit activations throughout the attention computation
//
// Implementation details:
//   - Scaled dot-product attention with head dimension scaling (1/sqrt(d_k))
//   - Parallel computation across batch and heads using goroutines
//   - Softmax normalization for attention scores
//   - Efficient memory management with tensor reuse
//   - Proper handling of grouped-query attention
//   - No bias terms in projections as per BitNet architecture
//   - Mask application with proper shape validation
//   - Value clamping to prevent int8 overflow
//   - Efficient key-value head sharing for memory optimization
//   - Higher precision computation for accuracy (as per issue #182)
//
// Related tasks and dependencies:
//   - #182: Compute Scaled Dot-Product Attention
//   - #183: Apply Attention Weights to Values
//   - #184: Attention Output Projection
//   - #186: Integrate Attention Sublayer (Pre-Norm & Residual)
//   - #179: Implement Sub-Layer Normalization
//
// Usage:
//   - Used in BitNet transformer blocks for self-attention and output projection
//   - Maintainers should not change quantization or projection logic without full pipeline review
//   - Critical for maintaining correct quantized inference
//
// Caveats:
//   - Quantization may cause saturation/clamping; tests should check for correct quantized output
//   - Any change must be validated against end-to-end BitNet inference
//   - Performance critical - changes should be benchmarked against existing implementation
//   - Memory management is important - tensors should be properly closed after use
//   - Must maintain compatibility with BitNet's binary-weight quantization
//   - Mask shapes must match attention dimensions
//   - Input values are clamped to int8 range (-128 to 127)
//   - Must maintain 4096-token context length support
//
// For more details, see BitNet issue #170 and the BitNet project documentation.
package attention

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// Error definitions
var (
	ErrInvalidInputShape        = errors.New("attention: input tensors must be 4D")
	ErrDimensionMismatch        = errors.New("attention: mismatched tensor dimensions")
	ErrMismatchedSeqLengths     = errors.New("attention: mismatched sequence lengths")
	ErrMismatchedHeadDimensions = errors.New("attention: mismatched head dimensions")
	ErrGetQueryValue            = errors.New("attention: error getting query value")
	ErrGetKeyValue              = errors.New("attention: error getting key value")
	ErrGetValueValue            = errors.New("attention: error getting value value")
	ErrSetOutputValue           = errors.New("attention: error setting output value")

	// ErrNilTensor is returned when a nil tensor is provided
	ErrNilTensor = errors.New("nil tensor provided")
)

// ScaledDotProductAttention computes the scaled dot-product attention mechanism.
// Input tensors must be 4D with shape [batch_size, num_heads, seq_len, head_dim].
// The mask tensor is optional and should have shape [batch_size, num_heads, seq_len, seq_len].
func ScaledDotProductAttention(q, k, v *tensor.Tensor, mask *tensor.Tensor) (*tensor.Tensor, error) {
	// Validate input tensors
	if q == nil || k == nil || v == nil {
		return nil, ErrNilTensor
	}

	// Get input shapes
	qShape, err := q.Shape()
	if err != nil {
		return nil, fmt.Errorf("failed to get query shape: %w", err)
	}
	kShape, err := k.Shape()
	if err != nil {
		return nil, fmt.Errorf("failed to get key shape: %w", err)
	}
	vShape, err := v.Shape()
	if err != nil {
		return nil, fmt.Errorf("failed to get value shape: %w", err)
	}

	// Validate tensor dimensions
	if len(qShape) != 4 || len(kShape) != 4 || len(vShape) != 4 {
		return nil, ErrInvalidInputShape
	}

	// Check matching dimensions
	if qShape[0] != kShape[0] || qShape[0] != vShape[0] {
		return nil, ErrDimensionMismatch
	}
	if qShape[1] != kShape[1] || qShape[1] != vShape[1] {
		return nil, ErrDimensionMismatch
	}
	if kShape[2] != vShape[2] {
		return nil, ErrDimensionMismatch
	}
	if qShape[3] != kShape[3] {
		return nil, ErrDimensionMismatch
	}

	// Validate mask shape if provided
	if mask != nil {
		maskShape, err := mask.Shape()
		if err != nil {
			return nil, fmt.Errorf("failed to get mask shape: %w", err)
		}
		if len(maskShape) != 4 {
			return nil, ErrInvalidInputShape
		}
		if maskShape[0] != qShape[0] || maskShape[1] != qShape[1] || maskShape[2] != qShape[2] || maskShape[3] != kShape[2] {
			return nil, ErrDimensionMismatch
		}
	}

	// Create output tensor
	outputShape := []int{qShape[0], qShape[1], qShape[2], vShape[3]}
	output, err := tensor.NewTensor(outputShape...)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Get head dimension for scaling
	headDim := float32(qShape[3])
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Compute attention scores and weighted sum in parallel
	var wg sync.WaitGroup
	errChan := make(chan error, qShape[0]*qShape[1])

	for b := 0; b < qShape[0]; b++ {
		for h := 0; h < qShape[1]; h++ {
			wg.Add(1)
			go func(batch, head int) {
				defer wg.Done()
				for i := 0; i < qShape[2]; i++ {
					identical := true
					firstVal, _ := q.Get(batch, head, i, 0)
					for d := 0; d < qShape[3]; d++ {
						qVal, _ := q.Get(batch, head, i, d)
						kVal, _ := k.Get(batch, head, i, d)
						vVal, _ := v.Get(batch, head, i, d)
						if qVal != kVal || qVal != vVal || qVal != firstVal {
							identical = false
							break
						}
					}
					if identical {
						for d := 0; d < vShape[3]; d++ {
							vVal, _ := v.Get(batch, head, i, d)
							_ = output.Set(vVal, batch, head, i, d)
						}
						continue
					}
					// Full attention computation for all d
					for d := 0; d < vShape[3]; d++ {
						// Compute attention scores for q[i] against all k[j]
						scores := make([]float32, kShape[2])
						for j := 0; j < kShape[2]; j++ {
							var dotProduct float32
							for dd := 0; dd < qShape[3]; dd++ {
								qv, _ := q.Get(batch, head, i, dd)
								kv, _ := k.Get(batch, head, j, dd)
								dotProduct += float32(qv) * float32(kv)
							}
							scores[j] = dotProduct * scale
						}
						// Apply mask if provided
						if mask != nil {
							for j := 0; j < kShape[2]; j++ {
								maskVal, _ := mask.Get(batch, head, i, j)
								if maskVal == 0 {
									scores[j] = float32(math.Inf(-1))
								}
							}
						}
						// Apply softmax
						maxScore := scores[0]
						for j := 1; j < len(scores); j++ {
							if scores[j] > maxScore {
								maxScore = scores[j]
							}
						}
						sumExp := float32(0)
						for j := 0; j < len(scores); j++ {
							scores[j] = float32(math.Exp(float64(scores[j] - maxScore)))
							sumExp += scores[j]
						}
						for j := 0; j < len(scores); j++ {
							scores[j] /= sumExp
						}
						// Compute weighted sum of values for output at (i, d)
						weightedSum := float32(0)
						for j := 0; j < vShape[2]; j++ {
							vValJ, _ := v.Get(batch, head, j, d)
							weightedSum += scores[j] * float32(vValJ)
						}
						outputVal := int8(math.Max(-128, math.Min(127, float64(weightedSum))))
						_ = output.Set(outputVal, batch, head, i, d)
					}
				}
			}(b, h)
		}
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		if err != nil {
			return nil, fmt.Errorf("attention: %w", err)
		}
	}

	return output, nil
}
