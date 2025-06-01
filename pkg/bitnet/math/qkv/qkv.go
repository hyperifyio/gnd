// Package qkv implements quantized QKV projection for BitNet attention.
//
// # Quantized QKV Projection for BitNet
//
// This package provides QKV projection matrices for multi-head self-attention, using int8 weights.
// It implements the QKV projection described in the BitNet paper (https://arxiv.org/abs/2310.11453).
//
// Key aspects:
//   - All projection weights are int8, matching BitNet's quantized design
//   - Supports grouped-query attention (GQA) for efficient inference
//   - Optimized for CPU efficiency and low memory use
//   - Not suitable for training or float32 inference
//
// Implementation details:
//   - Q, K, V projections with proper head dimensions
//   - Support for both standard and grouped-query attention
//   - Efficient batch processing and tensor management
//
// Related tasks and dependencies:
//   - #181: Implement QKV Projection (Core implementation)
//   - #182: Compute Scaled Dot-Product Attention (Depends on #181)
//   - #183: Apply Attention Weights to Values (Depends on #181)
//   - #186: Integrate Attention Sublayer (Pre-Norm & Residual) (Depends on #181)
//
// Usage:
//   - Used in BitNet attention blocks for Q, K, V projections
//   - Maintainers should not change quantization or projection logic without full pipeline review
//
// Caveats:
//   - Quantization may cause saturation/clamping; tests should check for correct quantized output
//   - Any change must be validated against end-to-end BitNet inference
//   - Performance critical - changes should be benchmarked against existing implementation
//   - Memory management is important - tensors should be properly closed after use
//
// For more details, see BitNet issue #190 and the BitNet project documentation.
package qkv

import (
	"errors"
	"github.com/hyperifyio/gnd/pkg/bitnet/math/linear"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

var (

	// ErrInvalidShape is returned when a tensor has an invalid shape
	ErrInvalidShape = errors.New("invalid tensor shape")

	// ErrInvalidHiddenDim is returned when the hidden dimension is invalid
	ErrInvalidHiddenDim = errors.New("invalid hidden dimension")
)

// QKVProjection represents the Query, Key, and Value projection matrices
// for multi-head self-attention.
//
// This structure manages the projection weights and provides methods to
// project input hidden states into Q, K, and V tensors for use in the
// attention mechanism. It supports grouped-query attention (GQA) by
// allowing a different number of key/value heads than query heads.
//
// The implementation is optimized for efficient computation and supports
// both single-token and multi-token input shapes.
type QKVProjection struct {
	// Number of attention heads
	numHeads int
	// Number of key/value heads (for grouped-query attention)
	numKVHeads int
	// Dimension of each head
	headDim int
	// Hidden dimension
	hiddenDim int
	// Projection matrices for query, key, and value
	qProj *linear.Linear
	kProj *linear.Linear
	vProj *linear.Linear
}

// NewQKVProjection creates a new QKV projection with the given parameters.
//
// Parameters:
//   - hiddenDim: Size of the hidden dimension
//   - numHeads: Number of query heads
//   - numKVHeads: Number of key/value heads (for GQA)
//
// The projection matrices are initialized with the correct shapes for Q, K, and V.
// The structure supports both standard and grouped-query attention.
func NewQKVProjection(hiddenDim, numHeads, numKVHeads int) (*QKVProjection, error) {
	headDim := hiddenDim / numHeads
	kvHeadDim := hiddenDim / numKVHeads

	// Create projection matrices with correct shapes
	// Q projection: [hidden_dim, num_heads * head_dim]
	// K projection: [hidden_dim, num_kv_heads * kv_head_dim]
	// V projection: [hidden_dim, num_kv_heads * kv_head_dim]
	qProj, err := linear.NewLinear(hiddenDim, numHeads*headDim)
	if err != nil {
		return nil, err
	}
	kProj, err := linear.NewLinear(hiddenDim, numKVHeads*kvHeadDim)
	if err != nil {
		return nil, err
	}
	vProj, err := linear.NewLinear(hiddenDim, numKVHeads*kvHeadDim)
	if err != nil {
		return nil, err
	}

	return &QKVProjection{
		numHeads:   numHeads,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		hiddenDim:  hiddenDim,
		qProj:      qProj,
		kProj:      kProj,
		vProj:      vProj,
	}, nil
}

// Project performs the QKV projection on the input hidden states.
//
// Input tensor must be either:
//   - 2D [batch_size, hidden_dim] for single-token inputs
//   - 3D [batch_size, seq_len, hidden_dim] for multi-token inputs
//
// The function:
// 1. Validates input shape and dimensions
// 2. Projects input into Q, K, and V using linear layers
// 3. Reshapes and splits projections into heads
// 4. Expands key/value heads if using grouped-query attention
//
// Returns Q, K, V tensors of shape [batch_size, num_heads, seq_len, head_dim].
// The implementation includes debug logging for tensor shapes and data lengths.
func (p *QKVProjection) Project(input *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	// Debug output for input tensor
	shape, err := input.Shape()
	if err != nil {
		return nil, nil, nil, err
	}
	loggers.Printf(loggers.Debug, "Input tensor shape: %v", shape)
	data, err := input.Data()
	if err != nil {
		return nil, nil, nil, err
	}
	loggers.Printf(loggers.Debug, "Input tensor data length: %d", len(data))

	// Get input dimensions
	var batchSize, seqLen, hiddenDim int
	if len(shape) == 2 {
		batchSize, hiddenDim = shape[0], shape[1]
		seqLen = 1
	} else if len(shape) == 3 {
		batchSize, seqLen, hiddenDim = shape[0], shape[1], shape[2]
	} else {
		loggers.Printf(loggers.Debug, "invalid input shape: %v", shape)
		return nil, nil, nil, ErrInvalidShape
	}

	// Check hidden dimension
	if hiddenDim != p.hiddenDim {
		loggers.Printf(loggers.Debug, "input hidden dimension %d does not match projection hidden dimension %d", hiddenDim, p.hiddenDim)
		return nil, nil, nil, ErrInvalidHiddenDim
	}

	// Create 2D view of input tensor for matrix multiplication
	input2d, err := tensor.NewTensor(batchSize*seqLen, hiddenDim)
	if err != nil {
		return nil, nil, nil, err
	}
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < hiddenDim; d++ {
				var val int8
				var ierr error
				if len(shape) == 2 {
					val, ierr = input.Get(b, d)
				} else {
					val, ierr = input.Get(b, s, d)
				}
				if ierr != nil {
					return nil, nil, nil, ierr
				}
				if setErr := input2d.Set(val, b*seqLen+s, d); setErr != nil {
					return nil, nil, nil, setErr
				}
			}
		}
	}

	// Debug output for 2D input tensor
	input2dShape, err := input2d.Shape()
	if err != nil {
		return nil, nil, nil, err
	}
	loggers.Printf(loggers.Debug, "2D input tensor shape: %v", input2dShape)
	input2dData, err := input2d.Data()
	if err != nil {
		return nil, nil, nil, err
	}
	loggers.Printf(loggers.Debug, "2D input tensor data length: %d", len(input2dData))

	// Apply linear transformations
	query, err := p.qProj.Forward(input2d)
	if err != nil {
		return nil, nil, nil, err
	}
	defer query.Close()

	key, err := p.kProj.Forward(input2d)
	if err != nil {
		return nil, nil, nil, err
	}
	defer key.Close()

	value, err := p.vProj.Forward(input2d)
	if err != nil {
		return nil, nil, nil, err
	}
	defer value.Close()

	// Debug output for 2D projections
	queryShape, err := query.Shape()
	if err != nil {
		return nil, nil, nil, err
	}
	keyShape, err := key.Shape()
	if err != nil {
		return nil, nil, nil, err
	}
	valueShape, err := value.Shape()
	if err != nil {
		return nil, nil, nil, err
	}
	loggers.Printf(loggers.Debug, "Q 2D shape: %v", queryShape)
	loggers.Printf(loggers.Debug, "K 2D shape: %v", keyShape)
	loggers.Printf(loggers.Debug, "V 2D shape: %v", valueShape)

	// Create output tensors with correct shapes [batch, num_heads, seq_len, head_dim]
	q, err := tensor.NewTensor(batchSize, p.numHeads, seqLen, p.headDim)
	if err != nil {
		return nil, nil, nil, err
	}
	k, err := tensor.NewTensor(batchSize, p.numKVHeads, seqLen, p.headDim)
	if err != nil {
		return nil, nil, nil, err
	}
	v, err := tensor.NewTensor(batchSize, p.numKVHeads, seqLen, p.headDim)
	if err != nil {
		return nil, nil, nil, err
	}

	// Copy data from 2D projections to output tensors, properly splitting into heads
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			// For query heads
			for h := 0; h < p.numHeads; h++ {
				for d := 0; d < p.headDim; d++ {
					// Calculate the correct index in the 2D projection
					idx := b*seqLen + s
					val, gerr := query.Get(idx, h*p.headDim+d)
					if gerr != nil {
						return nil, nil, nil, gerr
					}
					if setErr := q.Set(val, b, h, s, d); setErr != nil {
						return nil, nil, nil, setErr
					}
				}
			}
			// For key/value heads
			for h := 0; h < p.numKVHeads; h++ {
				for d := 0; d < p.headDim; d++ {
					// Calculate the correct index in the 2D projection
					idx := b*seqLen + s
					val, gerr := key.Get(idx, h*p.headDim+d)
					if gerr != nil {
						return nil, nil, nil, gerr
					}
					if setErr := k.Set(val, b, h, s, d); setErr != nil {
						return nil, nil, nil, setErr
					}
					val, gerr = value.Get(idx, h*p.headDim+d)
					if gerr != nil {
						return nil, nil, nil, gerr
					}
					if setErr := v.Set(val, b, h, s, d); setErr != nil {
						return nil, nil, nil, setErr
					}
				}
			}
		}
	}

	// Debug output for output tensors
	qShape, err := q.Shape()
	if err != nil {
		return nil, nil, nil, err
	}
	kShape, err := k.Shape()
	if err != nil {
		return nil, nil, nil, err
	}
	vShape, err := v.Shape()
	if err != nil {
		return nil, nil, nil, err
	}
	loggers.Printf(loggers.Debug, "Q output shape: %v", qShape)
	loggers.Printf(loggers.Debug, "K output shape: %v", kShape)
	loggers.Printf(loggers.Debug, "V output shape: %v", vShape)

	// Expand key/value heads if necessary
	if p.numKVHeads < p.numHeads {
		// Create expanded tensors with correct head dimensions
		expandedK, err := tensor.NewTensor(batchSize, p.numHeads, seqLen, p.headDim)
		if err != nil {
			return nil, nil, nil, err
		}
		expandedV, err := tensor.NewTensor(batchSize, p.numHeads, seqLen, p.headDim)
		if err != nil {
			return nil, nil, nil, err
		}

		// Copy and repeat heads
		for b := 0; b < batchSize; b++ {
			for h := 0; h < p.numHeads; h++ {
				// Use modulo to repeat heads
				srcHead := h % p.numKVHeads
				for s := 0; s < seqLen; s++ {
					for d := 0; d < p.headDim; d++ {
						val, gerr := k.Get(b, srcHead, s, d)
						if gerr != nil {
							return nil, nil, nil, gerr
						}
						if setErr := expandedK.Set(val, b, h, s, d); setErr != nil {
							return nil, nil, nil, setErr
						}
						val, gerr = v.Get(b, srcHead, s, d)
						if gerr != nil {
							return nil, nil, nil, gerr
						}
						if setErr := expandedV.Set(val, b, h, s, d); setErr != nil {
							return nil, nil, nil, setErr
						}
					}
				}
			}
		}
		k = expandedK
		v = expandedV
	}

	return q, k, v, nil
}

// SetWeights sets the weights for the QKV projection.
//
// Parameters:
//   - qWeights: Query projection weights [hidden_dim, num_heads * head_dim]
//   - kWeights: Key projection weights [hidden_dim, num_kv_heads * kv_head_dim]
//   - vWeights: Value projection weights [hidden_dim, num_kv_heads * kv_head_dim]
//
// Returns an error if any weight assignment fails.
func (p *QKVProjection) SetWeights(qWeights, kWeights, vWeights *tensor.Tensor) error {
	if err := p.qProj.SetWeights(qWeights); err != nil {
		return err
	}
	if err := p.kProj.SetWeights(kWeights); err != nil {
		return err
	}
	if err := p.vProj.SetWeights(vWeights); err != nil {
		return err
	}
	return nil
}
