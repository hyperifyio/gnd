package math

import (
	"fmt"
	"os"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// QKVProjection represents the Query, Key, and Value projection matrices
// for multi-head self-attention
type QKVProjection struct {
	// Number of attention heads
	numHeads int
	// Number of key/value heads (for grouped-query attention)
	numKVHeads int
	// Dimension of each head
	headDim int
	// Hidden dimension
	hiddenDim int
	// Query projection weights
	qProj *tensor.Tensor
	// Key projection weights
	kProj *tensor.Tensor
	// Value projection weights
	vProj *tensor.Tensor
}

// NewQKVProjection creates a new QKV projection with the given parameters
func NewQKVProjection(hiddenDim, numHeads, numKVHeads int) *QKVProjection {
	headDim := hiddenDim / numHeads

	// Create projection matrices with correct shapes
	// Each projection matrix is [hidden_size, hidden_size]
	qProj := tensor.NewTensor(hiddenDim, hiddenDim)
	kProj := tensor.NewTensor(hiddenDim, hiddenDim)
	vProj := tensor.NewTensor(hiddenDim, hiddenDim)

	return &QKVProjection{
		numHeads:   numHeads,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		hiddenDim:  hiddenDim,
		qProj:      qProj,
		kProj:      kProj,
		vProj:      vProj,
	}
}

// Project performs the QKV projection on the input hidden states
// input: [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
// Returns: Q, K, V tensors of shape [batch_size, num_heads, seq_len, head_dim]
func (p *QKVProjection) Project(input *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	// Debug output for input tensor
	fmt.Fprintf(os.Stderr, "[DEBUG] Input tensor shape: %v\n", input.Shape())
	fmt.Fprintf(os.Stderr, "[DEBUG] Input tensor data length: %d\n", len(input.Data()))

	// Get input dimensions
	var batchSize, seqLen, hiddenDim int
	if len(input.Shape()) == 2 {
		batchSize, hiddenDim = input.Shape()[0], input.Shape()[1]
		seqLen = 1
	} else if len(input.Shape()) == 3 {
		batchSize, seqLen, hiddenDim = input.Shape()[0], input.Shape()[1], input.Shape()[2]
	} else {
		panic(fmt.Sprintf("invalid input shape: %v", input.Shape()))
	}

	// Check hidden dimension
	if hiddenDim != p.hiddenDim {
		panic(fmt.Sprintf("input hidden dimension %d does not match projection hidden dimension %d", hiddenDim, p.hiddenDim))
	}

	// Create 2D view of input tensor for matrix multiplication
	input2d := tensor.NewTensor(batchSize*seqLen, hiddenDim)
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < hiddenDim; d++ {
				var val int8
				if len(input.Shape()) == 2 {
					val = input.Get(b, d)
				} else {
					val = input.Get(b, s, d)
				}
				input2d.Set(val, b*seqLen+s, d)
			}
		}
	}

	// Debug output for 2D input tensor
	fmt.Fprintf(os.Stderr, "[DEBUG] 2D input tensor shape: %v\n", input2d.Shape())
	fmt.Fprintf(os.Stderr, "[DEBUG] 2D input tensor data length: %d\n", len(input2d.Data()))

	// Apply projections
	q2d := tensor.BitLinear(input2d, p.qProj)
	k2d := tensor.BitLinear(input2d, p.kProj)
	v2d := tensor.BitLinear(input2d, p.vProj)

	// Debug output for 2D projections
	fmt.Fprintf(os.Stderr, "[DEBUG] Q 2D shape: %v\n", q2d.Shape())
	fmt.Fprintf(os.Stderr, "[DEBUG] K 2D shape: %v\n", k2d.Shape())
	fmt.Fprintf(os.Stderr, "[DEBUG] V 2D shape: %v\n", v2d.Shape())

	// Create output tensors with correct shapes [batch, num_heads, seq_len, head_dim]
	q := tensor.NewTensor(batchSize, p.numHeads, seqLen, p.headDim)
	k := tensor.NewTensor(batchSize, p.numKVHeads, seqLen, p.headDim)
	v := tensor.NewTensor(batchSize, p.numKVHeads, seqLen, p.headDim)

	// Copy data from 2D projections to output tensors, properly splitting into heads
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			// For query heads
			for h := 0; h < p.numHeads; h++ {
				for d := 0; d < p.headDim; d++ {
					// Calculate the correct index in the 2D projection
					idx := b*seqLen + s
					val := q2d.Get(idx, h*p.headDim+d)
					q.Set(val, b, h, s, d)
				}
			}
			// For key/value heads
			for h := 0; h < p.numKVHeads; h++ {
				for d := 0; d < p.headDim; d++ {
					// Calculate the correct index in the 2D projection
					idx := b*seqLen + s
					val := k2d.Get(idx, h*p.headDim+d)
					k.Set(val, b, h, s, d)
					val = v2d.Get(idx, h*p.headDim+d)
					v.Set(val, b, h, s, d)
				}
			}
		}
	}

	// Debug output for output tensors
	fmt.Fprintf(os.Stderr, "[DEBUG] Q output shape: %v\n", q.Shape())
	fmt.Fprintf(os.Stderr, "[DEBUG] K output shape: %v\n", k.Shape())
	fmt.Fprintf(os.Stderr, "[DEBUG] V output shape: %v\n", v.Shape())

	// Expand key/value heads if necessary
	if p.numKVHeads < p.numHeads {
		k = expandKVHeads(k, p.numHeads)
		v = expandKVHeads(v, p.numHeads)
	}

	return q, k, v
}

// expandKVHeads expands the number of key/value heads by repeating the existing heads
func expandKVHeads(t *tensor.Tensor, numHeads int) *tensor.Tensor {
	shape := t.Shape()
	if len(shape) != 4 {
		panic(fmt.Sprintf("invalid tensor shape for head expansion: %v", shape))
	}

	batchSize, seqLen, numKVHeads, headDim := shape[0], shape[1], shape[2], shape[3]
	if numKVHeads >= numHeads {
		return t
	}

	// Create expanded tensor
	expanded := tensor.NewTensor(batchSize, seqLen, numHeads, headDim)

	// Copy and repeat heads
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < numHeads; h++ {
				// Use modulo to repeat heads
				srcHead := h % numKVHeads
				for d := 0; d < headDim; d++ {
					val := t.Get(b, s, srcHead, d)
					expanded.Set(val, b, s, h, d)
				}
			}
		}
	}

	return expanded
}

// SetWeights sets the QKV projection weights
func (p *QKVProjection) SetWeights(qWeights, kWeights, vWeights *tensor.Tensor) {
	// Check tensor shapes
	if qWeights.Shape()[0] != p.hiddenDim || qWeights.Shape()[1] != p.hiddenDim {
		panic(fmt.Sprintf("invalid Q weights shape: got %v, want [%d, %d]", qWeights.Shape(), p.hiddenDim, p.hiddenDim))
	}
	if kWeights.Shape()[0] != p.hiddenDim || kWeights.Shape()[1] != p.hiddenDim {
		panic(fmt.Sprintf("invalid K weights shape: got %v, want [%d, %d]", kWeights.Shape(), p.hiddenDim, p.hiddenDim))
	}
	if vWeights.Shape()[0] != p.hiddenDim || vWeights.Shape()[1] != p.hiddenDim {
		panic(fmt.Sprintf("invalid V weights shape: got %v, want [%d, %d]", vWeights.Shape(), p.hiddenDim, p.hiddenDim))
	}

	// Set projection matrices
	p.qProj = qWeights
	p.kProj = kWeights
	p.vProj = vWeights

	// Debug output for QKV projection matrices
	fmt.Fprintf(os.Stderr, "[DEBUG] Q projection shape: %v\n", p.qProj.Shape())
	fmt.Fprintf(os.Stderr, "[DEBUG] K projection shape: %v\n", p.kProj.Shape())
	fmt.Fprintf(os.Stderr, "[DEBUG] V projection shape: %v\n", p.vProj.Shape())
}
