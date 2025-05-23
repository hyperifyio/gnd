package math

import (
	"fmt"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/hyperifyio/gnd/pkg/loggers"
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
	kvHeadDim := hiddenDim / numKVHeads

	// Create projection matrices with correct shapes
	// Q projection: [hidden_dim, num_heads * head_dim]
	// K projection: [hidden_dim, num_kv_heads * kv_head_dim]
	// V projection: [hidden_dim, num_kv_heads * kv_head_dim]
	qProj := tensor.NewTensor(hiddenDim, numHeads*headDim)
	kProj := tensor.NewTensor(hiddenDim, numKVHeads*kvHeadDim)
	vProj := tensor.NewTensor(hiddenDim, numKVHeads*kvHeadDim)

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
	loggers.Printf(loggers.Debug, "Input tensor shape: %v", input.Shape())
	loggers.Printf(loggers.Debug, "Input tensor data length: %d", len(input.Data()))

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
		loggers.Printf(loggers.Debug, "input hidden dimension %d does not match projection hidden dimension %d", hiddenDim, p.hiddenDim)
		panic("input hidden dimension does not match projection hidden dimension")
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
	loggers.Printf(loggers.Debug, "2D input tensor shape: %v", input2d.Shape())
	loggers.Printf(loggers.Debug, "2D input tensor data length: %d", len(input2d.Data()))

	// Apply projections
	q2d := tensor.BitLinear(input2d, p.qProj)
	k2d := tensor.BitLinear(input2d, p.kProj)
	v2d := tensor.BitLinear(input2d, p.vProj)

	// Debug output for 2D projections
	loggers.Printf(loggers.Debug, "Q 2D shape: %v", q2d.Shape())
	loggers.Printf(loggers.Debug, "K 2D shape: %v", k2d.Shape())
	loggers.Printf(loggers.Debug, "V 2D shape: %v", v2d.Shape())

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
	loggers.Printf(loggers.Debug, "Q output shape: %v", q.Shape())
	loggers.Printf(loggers.Debug, "K output shape: %v", k.Shape())
	loggers.Printf(loggers.Debug, "V output shape: %v", v.Shape())

	// Expand key/value heads if necessary
	if p.numKVHeads < p.numHeads {
		// Create expanded tensors with correct head dimensions
		expandedK := tensor.NewTensor(batchSize, p.numHeads, seqLen, p.headDim)
		expandedV := tensor.NewTensor(batchSize, p.numHeads, seqLen, p.headDim)

		// Copy and repeat heads
		for b := 0; b < batchSize; b++ {
			for h := 0; h < p.numHeads; h++ {
				// Use modulo to repeat heads
				srcHead := h % p.numKVHeads
				for s := 0; s < seqLen; s++ {
					for d := 0; d < p.headDim; d++ {
						val := k.Get(b, srcHead, s, d)
						expandedK.Set(val, b, h, s, d)
						val = v.Get(b, srcHead, s, d)
						expandedV.Set(val, b, h, s, d)
					}
				}
			}
		}

		k = expandedK
		v = expandedV
	}

	return q, k, v
}

// SetWeights sets the QKV projection weights
func (p *QKVProjection) SetWeights(qWeights, kWeights, vWeights *tensor.Tensor) {
	// Debug output for weight shapes
	loggers.Printf(loggers.Debug, "Q weights shape: %v", qWeights.Shape())
	loggers.Printf(loggers.Debug, "K weights shape: %v", kWeights.Shape())
	loggers.Printf(loggers.Debug, "V weights shape: %v", vWeights.Shape())
	loggers.Printf(loggers.Debug, "Expected Q shape: [%d, %d]", p.hiddenDim, p.numHeads*p.headDim)
	loggers.Printf(loggers.Debug, "Expected K shape: [%d, %d]", p.hiddenDim, p.numKVHeads*(p.hiddenDim/p.numKVHeads))
	loggers.Printf(loggers.Debug, "Expected V shape: [%d, %d]", p.hiddenDim, p.numKVHeads*(p.hiddenDim/p.numKVHeads))

	// Check tensor shapes
	if qWeights.Shape()[0] != p.hiddenDim || qWeights.Shape()[1] != p.numHeads*p.headDim {
		panic(fmt.Sprintf("invalid Q weights shape: got %v, want [%d, %d]", qWeights.Shape(), p.hiddenDim, p.numHeads*p.headDim))
	}
	if kWeights.Shape()[0] != p.hiddenDim || kWeights.Shape()[1] != p.numKVHeads*(p.hiddenDim/p.numKVHeads) {
		panic(fmt.Sprintf("invalid K weights shape: got %v, want [%d, %d]", kWeights.Shape(), p.hiddenDim, p.numKVHeads*(p.hiddenDim/p.numKVHeads)))
	}
	if vWeights.Shape()[0] != p.hiddenDim || vWeights.Shape()[1] != p.numKVHeads*(p.hiddenDim/p.numKVHeads) {
		panic(fmt.Sprintf("invalid V weights shape: got %v, want [%d, %d]", vWeights.Shape(), p.hiddenDim, p.numKVHeads*(p.hiddenDim/p.numKVHeads)))
	}

	p.qProj = qWeights
	p.kProj = kWeights
	p.vProj = vWeights
}
