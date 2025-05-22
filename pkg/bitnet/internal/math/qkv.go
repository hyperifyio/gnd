package math

import (
	"runtime"
	"sync"

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

	// Create projection matrices
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
// input: [batch_size, seq_len, hidden_dim]
// Returns: Q, K, V tensors of shape [batch_size, num_heads, seq_len, head_dim]
func (qkv *QKVProjection) Project(input *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	if len(input.Shape()) != 3 {
		panic("input must be 3D tensor [batch_size, seq_len, hidden_dim]")
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]
	hiddenDim := input.Shape()[2]

	flatInput := input.Reshape(batchSize*seqLen, hiddenDim)

	qProj := qkv.qProj.Reshape(qkv.numHeads*qkv.headDim, hiddenDim)
	kProj := qkv.kProj.Reshape(qkv.numKVHeads*qkv.headDim, hiddenDim)
	vProj := qkv.vProj.Reshape(qkv.numKVHeads*qkv.headDim, hiddenDim)

	q2d := tensor.BitLinear(flatInput, qProj)
	k2d := tensor.BitLinear(flatInput, kProj)
	v2d := tensor.BitLinear(flatInput, vProj)

	var q, k, v *tensor.Tensor

	q = q2d.Reshape(batchSize, qkv.numHeads, seqLen, qkv.headDim)
	k = k2d.Reshape(batchSize, qkv.numKVHeads, seqLen, qkv.headDim)
	v = v2d.Reshape(batchSize, qkv.numKVHeads, seqLen, qkv.headDim)

	if qkv.numKVHeads < qkv.numHeads {
		k = qkv.expandKVHeads(k)
		v = qkv.expandKVHeads(v)
	}

	return q, k, v
}

// expandKVHeads expands the key/value heads to match the number of query heads
// input: [batch_size, num_kv_heads, seq_len, head_dim]
// Returns: [batch_size, num_heads, seq_len, head_dim]
func (qkv *QKVProjection) expandKVHeads(input *tensor.Tensor) *tensor.Tensor {
	if len(input.Shape()) != 4 {
		panic("input must be 4D tensor [batch_size, num_kv_heads, seq_len, head_dim]")
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[2]
	headDim := input.Shape()[3]

	// Create output tensor
	output := tensor.NewTensor(batchSize, qkv.numHeads, seqLen, headDim)

	// Calculate number of heads per KV head
	headsPerKV := qkv.numHeads / qkv.numKVHeads

	// Process in parallel chunks
	var wg sync.WaitGroup
	chunkSize := batchSize / runtime.NumCPU()
	if chunkSize < 1 {
		chunkSize = 1
	}

	for i := 0; i < batchSize; i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > batchSize {
				end = batchSize
			}

			// For each batch element
			for b := start; b < end; b++ {
				// For each KV head
				for kv := 0; kv < qkv.numKVHeads; kv++ {
					// Expand to multiple query heads
					for h := 0; h < headsPerKV; h++ {
						headIdx := kv*headsPerKV + h
						// Copy KV head to all corresponding query heads
						for s := 0; s < seqLen; s++ {
							for d := 0; d < headDim; d++ {
								val := input.Get(b, kv, s, d)
								output.Set(val, b, headIdx, s, d)
							}
						}
					}
				}
			}
		}(i)
	}

	wg.Wait()
	return output
}

// SetWeights sets the projection weights
func (qkv *QKVProjection) SetWeights(qWeights, kWeights, vWeights *tensor.Tensor) {
	if qWeights.Shape()[0] != qkv.hiddenDim || qWeights.Shape()[1] != qkv.hiddenDim {
		panic("invalid Q weights shape")
	}
	// Allow K/V weights to be either [hiddenDim, hiddenDim] or [numKVHeads*headDim, hiddenDim]
	validKVShape := (kWeights.Shape()[0] == qkv.hiddenDim && kWeights.Shape()[1] == qkv.hiddenDim) ||
		(kWeights.Shape()[0] == qkv.numKVHeads*qkv.headDim && kWeights.Shape()[1] == qkv.hiddenDim)
	if !validKVShape {
		panic("invalid K weights shape")
	}
	validVShape := (vWeights.Shape()[0] == qkv.hiddenDim && vWeights.Shape()[1] == qkv.hiddenDim) ||
		(vWeights.Shape()[0] == qkv.numKVHeads*qkv.headDim && vWeights.Shape()[1] == qkv.hiddenDim)
	if !validVShape {
		panic("invalid V weights shape")
	}

	qkv.qProj = qWeights
	qkv.kProj = kWeights
	qkv.vProj = vWeights
}
