// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"fmt"
	"math"

	"github.com/hyperifyio/gnd/pkg/bitnet/errors"
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// AttentionSublayer implements the attention sublayer of a transformer block.
// It consists of:
// 1. Pre-norm layer normalization
// 2. Multi-head attention
// 3. Residual connection
type AttentionSublayer struct {
	// Hidden dimension of the model
	hiddenDim int
	// Number of attention heads
	numHeads int
	// Number of key-value heads (for grouped-query attention)
	numKVHeads int
	// Dimension of each attention head
	headDim int
	// Pre-norm layer normalization
	preNorm *LayerNorm
	// Query projection layer
	qProj *Linear
	// Key projection layer
	kProj *Linear
	// Value projection layer
	vProj *Linear
	// Output projection layer
	oProj *AttentionOutputProjection
	// Flag to track if the layer is closed
	closed bool
}

// NewAttentionSublayer creates a new attention sublayer.
//
// Parameters:
//   - hiddenDim: Size of the hidden dimension
//   - numHeads: Number of attention heads
//   - numKVHeads: Number of key-value heads (for grouped-query attention)
//
// The layer is initialized with:
// - Pre-norm layer normalization
// - Query, key, value projections
// - Output projection
func NewAttentionSublayer(hiddenDim, numHeads, numKVHeads int) (*AttentionSublayer, error) {
	if hiddenDim <= 0 {
		return nil, errors.ErrInvalidHiddenDim
	}
	if numHeads <= 0 {
		return nil, errors.ErrInvalidNumHeads
	}
	if numKVHeads <= 0 || numKVHeads > numHeads {
		return nil, errors.ErrInvalidNumKVHeads
	}
	if hiddenDim%numHeads != 0 {
		return nil, errors.ErrInvalidHeadDim
	}

	headDim := hiddenDim / numHeads
	kvHeadDim := hiddenDim / numKVHeads

	preNorm, err := NewLayerNorm(hiddenDim)
	if err != nil {
		return nil, err
	}
	qProj := NewLinear(hiddenDim, numHeads*headDim)
	kProj := NewLinear(hiddenDim, numKVHeads*kvHeadDim)
	vProj := NewLinear(hiddenDim, numKVHeads*kvHeadDim)
	oProj := NewAttentionOutputProjection(hiddenDim, numHeads)

	return &AttentionSublayer{
		hiddenDim:  hiddenDim,
		numHeads:   numHeads,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		preNorm:    preNorm,
		qProj:      qProj,
		kProj:      kProj,
		vProj:      vProj,
		oProj:      oProj,
	}, nil
}

// Forward performs the forward pass through the attention sublayer.
//
// Input tensor can be either:
//   - 2D [batch_size, hidden_dim]
//   - 3D [batch_size, seq_len, hidden_dim]
//
// The function performs the following steps:
//  1. Pre-norm layer normalization
//  2. Q, K, V projections
//  3. Scaled dot-product attention
//  4. Output projection
//  5. Residual connection
//
// Returns a tensor with the same shape as the input and an error if any step fails.
func (a *AttentionSublayer) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if a.closed {
		return nil, errors.ErrLayerClosed
	}
	if x == nil {
		return nil, errors.ErrNilTensor
	}

	// Get input shape
	shape, err := x.Shape()
	if err != nil {
		return nil, fmt.Errorf("failed to get input shape: %w", err)
	}
	if len(shape) < 2 {
		return nil, errors.ErrInvalidShape
	}

	hiddenDim := shape[len(shape)-1]

	// Validate hidden dimension
	if hiddenDim != a.hiddenDim {
		return nil, fmt.Errorf("tensor: invalid hidden dimension, got %d, want %d", hiddenDim, a.hiddenDim)
	}

	// Pre-norm layer normalization
	normed, err := a.preNorm.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("pre-norm forward: %w", err)
	}

	// Project to Q, K, V (do not close normed until all projections are done)
	qMat, err := a.qProj.Forward(normed)
	if err != nil {
		if err := normed.Close(); err != nil {
			return nil, fmt.Errorf("close normed: %w", err)
		}
		return nil, fmt.Errorf("q projection: %w", err)
	}
	kMat, err := a.kProj.Forward(normed)
	if err != nil {
		if err := normed.Close(); err != nil {
			return nil, fmt.Errorf("close normed: %w", err)
		}
		if err := qMat.Close(); err != nil {
			return nil, fmt.Errorf("close qMat: %w", err)
		}
		return nil, fmt.Errorf("k projection: %w", err)
	}
	vMat, err := a.vProj.Forward(normed)
	if err := normed.Close(); err != nil { // Now safe to close
		return nil, fmt.Errorf("close normed: %w", err)
	}
	if err != nil {
		if err := qMat.Close(); err != nil {
			return nil, fmt.Errorf("close qMat: %w", err)
		}
		if err := kMat.Close(); err != nil {
			return nil, fmt.Errorf("close kMat: %w", err)
		}
		return nil, fmt.Errorf("v projection: %w", err)
	}

	// Debug: print shapes after projection
	qShape, err := qMat.Shape()
	if err != nil {
		return nil, fmt.Errorf("failed to get Q shape: %w", err)
	}
	kShape, err := kMat.Shape()
	if err != nil {
		return nil, fmt.Errorf("failed to get K shape: %w", err)
	}
	vShape, err := vMat.Shape()
	if err != nil {
		return nil, fmt.Errorf("failed to get V shape: %w", err)
	}
	fmt.Printf("[DEBUG] Q shape: %v\n", qShape)
	fmt.Printf("[DEBUG] K shape: %v\n", kShape)
	fmt.Printf("[DEBUG] V shape: %v\n", vShape)

	// Reshape for attention
	qMatOp, err := transposeForAttention(qMat)
	if err != nil {
		if err := qMat.Close(); err != nil {
			return nil, fmt.Errorf("close qMat: %w", err)
		}
		if err := kMat.Close(); err != nil {
			return nil, fmt.Errorf("close kMat: %w", err)
		}
		if err := vMat.Close(); err != nil {
			return nil, fmt.Errorf("close vMat: %w", err)
		}
		return nil, fmt.Errorf("transpose q: %w", err)
	}
	qMat = qMatOp

	kMatOp, err := transposeForAttention(kMat)
	if err != nil {
		if err := qMat.Close(); err != nil {
			return nil, fmt.Errorf("close qMat: %w", err)
		}
		if err := kMat.Close(); err != nil {
			return nil, fmt.Errorf("close kMat: %w", err)
		}
		if err := vMat.Close(); err != nil {
			return nil, fmt.Errorf("close vMat: %w", err)
		}
		return nil, fmt.Errorf("transpose k: %w", err)
	}
	kMat = kMatOp

	vMatOp, err := transposeForAttention(vMat)
	if err != nil {
		if err := qMat.Close(); err != nil {
			return nil, fmt.Errorf("close qMat: %w", err)
		}
		if err := kMat.Close(); err != nil {
			return nil, fmt.Errorf("close kMat: %w", err)
		}
		if err := vMat.Close(); err != nil {
			return nil, fmt.Errorf("close vMat: %w", err)
		}
		return nil, fmt.Errorf("transpose v: %w", err)
	}
	vMat = vMatOp

	fmt.Printf("[DEBUG] Q reshaped: %v\n", qMat.Shape())
	fmt.Printf("[DEBUG] K reshaped: %v\n", kMat.Shape())
	fmt.Printf("[DEBUG] V reshaped: %v\n", vMat.Shape())

	// Compute attention scores
	kTransposedOp, err := transposeForAttentionK(kMat)
	if err := kMat.Close(); err != nil { // kMat is not used after this point
		return nil, fmt.Errorf("close kMat: %w", err)
	}
	if err != nil {
		if err := qMat.Close(); err != nil {
			return nil, fmt.Errorf("close qMat: %w", err)
		}
		if err := vMat.Close(); err != nil {
			return nil, fmt.Errorf("close vMat: %w", err)
		}
		return nil, fmt.Errorf("transpose k: %w", err)
	}
	kTransposed := kTransposedOp

	// Add debug output before MatMul
	fmt.Printf("[DEBUG] MatMul: qMat shape: %v, kTransposed shape: %v\n", qMat.Shape(), kTransposed.Shape())
	scores, err := qMat.MatMul(kTransposed)
	if err := qMat.Close(); err != nil { // qMat is not used after this point
		return nil, fmt.Errorf("close qMat: %w", err)
	}
	if err := kTransposed.Close(); err != nil { // kTransposed is not used after this point
		return nil, fmt.Errorf("close kTransposed: %w", err)
	}
	if err != nil {
		if err := vMat.Close(); err != nil {
			return nil, fmt.Errorf("close vMat: %w", err)
		}
		return nil, fmt.Errorf("attention scores: %w", err)
	}

	// Debug: print shape after matmul
	fmt.Printf("[DEBUG] scores shape: %v\n", scores.Shape())

	// Scale scores
	scale := float32(1.0 / math.Sqrt(float64(a.headDim)))
	scaled, err := scores.Scale(scale)
	if err != nil {
		if err := scores.Close(); err != nil {
			return nil, fmt.Errorf("close scores: %w", err)
		}
		return nil, fmt.Errorf("scale: %w", err)
	}

	// Apply softmax
	probs, err := scaled.Softmax(-1)
	if err := scaled.Close(); err != nil { // scaled is not used after this point
		return nil, fmt.Errorf("close scaled: %w", err)
	}
	if err != nil {
		if err := vMat.Close(); err != nil {
			return nil, fmt.Errorf("close vMat: %w", err)
		}
		return nil, fmt.Errorf("softmax: %w", err)
	}

	// Add debug output before MatMul
	fmt.Printf("[DEBUG] MatMul: probs shape: %v, vMat shape: %v\n", probs.Shape(), vMat.Shape())
	// Apply attention to values
	attn, err := probs.MatMul(vMat)
	if err := probs.Close(); err != nil { // probs is not used after this point
		return nil, fmt.Errorf("close probs: %w", err)
	}
	if err := vMat.Close(); err != nil { // vMat is not used after this point
		return nil, fmt.Errorf("close vMat: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("attention output: %w", err)
	}

	// Debug: print shape after attention matmul
	fmt.Printf("[DEBUG] attn shape: %v\n", attn.Shape())

	// Transpose back
	attnOp, err := transposeBack(attn)
	if err != nil {
		if err := attn.Close(); err != nil {
			return nil, fmt.Errorf("close attn: %w", err)
		}
		return nil, fmt.Errorf("transpose back: %w", err)
	}
	attn = attnOp
	fmt.Printf("[DEBUG] attn transposed back: %v\n", attn.Shape())

	// Project to output dimension
	output, err := a.oProj.Project(attn)
	if err := attn.Close(); err != nil { // attn is not used after this point
		return nil, fmt.Errorf("close attn: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("output projection: %w", err)
	}

	// Add residual connection
	result := output.Add(x)
	if err := output.Close(); err != nil { // output is not used after this point
		return nil, fmt.Errorf("close output: %w", err)
	}

	return result, nil
}

// SetWeights sets the weights for the attention sublayer.
//
// Parameters:
//   - queryWeights: Query projection weights [hidden_dim, hidden_dim]
//   - keyWeights: Key projection weights [hidden_dim, hidden_dim]
//   - valueWeights: Value projection weights [hidden_dim, hidden_dim]
//   - outWeights: Output projection weights [hidden_dim, hidden_dim]
//
// Returns an error if any weight assignment fails.
func (a *AttentionSublayer) SetWeights(queryWeights, keyWeights, valueWeights, outWeights *tensor.Tensor) error {
	// Check for nil weights
	if queryWeights == nil {
		return errors.ErrSetQueryWeights
	}
	if keyWeights == nil {
		return errors.ErrSetKeyWeights
	}
	if valueWeights == nil {
		return errors.ErrSetValueWeights
	}
	if outWeights == nil {
		return errors.ErrSetOutputWeights
	}

	// Check shapes
	queryShape := queryWeights.Shape()
	if len(queryShape) != 2 || queryShape[0] != a.hiddenDim || queryShape[1] != a.numHeads*a.headDim {
		return errors.ErrSetQueryWeights
	}
	keyShape := keyWeights.Shape()
	if len(keyShape) != 2 || keyShape[0] != a.hiddenDim || keyShape[1] != a.hiddenDim {
		return errors.ErrSetKeyWeights
	}
	valueShape := valueWeights.Shape()
	if len(valueShape) != 2 || valueShape[0] != a.hiddenDim || valueShape[1] != a.hiddenDim {
		return errors.ErrSetValueWeights
	}
	outShape := outWeights.Shape()
	if len(outShape) != 2 || outShape[0] != a.numHeads*a.headDim || outShape[1] != a.hiddenDim {
		return errors.ErrSetOutputWeights
	}

	// Set weights
	if err := a.qProj.SetWeights(queryWeights); err != nil {
		return errors.ErrSetQueryWeights
	}
	if err := a.kProj.SetWeights(keyWeights); err != nil {
		return errors.ErrSetKeyWeights
	}
	if err := a.vProj.SetWeights(valueWeights); err != nil {
		return errors.ErrSetValueWeights
	}
	if err := a.oProj.SetWeights(outWeights); err != nil {
		return errors.ErrSetOutputWeights
	}
	return nil
}

// SetGamma sets the scale parameter for the sublayer normalization.
//
// Parameters:
//   - gamma: Scale parameter tensor for layer normalization
//
// Returns an error if the gamma tensor is invalid.
func (a *AttentionSublayer) SetGamma(gamma *tensor.Tensor) error {
	if gamma == nil {
		return errors.ErrSetGamma
	}
	return a.preNorm.SetGamma(gamma)
}

// Close releases all resources associated with the attention sublayer.
// This includes closing all tensors and cleaning up memory.
func (a *AttentionSublayer) Close() error {
	var lastErr error
	if a.preNorm != nil {
		a.preNorm.Close()
	}
	if a.qProj != nil {
		a.qProj.Close()
	}
	if a.kProj != nil {
		a.kProj.Close()
	}
	if a.vProj != nil {
		a.vProj.Close()
	}
	if a.oProj != nil {
		if err := a.oProj.Close(); err != nil {
			lastErr = err
		}
	}
	a.closed = true
	return lastErr
}

// transposeForAttention reshapes a tensor for attention computation.
func transposeForAttention(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("nil tensor")
	}

	shape := t.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("invalid input shape: expected 3 dimensions, got %d", len(shape))
	}

	// Reshape to [batch_size, 1, seq_len]
	reshaped1 := t.Reshape(shape[0], 1, shape[1])
	if reshaped1 == nil {
		return nil, fmt.Errorf("failed to reshape tensor")
	}

	// Reshape to [batch_size, seq_len, head_dim, 64]
	reshaped2 := reshaped1.Reshape(shape[0], shape[1], shape[2]/64, 64)
	if reshaped2 == nil {
		return nil, fmt.Errorf("failed to reshape tensor")
	}

	return reshaped2, nil
}

// transposeForAttentionK reshapes a tensor for key attention computation.
func transposeForAttentionK(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("nil tensor")
	}

	shape := t.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("invalid input shape: expected 3 dimensions, got %d", len(shape))
	}

	// Reshape to [batch_size, 1, seq_len]
	reshaped1 := t.Reshape(shape[0], 1, shape[1])
	if reshaped1 == nil {
		return nil, fmt.Errorf("failed to reshape tensor")
	}

	// Reshape to [batch_size, seq_len, head_dim, 64]
	reshaped2 := reshaped1.Reshape(shape[0], shape[1], shape[2]/64, 64)
	if reshaped2 == nil {
		return nil, fmt.Errorf("failed to reshape tensor")
	}

	return reshaped2, nil
}

// transposeForAttentionV reshapes a tensor for value attention computation.
func transposeForAttentionV(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("nil tensor")
	}

	shape := t.Shape()
	if len(shape) != 4 {
		return nil, fmt.Errorf("invalid input shape: expected 4 dimensions, got %d", len(shape))
	}

	// Reshape to [batch_size, seq_len * head_dim]
	reshaped1 := t.Reshape(shape[0], shape[1]*shape[2])
	if reshaped1 == nil {
		return nil, fmt.Errorf("failed to reshape tensor")
	}

	// Reshape to [batch_size, seq_len, head_dim]
	reshaped2 := reshaped1.Reshape(shape[0], shape[1], shape[2]*shape[3])
	if reshaped2 == nil {
		return nil, fmt.Errorf("failed to reshape tensor")
	}

	return reshaped2, nil
}

func transposeBack(t *tensor.Tensor) (*tensor.Tensor, error) {
	shape := t.Shape()
	switch len(shape) {
	case 3:
		return t.Reshape(shape[0], shape[1]*shape[2]), nil
	case 4:
		return t.Reshape(shape[0], shape[1], shape[2]*shape[3]), nil
	default:
		return nil, errors.ErrInvalidShape
	}
}
