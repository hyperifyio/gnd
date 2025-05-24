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

	return &AttentionSublayer{
		hiddenDim:  hiddenDim,
		numHeads:   numHeads,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		preNorm:    NewLayerNorm(hiddenDim),
		qProj:      NewLinear(hiddenDim, numHeads*headDim),
		kProj:      NewLinear(hiddenDim, numKVHeads*kvHeadDim),
		vProj:      NewLinear(hiddenDim, numKVHeads*kvHeadDim),
		oProj:      NewAttentionOutputProjection(hiddenDim, numHeads),
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

	// Get input shape
	shape := x.Shape()
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
		normed.Close()
		return nil, fmt.Errorf("q projection: %w", err)
	}
	kMat, err := a.kProj.Forward(normed)
	if err != nil {
		normed.Close()
		qMat.Close()
		return nil, fmt.Errorf("k projection: %w", err)
	}
	vMat, err := a.vProj.Forward(normed)
	normed.Close() // Now safe to close
	if err != nil {
		qMat.Close()
		kMat.Close()
		return nil, fmt.Errorf("v projection: %w", err)
	}

	// Debug: print shapes after projection
	fmt.Printf("[DEBUG] Q shape: %v\n", qMat.Shape())
	fmt.Printf("[DEBUG] K shape: %v\n", kMat.Shape())
	fmt.Printf("[DEBUG] V shape: %v\n", vMat.Shape())

	// Reshape for attention
	qMat = transposeForAttention(qMat)
	kMat = transposeForAttention(kMat)
	vMat = transposeForAttention(vMat)
	fmt.Printf("[DEBUG] Q reshaped: %v\n", qMat.Shape())
	fmt.Printf("[DEBUG] K reshaped: %v\n", kMat.Shape())
	fmt.Printf("[DEBUG] V reshaped: %v\n", vMat.Shape())

	// Compute attention scores
	kTransposed := transposeForAttentionK(kMat)
	// kMat is not used after this point, safe to close
	kMat.Close() // Close kMat after transpose

	// Add debug output before MatMul
	fmt.Printf("[DEBUG] MatMul: qMat shape: %v, kTransposed shape: %v\n", qMat.Shape(), kTransposed.Shape())
	scores, err := qMat.MatMul(kTransposed)
	// qMat and kTransposed are not used after this point, safe to close
	qMat.Close()        // Close qMat after matmul
	kTransposed.Close() // Close kTransposed after matmul
	if err != nil {
		vMat.Close()
		return nil, fmt.Errorf("attention scores: %w", err)
	}

	// Debug: print shape after matmul
	fmt.Printf("[DEBUG] scores shape: %v\n", scores.Shape())

	// Scale scores
	scale := float32(1.0 / math.Sqrt(float64(a.headDim)))
	scores = scores.Scale(scale)

	// Apply softmax
	probs, err := scores.Softmax(-1)
	// scores is not used after this point, safe to close
	scores.Close() // Close scores after softmax
	if err != nil {
		vMat.Close()
		return nil, fmt.Errorf("softmax: %w", err)
	}

	// Add debug output before MatMul
	fmt.Printf("[DEBUG] MatMul: probs shape: %v, vMat shape: %v\n", probs.Shape(), vMat.Shape())
	// Apply attention to values
	attn, err := probs.MatMul(vMat)
	// probs and vMat are not used after this point, safe to close
	probs.Close() // Close probs after matmul
	vMat.Close()  // Close vMat after matmul
	if err != nil {
		return nil, fmt.Errorf("attention output: %w", err)
	}

	// Debug: print shape after attention matmul
	fmt.Printf("[DEBUG] attn shape: %v\n", attn.Shape())

	// Transpose back
	attn = transposeBack(attn)
	fmt.Printf("[DEBUG] attn transposed back: %v\n", attn.Shape())

	// Project to output dimension
	output, err := a.oProj.Project(attn)
	// attn is not used after this point, safe to close
	attn.Close() // Close attn after projection
	if err != nil {
		return nil, fmt.Errorf("output projection: %w", err)
	}

	// Add residual connection
	result := output.Add(x)
	// output is not used after this point, safe to close
	output.Close()

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
	if len(queryWeights.Shape()) != 2 || queryWeights.Shape()[0] != a.hiddenDim || queryWeights.Shape()[1] != a.numHeads*a.headDim {
		return errors.ErrSetQueryWeights
	}
	if len(keyWeights.Shape()) != 2 || keyWeights.Shape()[0] != a.hiddenDim || keyWeights.Shape()[1] != a.hiddenDim {
		return errors.ErrSetKeyWeights
	}
	if len(valueWeights.Shape()) != 2 || valueWeights.Shape()[0] != a.hiddenDim || valueWeights.Shape()[1] != a.hiddenDim {
		return errors.ErrSetValueWeights
	}
	if len(outWeights.Shape()) != 2 || outWeights.Shape()[0] != a.numHeads*a.headDim || outWeights.Shape()[1] != a.hiddenDim {
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

// Helper function for shape comparison
func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Close releases all resources associated with the attention sublayer.
// This includes closing all tensors and cleaning up memory.
func (a *AttentionSublayer) Close() {
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
		a.oProj.Close()
	}
	a.closed = true
}

// Helper functions for safe transpose
func transposeForAttention(t *tensor.Tensor) *tensor.Tensor {
	shape := t.Shape()
	switch len(shape) {
	case 2:
		// For 2D tensors, reshape to [batch, 1, hidden_dim]
		return t.Reshape(shape[0], 1, shape[1]).(*tensor.Tensor)
	case 3:
		// For 3D tensors, reshape to [batch, seq_len, num_heads, head_dim]
		return t.Reshape(shape[0], shape[1], shape[2]/64, 64).(*tensor.Tensor)
	case 4:
		return t.Transpose(0, 2, 1, 3)
	default:
		panic(fmt.Sprintf("transposeForAttention: unsupported tensor rank %d, shape %v", len(shape), shape))
	}
}

func transposeForAttentionK(t *tensor.Tensor) *tensor.Tensor {
	shape := t.Shape()
	switch len(shape) {
	case 2:
		// For 2D tensors, reshape to [batch, 1, hidden_dim]
		return t.Reshape(shape[0], 1, shape[1]).(*tensor.Tensor)
	case 3:
		// For 3D tensors, reshape to [batch, seq_len, num_heads, head_dim]
		return t.Reshape(shape[0], shape[1], shape[2]/64, 64).(*tensor.Tensor)
	case 4:
		return t.Transpose(0, 2, 3, 1)
	default:
		panic(fmt.Sprintf("transposeForAttentionK: unsupported tensor rank %d, shape %v", len(shape), shape))
	}
}

func transposeBack(t *tensor.Tensor) *tensor.Tensor {
	shape := t.Shape()
	switch len(shape) {
	case 3:
		// For 3D tensors, reshape to [batch, hidden_dim]
		return t.Reshape(shape[0], shape[2]).(*tensor.Tensor)
	case 4:
		// For 4D tensors, reshape to [batch, seq_len, hidden_dim]
		return t.Reshape(shape[0], shape[1], shape[2]*shape[3]).(*tensor.Tensor)
	default:
		panic(fmt.Sprintf("transposeBack: unsupported tensor rank %d, shape %v", len(shape), shape))
	}
}
