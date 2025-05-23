// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"errors"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

// DebugLog logs debug information with formatting.
// Used for internal debugging and diagnostics in the math package.
func DebugLog(format string, args ...interface{}) {
	loggers.Printf(loggers.Debug, format, args...)
}

var (
	// ErrInvalidHeadDimensions is returned when the head dimensions are invalid for attention.
	ErrInvalidHeadDimensions = errors.New("attention: invalid head dimensions")
	// ErrInvalidKVHeads is returned when numKVHeads > numHeads.
	ErrInvalidKVHeads = errors.New("attention: numKVHeads must be <= numHeads")
	// ErrNonDivisibleHeads is returned when numHeads is not divisible by numKVHeads.
	ErrNonDivisibleHeads = errors.New("attention: numHeads must be divisible by numKVHeads")
	// ErrPreNormForward is returned when the pre-norm layer normalization fails.
	ErrPreNormForward = errors.New("attention: pre-norm forward pass failed")
	// ErrQueryProjection is returned when the query projection fails.
	ErrQueryProjection = errors.New("attention: query projection failed")
	// ErrKeyProjection is returned when the key projection fails.
	ErrKeyProjection = errors.New("attention: key projection failed")
	// ErrValueProjection is returned when the value projection fails.
	ErrValueProjection = errors.New("attention: value projection failed")
	// ErrScaledDotProduct is returned when the scaled dot-product attention fails.
	ErrScaledDotProduct = errors.New("attention: scaled dot-product attention failed")
	// ErrSetQueryWeights is returned when setting query weights fails.
	ErrSetQueryWeights = errors.New("attention: failed to set query weights")
	// ErrSetKeyWeights is returned when setting key weights fails.
	ErrSetKeyWeights = errors.New("attention: failed to set key weights")
	// ErrSetValueWeights is returned when setting value weights fails.
	ErrSetValueWeights = errors.New("attention: failed to set value weights")
)

// AttentionSublayer implements the attention sublayer with pre-norm and residual connection
// as described in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).
//
// The sublayer consists of:
//   - Pre-norm layer normalization
//   - Multi-head attention with QKV projections
//   - Output projection
//   - Residual connection
//
// The sublayer supports both standard multi-head attention and grouped-query attention
// through the numKVHeads parameter. When numKVHeads < numHeads, it implements
// grouped-query attention where multiple query heads share the same key and value heads.
type AttentionSublayer struct {
	hiddenDim  int                        // Hidden dimension of the model
	numHeads   int                        // Number of attention heads
	numKVHeads int                        // Number of key/value heads (for grouped-query attention)
	preNorm    *LayerNorm                 // Pre-norm layer normalization
	qProj      *Linear                    // Query projection layer
	kProj      *Linear                    // Key projection layer
	vProj      *Linear                    // Value projection layer
	outProj    *AttentionOutputProjection // Output projection layer
}

// NewAttentionSublayer creates a new attention sublayer.
//
// Parameters:
//   - hiddenDim: Dimension of the hidden state
//   - numHeads: Number of attention heads
//   - numKVHeads: Number of key/value heads (for grouped-query attention)
//
// The function initializes:
//   - Pre-norm layer normalization
//   - QKV projection matrices
//   - Output projection
//
// Returns a pointer to the AttentionSublayer and an error if validation fails.
func NewAttentionSublayer(hiddenDim, numHeads, numKVHeads int) (*AttentionSublayer, error) {
	if err := ValidateHeadDimensions(hiddenDim, numHeads, hiddenDim/numHeads); err != nil {
		return nil, ErrInvalidHeadDimensions
	}

	if numKVHeads > numHeads {
		DebugLog("numKVHeads (%d) must be <= numHeads (%d)", numKVHeads, numHeads)
		return nil, ErrInvalidKVHeads
	}

	if numHeads%numKVHeads != 0 {
		DebugLog("numHeads (%d) must be divisible by numKVHeads (%d)", numHeads, numKVHeads)
		return nil, ErrNonDivisibleHeads
	}

	headDim := hiddenDim / numHeads
	kvHeadDim := hiddenDim / numKVHeads

	return &AttentionSublayer{
		hiddenDim:  hiddenDim,
		numHeads:   numHeads,
		numKVHeads: numKVHeads,
		preNorm:    NewLayerNorm(hiddenDim),
		qProj:      NewLinear(hiddenDim, numHeads*headDim),
		kProj:      NewLinear(hiddenDim, numKVHeads*kvHeadDim),
		vProj:      NewLinear(hiddenDim, numKVHeads*kvHeadDim),
		outProj:    NewAttentionOutputProjection(hiddenDim, numHeads),
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
	// Validate input shape
	if err := ValidateShape(x, 2, 3); err != nil {
		return nil, ErrInvalidInputShape
	}

	// Handle 2D input by adding sequence dimension
	var input *tensor.Tensor
	if len(x.Shape()) == 2 {
		hiddenDim := x.Shape()[1]
		if hiddenDim != a.hiddenDim {
			DebugLog("input hidden dimension (%d) must match sublayer hidden dimension (%d)", hiddenDim, a.hiddenDim)
			return nil, ErrHiddenDimMismatch
		}
		input = tensor.NewTensor(x.Shape()[0], 1, hiddenDim)
		for b := 0; b < x.Shape()[0]; b++ {
			for d := 0; d < hiddenDim; d++ {
				input.Set(x.Get(b, d), b, 0, d)
			}
		}
	} else {
		hiddenDim := x.Shape()[2]
		if hiddenDim != a.hiddenDim {
			DebugLog("input hidden dimension (%d) must match sublayer hidden dimension (%d)", hiddenDim, a.hiddenDim)
			return nil, ErrHiddenDimMismatch
		}
		input = x
	}

	// Pre-norm layer normalization
	normed, err := a.preNorm.Forward(input)
	if err != nil {
		return nil, ErrPreNormForward
	}

	// Project to Q, K, V
	q, err := a.qProj.Forward(normed)
	if err != nil {
		return nil, ErrQueryProjection
	}

	k, err := a.kProj.Forward(normed)
	if err != nil {
		return nil, ErrKeyProjection
	}

	v, err := a.vProj.Forward(normed)
	if err != nil {
		return nil, ErrValueProjection
	}

	// Reshape for attention
	headDim := a.hiddenDim / a.numHeads
	kvHeadDim := a.hiddenDim / a.numKVHeads

	// Reshape and transpose Q, K, V
	q = q.Reshape(input.Shape()[0], input.Shape()[1], a.numHeads, headDim).Transpose(0, 2, 1, 3)
	k = k.Reshape(input.Shape()[0], input.Shape()[1], a.numKVHeads, kvHeadDim).Transpose(0, 2, 1, 3)
	v = v.Reshape(input.Shape()[0], input.Shape()[1], a.numKVHeads, kvHeadDim).Transpose(0, 2, 1, 3)

	// For grouped-query attention, repeat K and V heads
	if a.numKVHeads < a.numHeads {
		repeats := a.numHeads / a.numKVHeads
		k = k.Repeat(1, repeats)
		v = v.Repeat(1, repeats)
	}

	// Compute attention
	attn, err := ScaledDotProductAttention(q, k, v)
	if err != nil {
		return nil, ErrScaledDotProduct
	}

	// Project output
	attn = attn.Transpose(0, 2, 1, 3).Reshape(input.Shape()[0], input.Shape()[1], a.hiddenDim)
	out := a.outProj.Project(attn)

	// Add residual connection
	if len(x.Shape()) == 2 {
		// For 2D input, take first sequence position
		res := tensor.NewTensor(input.Shape()[0], a.hiddenDim)
		for b := 0; b < input.Shape()[0]; b++ {
			for d := 0; d < a.hiddenDim; d++ {
				val := out.Get(b, 0, d) + x.Get(b, d)
				// Clamp to int8 range
				if val > 127 {
					val = 127
				} else if val < -128 {
					val = -128
				}
				res.Set(int8(val), b, d)
			}
		}
		return res, nil
	}

	// For 3D input, add residual connection
	res := tensor.NewTensor(input.Shape()[0], input.Shape()[1], a.hiddenDim)
	for b := 0; b < input.Shape()[0]; b++ {
		for s := 0; s < input.Shape()[1]; s++ {
			for d := 0; d < a.hiddenDim; d++ {
				val := out.Get(b, s, d) + x.Get(b, s, d)
				// Clamp to int8 range
				if val > 127 {
					val = 127
				} else if val < -128 {
					val = -128
				}
				res.Set(int8(val), b, s, d)
			}
		}
	}
	return res, nil
}

// SetWeights sets the weights for the attention sublayer.
//
// Parameters:
//   - qWeights: Query projection weights [hidden_dim, num_heads * head_dim]
//   - kWeights: Key projection weights [hidden_dim, num_kv_heads * kv_head_dim]
//   - vWeights: Value projection weights [hidden_dim, num_kv_heads * kv_head_dim]
//   - outWeights: Output projection weights [num_heads * head_dim, hidden_dim]
//
// Returns an error if any weight assignment fails.
func (a *AttentionSublayer) SetWeights(qWeights, kWeights, vWeights, outWeights *tensor.Tensor) error {
	if err := a.qProj.SetWeights(qWeights); err != nil {
		return ErrSetQueryWeights
	}
	if err := a.kProj.SetWeights(kWeights); err != nil {
		return ErrSetKeyWeights
	}
	if err := a.vProj.SetWeights(vWeights); err != nil {
		return ErrSetValueWeights
	}
	a.outProj.SetWeights(outWeights)
	return nil
}

// SetGamma sets the scale parameter for the sublayer normalization.
//
// Parameters:
//   - gamma: Scale parameter tensor for layer normalization
//
// Returns an error if the gamma tensor is invalid.
func (a *AttentionSublayer) SetGamma(gamma *tensor.Tensor) error {
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
