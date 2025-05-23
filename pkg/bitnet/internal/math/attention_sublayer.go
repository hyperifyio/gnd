// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"errors"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

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
	// ErrSetOutputWeights is returned when setting output weights fails.
	ErrSetOutputWeights = errors.New("attention: failed to set output weights")
	// ErrSetGamma is returned when setting the scale parameter fails.
	ErrSetGamma = errors.New("attention: failed to set gamma")
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
	if numHeads <= 0 {
		return nil, ErrInvalidHeadDimensions
	}
	if numKVHeads <= 0 {
		return nil, ErrInvalidKVHeads
	}

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
	if x == nil {
		return nil, ErrInvalidInputShape
	}

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
		defer input.Close()
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
	defer normed.Close()

	// Project to Q, K, V
	q, err := a.qProj.Forward(normed)
	if err != nil {
		return nil, ErrQueryProjection
	}
	defer q.Close()

	k, err := a.kProj.Forward(normed)
	if err != nil {
		return nil, ErrKeyProjection
	}
	defer k.Close()

	v, err := a.vProj.Forward(normed)
	if err != nil {
		return nil, ErrValueProjection
	}
	defer v.Close()

	// Reshape for attention
	headDim := a.hiddenDim / a.numHeads
	kvHeadDim := a.hiddenDim / a.numKVHeads

	// Reshape and transpose Q, K, V
	q = q.Reshape(input.Shape()[0], input.Shape()[1], a.numHeads, headDim).Transpose(0, 2, 1, 3)
	defer q.Close()

	k = k.Reshape(input.Shape()[0], input.Shape()[1], a.numKVHeads, kvHeadDim).Transpose(0, 2, 1, 3)
	defer k.Close()

	v = v.Reshape(input.Shape()[0], input.Shape()[1], a.numKVHeads, kvHeadDim).Transpose(0, 2, 1, 3)
	defer v.Close()

	// For grouped-query attention, repeat K and V heads
	if a.numKVHeads < a.numHeads {
		repeats := a.numHeads / a.numKVHeads
		k = k.Repeat(1, repeats)
		defer k.Close()
		v = v.Repeat(1, repeats)
		defer v.Close()
	}

	// Compute attention
	attn, err := ScaledDotProductAttention(q, k, v)
	if err != nil {
		return nil, ErrScaledDotProduct
	}
	defer attn.Close()

	// Project output
	attn = attn.Transpose(0, 2, 1, 3).Reshape(input.Shape()[0], input.Shape()[1], a.hiddenDim)
	defer attn.Close()

	out, err := a.outProj.Project(attn)
	if err != nil {
		return nil, err
	}
	defer out.Close()

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
//   - queryWeights: Query projection weights [hidden_dim, hidden_dim]
//   - keyWeights: Key projection weights [hidden_dim, hidden_dim]
//   - valueWeights: Value projection weights [hidden_dim, hidden_dim]
//   - outWeights: Output projection weights [hidden_dim, hidden_dim]
//
// Returns an error if any weight assignment fails.
func (a *AttentionSublayer) SetWeights(queryWeights, keyWeights, valueWeights, outWeights *tensor.Tensor) error {
	headDim := a.hiddenDim / a.numHeads
	kvHeadDim := a.hiddenDim / a.numKVHeads

	// Check for nil weights
	if queryWeights == nil {
		return ErrSetQueryWeights
	}
	if keyWeights == nil {
		return ErrSetKeyWeights
	}
	if valueWeights == nil {
		return ErrSetValueWeights
	}
	if outWeights == nil {
		return ErrSetOutputWeights
	}

	// Check shapes
	if len(queryWeights.Shape()) != 2 || queryWeights.Shape()[0] != a.hiddenDim || queryWeights.Shape()[1] != a.numHeads*headDim {
		return ErrSetQueryWeights
	}
	if len(keyWeights.Shape()) != 2 || keyWeights.Shape()[0] != a.hiddenDim || keyWeights.Shape()[1] != a.numKVHeads*kvHeadDim {
		return ErrSetKeyWeights
	}
	if len(valueWeights.Shape()) != 2 || valueWeights.Shape()[0] != a.hiddenDim || valueWeights.Shape()[1] != a.numKVHeads*kvHeadDim {
		return ErrSetValueWeights
	}
	if len(outWeights.Shape()) != 2 || outWeights.Shape()[0] != a.numHeads*headDim || outWeights.Shape()[1] != a.hiddenDim {
		return ErrSetOutputWeights
	}

	// Set weights
	if err := a.qProj.SetWeights(queryWeights); err != nil {
		return ErrSetQueryWeights
	}
	if err := a.kProj.SetWeights(keyWeights); err != nil {
		return ErrSetKeyWeights
	}
	if err := a.vProj.SetWeights(valueWeights); err != nil {
		return ErrSetValueWeights
	}
	if err := a.outProj.SetWeights(outWeights); err != nil {
		return ErrSetOutputWeights
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
		return ErrSetGamma
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
	if a.outProj != nil {
		a.outProj.Close()
	}
}
