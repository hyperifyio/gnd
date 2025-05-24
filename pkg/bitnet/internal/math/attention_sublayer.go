// Package math implements mathematical operations for the BitNet model, including
// attention mechanisms, feed-forward networks, and normalization layers.
// The package provides optimized implementations of transformer architecture
// components with support for ternary quantization.
package math

import (
	"errors"
	"fmt"
	"math"

	bitneterrors "github.com/hyperifyio/gnd/pkg/bitnet/errors"
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

// Common errors returned by attention sublayer operations
var (
	ErrOutputProjectionCreate = errors.New("attention: failed to create output projection")
	ErrInputShape             = errors.New("attention: failed to get input shape")
	ErrInvalidHiddenDim       = errors.New("attention: invalid hidden dimension")
	ErrPreNormForward         = errors.New("attention: pre-norm forward failed")
	ErrCloseNormed            = errors.New("attention: failed to close normed tensor")
	ErrQProjection            = errors.New("attention: q projection failed")
	ErrCloseQMat              = errors.New("attention: failed to close qMat tensor")
	ErrKProjection            = errors.New("attention: k projection failed")
	ErrCloseKMat              = errors.New("attention: failed to close kMat tensor")
	ErrVProjection            = errors.New("attention: v projection failed")
	ErrGetQShape              = errors.New("attention: failed to get Q shape")
	ErrGetKShape              = errors.New("attention: failed to get K shape")
	ErrGetVShape              = errors.New("attention: failed to get V shape")
	ErrTransposeQ             = errors.New("attention: failed to transpose Q")
	ErrTransposeK             = errors.New("attention: failed to transpose K")
	ErrTransposeV             = errors.New("attention: failed to transpose V")
	ErrAttentionScores        = errors.New("attention: failed to compute attention scores")
	ErrGetScoresShape         = errors.New("attention: failed to get scores shape")
	ErrCloseScores            = errors.New("attention: failed to close scores tensor")
	ErrScale                  = errors.New("attention: failed to scale scores")
	ErrCloseScaled            = errors.New("attention: failed to close scaled tensor")
	ErrSoftmax                = errors.New("attention: failed to apply softmax")
	ErrGetProbsShape          = errors.New("attention: failed to get probs shape")
	ErrCloseProbs             = errors.New("attention: failed to close probs tensor")
	ErrCloseVMat              = errors.New("attention: failed to close vMat tensor")
	ErrAttentionOutput        = errors.New("attention: failed to compute attention output")
	ErrGetAttnShape           = errors.New("attention: failed to get attention shape")
	ErrCloseAttn              = errors.New("attention: failed to close attention tensor")
	ErrTransposeBack          = errors.New("attention: failed to transpose back")
	ErrOutputProjection       = errors.New("attention: output projection failed")
	ErrCloseOutput            = errors.New("attention: failed to close output tensor")
	ErrAddResidual            = errors.New("attention: failed to add residual connection")
	ErrGetQueryWeightsShape   = errors.New("attention: failed to get query weights shape")
	ErrGetKeyWeightsShape     = errors.New("attention: failed to get key weights shape")
	ErrGetValueWeightsShape   = errors.New("attention: failed to get value weights shape")
	ErrGetOutputWeightsShape  = errors.New("attention: failed to get output weights shape")
	ErrGetTensorShape         = errors.New("attention: failed to get tensor shape")
	ErrReshapeTensor          = errors.New("attention: failed to reshape tensor")
	ErrReshapeFailed          = errors.New("attention: reshape operation failed")
	ErrCloseKTransposed       = errors.New("attention: failed to close kTransposed tensor")
	ErrCloseAttnTensor        = errors.New("attention: failed to close attention tensor")
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
		return nil, bitneterrors.ErrInvalidHiddenDim
	}
	if numHeads <= 0 {
		return nil, bitneterrors.ErrInvalidNumHeads
	}
	if numKVHeads <= 0 || numKVHeads > numHeads {
		return nil, bitneterrors.ErrInvalidNumKVHeads
	}
	if hiddenDim%numHeads != 0 {
		return nil, bitneterrors.ErrInvalidHeadDim
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
	oProj, err := NewAttentionOutputProjection(hiddenDim, numHeads)
	if err != nil {
		loggers.Printf(loggers.Debug, "create output projection: %v", err)
		return nil, ErrOutputProjectionCreate
	}

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
		return nil, bitneterrors.ErrLayerClosed
	}
	if x == nil {
		return nil, bitneterrors.ErrNilTensor
	}

	// Get input shape
	shape, err := x.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get input shape: %v", err)
		return nil, ErrInputShape
	}
	if len(shape) < 2 {
		return nil, bitneterrors.ErrInvalidShape
	}

	hiddenDim := shape[len(shape)-1]

	// Validate hidden dimension
	if hiddenDim != a.hiddenDim {
		loggers.Printf(loggers.Debug, "tensor: invalid hidden dimension, got %d, want %d", hiddenDim, a.hiddenDim)
		return nil, ErrInvalidHiddenDim
	}

	// Pre-norm layer normalization
	normed, err := a.preNorm.Forward(x)
	if err != nil {
		loggers.Printf(loggers.Debug, "pre-norm forward: %v", err)
		return nil, ErrPreNormForward
	}

	// Project to Q, K, V (do not close normed until all projections are done)
	qMat, err := a.qProj.Forward(normed)
	if err != nil {
		if err := normed.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close normed: %v", err)
			return nil, ErrCloseNormed
		}
		loggers.Printf(loggers.Debug, "q projection: %v", err)
		return nil, ErrQProjection
	}
	kMat, err := a.kProj.Forward(normed)
	if err != nil {
		if err := normed.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close normed: %v", err)
			return nil, ErrCloseNormed
		}
		if err := qMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close qMat: %v", err)
			return nil, ErrCloseQMat
		}
		loggers.Printf(loggers.Debug, "k projection: %v", err)
		return nil, ErrKProjection
	}
	vMat, err := a.vProj.Forward(normed)
	if err := normed.Close(); err != nil { // Now safe to close
		loggers.Printf(loggers.Debug, "close normed: %v", err)
		return nil, ErrCloseNormed
	}
	if err != nil {
		if err := qMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close qMat: %v", err)
			return nil, ErrCloseQMat
		}
		if err := kMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close kMat: %v", err)
			return nil, ErrCloseKMat
		}
		loggers.Printf(loggers.Debug, "v projection: %v", err)
		return nil, ErrVProjection
	}

	// Debug: print shapes after projection
	qShape, err := qMat.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get Q shape: %v", err)
		return nil, ErrGetQShape
	}
	kShape, err := kMat.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get K shape: %v", err)
		return nil, ErrGetKShape
	}
	vShape, err := vMat.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get V shape: %v", err)
		return nil, ErrGetVShape
	}
	loggers.Printf(loggers.Debug, "Q shape: %v", qShape)
	loggers.Printf(loggers.Debug, "K shape: %v", kShape)
	loggers.Printf(loggers.Debug, "V shape: %v", vShape)

	// Reshape for attention
	qMatOp, err := transposeForAttention(qMat)
	if err != nil {
		if err := qMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close qMat: %v", err)
			return nil, ErrTransposeQ
		}
		if err := kMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close kMat: %v", err)
			return nil, ErrTransposeK
		}
		if err := vMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close vMat: %v", err)
			return nil, ErrTransposeV
		}
		return nil, fmt.Errorf("transpose q: %w", err)
	}
	qMat = qMatOp

	kMatOp, err := transposeForAttention(kMat)
	if err != nil {
		if err := qMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close qMat: %v", err)
			return nil, ErrTransposeQ
		}
		if err := kMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close kMat: %v", err)
			return nil, ErrTransposeK
		}
		if err := vMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close vMat: %v", err)
			return nil, ErrTransposeV
		}
		return nil, fmt.Errorf("transpose k: %w", err)
	}
	kMat = kMatOp

	vMatOp, err := transposeForAttention(vMat)
	if err != nil {
		if err := qMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close qMat: %v", err)
			return nil, ErrTransposeQ
		}
		if err := kMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close kMat: %v", err)
			return nil, ErrTransposeK
		}
		if err := vMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close vMat: %v", err)
			return nil, ErrTransposeV
		}
		return nil, fmt.Errorf("transpose v: %w", err)
	}
	vMat = vMatOp

	// Debug: print shapes after reshaping
	qShape, err = qMat.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get Q shape: %v", err)
		return nil, ErrGetQShape
	}
	kShape, err = kMat.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get K shape: %v", err)
		return nil, ErrGetKShape
	}
	vShape, err = vMat.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get V shape: %v", err)
		return nil, ErrGetVShape
	}
	loggers.Printf(loggers.Debug, "Q reshaped: %v", qShape)
	loggers.Printf(loggers.Debug, "K reshaped: %v", kShape)
	loggers.Printf(loggers.Debug, "V reshaped: %v", vShape)

	// Compute attention scores
	kTransposedOp, err := transposeForAttentionK(kMat)
	if err := kMat.Close(); err != nil { // kMat is not used after this point
		loggers.Printf(loggers.Debug, "close kMat: %v", err)
		return nil, fmt.Errorf("close kMat: %w", err)
	}
	if err != nil {
		if err := qMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close qMat: %v", err)
			return nil, ErrTransposeQ
		}
		if err := vMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close vMat: %v", err)
			return nil, ErrTransposeV
		}
		return nil, fmt.Errorf("transpose k: %w", err)
	}
	kTransposed := kTransposedOp

	// Add debug output before MatMul
	qShape, err = qMat.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get Q shape: %v", err)
		return nil, ErrGetQShape
	}
	kShape, err = kTransposed.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get K shape: %v", err)
		return nil, ErrGetKShape
	}
	loggers.Printf(loggers.Debug, "MatMul: qMat shape: %v, kTransposed shape: %v", qShape, kShape)
	scores, err := qMat.MatMul(kTransposed)
	if err := qMat.Close(); err != nil { // qMat is not used after this point
		loggers.Printf(loggers.Debug, "close qMat: %v", err)
		return nil, ErrCloseQMat
	}
	if err := kTransposed.Close(); err != nil { // kTransposed is not used after this point
		loggers.Printf(loggers.Debug, "close kTransposed: %v", err)
		return nil, fmt.Errorf("close kTransposed: %w", err)
	}
	if err != nil {
		if err := vMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close vMat: %v", err)
			return nil, ErrCloseVMat
		}
		loggers.Printf(loggers.Debug, "attention scores: %v", err)
		return nil, ErrAttentionScores
	}

	// Debug: print shape after matmul
	scoresShape, err := scores.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get scores shape: %v", err)
		return nil, ErrGetScoresShape
	}
	loggers.Printf(loggers.Debug, "scores shape: %v", scoresShape)

	// Scale scores
	scale := float32(1.0 / math.Sqrt(float64(a.headDim)))
	scaled, err := scores.Scale(scale)
	if err != nil {
		if err := scores.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close scores: %v", err)
			return nil, ErrCloseScores
		}
		loggers.Printf(loggers.Debug, "scale: %v", err)
		return nil, ErrScale
	}

	// Apply softmax
	probs, err := scaled.Softmax(-1)
	if err := scaled.Close(); err != nil { // scaled is not used after this point
		loggers.Printf(loggers.Debug, "close scaled: %v", err)
		return nil, ErrCloseScaled
	}
	if err != nil {
		if err := vMat.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close vMat: %v", err)
			return nil, ErrCloseVMat
		}
		loggers.Printf(loggers.Debug, "softmax: %v", err)
		return nil, ErrSoftmax
	}

	// Add debug output before MatMul
	probsShape, err := probs.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get probs shape: %v", err)
		return nil, ErrGetProbsShape
	}
	vShape, err = vMat.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get V shape: %v", err)
		return nil, ErrGetVShape
	}
	loggers.Printf(loggers.Debug, "MatMul: probs shape: %v, vMat shape: %v", probsShape, vShape)
	// Apply attention to values
	attn, err := probs.MatMul(vMat)
	if err := probs.Close(); err != nil { // probs is not used after this point
		loggers.Printf(loggers.Debug, "close probs: %v", err)
		return nil, ErrCloseProbs
	}
	if err := vMat.Close(); err != nil { // vMat is not used after this point
		loggers.Printf(loggers.Debug, "close vMat: %v", err)
		return nil, ErrCloseVMat
	}
	if err != nil {
		loggers.Printf(loggers.Debug, "attention output: %v", err)
		return nil, ErrAttentionOutput
	}

	// Debug: print shape after attention matmul
	attnShape, err := attn.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get attn shape: %v", err)
		return nil, ErrGetAttnShape
	}
	loggers.Printf(loggers.Debug, "attn shape: %v", attnShape)

	// Transpose back
	attnOp, err := transposeBack(attn)
	if err != nil {
		if err := attn.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close attn: %v", err)
			return nil, ErrCloseAttn
		}
		loggers.Printf(loggers.Debug, "transpose back: %v", err)
		return nil, ErrTransposeBack
	}
	attn = attnOp
	attnShape, err = attn.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get attn shape: %v", err)
		return nil, ErrGetAttnShape
	}
	loggers.Printf(loggers.Debug, "attn transposed back: %v", attnShape)

	// Project to output dimension
	output, err := a.oProj.Project(attn)
	if err := attn.Close(); err != nil { // attn is not used after this point
		loggers.Printf(loggers.Debug, "close attn: %v", err)
		return nil, fmt.Errorf("close attn: %w", err)
	}
	if err != nil {
		loggers.Printf(loggers.Debug, "output projection: %v", err)
		return nil, ErrOutputProjection
	}

	// Add residual connection
	result, err := output.Add(x)
	if err != nil {
		if err := output.Close(); err != nil {
			loggers.Printf(loggers.Debug, "close output: %v", err)
			return nil, ErrCloseOutput
		}
		loggers.Printf(loggers.Debug, "add residual: %v", err)
		return nil, ErrAddResidual
	}
	if err := output.Close(); err != nil { // output is not used after this point
		loggers.Printf(loggers.Debug, "close output: %v", err)
		return nil, ErrCloseOutput
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
		return bitneterrors.ErrSetQueryWeights
	}
	if keyWeights == nil {
		return bitneterrors.ErrSetKeyWeights
	}
	if valueWeights == nil {
		return bitneterrors.ErrSetValueWeights
	}
	if outWeights == nil {
		return bitneterrors.ErrSetOutputWeights
	}

	// Check shapes
	queryShape, err := queryWeights.Shape()
	if err != nil {
		return fmt.Errorf("get query weights shape: %w", err)
	}
	if len(queryShape) != 2 || queryShape[0] != a.hiddenDim || queryShape[1] != a.numHeads*a.headDim {
		return bitneterrors.ErrSetQueryWeights
	}
	keyShape, err := keyWeights.Shape()
	if err != nil {
		return fmt.Errorf("get key weights shape: %w", err)
	}
	if len(keyShape) != 2 || keyShape[0] != a.hiddenDim || keyShape[1] != a.hiddenDim {
		return bitneterrors.ErrSetKeyWeights
	}
	valueShape, err := valueWeights.Shape()
	if err != nil {
		return fmt.Errorf("get value weights shape: %w", err)
	}
	if len(valueShape) != 2 || valueShape[0] != a.hiddenDim || valueShape[1] != a.hiddenDim {
		return bitneterrors.ErrSetValueWeights
	}
	outShape, err := outWeights.Shape()
	if err != nil {
		return fmt.Errorf("get output weights shape: %w", err)
	}
	if len(outShape) != 2 || outShape[0] != a.numHeads*a.headDim || outShape[1] != a.hiddenDim {
		return bitneterrors.ErrSetOutputWeights
	}

	// Set weights
	if err := a.qProj.SetWeights(queryWeights); err != nil {
		return bitneterrors.ErrSetQueryWeights
	}
	if err := a.kProj.SetWeights(keyWeights); err != nil {
		return bitneterrors.ErrSetKeyWeights
	}
	if err := a.vProj.SetWeights(valueWeights); err != nil {
		return bitneterrors.ErrSetValueWeights
	}
	if err := a.oProj.SetWeights(outWeights); err != nil {
		return bitneterrors.ErrSetOutputWeights
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
		return bitneterrors.ErrSetGamma
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
		return nil, bitneterrors.ErrNilTensor
	}

	shape, err := t.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "get tensor shape: %v", err)
		return nil, ErrGetTensorShape
	}
	if len(shape) != 3 {
		loggers.Printf(loggers.Debug, "invalid input shape: expected 3 dimensions, got %d", len(shape))
		return nil, bitneterrors.ErrInvalidShape
	}

	// Reshape to [batch_size, 1, seq_len]
	reshaped1, err := t.Reshape(shape[0], 1, shape[1])
	if err != nil {
		loggers.Printf(loggers.Debug, "reshape tensor: %v", err)
		return nil, ErrReshapeTensor
	}
	if reshaped1 == nil {
		return nil, ErrReshapeFailed
	}

	// Reshape to [batch_size, seq_len, head_dim, 64]
	reshaped2, err := reshaped1.Reshape(shape[0], shape[1], shape[2]/64, 64)
	if err != nil {
		loggers.Printf(loggers.Debug, "reshape tensor: %v", err)
		return nil, ErrReshapeTensor
	}
	if reshaped2 == nil {
		return nil, ErrReshapeFailed
	}

	return reshaped2, nil
}

// transposeForAttentionK reshapes a tensor for key attention computation.
func transposeForAttentionK(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t == nil {
		return nil, bitneterrors.ErrNilTensor
	}

	shape, err := t.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "get tensor shape: %v", err)
		return nil, ErrGetTensorShape
	}
	if len(shape) != 3 {
		loggers.Printf(loggers.Debug, "invalid input shape: expected 3 dimensions, got %d", len(shape))
		return nil, bitneterrors.ErrInvalidShape
	}

	// Reshape to [batch_size, 1, seq_len]
	reshaped1, err := t.Reshape(shape[0], 1, shape[1])
	if err != nil {
		loggers.Printf(loggers.Debug, "reshape tensor: %v", err)
		return nil, ErrReshapeTensor
	}
	if reshaped1 == nil {
		return nil, ErrReshapeFailed
	}

	// Reshape to [batch_size, seq_len, head_dim, 64]
	reshaped2, err := reshaped1.Reshape(shape[0], shape[1], shape[2]/64, 64)
	if err != nil {
		loggers.Printf(loggers.Debug, "reshape tensor: %v", err)
		return nil, ErrReshapeTensor
	}
	if reshaped2 == nil {
		return nil, ErrReshapeFailed
	}

	return reshaped2, nil
}

// transposeForAttentionV reshapes a tensor for value attention computation.
func transposeForAttentionV(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t == nil {
		return nil, bitneterrors.ErrNilTensor
	}

	shape, err := t.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "get tensor shape: %v", err)
		return nil, ErrGetTensorShape
	}
	if len(shape) != 4 {
		loggers.Printf(loggers.Debug, "invalid input shape: expected 4 dimensions, got %d", len(shape))
		return nil, bitneterrors.ErrInvalidShape
	}

	// Reshape to [batch_size, seq_len * head_dim]
	reshaped1, err := t.Reshape(shape[0], shape[1]*shape[2])
	if err != nil {
		loggers.Printf(loggers.Debug, "reshape tensor: %v", err)
		return nil, ErrReshapeTensor
	}
	if reshaped1 == nil {
		return nil, ErrReshapeFailed
	}

	// Reshape to [batch_size, seq_len, head_dim]
	reshaped2, err := reshaped1.Reshape(shape[0], shape[1], shape[2]*shape[3])
	if err != nil {
		loggers.Printf(loggers.Debug, "reshape tensor: %v", err)
		return nil, ErrReshapeTensor
	}
	if reshaped2 == nil {
		return nil, ErrReshapeFailed
	}

	return reshaped2, nil
}

func transposeBack(t *tensor.Tensor) (*tensor.Tensor, error) {
	shape, err := t.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "get tensor shape: %v", err)
		return nil, ErrGetTensorShape
	}
	switch len(shape) {
	case 3:
		result, err := t.Reshape(shape[0], shape[1]*shape[2])
		if err != nil {
			loggers.Printf(loggers.Debug, "reshape tensor: %v", err)
			return nil, ErrReshapeTensor
		}
		return result, nil
	case 4:
		result, err := t.Reshape(shape[0], shape[1], shape[2]*shape[3])
		if err != nil {
			loggers.Printf(loggers.Debug, "reshape tensor: %v", err)
			return nil, ErrReshapeTensor
		}
		return result, nil
	default:
		return nil, bitneterrors.ErrInvalidShape
	}
}
