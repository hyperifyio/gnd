// Package attention_sublayer implements the attention sublayer for BitNet transformer blocks.
//
// # Attention Sublayer for BitNet
//
// This package provides the complete attention sublayer implementation for BitNet,
// including pre-norm layer normalization, multi-head attention, and residual connections.
// The implementation follows BitNet's b1.58-2B 4T architecture specifications.
//
// Key aspects:
//   - All weights and activations are int8, matching BitNet's quantized design
//   - Supports grouped-query attention with 20 query heads and 5 key-value heads
//   - Pre-norm architecture with layer normalization (epsilon=1e-5)
//   - Efficient parallel processing for attention computation
//   - Handles 4096-token context length
//
// Implementation details:
//   - Pre-norm layer normalization with proper scaling
//   - Query, key, value projections with proper head dimensions
//   - Scaled dot-product attention with softmax
//   - Output projection back to hidden dimension (2560)
//   - Residual connection with proper tensor management
//   - Efficient memory management with proper tensor cleanup
//
// Related tasks and dependencies:
//   - #186: Integrate Attention Sublayer (Pre-Norm & Residual)
//   - #182: Compute Scaled Dot-Product Attention
//   - #183: Apply Attention Weights to Values
//   - #184: Attention Output Projection
//   - #179: Implement Sub-Layer Normalization
//
// Usage:
//   - Used in BitNet transformer blocks for self-attention
//   - Supports both single-token and multi-token inputs
//   - Maintainers should not change quantization or architecture without full pipeline review
//
// Caveats:
//   - Quantization may cause saturation/clamping; tests should check for correct quantized output
//   - Any change must be validated against end-to-end BitNet inference
//   - Performance critical - changes should be benchmarked against existing implementation
//   - Memory management is important - tensors should be properly closed after use
//   - Must maintain compatibility with BitNet's binary-weight quantization
//
// For more details, see BitNet issue #170 and the BitNet project documentation.
package attention_sublayer

import (
	"errors"
	"github.com/hyperifyio/gnd/pkg/bitnet/math/attention_output"
	"github.com/hyperifyio/gnd/pkg/bitnet/math/layer_norm"
	"github.com/hyperifyio/gnd/pkg/bitnet/math/linear"
	"math"

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

	// ErrInvalidNumHeads is returned when the number of attention heads is invalid
	ErrInvalidNumHeads = errors.New("invalid number of attention heads")

	// ErrInvalidNumKVHeads is returned when the number of key-value heads is invalid
	ErrInvalidNumKVHeads = errors.New("invalid number of key-value heads")

	// ErrInvalidHeadDim is returned when the head dimension is invalid
	ErrInvalidHeadDim = errors.New("invalid head dimension")

	// ErrLayerClosed is returned when a bitnet layer is closed
	ErrLayerClosed = errors.New("bitnet: layer is closed")

	// ErrNilTensor is returned when a nil tensor is provided
	ErrNilTensor = errors.New("nil tensor provided")

	// ErrInvalidShape is returned when a tensor has an invalid shape
	ErrInvalidShape = errors.New("invalid tensor shape")
	// ErrShapeMismatch is returned when tensor shapes do not match
	ErrShapeMismatch = errors.New("tensor shapes do not match")
	// ErrInvalidAxis is returned when an invalid axis is provided
	ErrInvalidAxis = errors.New("invalid axis")
	// ErrIndexOutOfRange is returned when an index is out of range
	ErrIndexOutOfRange = errors.New("index out of range")
	// ErrSetQueryWeights is returned when setting query weights fails
	ErrSetQueryWeights = errors.New("failed to set query weights")
	// ErrSetKeyWeights is returned when setting key weights fails
	ErrSetKeyWeights = errors.New("failed to set key weights")
	// ErrSetValueWeights is returned when setting value weights fails
	ErrSetValueWeights = errors.New("failed to set value weights")
	// ErrSetOutputWeights is returned when setting output weights fails
	ErrSetOutputWeights = errors.New("failed to set output weights")
	// ErrSetGamma is returned when setting the scale parameter fails
	ErrSetGamma = errors.New("failed to set gamma")
	// ErrTensorClosed is returned when a tensor is closed
	ErrTensorClosed = errors.New("tensor: operation attempted on closed tensor")
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
	preNorm *layer_norm.LayerNorm
	// Query projection layer
	qProj *linear.Linear
	// Key projection layer
	kProj *linear.Linear
	// Value projection layer
	vProj *linear.Linear
	// Output projection layer
	oProj *attention_output.AttentionOutputProjection
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
		return nil, ErrInvalidHiddenDim
	}
	if numHeads <= 0 {
		return nil, ErrInvalidNumHeads
	}
	if numKVHeads <= 0 || numKVHeads > numHeads {
		return nil, ErrInvalidNumKVHeads
	}
	if hiddenDim%numHeads != 0 {
		return nil, ErrInvalidHeadDim
	}

	headDim := hiddenDim / numHeads
	kvHeadDim := hiddenDim / numKVHeads

	preNorm, err := layer_norm.NewLayerNorm(hiddenDim)
	if err != nil {
		return nil, err
	}
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
	oProj, err := attention_output.NewAttentionOutputProjection(hiddenDim, numHeads)
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
		return nil, ErrLayerClosed
	}
	if x == nil {
		return nil, ErrNilTensor
	}

	// Get input shape
	shape, err := x.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get input shape: %v", err)
		return nil, ErrInputShape
	}
	if len(shape) < 2 {
		return nil, ErrInvalidShape
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
	defer normed.Close()

	// Project to Q, K, V
	qMat, err := a.qProj.Forward(normed)
	if err != nil {
		loggers.Printf(loggers.Debug, "q projection: %v", err)
		return nil, ErrQProjection
	}
	defer qMat.Close()

	kMat, err := a.kProj.Forward(normed)
	if err != nil {
		loggers.Printf(loggers.Debug, "k projection: %v", err)
		return nil, ErrKProjection
	}
	defer kMat.Close()

	vMat, err := a.vProj.Forward(normed)
	if err != nil {
		loggers.Printf(loggers.Debug, "v projection: %v", err)
		return nil, ErrVProjection
	}
	defer vMat.Close()

	// Get shape for reshaping
	qShape, err := qMat.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get Q shape: %v", err)
		return nil, ErrGetQShape
	}

	batchSize := qShape[0]
	seqLen := qShape[1]

	// Reshape Q, K, V for attention computation
	qReshaped, err := qMat.Reshape(batchSize, seqLen, a.numHeads, a.headDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to reshape Q: %v", err)
		return nil, ErrReshapeTensor
	}
	defer qReshaped.Close()

	kReshaped, err := kMat.Reshape(batchSize, seqLen, a.numKVHeads, a.headDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to reshape K: %v", err)
		return nil, ErrReshapeTensor
	}
	defer kReshaped.Close()

	vReshaped, err := vMat.Reshape(batchSize, seqLen, a.numKVHeads, a.headDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to reshape V: %v", err)
		return nil, ErrReshapeTensor
	}
	defer vReshaped.Close()

	// Transpose for attention computation
	qTransposed, err := qReshaped.Transpose(0, 2, 1, 3)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to transpose Q: %v", err)
		return nil, ErrTransposeQ
	}
	defer qTransposed.Close()

	kTransposed, err := kReshaped.Transpose(0, 2, 1, 3)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to transpose K: %v", err)
		return nil, ErrTransposeK
	}
	defer kTransposed.Close()

	vTransposed, err := vReshaped.Transpose(0, 2, 1, 3)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to transpose V: %v", err)
		return nil, ErrTransposeV
	}
	defer vTransposed.Close()

	// Compute attention scores
	kTransposedForScores, err := kTransposed.Transpose(0, 1, 3, 2)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to transpose K for scores: %v", err)
		return nil, ErrTransposeK
	}
	defer kTransposedForScores.Close()

	scores, err := qTransposed.MatMul(kTransposedForScores)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to compute attention scores: %v", err)
		return nil, ErrAttentionScores
	}
	defer scores.Close()

	// Scale scores
	scaled, err := scores.Scale(float32(1.0 / math.Sqrt(float64(a.headDim))))
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to scale scores: %v", err)
		return nil, ErrScale
	}
	defer scaled.Close()

	// Apply softmax
	probs, err := scaled.Softmax(-1)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to apply softmax: %v", err)
		return nil, ErrSoftmax
	}
	defer probs.Close()

	// Compute attention output
	attn, err := probs.MatMul(vTransposed)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to compute attention output: %v", err)
		return nil, ErrAttentionOutput
	}
	defer attn.Close()

	// Transpose back
	attnTransposed, err := attn.Transpose(0, 2, 1, 3)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to transpose back: %v", err)
		return nil, ErrTransposeBack
	}
	defer attnTransposed.Close()

	// Reshape for output projection
	attnReshaped, err := attnTransposed.Reshape(batchSize, seqLen, a.numHeads*a.headDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to reshape attention output: %v", err)
		return nil, ErrReshapeTensor
	}
	defer attnReshaped.Close()

	// Apply output projection
	output, err := a.oProj.Project(attnReshaped)
	if err != nil {
		loggers.Printf(loggers.Debug, "output projection: %v", err)
		return nil, ErrOutputProjection
	}

	// Add residual connection
	result, err := output.Add(x)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to add residual connection: %v", err)
		output.Close()
		return nil, ErrAddResidual
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
	queryShape, err := queryWeights.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "get query weights shape: %v", err)
		return ErrGetQueryWeightsShape
	}
	if len(queryShape) != 2 || queryShape[0] != a.hiddenDim || queryShape[1] != a.numHeads*a.headDim {
		return ErrSetQueryWeights
	}
	keyShape, err := keyWeights.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "get key weights shape: %v", err)
		return ErrGetKeyWeightsShape
	}
	if len(keyShape) != 2 || keyShape[0] != a.hiddenDim || keyShape[1] != a.hiddenDim {
		return ErrSetKeyWeights
	}
	valueShape, err := valueWeights.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "get value weights shape: %v", err)
		return ErrGetValueWeightsShape
	}
	if len(valueShape) != 2 || valueShape[0] != a.hiddenDim || valueShape[1] != a.hiddenDim {
		return ErrSetValueWeights
	}
	outShape, err := outWeights.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "get output weights shape: %v", err)
		return ErrGetOutputWeightsShape
	}
	if len(outShape) != 2 || outShape[0] != a.numHeads*a.headDim || outShape[1] != a.hiddenDim {
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
	if err := a.oProj.SetWeights(outWeights); err != nil {
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
		return nil, ErrNilTensor
	}

	shape, err := t.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "get tensor shape: %v", err)
		return nil, ErrGetTensorShape
	}
	if len(shape) != 3 {
		loggers.Printf(loggers.Debug, "invalid input shape: expected 3 dimensions, got %d", len(shape))
		return nil, ErrInvalidShape
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
		return nil, ErrNilTensor
	}

	shape, err := t.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "get tensor shape: %v", err)
		return nil, ErrGetTensorShape
	}
	if len(shape) != 3 {
		loggers.Printf(loggers.Debug, "invalid input shape: expected 3 dimensions, got %d", len(shape))
		return nil, ErrInvalidShape
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
		return nil, ErrNilTensor
	}

	shape, err := t.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "get tensor shape: %v", err)
		return nil, ErrGetTensorShape
	}
	if len(shape) != 4 {
		loggers.Printf(loggers.Debug, "invalid input shape: expected 4 dimensions, got %d", len(shape))
		return nil, ErrInvalidShape
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
		return nil, ErrInvalidShape
	}
}
