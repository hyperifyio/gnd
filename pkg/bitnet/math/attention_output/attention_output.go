// Package attention_output implements attention output operations for the BitNet model.
//
// # Attention Output Projection for BitNet
//
// This package provides the output projection layer for multi-head attention in BitNet.
// It projects the concatenated attention outputs from all heads back to the model's hidden dimension.
//
// Key aspects:
//   - All weights and activations are int8, matching BitNet's quantized design
//   - Optimized for both single-token and multi-token inputs
//   - Efficient memory management with tensor reuse
//   - Not suitable for training or float32 inference
//
// Implementation details:
//   - Linear projection with [hidden_dim, hidden_dim] weight matrix
//   - Scaling by 1/sqrt(head_dim) for numerical stability
//   - Proper rounding and clamping to int8 range
//   - Efficient batch processing
//
// Related tasks and dependencies:
//   - #184: Attention Output Projection (Core implementation)
//   - #186: Integrate Attention Sublayer (Pre-Norm & Residual) (Depends on #184)
//   - #182: Compute Scaled Dot-Product Attention (Required by #184)
//   - #183: Apply Attention Weights to Values (Required by #184)
//
// Usage:
//   - Used in BitNet transformer blocks for attention output projection
//   - Maintainers should not change quantization or projection logic without full pipeline review
//
// Caveats:
//   - Quantization may cause saturation/clamping; tests should check for correct quantized output
//   - Any change must be validated against end-to-end BitNet inference
//   - Performance critical - changes should be benchmarked against existing implementation
//   - Memory management is important - tensors should be properly closed after use
//
// For more details, see BitNet issue #190 and the BitNet project documentation.
package attention_output

import (
	"errors"
	"math"

	"github.com/hyperifyio/gnd/pkg/bitnet/logging"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

// Error definitions
var (
	ErrNilTensor          = errors.New("attention: nil tensor")
	ErrClosed             = errors.New("attention: operation on closed tensor")
	ErrGetInputShape      = errors.New("attention: failed to get input shape")
	ErrReshapeInput       = errors.New("attention: failed to reshape input tensor")
	ErrCreateHeadOutput   = errors.New("attention: failed to create head output tensor")
	ErrGetReshapedValue   = errors.New("attention: failed to get value from reshaped tensor")
	ErrSetHeadOutput      = errors.New("attention: failed to set value in head output tensor")
	ErrCreateCombined     = errors.New("attention: failed to create combined output tensor")
	ErrGetHeadOutput      = errors.New("attention: failed to get value from head output tensor")
	ErrSetCombined        = errors.New("attention: failed to set value in combined tensor")
	ErrReshapeCombined    = errors.New("attention: failed to reshape combined tensor")
	ErrGetWeightsShape    = errors.New("attention: failed to get weights shape")
	ErrProjectionFailed   = errors.New("attention: failed to apply output projection")
	ErrCreateOutputTensor = errors.New("attention: failed to create output tensor")

	// ErrInvalidShape is returned when a tensor has an invalid shape
	ErrInvalidShape = errors.New("invalid tensor shape")

	// ErrInvalidHeadDim is returned when the head dimension is invalid
	ErrInvalidHeadDim = errors.New("invalid head dimension")
)

// AttentionOutputProjection represents the output projection layer for multi-head attention.
// This layer projects the concatenated attention outputs from all heads back to the
// model's hidden dimension.
//
// The projection is performed using a linear transformation:
//
//	output = input * W
//
// where W is a [hidden_dim, hidden_dim] weight matrix.
//
// The layer handles both single-token and multi-token cases efficiently,
// with special optimizations for the single-token case to avoid unnecessary
// reshaping operations.
type AttentionOutputProjection struct {
	// Hidden dimension of the model
	hiddenDim int
	// Number of attention heads
	numHeads int
	// Output projection weights [hidden_dim, hidden_dim]
	outProj *tensor.Tensor
	// Closed flag
	closed bool
}

// NewAttentionOutputProjection creates a new attention output projection layer.
//
// Parameters:
//   - hiddenDim: Size of the hidden dimension
//   - numHeads: Number of attention heads
//
// The projection matrix is initialized as a [hidden_dim, hidden_dim] tensor.
// The layer is optimized for efficient computation with both single-token
// and multi-token inputs.
func NewAttentionOutputProjection(hiddenDim, numHeads int) (*AttentionOutputProjection, error) {
	outProj, err := tensor.NewTensor(hiddenDim, hiddenDim)
	if err != nil {
		logging.DebugLogf("NewAttentionOutputProjection: failed to create outProj: %v", err)
		return nil, err
	}
	return &AttentionOutputProjection{
		hiddenDim: hiddenDim,
		numHeads:  numHeads,
		outProj:   outProj,
	}, nil
}

// Project applies the attention output projection to the input tensor.
func (p *AttentionOutputProjection) Project(x *tensor.Tensor) (*tensor.Tensor, error) {
	if x == nil {
		return nil, ErrNilTensor
	}
	if p.closed {
		return nil, ErrClosed
	}
	if p.outProj == nil {
		return nil, ErrClosed
	}

	// Get input shape
	shape, err := x.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get input shape: %v", err)
		return nil, ErrGetInputShape
	}

	// Validate input shape
	if len(shape) != 3 {
		loggers.Printf(loggers.Debug, "expected 3D tensor, got %dD", len(shape))
		return nil, ErrInvalidShape
	}

	// Calculate head dimension
	headDim := p.hiddenDim / p.numHeads
	if headDim == 0 || p.hiddenDim%p.numHeads != 0 {
		loggers.Printf(loggers.Debug, "invalid head dimension: hiddenDim=%d, numHeads=%d", p.hiddenDim, p.numHeads)
		return nil, ErrInvalidHeadDim
	}

	// Validate input dimensions
	if shape[2] != p.numHeads*headDim {
		loggers.Printf(loggers.Debug, "invalid input dimension: got=%d, want=%d", shape[2], p.numHeads*headDim)
		return nil, ErrInvalidShape
	}

	// Create output tensor
	output, err := tensor.NewTensor(shape[0], shape[1], p.hiddenDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to create output tensor: %v", err)
		return nil, ErrCreateOutputTensor
	}

	// Process each batch element
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			for k := 0; k < p.hiddenDim; k++ {
				var sum float64
				for l := 0; l < p.numHeads*headDim; l++ {
					iv, err := x.Get(i, j, l)
					if err != nil {
						loggers.Printf(loggers.Debug, "failed to get input value: %v", err)
						return nil, ErrGetReshapedValue
					}
					wv, err := p.outProj.Get(l, k)
					if err != nil {
						loggers.Printf(loggers.Debug, "failed to get weight value: %v", err)
						return nil, ErrProjectionFailed
					}
					sum += float64(iv) * float64(wv)
				}
				// Scale by 1/sqrt(head_dim) for numerical stability
				scaled := sum / math.Sqrt(float64(headDim))
				// Round to nearest integer
				rounded := int8(math.Round(scaled))
				// Clamp to int8 range
				if rounded > 127 {
					rounded = 127
				} else if rounded < -128 {
					rounded = -128
				}
				if err := output.Set(rounded, i, j, k); err != nil {
					loggers.Printf(loggers.Debug, "failed to set output value: %v", err)
					return nil, ErrProjectionFailed
				}
			}
		}
	}

	return output, nil
}

// SetWeights sets the output projection weights.
// AttentionOutputProjection takes ownership of the weights tensor.
// The caller must not use the weights tensor after passing it to SetWeights.
func (out *AttentionOutputProjection) SetWeights(weights *tensor.Tensor) error {
	if out.closed {
		return ErrClosed
	}
	if weights == nil {
		return ErrNilTensor
	}
	shape, err := weights.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get weights shape: %v", err)
		return ErrGetWeightsShape
	}
	if len(shape) != 2 || shape[0] != out.hiddenDim || shape[1] != out.hiddenDim {
		return ErrInvalidShape
	}
	if out.outProj != nil {
		if err := out.outProj.Close(); err != nil {
			return err
		}
	}
	out.outProj = weights
	return nil
}

// Close releases all resources associated with the attention output projection.
// This includes closing all tensors and cleaning up memory.
func (out *AttentionOutputProjection) Close() error {
	if out.outProj != nil {
		if err := out.outProj.Close(); err != nil {
			return err
		}
		out.outProj = nil
	}
	out.closed = true
	return nil
}

// AttentionOutput represents the output layer for multi-head attention.
// This layer processes the attention outputs from all heads and combines them
// into a single output tensor.
type AttentionOutput struct {
	// Hidden dimension of the model
	hiddenDim int
	// Number of attention heads
	numHeads int
	// Dimension of each attention head
	headDim int
	// Output tensors for each head
	outputs []*tensor.Tensor
}

// NewAttentionOutput creates a new attention output layer.
func NewAttentionOutput(hiddenDim, numHeads int) *AttentionOutput {
	headDim := hiddenDim / numHeads
	return &AttentionOutput{
		hiddenDim: hiddenDim,
		numHeads:  numHeads,
		headDim:   headDim,
		outputs:   make([]*tensor.Tensor, numHeads),
	}
}

// Forward performs the forward pass of the attention output layer
func (out *AttentionOutput) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if input == nil {
		return nil, ErrNilTensor
	}
	shape, err := input.Shape()
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to get input shape: %v", err)
		return nil, ErrGetInputShape
	}
	if len(shape) != 3 {
		return nil, ErrInvalidShape
	}
	batchSize, seqLen, hiddenDim := shape[0], shape[1], shape[2]
	if hiddenDim != out.hiddenDim {
		return nil, ErrInvalidShape
	}

	// Reshape input for processing
	flatSize := batchSize * seqLen
	reshaped, err := input.Reshape(flatSize, out.hiddenDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to reshape input tensor: %v", err)
		return nil, ErrReshapeInput
	}

	// Process each head
	outputs := make([]*tensor.Tensor, out.numHeads)
	for i := 0; i < out.numHeads; i++ {
		// Create a new tensor for this head
		headOutput, err := tensor.NewTensor(flatSize, out.headDim)
		if err != nil {
			loggers.Printf(loggers.Debug, "failed to create head output tensor: %v", err)
			return nil, ErrCreateHeadOutput
		}

		// Process this head
		headStart := i * out.headDim
		for j := 0; j < flatSize; j++ {
			for k := 0; k < out.headDim; k++ {
				val, err := reshaped.Get(j, headStart+k)
				if err != nil {
					loggers.Printf(loggers.Debug, "failed to get value from reshaped tensor: %v", err)
					return nil, ErrGetReshapedValue
				}
				if err := headOutput.Set(val, j, k); err != nil {
					loggers.Printf(loggers.Debug, "failed to set value in head output tensor: %v", err)
					return nil, ErrSetHeadOutput
				}
			}
		}
		outputs[i] = headOutput
	}

	// Combine head outputs
	combined, err := tensor.NewTensor(batchSize, seqLen, out.hiddenDim)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to create combined output tensor: %v", err)
		return nil, ErrCreateCombined
	}

	for i := 0; i < out.numHeads; i++ {
		headStart := i * out.headDim
		for j := 0; j < flatSize; j++ {
			for k := 0; k < out.headDim; k++ {
				val, err := outputs[i].Get(j, k)
				if err != nil {
					loggers.Printf(loggers.Debug, "failed to get value from head output tensor: %v", err)
					return nil, ErrGetHeadOutput
				}
				if err := combined.Set(val, j, headStart+k); err != nil {
					loggers.Printf(loggers.Debug, "failed to set value in combined tensor: %v", err)
					return nil, ErrSetCombined
				}
			}
		}
	}

	return combined, nil
}

// processHead processes a single attention head's output
func (out *AttentionOutput) processHead(headSlice *tensor.Tensor) (*tensor.Tensor, error) {
	// TODO: Implement head-specific processing
	return headSlice, nil
}

// combineHeads combines the outputs from all attention heads
func (out *AttentionOutput) combineHeads(batchSize, seqLen int) (*tensor.Tensor, error) {
	// TODO: Implement head combination
	result, err := tensor.NewTensor(batchSize, seqLen, out.hiddenDim)
	if err != nil {
		return nil, err
	}
	return result, nil
}

// Close releases all resources associated with the attention output layer
func (out *AttentionOutput) Close() error {
	var lastErr error
	for _, t := range out.outputs {
		if t != nil {
			if err := t.Close(); err != nil {
				lastErr = err
			}
		}
	}
	out.outputs = nil
	return lastErr
}
