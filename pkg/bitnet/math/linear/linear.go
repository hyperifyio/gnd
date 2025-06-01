// Package linear provides linear layer operations for BitNet math operations.
//
// # Quantized Linear Layer for BitNet
//
// This package provides a linear transformation layer using int8 weights and activations.
// It implements the BitLinear operation described in the BitNet paper (https://arxiv.org/abs/2310.11453).
//
// Key aspects:
//   - All weights and activations are int8, as required by BitNet
//   - The layer is optimized for both single-token and multi-token inference
//   - Efficient memory management with tensor reuse
//   - Not suitable for training or float32 inference
//
// Implementation details:
//   - Matrix multiplication with int8 weights
//   - Support for both 2D and 3D input tensors
//   - Efficient reshaping for batch processing
//   - Proper tensor cleanup and resource management
//
// Related tasks and dependencies:
//   - #178: Implement BitLinear Layer (Core implementation)
//   - #182: Compute Scaled Dot-Product Attention (Required by #178)
//   - #185: Feed-Forward Network (FFN) Sublayer (Required by #178)
//   - #186: Integrate Attention Sublayer (Pre-Norm & Residual) (Required by #178)
//   - #187: Integrate Feed-Forward Sublayer (Pre-Norm & Residual) (Required by #178)
//
// Usage:
//   - Used for projections in attention and FFN sublayers
//   - Supports both single-token and multi-token inputs
//   - Maintainers should not change quantization logic without full pipeline review
//
// Caveats:
//   - Quantization may cause saturation/clamping; tests should check for correct quantized output
//   - Any change must be validated against end-to-end BitNet inference
//   - Performance critical - changes should be benchmarked against existing implementation
//   - Memory management is important - tensors should be properly closed after use
//
// For more details, see BitNet issue #190 and the BitNet project documentation.
package linear

import (
	"errors"
	"github.com/hyperifyio/gnd/pkg/bitnet/logging"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

var (
	ErrLinearInputShape      = errors.New("linear: input must be 2D or 3D tensor")
	ErrLinearWeightsShape    = errors.New("linear: invalid weights shape")
	ErrLinearClosed          = errors.New("linear: operation called on closed layer")
	ErrLinearInputDimension  = errors.New("linear: input dimension must be positive")
	ErrLinearOutputDimension = errors.New("linear: output dimension must be positive")
	ErrLinearWeightsCreation = errors.New("linear: failed to create weights tensor")
)

// Linear represents a linear transformation layer.
// It applies a weight matrix to the input tensor.
type Linear struct {
	inDim   int
	outDim  int
	weights *tensor.Tensor
	closed  bool
}

// NewLinear creates a new linear layer with the given input and output dimensions.
func NewLinear(inDim, outDim int) (*Linear, error) {
	if inDim <= 0 {
		logging.DebugLogf("linear: input dimension must be positive, got %d", inDim)
		return nil, ErrLinearInputDimension
	}
	if outDim <= 0 {
		logging.DebugLogf("linear: output dimension must be positive, got %d", outDim)
		return nil, ErrLinearOutputDimension
	}
	weights, err := tensor.NewTensor(outDim, inDim)
	if err != nil {
		logging.DebugLogf("linear: failed to create weights tensor: %v", err)
		return nil, ErrLinearWeightsCreation
	}
	return &Linear{
		inDim:   inDim,
		outDim:  outDim,
		weights: weights,
	}, nil
}

// Forward applies the linear transformation to the input tensor.
// Returns a tensor with the same shape as input but with out_dim as the last dimension.
// The implementation handles both single-token and multi-token cases efficiently.
func (l *Linear) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if l.closed {
		return nil, ErrLinearClosed
	}

	// Validate input shape
	if err := tensor.ValidateTensorShape(x); err != nil {
		logging.DebugLogf("input shape validation failed: %v", err)
		return nil, ErrLinearInputShape
	}

	// Get input dimensions
	shape, err := x.Shape()
	if err != nil {
		return nil, err
	}
	var batchSize, seqLen, inDim int
	if len(shape) == 2 {
		batchSize, inDim = shape[0], shape[1]
		seqLen = 1
	} else {
		batchSize, seqLen, inDim = shape[0], shape[1], shape[2]
	}

	if inDim != l.inDim {
		logging.DebugLogf("input dimension (%d) must match layer input dimension (%d)", inDim, l.inDim)
		return nil, ErrLinearInputDimension
	}

	// Create 2D view of input tensor for matrix multiplication
	input2d, err := tensor.NewTensor(batchSize*seqLen, inDim)
	if err != nil {
		return nil, err
	}
	defer input2d.Close()

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < inDim; d++ {
				var val int8
				var ierr error
				if len(shape) == 2 {
					val, ierr = x.Get(b, d)
				} else {
					val, ierr = x.Get(b, s, d)
				}
				if ierr != nil {
					return nil, ierr
				}
				if setErr := input2d.Set(val, b*seqLen+s, d); setErr != nil {
					return nil, setErr
				}
			}
		}
	}

	// Apply linear transformation
	output2d, err := tensor.BitLinear(input2d, l.weights)
	if err != nil {
		return nil, err
	}
	defer output2d.Close()

	// Create output tensor with correct shape
	var output *tensor.Tensor
	if len(shape) == 2 {
		output, err = tensor.NewTensor(batchSize, l.outDim)
		if err != nil {
			return nil, err
		}
	} else {
		output, err = tensor.NewTensor(batchSize, seqLen, l.outDim)
		if err != nil {
			return nil, err
		}
	}

	// Copy data from output2d to output
	if len(shape) == 2 {
		// Input was 2D, output should be 2D
		for b := 0; b < batchSize; b++ {
			for d := 0; d < l.outDim; d++ {
				val, gerr := output2d.Get(b, d)
				if gerr != nil {
					return nil, gerr
				}
				if setErr := output.Set(val, b, d); setErr != nil {
					return nil, setErr
				}
			}
		}
	} else {
		// Input was 3D, output should be 3D
		for b := 0; b < batchSize; b++ {
			for s := 0; s < seqLen; s++ {
				for d := 0; d < l.outDim; d++ {
					val, gerr := output2d.Get(b*seqLen+s, d)
					if gerr != nil {
						return nil, gerr
					}
					if setErr := output.Set(val, b, s, d); setErr != nil {
						return nil, setErr
					}
				}
			}
		}
	}

	return output, nil
}

// SetWeights sets the weight matrix for the linear transformation.
// Linear takes ownership of the weights tensor and will close it when Linear is closed.
// The caller must not close the tensor after passing it to SetWeights.
func (l *Linear) SetWeights(weights *tensor.Tensor) error {
	if l.closed {
		return ErrLinearClosed
	}
	if weights == nil {
		return ErrLinearWeightsShape
	}
	shape, err := weights.Shape()
	if err != nil {
		return err
	}
	if len(shape) != 2 || shape[0] != l.outDim || shape[1] != l.inDim {
		logging.DebugLogf("weights must be 2D tensor with shape [%d, %d], got %v", l.outDim, l.inDim, shape)
		return ErrLinearWeightsShape
	}
	if l.weights != nil {
		l.weights.Close()
	}
	l.weights = weights
	return nil
}

// GetWeights returns the current weight matrix.
//
// Returns the weight tensor with shape [out_dim, in_dim].
// This is the matrix used for the linear transformation.
func (l *Linear) GetWeights() (*tensor.Tensor, error) {
	if l.closed {
		return nil, ErrLinearClosed
	}
	return l.weights, nil
}

// Close releases all resources associated with the linear layer.
// This includes closing all tensors and cleaning up memory.
func (l *Linear) Close() error {
	if !l.closed {
		if l.weights != nil {
			if err := l.weights.Close(); err != nil {
				return err
			}
		}
		l.closed = true
	}
	return nil
}
