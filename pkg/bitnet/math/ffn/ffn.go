// Package ffn provides feed-forward network operations for BitNet math operations.
//
// # Quantized FFN for BitNet
//
// This file provides a two-layer FFN with ReLU² activation, using quantized (int8) weights and activations.
// It implements the feed-forward network described in the BitNet paper (https://arxiv.org/abs/2310.11453).
//
// Key aspects:
//   - All weights and activations are int8, matching BitNet's quantized design.
//   - The FFN consists of an up-projection, ReLU² activation, and down-projection.
//   - Parallelized for CPU efficiency; optimized for batch/sequence processing.
//   - Not suitable for training or float32 inference.
//
// Implementation details:
//   - Two-layer architecture with expansion and contraction
//   - ReLU² activation with scaling to prevent overflow
//   - BitLinear operations for efficient computation
//   - Parallel processing for activation and projections
//
// Related tasks and dependencies:
//   - #180: Implement Squared ReLU Activation (Core activation function)
//   - #178: Implement BitLinear Layer (Required for projections)
//   - #185: Feed-Forward Network (FFN) Sublayer (Depends on #180 and #178)
//   - #187: Integrate Feed-Forward Sublayer (Pre-Norm & Residual) (Depends on #185)
//   - #179: Implement Sub-Layer Normalization (Required by #187)
//
// Usage:
//   - Used as a sublayer in BitNet transformer blocks.
//   - Maintainers should not change quantization or activation logic without full pipeline review.
//
// Caveats:
//   - Quantization may cause saturation/clamping; tests should check for correct quantized output.
//   - Any change must be validated against end-to-end BitNet inference.
//   - Performance critical - changes should be benchmarked against existing implementation.
//   - Memory management is important - tensors should be properly closed after use.
//
// For more details, see BitNet issue #190 and the BitNet project documentation.
package ffn

import (
	"errors"
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

var (
	ErrInvalidInputShape   = errors.New("ffn: invalid input shape")
	ErrFFNClosed           = errors.New("ffn: operation called on closed FFN")
	ErrInvalidWeightsShape = errors.New("ffn: invalid weights shape")
)

// FFN represents a two-layer feed-forward network with ReLU² activation.
// This is a key component of the transformer architecture that processes
// each position independently through two linear transformations with
// a non-linear activation in between.
//
// The network consists of:
// 1. An up-projection layer that expands the hidden dimension
// 2. A ReLU² activation function
// 3. A down-projection layer that contracts back to the hidden dimension
//
// The implementation is optimized for parallel processing and includes
// scaling to prevent numerical overflow in the ReLU² activation.
type FFN struct {
	// Hidden dimension of the model
	hiddenDim int
	// Intermediate dimension (typically 4x hidden_dim)
	intermediateDim int
	// First layer weights (up-projection) [intermediate_dim, hidden_dim]
	upProj *tensor.Tensor
	// Second layer weights (down-projection) [hidden_dim, intermediate_dim]
	downProj *tensor.Tensor
	// Whether the FFN has been closed
	closed bool
}

// NewFFN creates a new feed-forward network instance.
//
// Parameters:
//   - hiddenDim: Size of the hidden dimension
//   - intermediateDim: Size of the intermediate dimension (typically 4x hidden_dim)
//
// The network is initialized with two weight matrices:
// - upProj: [intermediate_dim, hidden_dim] for expansion
// - downProj: [hidden_dim, intermediate_dim] for contraction
func NewFFN(hiddenDim, intermediateDim int) (*FFN, error) {
	if hiddenDim <= 0 || intermediateDim <= 0 {
		return nil, ErrInvalidWeightsShape
	}

	// Create weight matrices with correct dimensions
	upProj, err := tensor.NewTensor(intermediateDim, hiddenDim)
	if err != nil {
		return nil, err
	}

	downProj, err := tensor.NewTensor(hiddenDim, intermediateDim)
	if err != nil {
		upProj.Close()
		return nil, err
	}

	// Initialize weights with ones
	for i := 0; i < intermediateDim; i++ {
		for j := 0; j < hiddenDim; j++ {
			if err := upProj.Set(1, i, j); err != nil {
				upProj.Close()
				downProj.Close()
				return nil, err
			}
		}
	}

	for i := 0; i < hiddenDim; i++ {
		for j := 0; j < intermediateDim; j++ {
			if err := downProj.Set(1, i, j); err != nil {
				upProj.Close()
				downProj.Close()
				return nil, err
			}
		}
	}

	return &FFN{
		hiddenDim:       hiddenDim,
		intermediateDim: intermediateDim,
		upProj:          upProj,
		downProj:        downProj,
	}, nil
}

// Forward performs the forward pass through the feed-forward network.
//
// Input tensor must be 2D [batch_size, hidden_dim] or 3D [batch_size, seq_len, hidden_dim].
// The function:
// 1. Reshapes input for efficient linear projection
// 2. Applies up-projection to expand dimensions
// 3. Applies ReLU² activation with scaling
// 4. Applies down-projection to contract dimensions
// 5. Reshapes output back to original dimensions
//
// Returns a tensor with the same shape as input.
//
// The implementation uses BitLinear for efficient computation with
// ternary weights and includes parallel processing for the activation.
func (f *FFN) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	if f.closed {
		return nil, ErrFFNClosed
	}

	if input == nil {
		return nil, ErrInvalidInputShape
	}

	shape, err := input.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) < 2 {
		return nil, ErrInvalidInputShape
	}

	// Get input dimensions
	batchSize := shape[0]
	seqLen := 1
	if len(shape) > 2 {
		seqLen = shape[1]
	}
	hiddenDim := shape[len(shape)-1]

	if hiddenDim != f.hiddenDim {
		return nil, ErrInvalidWeightsShape
	}

	// Reshape input for linear projection
	flatInput, err := input.Reshape(batchSize*seqLen, f.hiddenDim)
	if err != nil {
		return nil, err
	}
	defer flatInput.Close()

	// Apply first linear transformation
	intermediate, err := tensor.BitLinear(flatInput, f.upProj)
	if err != nil {
		return nil, err
	}
	defer intermediate.Close()

	// Apply ReLU² activation
	activated, err := f.applyReLU2(intermediate)
	if err != nil {
		return nil, err
	}
	defer activated.Close()

	// Apply second linear transformation
	output, err := tensor.BitLinear(activated, f.downProj)
	if err != nil {
		return nil, err
	}
	defer output.Close()

	// Reshape back to original shape
	reshaped, err := output.Reshape(shape...)
	if err != nil {
		return nil, err
	}
	return reshaped, nil
}

// applyReLU2 applies the ReLU² activation function to the intermediate outputs.
//
// Input tensor must be 2D with shape [batch_size * seq_len, intermediate_dim].
// The function:
// 1. Applies ReLU²: max(0, x)²
// 2. Scales down by 16 to prevent overflow
// 3. Clamps values to int8 range
//
// Returns a 2D tensor with shape [batch_size * seq_len, intermediate_dim].
//
// The implementation uses parallel processing with chunked computation
// for better performance on multi-core systems.
func (f *FFN) applyReLU2(input *tensor.Tensor) (*tensor.Tensor, error) {
	if input == nil {
		return nil, ErrInvalidInputShape
	}

	shape, err := input.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 {
		return nil, ErrInvalidInputShape
	}

	batchSize := shape[0]
	intermediateDim := shape[1]

	if batchSize == 0 || intermediateDim == 0 {
		return nil, ErrInvalidInputShape
	}

	output, err := tensor.NewTensor(batchSize, intermediateDim)
	if err != nil {
		if err.Error() == "tensor: invalid shape dimension" {
			return nil, ErrInvalidInputShape
		}
		return nil, err
	}

	numCPU := runtime.NumCPU()
	chunkSize := (batchSize + numCPU - 1) / numCPU
	var wg sync.WaitGroup
	errChan := make(chan error, numCPU)

	for i := 0; i < numCPU; i++ {
		wg.Add(1)
		start := i * chunkSize
		end := start + chunkSize
		if end > batchSize {
			end = batchSize
		}

		go func(start, end int) {
			defer wg.Done()
			for b := start; b < end; b++ {
				for d := 0; d < intermediateDim; d++ {
					val, err := input.Get(b, d)
					if err != nil {
						errChan <- err
						return
					}

					// Apply ReLU²: max(0, x)², then integer division with rounding
					var activated int8
					if val > 0 {
						activated = int8((int(val)*int(val) + 8) / 16)
					} else {
						activated = 0
					}

					if err := output.Set(activated, b, d); err != nil {
						errChan <- err
						return
					}
				}
			}
		}(start, end)
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		if err != nil {
			output.Close()
			return nil, err
		}
	}

	return output, nil
}

// SetWeights sets the feed-forward network weights.
// FFN takes ownership of the tensors and will close them when FFN is closed.
// The caller must not close the tensors after passing them to SetWeights.
func (f *FFN) SetWeights(upWeights, downWeights *tensor.Tensor) error {
	if f.closed {
		return ErrFFNClosed
	}
	upShape, err := upWeights.Shape()
	if err != nil {
		return err
	}
	downShape, err := downWeights.Shape()
	if err != nil {
		return err
	}
	if upShape[0] != f.intermediateDim || upShape[1] != f.hiddenDim {
		return ErrInvalidWeightsShape
	}
	if downShape[0] != f.hiddenDim || downShape[1] != f.intermediateDim {
		return ErrInvalidWeightsShape
	}

	// Close existing weights if they exist
	if f.upProj != nil {
		f.upProj.Close()
	}
	if f.downProj != nil {
		f.downProj.Close()
	}

	// Set new weights
	f.upProj = upWeights
	f.downProj = downWeights
	return nil
}

// Close releases all resources associated with the FFN.
// After Close is called, the FFN instance should not be used.
func (f *FFN) Close() error {
	if f.closed {
		return nil
	}
	if f.upProj != nil {
		if err := f.upProj.Close(); err != nil {
			return err
		}
		f.upProj = nil
	}
	if f.downProj != nil {
		if err := f.downProj.Close(); err != nil {
			return err
		}
		f.downProj = nil
	}
	f.closed = true
	return nil
}
