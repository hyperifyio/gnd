// Package tensor_ops provides core tensor operations for BitNet math operations.
//
// # Quantized Tensor Utilities for BitNet
//
// This package provides helpers for reshaping, copying, and pooling quantized tensors.
//
// Key aspects:
//   - All tensors are int8, matching BitNet's quantized design
//   - Utilities are optimized for CPU efficiency and low memory use
//   - Not suitable for training or float32 inference
//
// Implementation details:
//   - Memory pooling for efficient tensor reuse
//   - Utilities for reshaping, copying, and extracting hidden states
//   - Proper clamping and quantization of values
//
// Related tasks and dependencies:
//   - #175: Implement Tensor Utility Functions (Core implementation)
//   - #182: Compute Scaled Dot-Product Attention (Depends on #175)
//   - #185: Feed-Forward Network (FFN) Sublayer (Depends on #175)
//   - #186: Integrate Attention Sublayer (Pre-Norm & Residual) (Depends on #175)
//   - #187: Integrate Feed-Forward Sublayer (Pre-Norm & Residual) (Depends on #175)
//
// Usage:
//   - Used throughout BitNet math package for tensor management
//   - Maintainers should not change quantization or pooling logic without full pipeline review
//
// Caveats:
//   - Quantization may cause saturation/clamping; tests should check for correct quantized output
//   - Any change must be validated against end-to-end BitNet inference
//   - Performance critical - changes should be benchmarked against existing implementation
//
// For more details, see BitNet issue #190 and the BitNet project documentation.
package tensor_ops

import (
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// TensorOps provides utility functions for common tensor operations
type TensorOps struct {
	// Memory pool for intermediate tensors
	tensorPool sync.Pool
}

// NewTensorOps creates a new TensorOps instance
func NewTensorOps(maxSeqLength, hiddenSize int) (*TensorOps, error) {
	// Create a sample tensor to verify dimensions
	_, err := tensor.NewTensor(1, maxSeqLength, hiddenSize)
	if err != nil {
		return nil, err
	}
	return &TensorOps{
		tensorPool: sync.Pool{
			New: func() interface{} {
				t, _ := tensor.NewTensor(1, maxSeqLength, hiddenSize)
				return t
			},
		},
	}, nil
}

// ReshapeAndCopy creates a new tensor with the given shape and copies data from a float32 slice
func (t *TensorOps) ReshapeAndCopy(data [][]float32, batchSize, seqLength, hiddenSize int) (*tensor.Tensor, error) {
	newTensor, err := tensor.NewTensor(batchSize, seqLength, hiddenSize)
	if err != nil {
		return nil, err
	}
	// Copy data into tensor
	for i := 0; i < seqLength; i++ {
		for j := 0; j < hiddenSize; j++ {
			val := data[i][j]
			if val > 127 {
				val = 127
			} else if val < -128 {
				val = -128
			}
			if err := newTensor.Set(int8(val), 0, i, j); err != nil {
				return nil, err
			}
		}
	}
	return newTensor, nil
}

// GetLastHiddenState extracts the last hidden state from a tensor
func (t *TensorOps) GetLastHiddenState(tensor *tensor.Tensor, seqLength, hiddenSize int) ([]float32, error) {
	lastHiddenState := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		val, err := tensor.Get(0, seqLength-1, i)
		if err != nil {
			return nil, err
		}
		lastHiddenState[i] = float32(val)
	}
	return lastHiddenState, nil
}

// GetTensorFromPool gets a tensor from the pool
func (t *TensorOps) GetTensorFromPool() *tensor.Tensor {
	return t.tensorPool.Get().(*tensor.Tensor)
}

// PutTensorToPool returns a tensor to the pool
func (t *TensorOps) PutTensorToPool(tensor *tensor.Tensor) {
	t.tensorPool.Put(tensor)
}

// Close releases resources used by TensorOps
func (t *TensorOps) Close() {
	// Clear the pool
	t.tensorPool = sync.Pool{}
}
