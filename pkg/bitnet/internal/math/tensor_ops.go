package math

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
func NewTensorOps(maxSeqLength, hiddenSize int) *TensorOps {
	return &TensorOps{
		tensorPool: sync.Pool{
			New: func() interface{} {
				return tensor.NewTensor(1, maxSeqLength, hiddenSize)
			},
		},
	}
}

// ReshapeAndCopy creates a new tensor with the given shape and copies data from a float32 slice
func (t *TensorOps) ReshapeAndCopy(data [][]float32, batchSize, seqLength, hiddenSize int) tensor.TensorOperations {
	newTensor := tensor.NewTensor(batchSize, seqLength, hiddenSize)
	// Copy data into tensor
	for i := 0; i < seqLength; i++ {
		for j := 0; j < hiddenSize; j++ {
			val := data[i][j]
			if val > 127 {
				val = 127
			} else if val < -128 {
				val = -128
			}
			newTensor.Set(int8(val), 0, i, j)
		}
	}
	return newTensor
}

// GetLastHiddenState extracts the last hidden state from a tensor
func (t *TensorOps) GetLastHiddenState(tensor tensor.TensorReader, seqLength, hiddenSize int) []float32 {
	lastHiddenState := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		lastHiddenState[i] = float32(tensor.Get(0, seqLength-1, i))
	}
	return lastHiddenState
}

// GetTensorFromPool gets a tensor from the pool
func (t *TensorOps) GetTensorFromPool() tensor.TensorOperations {
	return t.tensorPool.Get().(tensor.TensorOperations)
}

// PutTensorToPool returns a tensor to the pool
func (t *TensorOps) PutTensorToPool(tensor tensor.TensorOperations) {
	t.tensorPool.Put(tensor)
}

// Close releases resources used by TensorOps
func (t *TensorOps) Close() {
	// Clear the pool
	t.tensorPool = sync.Pool{}
}
