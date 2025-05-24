package math

import (
	"testing"
)

func TestTensorOps(t *testing.T) {
	// Create test instance
	ops := NewTensorOps(10, 8) // maxSeqLength=10, hiddenSize=8
	defer ops.Close()

	// Test ReshapeAndCopy
	t.Run("ReshapeAndCopy", func(t *testing.T) {
		// Create test data
		data := make([][]float32, 5)
		for i := range data {
			data[i] = make([]float32, 8)
			for j := range data[i] {
				data[i][j] = float32(i + j)
			}
		}

		// Test reshape and copy
		tensor := ops.ReshapeAndCopy(data, 1, 5, 8)

		// Verify tensor contents
		for i := 0; i < 5; i++ {
			for j := 0; j < 8; j++ {
				expected := float32(i + j)
				actual := float32(tensor.Get(0, i, j))
				if actual != expected {
					t.Errorf("tensor[%d][%d] = %f, want %f", i, j, actual, expected)
				}
			}
		}
	})

	// Test GetLastHiddenState
	t.Run("GetLastHiddenState", func(t *testing.T) {
		// Create test tensor
		tensor := ops.GetTensorFromPool()
		defer ops.PutTensorToPool(tensor)

		// Fill tensor with test data
		for i := 0; i < 5; i++ {
			for j := 0; j < 8; j++ {
				tensor.Set(int8(i+j), 0, i, j)
			}
		}

		// Get last hidden state
		lastHiddenState := ops.GetLastHiddenState(tensor, 5, 8)

		// Verify last hidden state
		if len(lastHiddenState) != 8 {
			t.Errorf("len(lastHiddenState) = %d, want 8", len(lastHiddenState))
		}

		for j := 0; j < 8; j++ {
			expected := float32(4 + j) // Last row (i=4) + column value
			if lastHiddenState[j] != expected {
				t.Errorf("lastHiddenState[%d] = %f, want %f", j, lastHiddenState[j], expected)
			}
		}
	})

	// Test tensor pool
	t.Run("TensorPool", func(t *testing.T) {
		// Get tensor from pool
		tensor1 := ops.GetTensorFromPool()
		if tensor1 == nil {
			t.Error("GetTensorFromPool returned nil")
		}

		// Put tensor back in pool
		ops.PutTensorToPool(tensor1)

		// Get another tensor from pool
		tensor2 := ops.GetTensorFromPool()
		if tensor2 == nil {
			t.Error("GetTensorFromPool returned nil")
		}

		// Verify tensors are different instances
		if tensor1 == tensor2 {
			t.Error("GetTensorFromPool returned same tensor instance")
		}
	})
}
