package testutil

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/stretchr/testify/require"
)

// CreateTensor creates a tensor with the given data and dimensions.
// It is a helper function for tests.
func CreateTensor(t *testing.T, data []int8, dims ...int) *tensor.Tensor {
	t.Helper()
	tensor, err := tensor.NewTensorFromData(data, dims[0])
	require.NoError(t, err)
	return tensor
}
