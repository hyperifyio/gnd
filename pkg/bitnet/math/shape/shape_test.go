package shape_test

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/math/shape"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestValidateShape(t *testing.T) {
	tests := []struct {
		name         string
		shape        []int
		expectedDims int
		expectedErr  error
	}{
		{
			name:         "valid shape",
			shape:        []int{2, 3, 4},
			expectedDims: 3,
			expectedErr:  nil,
		},
		{
			name:         "invalid dimensions",
			shape:        []int{2, 3},
			expectedDims: 3,
			expectedErr:  shape.ErrInvalidDimensions,
		},
		{
			name:         "empty shape",
			shape:        []int{},
			expectedDims: 3,
			expectedErr:  shape.ErrInvalidDimensions,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := shape.ValidateShape(tt.shape, tt.expectedDims)
			if tt.expectedErr != nil {
				assert.ErrorIs(t, err, tt.expectedErr)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateBatchSeqHidden(t *testing.T) {
	tests := []struct {
		name        string
		shape       []int
		expectedErr error
	}{
		{
			name:        "valid shape",
			shape:       []int{2, 3, 4},
			expectedErr: nil,
		},
		{
			name:        "invalid dimensions",
			shape:       []int{2, 3},
			expectedErr: shape.ErrInvalidDimensions,
		},
		{
			name:        "invalid batch size",
			shape:       []int{0, 3, 4},
			expectedErr: shape.ErrInvalidInputShape,
		},
		{
			name:        "invalid sequence length",
			shape:       []int{2, 0, 4},
			expectedErr: shape.ErrInvalidInputShape,
		},
		{
			name:        "invalid hidden dim",
			shape:       []int{2, 3, 0},
			expectedErr: shape.ErrInvalidInputShape,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := shape.ValidateBatchSeqHiddenShape(tt.shape, "test_tensor")
			if tt.expectedErr != nil {
				assert.ErrorIs(t, err, tt.expectedErr)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateBatchHeadsSeqHead(t *testing.T) {
	tests := []struct {
		name        string
		shape       []int
		expectedErr error
	}{
		{
			name:        "valid shape",
			shape:       []int{2, 4, 3, 8},
			expectedErr: nil,
		},
		{
			name:        "invalid dimensions",
			shape:       []int{2, 4, 3},
			expectedErr: shape.ErrInvalidDimensions,
		},
		{
			name:        "invalid batch size",
			shape:       []int{0, 4, 3, 8},
			expectedErr: shape.ErrInvalidInputShape,
		},
		{
			name:        "invalid head count",
			shape:       []int{2, 0, 3, 8},
			expectedErr: shape.ErrInvalidHeadCount,
		},
		{
			name:        "invalid sequence length",
			shape:       []int{2, 4, 0, 8},
			expectedErr: shape.ErrInvalidInputShape,
		},
		{
			name:        "invalid head dimension",
			shape:       []int{2, 4, 3, 0},
			expectedErr: shape.ErrInvalidHeadDimension,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := shape.ValidateBatchHeadsSeqHeadShape(tt.shape, "test_tensor")
			if tt.expectedErr != nil {
				assert.ErrorIs(t, err, tt.expectedErr)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateHiddenHidden(t *testing.T) {
	tests := []struct {
		name        string
		shape       []int
		expectedErr error
	}{
		{
			name:        "valid shape",
			shape:       []int{4, 4},
			expectedErr: nil,
		},
		{
			name:        "invalid dimensions",
			shape:       []int{4},
			expectedErr: shape.ErrInvalidDimensions,
		},
		{
			name:        "non-square matrix",
			shape:       []int{4, 5},
			expectedErr: shape.ErrNonSquareMatrix,
		},
		{
			name:        "zero dimensions",
			shape:       []int{0, 0},
			expectedErr: shape.ErrInvalidInputShape,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := shape.ValidateHiddenHiddenShape(tt.shape, "test_tensor")
			if tt.expectedErr != nil {
				assert.ErrorIs(t, err, tt.expectedErr)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateMatchingShapes(t *testing.T) {
	tests := []struct {
		name        string
		shape1      []int
		shape2      []int
		expectedErr error
	}{
		{
			name:        "matching shapes",
			shape1:      []int{2, 3, 4},
			shape2:      []int{2, 3, 4},
			expectedErr: nil,
		},
		{
			name:        "different dimensions",
			shape1:      []int{2, 3},
			shape2:      []int{2, 3, 4},
			expectedErr: shape.ErrDimensionMismatch,
		},
		{
			name:        "different sizes",
			shape1:      []int{2, 3, 4},
			shape2:      []int{2, 3, 5},
			expectedErr: shape.ErrDimensionMismatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := shape.ValidateMatchingShapes(tt.shape1, tt.shape2, "test_tensor1", "test_tensor2")
			if tt.expectedErr != nil {
				assert.ErrorIs(t, err, tt.expectedErr)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateHeadDimensions(t *testing.T) {
	tests := []struct {
		name        string
		hiddenDim   int
		numHeads    int
		headDim     int
		expectedErr error
	}{
		{
			name:        "valid dimensions",
			hiddenDim:   64,
			numHeads:    4,
			headDim:     16,
			expectedErr: nil,
		},
		{
			name:        "invalid head count",
			hiddenDim:   64,
			numHeads:    0,
			headDim:     16,
			expectedErr: shape.ErrInvalidHeadCount,
		},
		{
			name:        "invalid head dimension",
			hiddenDim:   64,
			numHeads:    4,
			headDim:     0,
			expectedErr: shape.ErrInvalidHeadDimension,
		},
		{
			name:        "dimension mismatch",
			hiddenDim:   64,
			numHeads:    4,
			headDim:     15,
			expectedErr: shape.ErrHiddenDimMismatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := shape.ValidateHeadDimensions(tt.hiddenDim, tt.numHeads, tt.headDim)
			if tt.expectedErr != nil {
				assert.ErrorIs(t, err, tt.expectedErr)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
