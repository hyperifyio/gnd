package math

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

func TestValidateShape(t *testing.T) {
	tests := []struct {
		name        string
		shape       []int
		expectedDim int
		wantErr     bool
	}{
		{
			name:        "valid shape",
			shape:       []int{2, 3, 4},
			expectedDim: 3,
			wantErr:     false,
		},
		{
			name:        "empty shape",
			shape:       []int{},
			expectedDim: 3,
			wantErr:     true,
		},
		{
			name:        "zero dimension",
			shape:       []int{2, 0, 4},
			expectedDim: 3,
			wantErr:     false,
		},
		{
			name:        "negative dimension",
			shape:       []int{2, -3, 4},
			expectedDim: 3,
			wantErr:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.name == "negative dimension" || tt.name == "zero dimension" {
				defer func() {
					if r := recover(); r == nil {
						t.Errorf("expected panic for %s, but did not panic", tt.name)
					}
				}()
			}
			tensor := tensor.NewTensor(tt.shape...)
			if tt.name != "negative dimension" && tt.name != "zero dimension" {
				err := ValidateShape(tensor, tt.expectedDim)
				if (err != nil) != tt.wantErr {
					t.Errorf("ValidateShape() error = %v, wantErr %v", err, tt.wantErr)
				}
			}
		})
	}
}

func TestValidateBatchSeqHidden(t *testing.T) {
	tests := []struct {
		name    string
		shape   []int
		wantErr bool
	}{
		{
			name:    "valid shape",
			shape:   []int{2, 3, 4},
			wantErr: false,
		},
		{
			name:    "wrong dimensions",
			shape:   []int{2, 3},
			wantErr: true,
		},
		{
			name:    "too many dimensions",
			shape:   []int{2, 3, 4, 5},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := tensor.NewTensor(tt.shape...)
			err := ValidateBatchSeqHidden(tensor, "test")
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateBatchSeqHidden() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateBatchHeadsSeqHead(t *testing.T) {
	tests := []struct {
		name    string
		shape   []int
		wantErr bool
	}{
		{
			name:    "valid shape",
			shape:   []int{2, 4, 3, 5},
			wantErr: false,
		},
		{
			name:    "wrong dimensions",
			shape:   []int{2, 4, 3},
			wantErr: true,
		},
		{
			name:    "too many dimensions",
			shape:   []int{2, 4, 3, 5, 6},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := tensor.NewTensor(tt.shape...)
			err := ValidateBatchHeadsSeqHead(tensor, "test")
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateBatchHeadsSeqHead() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateHiddenHidden(t *testing.T) {
	tests := []struct {
		name    string
		shape   []int
		wantErr bool
	}{
		{
			name:    "valid shape",
			shape:   []int{4, 4},
			wantErr: false,
		},
		{
			name:    "wrong dimensions",
			shape:   []int{4},
			wantErr: true,
		},
		{
			name:    "non-square matrix",
			shape:   []int{4, 5},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := tensor.NewTensor(tt.shape...)
			err := ValidateHiddenHidden(tensor, "test")
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateHiddenHidden() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateMatchingShapes(t *testing.T) {
	tests := []struct {
		name    string
		shape1  []int
		shape2  []int
		wantErr bool
	}{
		{
			name:    "matching shapes",
			shape1:  []int{2, 3, 4},
			shape2:  []int{2, 3, 4},
			wantErr: false,
		},
		{
			name:    "different shapes",
			shape1:  []int{2, 3, 4},
			shape2:  []int{2, 3, 5},
			wantErr: true,
		},
		{
			name:    "different dimensions",
			shape1:  []int{2, 3, 4},
			shape2:  []int{2, 3},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor1 := tensor.NewTensor(tt.shape1...)
			tensor2 := tensor.NewTensor(tt.shape2...)
			err := ValidateMatchingShapes(tensor1, tensor2, "test1", "test2")
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateMatchingShapes() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateHeadDimensions(t *testing.T) {
	tests := []struct {
		name    string
		hidden  int
		heads   int
		headDim int
		wantErr bool
	}{
		{
			name:    "valid dimensions",
			hidden:  64,
			heads:   8,
			headDim: 8,
			wantErr: false,
		},
		{
			name:    "invalid division",
			hidden:  65,
			heads:   8,
			headDim: 8,
			wantErr: true,
		},
		{
			name:    "too few heads",
			hidden:  64,
			heads:   0,
			headDim: 8,
			wantErr: true,
		},
		{
			name:    "too many heads",
			hidden:  64,
			heads:   33,
			headDim: 8,
			wantErr: true,
		},
		{
			name:    "head dim too small",
			hidden:  64,
			heads:   8,
			headDim: 7,
			wantErr: true,
		},
		{
			name:    "head dim too large",
			hidden:  64,
			heads:   8,
			headDim: 257,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateHeadDimensions(tt.hidden, tt.heads, tt.headDim)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateHeadDimensions() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
