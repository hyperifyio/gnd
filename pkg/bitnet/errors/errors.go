package errors

import "errors"

var (
	// ErrInvalidHiddenDim is returned when the hidden dimension is invalid
	ErrInvalidHiddenDim = errors.New("invalid hidden dimension")
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
	// ErrInvalidNumHeads is returned when the number of attention heads is invalid
	ErrInvalidNumHeads = errors.New("invalid number of attention heads")
	// ErrInvalidNumKVHeads is returned when the number of key-value heads is invalid
	ErrInvalidNumKVHeads = errors.New("invalid number of key-value heads")
	// ErrInvalidHeadDim is returned when the head dimension is invalid
	ErrInvalidHeadDim = errors.New("invalid head dimension")
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
	// ErrLayerClosed is returned when a bitnet layer is closed
	ErrLayerClosed = errors.New("bitnet: layer is closed")
	// ErrTensorClosed is returned when a tensor is closed
	ErrTensorClosed = errors.New("tensor: operation attempted on closed tensor")
)
