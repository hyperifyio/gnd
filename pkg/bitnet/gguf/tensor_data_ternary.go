package gguf

import (
	"encoding/binary"
	"errors"
	"math"
)

var (
	ErrTernaryTensorDataIndexOutOfRange = errors.New("TernaryTensorData index out of range")
)

type TernaryTensorData struct {
	bytes    []byte
	elements uint64
}

// NewTernaryTensorData constructs interface to BitNet I2S tersor data
func NewTernaryTensorData(
	bytes []byte,
	elements uint64,
) *TernaryTensorData {
	return &TernaryTensorData{
		bytes,
		elements,
	}
}

var _ TensorData = &TernaryTensorData{}

// ValueFloat32 returns the value at index as a 32-bit floating point number, scaled if a scale exists
func (d *TernaryTensorData) ValueFloat32(idx uint64) (float32, error) {
	if idx >= d.elements {
		return 0, ErrTernaryTensorDataIndexOutOfRange
	}

	var err error
	var t uint8
	var s float32

	t, err = d.ValueTernary(idx)
	if err != nil {
		return 0, err
	}

	s, err = d.Scale(idx)
	if err != nil {
		return 0, err
	}

	return float32(t) * s, nil
}

// Value returns the internal value at index as the internal value type
func (d *TernaryTensorData) Value(idx uint64) (interface{}, error) {
	if idx >= d.elements {
		return 0, ErrTernaryTensorDataIndexOutOfRange
	}
	return d.ValueTernary(idx)
}

// ValueTernary returns the internal value at index as a three-option value 0 = -1, 1 = 0 or 2 = 1
func (d *TernaryTensorData) ValueTernary(idx uint64) (uint8, error) {
	if idx >= d.elements {
		return 0, ErrTernaryTensorDataIndexOutOfRange
	}
	byteIdx := idx / 4
	bitIdx := idx % 4
	b := d.bytes[byteIdx]
	w := (b >> bitIdx * 2) & 0x03
	return w, nil
}

// Scale returns the internal scale for the tensor values, otherwise 1 if no scale
func (d *TernaryTensorData) Scale(idx uint64) (float32, error) {
	if idx >= d.elements {
		return 0, ErrTernaryTensorDataIndexOutOfRange
	}
	byteIdx := d.elements / 4
	data := d.bytes[byteIdx : byteIdx+4]
	bits := binary.LittleEndian.Uint32(data)
	return math.Float32frombits(bits), nil
}
