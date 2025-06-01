package gguf

import (
	"encoding/binary"
	"errors"
	"math"
)

var (
	ErrFloat32TensorDataIndexOutOfRange = errors.New("Float32TensorData index out of range")
)

type Float32TensorData struct {
	bytes    []byte
	elements uint64
}

// NewFloat32TensorData constructs interface to BitNet I2S tersor data
func NewFloat32TensorData(
	bytes []byte,
	elements uint64,
) *Float32TensorData {
	return &Float32TensorData{
		bytes:    bytes,
		elements: elements,
	}
}

var _ TensorData = &Float32TensorData{}

// ValueFloat32 returns the value at index as a 32-bit floating point number, scaled if a scale exists
func (d *Float32TensorData) ValueFloat32(idx uint64) (float32, error) {
	if idx >= d.elements {
		return 0, ErrFloat32TensorDataIndexOutOfRange
	}
	byteIdx := idx * 4
	data := d.bytes[byteIdx : byteIdx+4]
	bits := binary.LittleEndian.Uint32(data)
	return math.Float32frombits(bits), nil
}

// Value returns the internal value at index as the internal value type
func (d *Float32TensorData) Value(idx uint64) (interface{}, error) {
	if idx >= d.elements {
		return 0, ErrFloat32TensorDataIndexOutOfRange
	}
	return d.ValueFloat32(idx)
}

// ValueTernary returns the internal value at index as a three-option value 0 = -1, 1 = 0 or 2 = 1
func (d *Float32TensorData) ValueTernary(idx uint64) (uint8, error) {
	if idx >= d.elements {
		return 0, ErrFloat32TensorDataIndexOutOfRange
	}
	var v float32
	var err error
	v, err = d.ValueFloat32(idx)
	if err != nil {
		return 0, nil
	}
	if v < 0 {
		return 0, nil
	}
	if v == 0 {
		return 1, nil
	}
	return 2, nil
}

// Scale returns the internal scale for the tensor values for ValueTernary()
func (d *Float32TensorData) Scale(idx uint64) (float32, error) {
	if idx >= d.elements {
		return 0, ErrFloat32TensorDataIndexOutOfRange
	}
	var v float32
	var err error
	v, err = d.ValueFloat32(idx)
	if err != nil {
		return 0, nil
	}
	if v < 0 {
		return -v, nil
	}
	return v, nil
}
