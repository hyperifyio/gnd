package gguf

import (
	"encoding/binary"
	"errors"
)

var (
	ErrFloat16TensorDataIndexOutOfRange = errors.New("Float16TensorData index out of range")
)

type Float16TensorData struct {
	bytes    []byte
	elements uint64
}

// NewFloat16TensorData constructs interface to BitNet I2S tersor data
func NewFloat16TensorData(
	bytes []byte,
	elements uint64,
) *Float16TensorData {
	return &Float16TensorData{
		bytes:    bytes,
		elements: elements,
	}
}

var _ TensorData = &Float16TensorData{}

// ValueFloat16 returns the value at index as a 32-bit floating point number, scaled if a scale exists
func (d *Float16TensorData) ValueFloat16(idx uint64) (uint16, error) {
	if idx >= d.elements {
		return 0, ErrFloat16TensorDataIndexOutOfRange
	}
	byteIdx := idx * 2
	data := d.bytes[byteIdx : byteIdx+2]
	bits := binary.LittleEndian.Uint16(data)
	return bits, nil
}

// ValueFloat32 returns the value at index as a 32-bit floating point number, scaled if a scale exists
func (d *Float16TensorData) ValueFloat32(idx uint64) (float32, error) {
	f, err := d.ValueFloat16(idx)
	if err != nil {
		return 0, err
	}
	return float16ToFloat32(f), nil
}

// Value returns the internal value at index as the internal value type
func (d *Float16TensorData) Value(idx uint64) (interface{}, error) {
	return d.ValueFloat16(idx)
}

// ValueTernary returns the internal value at index as a three-option value 0 = -1, 1 = 0 or 2 = 1
func (d *Float16TensorData) ValueTernary(idx uint64) (uint8, error) {
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
func (d *Float16TensorData) Scale(idx uint64) (float32, error) {
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
