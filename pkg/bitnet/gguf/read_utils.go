package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
)

// readUint8 reads 8-bit unsigned integer
func readUint8(r io.Reader, value *uint8) error {
	if err := binary.Read(r, binary.LittleEndian, value); err != nil {
		return fmt.Errorf("readUint8: failed: %w", err)
	}
	return nil
}

// readInt8 reads 8-bit signed integer
func readInt8(r io.Reader, value *int8) error {
	if err := binary.Read(r, binary.LittleEndian, value); err != nil {
		return fmt.Errorf("readInt8: failed: %w", err)
	}
	return nil
}

// / readUint16 reads 16-bit unsigned integer
func readUint16(r io.Reader, value *uint16) error {
	if err := binary.Read(r, binary.LittleEndian, value); err != nil {
		return fmt.Errorf("readUint16: failed: %w", err)
	}
	return nil
}

// readInt16 reads 16-bit signed integer
func readInt16(r io.Reader, value *int16) error {
	if err := binary.Read(r, binary.LittleEndian, value); err != nil {
		return fmt.Errorf("readInt16: failed: %w", err)
	}
	return nil
}

// readUint32 reads 32-bit unsigned integer
func readUint32(r io.Reader, value *uint32) error {
	if err := binary.Read(r, binary.LittleEndian, value); err != nil {
		return fmt.Errorf("readUint32: failed: %w", err)
	}
	return nil
}

// readInt32 reads 32-bit signed integer
func readInt32(r io.Reader, value *int32) error {
	if err := binary.Read(r, binary.LittleEndian, value); err != nil {
		return fmt.Errorf("readInt32: failed: %w", err)
	}
	return nil
}

// readInt64 reads 64-bit unsigned integer
func readInt64(r io.Reader, value *int64) error {
	if err := binary.Read(r, binary.LittleEndian, value); err != nil {
		return fmt.Errorf("readInt64: failed: %w", err)
	}
	return nil
}

// readUint64 reads 64-bit unsigned integer
func readUint64(r io.Reader, value *uint64) error {
	if err := binary.Read(r, binary.LittleEndian, value); err != nil {
		return fmt.Errorf("readUint64: failed: %w", err)
	}
	return nil
}

// readFloat32 reads 32-bit floating point number
func readFloat32(r io.Reader, value *float32) error {
	if err := binary.Read(r, binary.LittleEndian, value); err != nil {
		return fmt.Errorf("readFloat32: failed: %w", err)
	}
	return nil
}

// readFloat64
func readFloat64(r io.Reader, value *float64) error {
	if err := binary.Read(r, binary.LittleEndian, value); err != nil {
		return fmt.Errorf("readFloat64: failed: %w", err)
	}
	return nil
}

// readSliceOfFloat32 reads 32-bit floating point numbers
func readSliceOfFloat32(r io.Reader, data []float32) error {
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		return fmt.Errorf("readSliceOfFloat32: failed: %w", err)
	}
	return nil
}

// readSliceOfUint8 reads 8-bit unsigned integers
func readSliceOfUint8(r io.Reader, data []uint8) error {
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		return fmt.Errorf("readSliceOfUint8: failed: %w", err)
	}
	return nil
}

// readSliceOfUint16 reads 16-bit unsigned integers
func readSliceOfUint16(r io.Reader, data []uint16) error {
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		return fmt.Errorf("readSliceOfUint16: failed: %w", err)
	}
	return nil
}

// readSliceOfFloat16 reads 16-bit floating point numbers
func readSliceOfFloat16(r io.Reader, data []float32) error {

	size := len(data)

	// Read F16 data
	// FIXME: We could optimize this by reading directly into a slice of float32
	h := make([]uint16, size)
	err := readSliceOfUint16(r, h)
	if err != nil {
		return fmt.Errorf("readSliceOfFloat16: failed: %w", err)
	}

	// Convert to float32
	for i, v := range h {
		data[i] = float16ToFloat32(v)
	}

	return nil
}

func sliceOfByte(buf []byte, offset, length uint64) []byte {
	return buf[offset : offset+length]
}
