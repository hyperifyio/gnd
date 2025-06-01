package gguf

import (
	"fmt"
)

// calculateTensorElements calculates number of elements
func calculateTensorElements(tensor *TensorInfo) uint64 {
	N := uint64(1)
	for _, dim := range tensor.Shape {
		N *= dim
	}
	return N
}

// calculateTensorRowCount calculates number of rows
func calculateTensorRowCount(tensor *TensorInfo) uint64 {
	length := len(tensor.Shape)
	if length == 0 {
		return uint64(0)
	}
	if length == 1 {
		return uint64(1)
	}
	prod := uint64(1)
	for i := 0; i < length-1; i++ {
		prod *= tensor.Shape[i]
	}
	return prod
}

// calculateTensorColumnCount calculates number of columns on a row
func calculateTensorColumnCount(tensor *TensorInfo) uint64 {
	length := len(tensor.Shape)
	if length == 0 {
		return uint64(0)
	}
	return tensor.Shape[length-1]
}

// calculateTensorRowSize calculates tensor row size in bytes from column size
func calculateTensorRowSize(tensor *TensorInfo, n uint64) (uint64, error) {
	tensorType := tensor.Type
	if n == 0 {
		return 0, fmt.Errorf("gguf: tensor of type %d has no data", tensorType)
	}
	switch tensorType {
	case GGML_TYPE_F32:
		return n * 4, nil // 4 bytes per float32
	case GGML_TYPE_F16:
		return n * 2, nil // 2 bytes per float16
	case GGML_TYPE_I2_S:
		return n / 4, nil // `n` * 2 bits + 4 bytes (float32) aligned to next 32 bytes`
	default:
		return 0, fmt.Errorf("gguf: unsupported tensor type %d", tensorType)
	}
}

// calculateTensorDataSize calculates tensor row size in bytes from column size
func calculateTensorDataSize(tensor *TensorInfo, n, alignment uint64) (uint64, error) {
	tensorType := tensor.Type
	if n == 0 {
		return 0, fmt.Errorf("gguf: tensor of type %d has no data", tensorType)
	}
	switch tensorType {
	case GGML_TYPE_F32:
		return n * 4, nil // 4 bytes per float32
	case GGML_TYPE_F16:
		return n * 2, nil // 2 bytes per float16
	case GGML_TYPE_I2_S:
		return ((n/4 + 32) / alignment) * alignment, nil // `n` * 2 bits + 4 bytes (float32) aligned to next 32 bytes`
	default:
		return 0, fmt.Errorf("gguf: unsupported tensor type %d", tensorType)
	}
}
