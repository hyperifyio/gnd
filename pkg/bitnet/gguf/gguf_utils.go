package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
)

// readLength reads 64-bit unsigned integer
func readLength(r io.Reader, logger *log.Logger, maxLength uint64) (uint64, error) {

	var keyLen uint64

	err := readUint64(r, &keyLen)
	if err != nil {
		return 0, fmt.Errorf("readLength: failed to read key length: %w", err)
	}

	// Safety check - cap at 1MB
	if keyLen > maxLength {
		logger.Printf("[DEBUG] Length too large: %d > %d", keyLen, maxLength)
		return 0, fmt.Errorf("readLength: string too long")
	}

	return keyLen, nil
}

// readLengthAndString reads a string with 64-bit length and UTF-8 string
func readLengthAndString(r io.Reader, logger *log.Logger, maxLength uint64) (uint64, string, error) {

	strlen, err := readLength(r, logger, maxLength)
	if err != nil {
		return 0, "", fmt.Errorf("readLengthAndString: failed to read length: %w", err)
	}

	// Handle empty strings - no need to read data
	if strlen == 0 {
		//logger.Printf("[DEBUG] Successfully read empty string")
		return 0, "", nil
	}

	bytes := make([]byte, strlen)
	if _, err = io.ReadFull(r, bytes); err != nil {
		return 0, "", fmt.Errorf("readLengthAndString: failed to read bytes: %w", err)
	}

	//logger.Printf("[DEBUG] Successfully read bytes: %s (length: %d)", string(bytes), strlen)
	return strlen, string(bytes), nil
}

// readMetadataKey reads 64-bit length and UTF-8 string
func readMetadataKey(r io.Reader, logger *log.Logger) (string, error) {
	_, key, err := readLengthAndString(r, logger, 1024*1024)
	if err != nil {
		return "", fmt.Errorf("failed to read key: %w", err)
	}
	return key, nil
}

// readMetadataValueType reads a metadata value type from the reader.
func readMetadataValueType(r io.Reader) (MetadataValueType, error) {

	var valueType uint32

	err := readUint32(r, &valueType)
	if err != nil {
		return 0, ErrReadMetadataValueType
	}

	// Handle BitNet-specific types (>= 128) by treating them as int32
	if valueType >= 128 {
		log.Printf("[DEBUG] Converting BitNet-specific type %d to int32", valueType)
		return MetadataValueTypeInt32, nil
	}

	//// Handle special token type (21) as a special case
	//if valueType == 21 {
	//	log.Printf("[DEBUG] Detected special token type 21")
	//	return MetadataValueTypeInt32, nil // Treat as int32 for now
	//}

	// Log unexpected value types between 14 and 127 for debugging
	if valueType > 13 {
		log.Printf("[DEBUG] Unexpected metadata value type: %d", valueType)
	}

	return MetadataValueType(valueType), nil
}

// readArrayCount reads the array count from the reader (64-bit unsigned int).
func readArrayCount(r io.Reader) (uint64, error) {
	var count uint64
	err := readUint64(r, &count)
	if err != nil {
		return 0, fmt.Errorf("gguf: failed to read array count: %w", err)
	}
	return count, nil
}

// readMetadataArray reads an array value from the reader.
func readMetadataArray(r io.Reader, logger *log.Logger) (interface{}, error) {

	// Read array element type
	elementType, e1 := readMetadataValueType(r)
	if e1 != nil {
		log.Printf("[DEBUG] Failed to read array element type: %v", e1)
		return nil, e1
	}

	// Read array count
	count, e2 := readArrayCount(r)
	if e2 != nil {
		log.Printf("[DEBUG] Failed to read array count: %v", e2)
		return nil, e2
	}

	if elementType == MetadataValueTypeString {
		return parseStringArray(r, count, logger)
	}

	return parseTypedArray(r, elementType, count)
}

// readMetadataValue reads a metadata value (non-array) from the reader.
func readMetadataValue(r io.Reader, logger *log.Logger, valueType MetadataValueType) (interface{}, error) {

	if valueType == MetadataValueTypeArray {
		return readMetadataArray(r, logger)
	}

	var value interface{}
	var err error

	switch valueType {

	case MetadataValueTypeUint8:
		var v uint8
		err = readUint8(r, &v)
		value = v
		break

	case MetadataValueTypeInt8:
		var v int8
		err = readInt8(r, &v)
		value = v
		break

	case MetadataValueTypeUint16:
		var v uint16
		err = readUint16(r, &v)
		value = v
		break

	case MetadataValueTypeInt16:
		var v int16
		err = readInt16(r, &v)
		value = v
		break

	case MetadataValueTypeUint32:
		var v uint32
		err = readUint32(r, &v)
		value = v
		break

	case MetadataValueTypeInt32:
		var v int32
		err = readInt32(r, &v)
		value = v
		break

	case MetadataValueTypeFloat32:
		var v float32
		err = readFloat32(r, &v)
		value = v
		break

	case MetadataValueTypeBool:
		var v uint8
		err = readUint8(r, &v)
		if err == nil {
			value = v != 0 // Convert to bool
		}
		break

	case MetadataValueTypeString:
		_, value, err = readLengthAndString(r, logger, 1024*1024)
		break

	case MetadataValueTypeUint64:
		var v uint64
		err = readUint64(r, &v)
		value = v
		break

	case MetadataValueTypeInt64:
		var v int64
		err = readInt64(r, &v)
		value = v
		break

	case MetadataValueTypeFloat64:
		var v float64
		err = readFloat64(r, &v)
		value = v
		break

	case MetadataValueTypeBinary:

		// For binary, read length-prefixed bytes (uint32)
		var length uint32
		lenErr := readUint32(r, &length)
		if lenErr != nil {
			log.Printf("[DEBUG] Failed to read binary length: %v", lenErr)
			return nil, ErrReadValueBytes
		}
		if length > 128<<20 { // 128MB safety cap
			log.Printf("[DEBUG] Binary data too large: %d bytes", length)
			return nil, ErrStringTooLong
		}

		buf := make([]byte, length)
		if _, bufErr := io.ReadFull(r, buf); bufErr != nil {
			log.Printf("[DEBUG] Failed to read binary data: %v", bufErr)
			return nil, ErrReadValueBytes
		}
		return buf, nil

	default:
		log.Printf("[DEBUG] Unsupported metadata value type: %d", valueType)
		return nil, ErrUnsupportedMetadataValueType
	}

	if err != nil {
		log.Printf("[DEBUG] Failed to read value: %v", err)
		return nil, ErrReadValueBytes
	}
	return value, nil

}

// getElementSize returns the size of an element based on its type.
func getElementSize(elementType MetadataValueType) (int, error) {
	switch elementType {
	case 0: // uint8
		return 1, nil
	case 1: // int8
		return 1, nil
	case 2: // uint16
		return 2, nil
	case 3: // int16
		return 2, nil
	case 4: // uint32
		return 4, nil
	case 5: // int32
		return 4, nil
	case 6: // float32
		return 4, nil
	case 7: // bool
		return 1, nil
	case 10: // uint64
		return 8, nil
	case 11: // int64
		return 8, nil
	case 12: // float64
		return 8, nil
	case 36: // I2_S (BitNet ternary)
		return 1, nil // Each byte contains 4 ternary weights
	default:
		if elementType >= 128 {
			return 4, nil // BitNet/LLM GGUF quirk: treat as int32
		}
		return 1, nil // fallback
	}
}

// parseStringArray reads an array of length-prefixed strings from the reader.
func parseStringArray(r io.Reader, count uint64, logger *log.Logger) ([]string, error) {

	if count == 0 {
		return nil, nil
	}

	// Safety check for array size
	if count > 1024*1024 {
		return nil, ErrArrayAllocationTooLarge
	}

	arr := make([]string, int(count))

	// Track total bytes read for debugging
	var totalBytes uint64

	for i := uint64(0); i < count; i++ {

		strlen, value, err := readLengthAndString(r, logger, 1024*1024)
		if err != nil {
			log.Printf("[DEBUG] Failed to read string bytes at index %d: %v", i, err)
			return nil, fmt.Errorf("gguf: failed to read length and string: %w", err)
		}

		arr[i] = value
		totalBytes += strlen
	}

	return arr, nil
}

// parseTypedArray reads an array of values of a specific type.
func parseTypedArray(r io.Reader, elementType MetadataValueType, count uint64) (interface{}, error) {

	// Get element size
	elementSize, err := getElementSize(elementType)
	if err != nil {
		return nil, err
	}

	// Safety check for total size
	if count > math.MaxInt32 {
		return nil, ErrArrayAllocationTooLarge
	}

	// Calculate total bytes needed, checking for overflow
	if count > math.MaxInt32/uint64(elementSize) {
		return nil, fmt.Errorf("gguf: array size too large: %d * %d", count, elementSize)
	}
	totalBytes := int(count) * elementSize

	// Safety check for maximum allocation size (1GB)
	if totalBytes > 1024*1024*1024 {
		return nil, fmt.Errorf("gguf: array allocation too large: %d bytes", totalBytes)
	}

	valueBytes := make([]byte, totalBytes)
	if _, err := io.ReadFull(r, valueBytes); err != nil {
		return nil, err
	}

	// Helper function to check bounds
	checkBounds := func(idx int) bool {
		return idx*elementSize+elementSize <= len(valueBytes)
	}

	// Parse based on element type
	switch elementType {
	case MetadataValueTypeUint8:
		arr := make([]uint8, int(count))
		for i := range arr {
			if !checkBounds(i) {
				return nil, fmt.Errorf("gguf: value bytes too short")
			}
			arr[i] = valueBytes[i*elementSize]
		}
		return arr, nil
	case MetadataValueTypeInt8:
		arr := make([]int8, int(count))
		for i := range arr {
			if !checkBounds(i) {
				return nil, fmt.Errorf("gguf: value bytes too short")
			}
			arr[i] = int8(valueBytes[i*elementSize])
		}
		return arr, nil
	case MetadataValueTypeUint16:
		arr := make([]uint16, int(count))
		for i := range arr {
			if !checkBounds(i) {
				return nil, fmt.Errorf("gguf: value bytes too short")
			}
			arr[i] = binary.LittleEndian.Uint16(valueBytes[i*elementSize:])
		}
		return arr, nil
	case MetadataValueTypeInt16:
		arr := make([]int16, int(count))
		for i := range arr {
			if !checkBounds(i) {
				return nil, fmt.Errorf("gguf: value bytes too short")
			}
			arr[i] = int16(binary.LittleEndian.Uint16(valueBytes[i*elementSize:]))
		}
		return arr, nil
	case MetadataValueTypeUint32:
		arr := make([]uint32, int(count))
		for i := range arr {
			if !checkBounds(i) {
				return nil, fmt.Errorf("gguf: value bytes too short")
			}
			arr[i] = binary.LittleEndian.Uint32(valueBytes[i*elementSize:])
		}
		return arr, nil
	case MetadataValueTypeInt32:
		arr := make([]int32, int(count))
		for i := range arr {
			if !checkBounds(i) {
				return nil, fmt.Errorf("gguf: value bytes too short")
			}
			arr[i] = int32(binary.LittleEndian.Uint32(valueBytes[i*elementSize:]))
		}
		return arr, nil
	case MetadataValueTypeFloat32:
		arr := make([]float32, int(count))
		for i := range arr {
			if !checkBounds(i) {
				return nil, fmt.Errorf("gguf: value bytes too short")
			}
			arr[i] = math.Float32frombits(binary.LittleEndian.Uint32(valueBytes[i*elementSize:]))
		}
		return arr, nil
	case MetadataValueTypeBool:
		arr := make([]bool, int(count))
		for i := range arr {
			if !checkBounds(i) {
				return nil, fmt.Errorf("gguf: value bytes too short")
			}
			arr[i] = valueBytes[i*elementSize] != 0
		}
		return arr, nil
	case MetadataValueTypeUint64:
		arr := make([]uint64, int(count))
		for i := range arr {
			if !checkBounds(i) {
				return nil, fmt.Errorf("gguf: value bytes too short")
			}
			arr[i] = binary.LittleEndian.Uint64(valueBytes[i*elementSize:])
		}
		return arr, nil
	case MetadataValueTypeInt64:
		arr := make([]int64, int(count))
		for i := range arr {
			if !checkBounds(i) {
				return nil, fmt.Errorf("gguf: value bytes too short")
			}
			arr[i] = int64(binary.LittleEndian.Uint64(valueBytes[i*elementSize:]))
		}
		return arr, nil
	case MetadataValueTypeFloat64:
		arr := make([]float64, int(count))
		for i := range arr {
			if !checkBounds(i) {
				return nil, fmt.Errorf("gguf: value bytes too short")
			}
			arr[i] = math.Float64frombits(binary.LittleEndian.Uint64(valueBytes[i*elementSize:]))
		}
		return arr, nil
	default:
		// Handle BitNet/LLM GGUF files where element types >= 128 are treated as int32
		if elementType >= 128 {
			arr := make([]int32, int(count))
			for i := range arr {
				if !checkBounds(i) {
					return nil, fmt.Errorf("gguf: value bytes too short")
				}
				arr[i] = int32(binary.LittleEndian.Uint32(valueBytes[i*elementSize:]))
			}
			return arr, nil
		}
		return nil, fmt.Errorf("gguf: unsupported array element type: %d", elementType)
	}
}

// parseTensorInfo
func parseTensorInfo(r io.Reader, logger *log.Logger, tensor *TensorInfo, alignment uint64) error {

	// Read tensor name
	_, name, nameErr := readLengthAndString(r, logger, 1024)
	if nameErr != nil {
		log.Printf("[DEBUG] Failed to read tensor name: %v", nameErr)
		return ErrReadTensorName
	}
	tensor.Name = name

	// Read number of dimensions (uint32)
	var numDims uint32
	numDimsErr := readUint32(r, &numDims)
	if numDimsErr != nil {
		log.Printf("[DEBUG] Failed to read number of dimensions: %v", numDimsErr)
		return ErrReadNumDimensions
	}

	// GGUF v3: shape as int64
	tensor.Shape = make([]uint64, numDims)
	for j := uint32(0); j < numDims; j++ {
		dimErr := readUint64(r, &tensor.Shape[j])
		if dimErr != nil {
			log.Printf("[DEBUG] Failed to read shape dimension: %v", dimErr)
			return ErrReadShapeDimension
		}
	}

	// Read tensor type (ggmltype)
	tensorTypeErr := readUint32(r, &tensor.Type)
	if tensorTypeErr != nil {
		log.Printf("[DEBUG] Failed to read tensor type: %v", tensorTypeErr)
		return ErrReadTensorType
	}

	// Read offset
	offsetErr := readUint64(r, &tensor.Offset)
	if offsetErr != nil {
		log.Printf("[DEBUG] Failed to read tensor offset: %v", offsetErr)
		return ErrReadTensorOffset
	}

	if tensor.Offset%alignment != 0 {
		log.Printf("[DEBUG] Invalid tensor offset: %v", offsetErr)
		return ErrReadTensorOffset
	}

	// Calculate number of elements
	tensor.N = calculateTensorElements(tensor)
	tensor.RowCount = calculateTensorRowCount(tensor)

	tensor.ColCount = calculateTensorColumnCount(tensor)

	var err error
	tensor.RowSize, err = calculateTensorRowSize(tensor, tensor.ColCount)
	if err != nil {
		return err
	}

	tensor.DataSize, err = calculateTensorDataSize(tensor, tensor.N, alignment)
	if err != nil {
		return err
	}

	return nil
}

func parseSliceOfTensorInfo(r io.ReadSeeker, logger *log.Logger, arr []TensorInfo, alignment uint64) error {

	numTensors := len(arr)
	if numTensors <= 0 {
		log.Printf("[DEBUG] Failed to read tensors: No space in the array")
		return fmt.Errorf("parseSliceOfTensorInfo: No space in array")
	}

	log.Printf("[DEBUG] Tensors to read: %d", numTensors)
	for i := 0; i < numTensors; i++ {
		tensor := &arr[i]

		tensorErr := parseTensorInfo(r, logger, tensor, alignment)
		if tensorErr != nil {
			log.Printf("[DEBUG] Failed to read tensor: %v", tensorErr)
			return tensorErr
		}

		log.Printf("[DEBUG] Tensor %d: name=%s, type=%d, shape=%v, offset=%d, dataSize=%d, n=%d, rows=%d, cols=%d, rowSize=%d",
			i, tensor.Name, tensor.Type, tensor.Shape, tensor.Offset, tensor.DataSize, tensor.N, tensor.RowCount, tensor.ColCount, tensor.RowSize)

		if i != 0 {
			arr[i-1].EndOffset = tensor.Offset
		}

	}

	return nil
}
