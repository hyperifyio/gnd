// Package gguf implements a parser for the GGUF (GGML Universal File) format.
//
// It can read GGUF v3 model files—the first version that explicitly records
// endianness—covering headers, metadata, tensor descriptors and the raw tensor
// data. This implementation targets v3 checkpoints that use 2-bit ternary
// (`i2_s`, ≈1.58 bits/value) block-quantised tensors such as those produced by
// BitNet.
//
// The GGUF file is a binary container for neural-network weights, especially
// language-model checkpoints.  Its top-level structure is
//
//   - **Header** (24 B) — magic, version, `n_tensors`, `n_kv`
//     (`u32, u32, u64, u64`, little-endian)
//   - **Key–value (KV) table** with `n_kv` entries
//   - **Tensor-descriptor array** with `n_tensors` entries
//   - **Tensor-data blob**
//
// ### Alignment rules (v3)
//
// **Alignment (`A`)**
//
// `A` is a *byte* boundary used only for padding—
// it defaults to 32 bytes (`GGUF_DEFAULT_ALIGNMENT`),
// but a model can override it with the `general.alignment` metadata key
// (the value of that key is a **u32**, and its numeric value must be a power-of-two).
//
// Alignment has no connection to the width of the integers you read elsewhere:
// all length/count fields in the header (`n_tensors`, `n_kv`) and inside
// strings/arrays are still **u64** in GGUF v3.
// Only the start offset of each KV entry, tensor-info record, the data-blob
// itself, and each individual tensor must be rounded up to the next multiple of `A`.
//
// 1. each **KV entry** (key + type + payload)
// 2. each **tensor descriptor**
// 3. the **start of the tensor-data blob**
// 4. every **individual tensor's data** inside that blob
//
// ### Tensor size and next offset
//
// For a tensor with **N** logical elements stored in a block-quantised type:
//
// ```text
// bytes       = ceil(N / bs) × ts
// next_offset = align_up(curr_offset + bytes, A)
// ````
//
// where
//
// * `bs` = elements per quantisation block
// * `ts` = bytes per block (e.g. 16 B for `i2_s`)
//
// The tensor's stored size is `bytes` rounded up to the next multiple of `A`.
//
// ### Further reading
//
// * Specification: [https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md)
// * Reference C header (`ggml.h`): [https://github.com/ggerganov/ggml/blob/master/include/ggml/ggml.h](https://github.com/ggerganov/ggml/blob/master/include/ggml/ggml.h)
package gguf

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
)

// Static error definitions for GGUF parsing operations.
// All errors are prefixed with "gguf:" to ensure uniqueness across the codebase.
var (
	// ErrReadHeader indicates a failure to read the GGUF file header
	ErrReadHeader = errors.New("gguf: failed to read header")

	// ErrReadMagicNumber indicates a failure to read the GGUF magic number
	ErrReadMagicNumber = errors.New("gguf: failed to read magic number")

	// ErrInvalidMagicNumber indicates that the file's magic number is not "GGUF"
	ErrInvalidMagicNumber = errors.New("gguf: invalid magic number")

	// ErrReadVersion indicates a failure to read the GGUF version number
	ErrReadVersion = errors.New("gguf: failed to read version")

	// ErrReadMetadataCount indicates a failure to read the metadata entry count
	ErrReadMetadataCount = errors.New("gguf: failed to read metadata count")

	// ErrReadTensorCount indicates a failure to read the tensor count
	ErrReadTensorCount = errors.New("gguf: failed to read tensor count")

	// ErrUnsupportedVersion indicates that the GGUF version is not supported
	ErrUnsupportedVersion = errors.New("gguf: unsupported version")

	// ErrLoadMetadata indicates a failure to load the metadata section
	ErrLoadMetadata = errors.New("gguf: failed to load metadata")

	// ErrReadTensMarker indicates a failure to read the TENS marker
	ErrReadTensMarker = errors.New("gguf: failed to read TENS marker")

	// ErrMissingTensMarker indicates that the TENS marker is missing after metadata
	ErrMissingTensMarker = errors.New("gguf: missing TENS marker after metadata")

	// ErrLoadTensors indicates a failure to load the tensor section
	ErrLoadTensors = errors.New("gguf: failed to load tensors")

	// ErrReadTensorName indicates a failure to read a tensor's name
	ErrReadTensorName = errors.New("gguf: failed to read tensor name")

	// ErrReadTensorType indicates a failure to read a tensor's type
	ErrReadTensorType = errors.New("gguf: failed to read tensor type")

	// ErrReadNumDimensions indicates a failure to read the number of dimensions
	ErrReadNumDimensions = errors.New("gguf: failed to read number of dimensions")

	// ErrReadShapeDimension indicates a failure to read a shape dimension
	ErrReadShapeDimension = errors.New("gguf: failed to read shape dimension")

	// ErrReadTensorOffset indicates a failure to read a tensor's offset
	ErrReadTensorOffset = errors.New("gguf: failed to read tensor offset")

	// ErrReadTensorSize indicates a failure to read a tensor's size
	ErrReadTensorSize = errors.New("gguf: failed to read tensor size")

	// ErrReadTensorAlignment indicates a failure to read a tensor's alignment
	ErrReadTensorAlignment = errors.New("gguf: failed to read tensor alignment")

	// ErrReadMetadataValueType indicates a failure to read a metadata value type
	ErrReadMetadataValueType = errors.New("gguf: failed to read metadata value type")

	// ErrArrayAllocationTooLarge indicates that the array allocation is too large
	ErrArrayAllocationTooLarge = errors.New("gguf: array allocation too large")

	// ErrStringTooLong indicates that a string is too long
	ErrStringTooLong = errors.New("gguf: string too long")

	// ErrReadValueBytes indicates a failure to read value bytes
	ErrReadValueBytes = errors.New("gguf: failed to read value bytes")

	// ErrUnsupportedMetadataValueType indicates an unsupported metadata value type
	ErrUnsupportedMetadataValueType = errors.New("gguf: unsupported metadata value type")
)

const (

	// HeaderGGUFMagic is "GGUF" in little-endian
	HeaderGGUFMagic = uint32(0x46554747)

	// Alignment-related constants

	// DefaultAlignment is the default alignment value for GGUF v3 files
	DefaultAlignment = 32

	// GeneralAlignmentKey is the metadata key used to override the default alignment
	GeneralAlignmentKey = "general.alignment"

	// FormatVersion3 is current GGUF version 3 with improved alignment and metadata support
	FormatVersion3 uint32 = 3
)

// Track required metadata keys
var requiredMetadataKeys = []string{
	"general.architecture",
	"general.name",
	"general.file_type",
}

// MetadataValueType represents the type of a metadata value in the GGUF file.
type MetadataValueType uint32

// Supported metadata value types
const (
	MetadataValueTypeUint8   MetadataValueType = 0
	MetadataValueTypeInt8    MetadataValueType = 1
	MetadataValueTypeUint16  MetadataValueType = 2
	MetadataValueTypeInt16   MetadataValueType = 3
	MetadataValueTypeUint32  MetadataValueType = 4
	MetadataValueTypeInt32   MetadataValueType = 5
	MetadataValueTypeFloat32 MetadataValueType = 6
	MetadataValueTypeBool    MetadataValueType = 7
	MetadataValueTypeString  MetadataValueType = 8
	MetadataValueTypeArray   MetadataValueType = 9
	MetadataValueTypeUint64  MetadataValueType = 10
	MetadataValueTypeInt64   MetadataValueType = 11
	MetadataValueTypeFloat64 MetadataValueType = 12
	MetadataValueTypeBinary  MetadataValueType = 13
)

// Supported value types for tensors
const (
	// GGML_TYPE_F32 is 32-bit IEEE-754 float
	GGML_TYPE_F32 = uint32(0)

	// GGML_TYPE_F16 is 16-bit IEEE-754 half
	GGML_TYPE_F16 = uint32(1)

	// GGML_TYPE_I2_S is 2-bit signed ternary (BitNet)
	GGML_TYPE_I2_S = uint32(36)

	// Block size constants for I2_S
	qkI2S = 32 // elements per block
	tsI2S = 16 // bytes per block
)

// Header represents the GGUF file header.
// It contains the magic number, version, and counts of tensors and metadata entries.
type Header struct {
	Magic       uint32 // Magic number (same as "GGUF")
	Version     uint32 // Version is GGUF format version
	NumTensors  uint64 // NumTensors is Number of tensors in the file
	NumMetadata uint64 // NumMetadata is Number of metadata entries
}

// TensorInfo represents metadata about a tensor in the GGUF file.
// It includes the tensor's name, type, shape, and location in the file.
type TensorInfo struct {
	Name     string   // Name of tensor
	Type     uint32   // Type of tensor data
	Shape    []uint64 // Shape is tensor dimensions
	Offset   uint64   // Offset in the file where tensor data begins, counted from the start of the tensor data, which is the region following the tensor info array.
	N        uint64   // N is the number of tensor elements
	RowCount uint64   // RowCount is how many rows the tensor has
	ColCount uint64   // ColCount is how many columns one row has
	RowSize  uint64   // RowSize is how many bytes single row contains
	DataSize uint64   // DataSize is how many bytes tensor block contains (all rows)
}

// Model represents a loaded GGUF model.
// It contains the file header, tensor information, metadata, and file handle.
type Model struct {
	Header    Header                 // GGUF file header
	Tensors   []TensorInfo           // Tensor information
	Metadata  map[string]interface{} // Model metadata
	Alignment uint64                 // Current alignment value (defaults to DefaultAlignment)
	DataStart uint64                 // Offset where tensor data begins
	DataEnd   uint64                 // Offset where tensor data ends

	modelData  []byte        // Internal model bytes
	fileHandle io.ReadSeeker // File handle for reading tensor data
}

// NewModel creates a new Model instance from a reader.
// It reads and validates the GGUF file header.
func NewModel(modelData []byte) (*Model, error) {

	reader := bytes.NewReader(modelData)

	model := &Model{
		Metadata:  make(map[string]interface{}),
		Alignment: DefaultAlignment,

		modelData:  modelData,
		fileHandle: reader,
	}

	if err := model.readHeader(); err != nil {
		log.Printf("[DEBUG] Failed to read header: %v", err)
		return nil, err
	}

	return model, nil
}

// LoadModel reads a GGUF model from bytes slice
func LoadModel(modelData []byte) (*Model, error) {

	totalBytes := len(modelData)

	// Create a new model
	model, newModelErr := NewModel(modelData)

	// Read the header
	if newModelErr != nil {
		return nil, fmt.Errorf("%w: %v", ErrReadHeader, newModelErr)
	}

	// Log debug info
	log.Printf("[DEBUG] After header: pos=0, tensors=%d, metadata=%d", model.Header.NumTensors, model.Header.NumMetadata)
	log.Printf("[DEBUG] Header raw: NumMetadata=%d, NumTensors=%d", model.Header.NumMetadata, model.Header.NumTensors)

	// Load metadata first to get alignment
	if loadMetadatErr := model.loadMetadata(); loadMetadatErr != nil {
		return nil, fmt.Errorf("%w: %v", ErrLoadMetadata, loadMetadatErr)
	}

	// Load tensor info
	if loadTensorErr := model.loadTensorInfoArray(); loadTensorErr != nil {
		return nil, fmt.Errorf("%w: %v", ErrLoadTensors, loadTensorErr)
	}

	// Determine the aligned data start position
	pos, getPosErr := getCurrentPosition(model.fileHandle)
	if getPosErr != nil {
		return nil, fmt.Errorf("failed to get current position: %v", getPosErr)
	}
	log.Printf("[DEBUG] Position just after header: %d", pos)
	log.Printf("[DEBUG] model.Alignment: %d", model.Alignment)

	if rem := pos % model.Alignment; rem != 0 {
		model.DataStart = pos - rem + model.Alignment
		if seekErr := seekToPosition(model.fileHandle, model.DataStart); seekErr != nil {
			return nil, fmt.Errorf("failed to seek to alignment: %v", seekErr)
		}
	} else {
		model.DataStart = pos
	}
	if model.DataStart > uint64(totalBytes) {
		return nil, fmt.Errorf("the data start position out of bounds: %d > %d", model.DataStart, totalBytes)
	}
	log.Printf("[DEBUG] DataStart: %d", model.DataStart)

	endPos, endPosErr := getEndPosition(model.fileHandle)
	if endPosErr != nil {
		return nil, fmt.Errorf("failed to get end position: %v", endPosErr)
	}
	if endPos > uint64(totalBytes) {
		return nil, fmt.Errorf("the data end position out of bounds: %d > %d", endPos, totalBytes)
	}
	model.DataEnd = endPos
	log.Printf("[DEBUG] DataEnd: %d", model.DataEnd)

	// FIXME: Do this check only if we have tensor types of 36 (I2_S)
	if q := model.quantizationVersion(); q != 2 {
		log.Printf("[DEBUG] Invalid quantizationVersion number: %x", q)
		return nil, fmt.Errorf("invalid quantization_version: %d", q)
	}
	log.Printf("[DEBUG] Quantization version number: 2")

	for idx, tensor := range model.Tensors {

		startOffset := model.DataStart + tensor.Offset
		if startOffset < model.DataStart || startOffset > model.DataEnd {
			log.Printf("[DEBUG] Invalid tensor %d offset: %d not between %d .. %d", idx, startOffset, model.DataStart, model.DataEnd)
			return nil, ErrReadTensorOffset
		}

		endOffset := model.DataStart + tensor.Offset + tensor.DataSize
		if endOffset < model.DataStart || endOffset > model.DataEnd {
			log.Printf("[DEBUG] Invalid tensor %d end offset: %d (%d + %d + %d) not between %d .. %d", idx, endOffset, model.DataStart, tensor.Offset, tensor.DataSize, model.DataStart, model.DataEnd)
			return nil, ErrReadTensorOffset
		}
	}

	return model, nil
}

// readHeader reads and validates the GGUF file header.
// It checks the magic number and version, and reads the tensor and metadata counts.
func (m *Model) readHeader() error {

	var magic uint32

	// Read magic number
	err := readUint32(m.fileHandle, &magic)
	if err != nil {
		log.Printf("[DEBUG] Failed to read magic number: %v", err)
		return ErrReadMagicNumber
	}

	// Compare magic number
	if magic != HeaderGGUFMagic {
		log.Printf("[DEBUG] Invalid magic number: %x", magic)
		return ErrInvalidMagicNumber
	}

	// Read and verify version
	err = readUint32(m.fileHandle, &m.Header.Version)
	if err != nil {
		log.Printf("[DEBUG] Failed to read version: %v", err)
		return ErrReadVersion
	}

	if m.Header.Version != FormatVersion3 {
		log.Printf("[DEBUG] Unsupported GGUF version: %d, only version 3 is supported", m.Header.Version)
		return ErrUnsupportedVersion
	}

	// Read count of tensors
	err = readUint64(m.fileHandle, &m.Header.NumTensors)
	if err != nil {
		log.Printf("[DEBUG] Failed to read tensor count: %v", err)
		return ErrReadTensorCount
	}

	// Read count of metadata
	err = readUint64(m.fileHandle, &m.Header.NumMetadata)
	if err != nil {
		log.Printf("[DEBUG] Failed to read metadata count: %v", err)
		return ErrReadMetadataCount
	}

	return nil
}

// loadMetadata loads the metadata section from the GGUF file.
func (m *Model) loadMetadata() error {

	logger := log.Default()

	for i := uint64(0); i < m.Header.NumMetadata; i++ {

		start, err := m.fileHandle.Seek(0, io.SeekCurrent)
		if err != nil {
			return fmt.Errorf("gguf: failed to get file position before metadata entry: %v", err)
		}

		// Read key
		key, keyErr := readMetadataKey(m.fileHandle, logger)
		if keyErr != nil {
			if keyErr == io.EOF {
				break
			}
			log.Printf("[DEBUG] Failed to read metadata key: %v", keyErr)
			return keyErr
		}

		// Read value type
		valueType, valueTypeErr := readMetadataValueType(m.fileHandle)
		if valueTypeErr != nil {
			log.Printf("[DEBUG] Failed to read metadata value type: %v", valueTypeErr)
			return valueTypeErr
		}

		// Read array values
		value, valueErr := readMetadataValue(m.fileHandle, logger, valueType)
		if valueErr != nil {
			log.Printf("[DEBUG] Failed to read metadata value: %v", valueErr)
			return valueErr
		}

		// Store the value
		m.Metadata[key] = value

		log.Printf("[DEBUG] Metadata entry %d: start=%d; key=%s; valueType=%d; %s", i, start, key, valueType, valueToString(value))

		// Check for alignment update
		if key == GeneralAlignmentKey {
			alignment, alignmentError := toUint64(value)
			if alignmentError != nil {
				return fmt.Errorf("gguf: failed to parse alignment value: %v", alignmentError)
			}
			m.updateAlignment(alignment)
			log.Printf("[DEBUG] Updated alignment to %d after entry %d", value, i)
		}

	}

	// Verify all required keys were found
	for _, key := range requiredMetadataKeys {
		if _, found := m.Metadata[key]; !found {
			return fmt.Errorf("gguf: missing required metadata key: %s", key)
		}
	}

	return nil
}

// quantizationVersion returns general.quantization_version metadata value
func (m *Model) quantizationVersion() uint32 {
	var found bool
	var quantizationVersion uint32
	if quantizationVersion, found = m.Metadata["general.quantization_version"].(uint32); !found {
		return 0
	}
	return quantizationVersion
}

// loadTensorInfoArray loads the tensor information from the GGUF file.
func (m *Model) loadTensorInfoArray() error {
	var logger = log.Default()
	numTensors := m.Header.NumTensors
	if numTensors == 0 {
		return fmt.Errorf("gguf: tensor count is zero")
	}
	m.Tensors = make([]TensorInfo, numTensors)
	log.Printf("[DEBUG] Tensors to read: %d", numTensors)
	if err := parseSliceOfTensorInfo(m.fileHandle, logger, m.Tensors, m.Alignment); err != nil {
		log.Printf("[DEBUG] Failed to read tensor: %v", err)
		return err
	}
	return nil
}

// GetTensorData returns a slice to tensor data
func (m *Model) GetTensorData(tensor *TensorInfo) ([]byte, error) {
	length := tensor.DataSize
	offset := m.DataStart + tensor.Offset
	log.Printf("[DEBUG] Reading tensor data (name=%s, type=%d, shape=%v, offset=%d, rowCount=%d, rowSize=%d, N=%d), model (dataStart=%d)",
		tensor.Name, tensor.Type, tensor.Shape, tensor.Offset, tensor.RowCount, tensor.RowSize, tensor.N,
		m.DataStart,
	)
	slice := sliceOfByte(m.modelData, offset, length)
	return slice, nil
}

// updateAlignment updates the model's alignment value.
// According to GGUF v3 spec, alignment can only be changed via the general.alignment KV.
func (m *Model) updateAlignment(alignment uint64) {

	// Validate alignment
	if !validateAlignment(alignment) {
		log.Printf("[WARN] Invalid alignment value: %d", alignment)
		return
	}

	m.Alignment = alignment
	log.Printf("[DEBUG] Updated alignment to %d via general.alignment KV", alignment)
}

// updateMetadata updates a metadata field in the model
func (m *Model) updateMetadata(key string, value interface{}) error {
	m.Metadata[key] = value
	log.Printf("[DEBUG] Updated metadata key %s with value %v", key, value)
	return nil
}
