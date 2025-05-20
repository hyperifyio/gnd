package model

import (
	"bufio"
	"encoding/binary"
	"io"
	"io/fs"
	"sync"
)

// GGUFHeader represents the header of a GGUF format file
type GGUFHeader struct {
	Magic       uint32
	Version     uint32
	TensorCount uint64
	KVCount     uint64
}

// ModelLoader handles loading and managing the BitNet model file in GGUF format.
type ModelLoader struct {
	fs         fs.FS
	modelPath  string
	bufferSize int
	chunkPool  sync.Pool
	header     *GGUFHeader
}

// NewModelLoader creates a new ModelLoader instance.
func NewModelLoader(filesystem fs.FS, modelPath string) (*ModelLoader, error) {
	if filesystem == nil {
		return nil, ErrFSNotSet
	}

	if modelPath == "" {
		return nil, ErrPathEmpty
	}

	// Create a memory pool for chunks
	chunkPool := sync.Pool{
		New: func() interface{} {
			return make([]byte, 1024*1024) // 1MB default chunk size
		},
	}

	loader := &ModelLoader{
		fs:         filesystem,
		modelPath:  modelPath,
		bufferSize: 1024 * 1024, // 1MB buffer size
		chunkPool:  chunkPool,
	}

	// Load and validate the GGUF header
	if err := loader.loadHeader(); err != nil {
		return nil, err
	}

	return loader, nil
}

// loadHeader reads and validates the GGUF file header
func (l *ModelLoader) loadHeader() error {
	file, err := l.fs.Open(l.modelPath)
	if err != nil {
		return ErrModelNotFound
	}
	defer file.Close()

	header := &GGUFHeader{}
	if err := binary.Read(file, binary.LittleEndian, header); err != nil {
		return err
	}

	// Validate GGUF magic number (0x46554747)
	if header.Magic != 0x46554747 {
		return ErrInvalidGGUF
	}

	l.header = header
	return nil
}

// LoadModel opens the model file and returns a file handle.
// The caller is responsible for closing the file.
func (l *ModelLoader) LoadModel() (fs.File, error) {
	if l.modelPath == "" {
		return nil, ErrModelNotSet
	}
	return l.fs.Open(l.modelPath)
}

// GetModelSize returns the size of the model file in bytes.
func (l *ModelLoader) GetModelSize() (int64, error) {
	file, err := l.fs.Open(l.modelPath)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

// GetModelPath returns the current model file path.
func (l *ModelLoader) GetModelPath() string {
	return l.modelPath
}

// GetHeader returns the GGUF header information.
func (l *ModelLoader) GetHeader() *GGUFHeader {
	return l.header
}

// LoadModelStream returns a buffered reader for the model file.
// The caller is responsible for closing the reader.
func (l *ModelLoader) LoadModelStream() (*bufio.Reader, fs.File, error) {
	if l.modelPath == "" {
		return nil, nil, ErrModelNotSet
	}

	file, err := l.fs.Open(l.modelPath)
	if err != nil {
		return nil, nil, err
	}

	return bufio.NewReaderSize(file, l.bufferSize), file, nil
}

// LoadModelChunk reads a chunk of the model file.
func (l *ModelLoader) LoadModelChunk(reader *bufio.Reader, chunkSize int) ([]byte, error) {
	if reader == nil {
		return nil, ErrReaderNil
	}

	chunk := make([]byte, chunkSize)
	n, err := reader.Read(chunk)
	if err != nil && err != io.EOF {
		return nil, err
	}

	return chunk[:n], nil
}
