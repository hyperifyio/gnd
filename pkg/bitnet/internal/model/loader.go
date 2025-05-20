package model

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"
)

var (
	ErrModelNotFound = errors.New("model file not found")
	ErrInvalidGGUF   = errors.New("invalid GGUF magic number")
	ErrModelNotSet   = errors.New("model path not set")
	ErrReaderNil     = errors.New("reader is nil")
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
	modelPath  string
	bufferSize int
	chunkPool  sync.Pool
	header     *GGUFHeader
}

// NewModelLoader creates a new ModelLoader instance.
func NewModelLoader() (*ModelLoader, error) {
	// Get the absolute path to the model file
	modelPath := filepath.Join("pkg", "bitnet", "internal", "assets", "models", "BitNet-b1.58-2B-4T", "model.bin")
	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, err
	}

	if _, err := os.Stat(absPath); err != nil {
		return nil, ErrModelNotFound
	}

	// Create a memory pool for chunks
	chunkPool := sync.Pool{
		New: func() interface{} {
			buf := make([]byte, 1024*1024) // 1MB default chunk size
			return &buf
		},
	}

	loader := &ModelLoader{
		modelPath:  absPath,
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
	file, err := os.Open(l.modelPath)
	if err != nil {
		return err
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
func (l *ModelLoader) LoadModel() (*os.File, error) {
	if l.modelPath == "" {
		return nil, ErrModelNotSet
	}
	return os.Open(l.modelPath)
}

// GetModelSize returns the size of the model file in bytes.
func (l *ModelLoader) GetModelSize() (int64, error) {
	info, err := os.Stat(l.modelPath)
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
func (l *ModelLoader) LoadModelStream() (*bufio.Reader, *os.File, error) {
	if l.modelPath == "" {
		return nil, nil, ErrModelNotSet
	}

	file, err := os.Open(l.modelPath)
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

	bufPtr := l.chunkPool.Get()
	if bufPtr == nil {
		buf := make([]byte, chunkSize)
		bufPtr = &buf
	}
	buf := *(bufPtr.(*[]byte))

	if cap(buf) < chunkSize {
		buf = make([]byte, chunkSize)
	}
	buf = buf[:chunkSize]

	n, err := io.ReadFull(reader, buf)
	if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
		l.chunkPool.Put(&buf)
		return nil, err
	}

	chunk := make([]byte, n)
	copy(chunk, buf[:n])
	l.chunkPool.Put(&buf)

	return chunk, nil
}
