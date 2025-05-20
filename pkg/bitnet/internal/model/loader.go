package model

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
)

// ModelLoader handles loading and managing the BitNet model file.
type ModelLoader struct {
	modelPath  string
	bufferSize int
	chunkPool  sync.Pool
}

// NewModelLoader creates a new ModelLoader instance.
func NewModelLoader() (*ModelLoader, error) {
	// Try to find the model file in different possible locations
	possiblePaths := []string{
		filepath.Join("assets", "bitnet", "models", "BitNet-b1.58-2B-4T", "ggml-model-i2_s.gguf"),
		filepath.Join("..", "..", "..", "..", "assets", "bitnet", "models", "BitNet-b1.58-2B-4T", "ggml-model-i2_s.gguf"),
		filepath.Join("..", "..", "..", "assets", "bitnet", "models", "BitNet-b1.58-2B-4T", "ggml-model-i2_s.gguf"),
	}

	var foundPath string
	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			foundPath = path
			break
		}
	}

	if foundPath == "" {
		return nil, fmt.Errorf("model file not found in any of the expected locations: %v", possiblePaths)
	}

	// Create a memory pool for chunks
	chunkPool := sync.Pool{
		New: func() interface{} {
			return make([]byte, 1024*1024) // 1MB default chunk size
		},
	}

	return &ModelLoader{
		modelPath:  foundPath,
		bufferSize: 1024 * 1024, // 1MB buffer size
		chunkPool:  chunkPool,
	}, nil
}

// LoadModel opens the model file and returns a file handle.
// The caller is responsible for closing the file.
func (l *ModelLoader) LoadModel() (*os.File, error) {
	if l.modelPath == "" {
		return nil, fmt.Errorf("model path not set")
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

// LoadModelStream returns a buffered reader for the model file.
// The caller is responsible for closing the reader.
func (l *ModelLoader) LoadModelStream() (*bufio.Reader, *os.File, error) {
	if l.modelPath == "" {
		return nil, nil, fmt.Errorf("model path not set")
	}

	file, err := os.Open(l.modelPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open model file: %w", err)
	}

	reader := bufio.NewReaderSize(file, l.bufferSize)
	return reader, file, nil
}

// LoadModelChunk reads a chunk of the model file.
func (l *ModelLoader) LoadModelChunk(reader *bufio.Reader, chunkSize int) ([]byte, error) {
	if reader == nil {
		return nil, fmt.Errorf("reader is nil")
	}

	chunk := make([]byte, chunkSize)
	n, err := reader.Read(chunk)
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("failed to read model chunk: %w", err)
	}

	return chunk[:n], nil
}
