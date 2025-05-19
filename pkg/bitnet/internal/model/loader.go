package model

import (
	"os"
	"path/filepath"
)

// ModelLoader handles loading and managing the BitNet model file.
type ModelLoader struct {
	modelPath string
}

// NewModelLoader creates a new ModelLoader instance.
func NewModelLoader() *ModelLoader {
	// Try to find the model file in different possible locations
	possiblePaths := []string{
		filepath.Join("assets", "bitnet", "models", "BitNet-b1.58-2B-4T", "ggml-model-i2_s.gguf"),
		filepath.Join("..", "..", "..", "..", "assets", "bitnet", "models", "BitNet-b1.58-2B-4T", "ggml-model-i2_s.gguf"),
		filepath.Join("..", "..", "..", "assets", "bitnet", "models", "BitNet-b1.58-2B-4T", "ggml-model-i2_s.gguf"),
	}

	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			return &ModelLoader{modelPath: path}
		}
	}

	// Default to the first path if none found
	return &ModelLoader{
		modelPath: possiblePaths[0],
	}
}

// LoadModel opens the model file and returns a file handle.
// The caller is responsible for closing the file.
func (l *ModelLoader) LoadModel() (*os.File, error) {
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
