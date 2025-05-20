package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Tokenizer handles loading and using the BitNet tokenizer.
type Tokenizer struct {
	modelPath string
	vocab     map[string]int
}

// NewTokenizer creates a new Tokenizer instance.
func NewTokenizer() (*Tokenizer, error) {
	// Try to find the tokenizer file in different possible locations
	possiblePaths := []string{
		filepath.Join("pkg", "bitnet", "internal", "assets", "models", "BitNet-b1.58-2B-4T", "tokenizer.json"),
		filepath.Join("..", "..", "..", "..", "pkg", "bitnet", "internal", "assets", "models", "BitNet-b1.58-2B-4T", "tokenizer.json"),
		filepath.Join("..", "..", "..", "pkg", "bitnet", "internal", "assets", "models", "BitNet-b1.58-2B-4T", "tokenizer.json"),
	}

	var foundPath string
	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			foundPath = path
			break
		}
	}

	if foundPath == "" {
		return nil, fmt.Errorf("tokenizer file not found in any of the expected locations: %v", possiblePaths)
	}

	return &Tokenizer{
		modelPath: foundPath,
		vocab:     make(map[string]int),
	}, nil
}

// Load loads the tokenizer vocabulary from the JSON file.
func (t *Tokenizer) Load() error {
	file, err := os.Open(t.modelPath)
	if err != nil {
		return fmt.Errorf("failed to open tokenizer file: %w", err)
	}
	defer file.Close()

	var data struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}

	if err := json.NewDecoder(file).Decode(&data); err != nil {
		return fmt.Errorf("failed to decode tokenizer JSON: %w", err)
	}

	t.vocab = data.Model.Vocab
	return nil
}

// GetVocab returns the tokenizer vocabulary.
func (t *Tokenizer) GetVocab() map[string]int {
	return t.vocab
}

// GetModelPath returns the current tokenizer file path.
func (t *Tokenizer) GetModelPath() string {
	return t.modelPath
}
