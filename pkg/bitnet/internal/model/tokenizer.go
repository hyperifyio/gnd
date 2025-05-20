package model

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

var (
	ErrTokenizerNotFound = errors.New("tokenizer file not found")
	ErrVocabNotLoaded    = errors.New("vocabulary not loaded")
)

// Tokenizer handles loading and using the BitNet tokenizer.
type Tokenizer struct {
	modelPath string
	vocab     map[string]int
}

// NewTokenizer creates a new Tokenizer instance.
func NewTokenizer() (*Tokenizer, error) {
	tokenizerPath := filepath.Join("pkg", "bitnet", "internal", "assets", "models", "BitNet-b1.58-2B-4T", "tokenizer.json")

	if _, err := os.Stat(tokenizerPath); err != nil {
		return nil, ErrTokenizerNotFound
	}

	tokenizer := &Tokenizer{
		modelPath: tokenizerPath,
	}

	if err := tokenizer.loadVocab(); err != nil {
		return nil, err
	}

	return tokenizer, nil
}

// loadVocab loads the vocabulary from the tokenizer file
func (t *Tokenizer) loadVocab() error {
	file, err := os.Open(t.modelPath)
	if err != nil {
		return err
	}
	defer file.Close()

	if err := json.NewDecoder(file).Decode(&t.vocab); err != nil {
		return err
	}

	return nil
}

// Tokenize converts text into token IDs
func (t *Tokenizer) Tokenize(text string) ([]int, error) {
	if t.vocab == nil {
		return nil, ErrVocabNotLoaded
	}

	// TODO: Implement actual tokenization logic
	return nil, nil
}

// GetVocab returns the tokenizer vocabulary.
func (t *Tokenizer) GetVocab() map[string]int {
	return t.vocab
}

// GetModelPath returns the current tokenizer file path.
func (t *Tokenizer) GetModelPath() string {
	return t.modelPath
}
