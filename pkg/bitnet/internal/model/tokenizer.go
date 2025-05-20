package model

import (
	"encoding/json"
	"io/fs"
	"strings"
)

// Tokenizer handles loading and using the BitNet tokenizer.
type Tokenizer struct {
	fs            fs.FS
	modelPath     string
	Vocab         map[string]int    `json:"vocab"`
	Merges        map[string]string `json:"merges"`
	SpecialTokens map[string]int    `json:"special_tokens"`
}

// NewTokenizer creates a new Tokenizer instance.
func NewTokenizer(filesystem fs.FS, modelPath string) (*Tokenizer, error) {
	if filesystem == nil {
		return nil, ErrFSNotSet
	}

	if modelPath == "" {
		return nil, ErrPathEmpty
	}

	tokenizer := &Tokenizer{
		fs:        filesystem,
		modelPath: modelPath,
	}

	if err := tokenizer.load(); err != nil {
		return nil, err
	}

	return tokenizer, nil
}

// load reads and decodes the tokenizer file
func (t *Tokenizer) load() error {
	file, err := t.fs.Open(t.modelPath)
	if err != nil {
		return ErrTokenizerNotFound
	}
	defer file.Close()

	if err := json.NewDecoder(file).Decode(t); err != nil {
		return ErrDecodeFailed
	}

	return nil
}

// Tokenize converts text into token IDs
func (t *Tokenizer) Tokenize(text string) ([]int, error) {
	if t.Vocab == nil {
		return nil, ErrVocabNotLoaded
	}

	// Split text into words
	words := strings.Fields(text)
	tokens := make([]int, 0, len(words))

	for _, word := range words {
		// Check if word exists in vocabulary
		if id, ok := t.Vocab[word]; ok {
			tokens = append(tokens, id)
			continue
		}

		// Apply BPE merges
		subwords := t.applyBPE(word)
		for _, subword := range subwords {
			if id, ok := t.Vocab[subword]; ok {
				tokens = append(tokens, id)
			} else if id, ok := t.SpecialTokens["[UNK]"]; ok {
				tokens = append(tokens, id)
			} else {
				return nil, ErrUnknownToken
			}
		}
	}

	return tokens, nil
}

// applyBPE applies Byte Pair Encoding to split unknown words
func (t *Tokenizer) applyBPE(word string) []string {
	// TODO: Implement BPE algorithm
	return []string{word}
}

// Detokenize converts token IDs back into text
func (t *Tokenizer) Detokenize(ids []int) (string, error) {
	if t.Vocab == nil {
		return "", ErrVocabNotLoaded
	}

	// Create reverse mapping
	reverseVocab := make(map[int]string)
	for token, id := range t.Vocab {
		reverseVocab[id] = token
	}

	// Convert IDs to tokens
	var tokens []string
	for _, id := range ids {
		if token, ok := reverseVocab[id]; ok {
			tokens = append(tokens, token)
		} else {
			return "", ErrUnknownTokenID
		}
	}

	return strings.Join(tokens, " "), nil
}

// GetVocab returns the tokenizer vocabulary.
func (t *Tokenizer) GetVocab() map[string]int {
	return t.Vocab
}

// GetModelPath returns the current tokenizer file path.
func (t *Tokenizer) GetModelPath() string {
	return t.modelPath
}
