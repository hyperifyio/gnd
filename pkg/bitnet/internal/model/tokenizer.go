package model

import (
	"bufio"
	"encoding/json"
	"io/fs"
	"strings"
	"unicode/utf8"
)

// Tokenizer handles loading and using the BitNet tokenizer.
type Tokenizer struct {
	fs            fs.FS
	modelPath     string
	Vocab         map[string]int    `json:"vocab"`
	Merges        []string          // Ordered list of merge pairs
	MergeMap      map[string]string // Map for fast lookup
	SpecialTokens map[string]int    `json:"special_tokens"`
	MaxTokens     int               `json:"max_tokens"`
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
		MaxTokens: 4096, // Default max sequence length
	}

	if err := tokenizer.load(); err != nil {
		return nil, err
	}

	return tokenizer, nil
}

// load reads and decodes the tokenizer files
func (t *Tokenizer) load() error {
	// Load vocabulary
	vocabFile, err := t.fs.Open(t.modelPath + "/vocab.json")
	if err != nil {
		return ErrTokenizerNotFound
	}
	defer vocabFile.Close()

	if err := json.NewDecoder(vocabFile).Decode(&t.Vocab); err != nil {
		return ErrDecodeFailed
	}

	// Load merges
	mergesFile, err := t.fs.Open(t.modelPath + "/merges.txt")
	if err != nil {
		return ErrTokenizerNotFound
	}
	defer mergesFile.Close()

	t.Merges = make([]string, 0)
	t.MergeMap = make(map[string]string)
	scanner := bufio.NewScanner(mergesFile)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			continue
		}
		pair := parts[0]
		merge := parts[1]
		t.Merges = append(t.Merges, pair)
		t.MergeMap[pair] = merge
	}

	if err := scanner.Err(); err != nil {
		return ErrDecodeFailed
	}

	// Load special tokens
	specialFile, err := t.fs.Open(t.modelPath + "/special_tokens.json")
	if err != nil {
		return ErrTokenizerNotFound
	}
	defer specialFile.Close()

	if err := json.NewDecoder(specialFile).Decode(&t.SpecialTokens); err != nil {
		return ErrDecodeFailed
	}

	return nil
}

// Tokenize converts text into token IDs using BPE
func (t *Tokenizer) Tokenize(text string) ([]int, error) {
	if t.Vocab == nil {
		return nil, ErrVocabNotLoaded
	}

	// Split text into words and handle special tokens
	words := t.splitText(text)
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

	// Check sequence length
	if len(tokens) > t.MaxTokens {
		return nil, ErrSequenceTooLong
	}

	return tokens, nil
}

// splitText splits text into words and handles special tokens
func (t *Tokenizer) splitText(text string) []string {
	var words []string
	var current strings.Builder

	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		i += size

		// Handle special tokens
		if r == '[' {
			// Check for special token
			end := strings.Index(text[i:], "]")
			if end != -1 {
				token := text[i-1 : i+end+1]
				if _, ok := t.SpecialTokens[token]; ok {
					if current.Len() > 0 {
						words = append(words, current.String())
						current.Reset()
					}
					words = append(words, token)
					i += end + 1
					continue
				}
			}
		}

		// Handle whitespace
		if r == ' ' || r == '\t' || r == '\n' {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
			continue
		}

		current.WriteRune(r)
	}

	if current.Len() > 0 {
		words = append(words, current.String())
	}

	return words
}

// applyBPE applies Byte Pair Encoding to split unknown words
func (t *Tokenizer) applyBPE(word string) []string {
	if len(word) == 0 {
		return nil
	}

	// Convert word to bytes for BPE
	bytes := []byte(word)
	symbols := make([]string, len(bytes))
	for i := 0; i < len(bytes); i++ {
		symbols[i] = string(bytes[i : i+1])
	}

	// Keep merging pairs as long as possible
	for {
		found := false
		for _, pair := range t.Merges {
			// Find the pair in the current symbols
			for i := 0; i < len(symbols)-1; i++ {
				if symbols[i]+symbols[i+1] == pair {
					// Merge the pair
					merged := t.MergeMap[pair]
					symbols = append(symbols[:i], append([]string{merged}, symbols[i+2:]...)...)
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		if !found {
			break
		}
	}

	return symbols
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

	// Join tokens and handle special cases
	text := strings.Join(tokens, "")
	text = strings.ReplaceAll(text, "▁", " ") // Replace special space token
	text = strings.TrimSpace(text)

	return text, nil
}

// GetVocab returns the tokenizer vocabulary.
func (t *Tokenizer) GetVocab() map[string]int {
	return t.Vocab
}

// GetModelPath returns the current tokenizer file path.
func (t *Tokenizer) GetModelPath() string {
	return t.modelPath
}
