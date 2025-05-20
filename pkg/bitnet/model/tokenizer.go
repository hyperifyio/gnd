package model

import (
	"errors"
	"strings"
)

var (
	ErrVocabNotLoaded = errors.New("vocabulary not loaded")
	ErrUnknownToken   = errors.New("unknown token")
	ErrUnknownTokenID = errors.New("unknown token ID")
)

// Tokenizer represents the model's tokenizer
type Tokenizer struct {
	Vocab         map[string]int    `json:"vocab"`
	Merges        map[string]string `json:"merges"`
	SpecialTokens map[string]int    `json:"special_tokens"`
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
	if t.Merges == nil {
		return []string{word}
	}

	// Start with individual characters
	subwords := make([]string, len(word))
	for i, char := range word {
		subwords[i] = string(char)
	}

	// Apply merges
	for {
		merged := false
		for i := 0; i < len(subwords)-1; i++ {
			pair := subwords[i] + subwords[i+1]
			if merge, ok := t.Merges[pair]; ok {
				// Replace the pair with the merged token
				subwords = append(subwords[:i], append([]string{merge}, subwords[i+2:]...)...)
				merged = true
				break
			}
		}
		if !merged {
			break
		}
	}

	// Add BPE markers to all subwords except the first one
	for i := 1; i < len(subwords); i++ {
		if !strings.HasPrefix(subwords[i], "##") {
			subwords[i] = "##" + subwords[i]
		}
	}

	return subwords
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int) (string, error) {
	if t.Vocab == nil {
		return "", ErrVocabNotLoaded
	}

	// Create reverse vocabulary mapping
	reverseVocab := make(map[int]string)
	for token, id := range t.Vocab {
		reverseVocab[id] = token
	}

	// Convert IDs to tokens
	tokens := make([]string, len(ids))
	for i, id := range ids {
		if token, ok := reverseVocab[id]; ok {
			tokens[i] = token
		} else {
			return "", ErrUnknownTokenID
		}
	}

	// Join tokens and clean up
	text := strings.Join(tokens, "")
	text = strings.ReplaceAll(text, "##", "") // Remove BPE markers
	return text, nil
}

// Detokenize converts token IDs back into text
func (t *Tokenizer) Detokenize(ids []int) (string, error) {
	if t.Vocab == nil {
		return "", ErrVocabNotLoaded
	}

	// TODO: Implement actual detokenization logic
	return "", nil
}

// GetVocab returns the tokenizer vocabulary
func (t *Tokenizer) GetVocab() map[string]int {
	return t.Vocab
}

// GetMerges returns the tokenizer merges
func (t *Tokenizer) GetMerges() map[string]string {
	return t.Merges
}

// GetSpecialTokens returns the tokenizer special tokens
func (t *Tokenizer) GetSpecialTokens() map[string]int {
	return t.SpecialTokens
}
