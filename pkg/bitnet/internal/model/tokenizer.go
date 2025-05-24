package model

import (
	"encoding/json"
	"io/fs"
	"strings"
	"unicode/utf8"

	"github.com/hyperifyio/gnd/pkg/loggers"
)

// Tokenizer handles loading and using the BitNet tokenizer.
type Tokenizer struct {
	fs            fs.FS
	modelPath     string
	Vocab         map[string]int
	Merges        []string
	MergeMap      map[string]string
	SpecialTokens map[string]int
	MaxTokens     int
}

// NewTokenizer creates a new Tokenizer instance.
func NewTokenizer(fs fs.FS, modelPath string) (*Tokenizer, error) {
	if fs == nil {
		return nil, ErrFSNotSet
	}
	if modelPath == "" {
		return nil, ErrPathEmpty
	}

	t := &Tokenizer{
		fs:        fs,
		modelPath: modelPath,
		MaxTokens: 4096,
	}

	if err := t.load(); err != nil {
		loggers.Printf(loggers.Debug, "failed to load tokenizer: %v", err)
		return nil, ErrTokenizerNotFound
	}

	return t, nil
}

// load reads and decodes the tokenizer files
func (t *Tokenizer) load() error {
	// Read vocabulary
	vocabData, err := fs.ReadFile(t.fs, t.modelPath+"/vocab.json")
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to read vocabulary file: %v", err)
		return ErrVocabRead
	}

	if err := json.Unmarshal(vocabData, &t.Vocab); err != nil {
		loggers.Printf(loggers.Debug, "failed to parse vocabulary file: %v", err)
		return ErrVocabParse
	}

	// Read merges
	mergesData, err := fs.ReadFile(t.fs, t.modelPath+"/merges.txt")
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to read merges file: %v", err)
		return ErrMergesRead
	}

	// Parse merges into ordered list and map
	merges := strings.Split(string(mergesData), "\n")
	t.Merges = make([]string, 0, len(merges))
	t.MergeMap = make(map[string]string)

	for _, merge := range merges {
		if merge == "" {
			continue
		}
		t.Merges = append(t.Merges, merge)
		parts := strings.Split(merge, " ")
		if len(parts) == 2 {
			t.MergeMap[parts[0]+" "+parts[1]] = parts[0] + parts[1]
		}
	}

	// Read special tokens
	specialData, err := fs.ReadFile(t.fs, t.modelPath+"/special_tokens.json")
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to read special tokens file: %v", err)
		return ErrSpecialRead
	}

	if err := json.Unmarshal(specialData, &t.SpecialTokens); err != nil {
		loggers.Printf(loggers.Debug, "failed to parse special tokens file: %v", err)
		return ErrSpecialParse
	}

	return nil
}

// Tokenize converts text into token IDs using BPE
func (t *Tokenizer) Tokenize(text string) ([]int, error) {
	if t.Vocab == nil {
		return nil, ErrVocabNotLoaded
	}

	if text == "" {
		return []int{}, nil
	}

	// Split text into words and add space tokens
	words := t.splitText(text)
	tokens := make([]int, 0, len(words)*2)

	for i, word := range words {
		// Add space token between words (except for the first word)
		if i > 0 {
			if spaceID, ok := t.Vocab["▁"]; ok {
				tokens = append(tokens, spaceID)
			}
		}

		// Handle special tokens
		if id, ok := t.SpecialTokens[word]; ok {
			tokens = append(tokens, id)
			continue
		}

		// Apply BPE to the word
		subTokens := t.applyBPE(word)
		allKnown := true
		for _, subToken := range subTokens {
			if _, ok := t.Vocab[subToken]; !ok {
				allKnown = false
				break
			}
		}
		if allKnown {
			for _, subToken := range subTokens {
				id := t.Vocab[subToken]
				tokens = append(tokens, id)
			}
		} else {
			if unkID, ok := t.SpecialTokens["<unk>"]; ok {
				tokens = append(tokens, unkID)
			} else {
				loggers.Printf(loggers.Debug, "unknown token encountered: %s", word)
				return nil, ErrUnknownToken
			}
		}
	}

	// Check sequence length
	if len(tokens) > t.MaxTokens {
		loggers.Printf(loggers.Debug, "sequence length %d exceeds maximum %d", len(tokens), t.MaxTokens)
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

	// Strip trailing punctuation from each word
	for i, word := range words {
		words[i] = strings.TrimRight(word, ",.!?;:")
	}

	return words
}

// applyBPE applies Byte Pair Encoding to split unknown words
func (t *Tokenizer) applyBPE(word string) []string {
	if word == "" {
		return nil
	}

	// Split on word boundaries (apostrophes, hyphens, etc.)
	parts := strings.FieldsFunc(word, func(r rune) bool {
		return r == '\'' || r == '-' || r == '_'
	})

	if len(parts) > 1 {
		// If we have multiple parts, process each one
		var result []string
		for i, part := range parts {
			if i > 0 {
				// Add the separator back
				result = append(result, string(word[len(result)]))
			}
			result = append(result, t.applyBPE(part)...)
		}
		return result
	}

	// Start with individual characters
	symbols := make([]string, 0, len(word))
	for _, r := range word {
		symbols = append(symbols, string(r))
	}

	// Apply merges in order until no more can be applied
	for {
		// Find the first merge that can be applied
		bestPos := -1
		bestMerge := ""

		// Check each merge in order
		for _, merge := range t.Merges {
			parts := strings.Split(merge, " ")
			if len(parts) != 2 {
				continue
			}
			// Look for this merge in the current symbols
			for i := 0; i < len(symbols)-1; i++ {
				if symbols[i] == parts[0] && symbols[i+1] == parts[1] {
					bestPos = i
					bestMerge = t.MergeMap[merge]
					break
				}
			}
			if bestPos != -1 {
				break // Found the first valid merge
			}
		}

		if bestPos == -1 {
			break // No more merges can be applied
		}

		// Apply the merge
		symbols[bestPos] = bestMerge
		symbols = append(symbols[:bestPos+1], symbols[bestPos+2:]...)
	}

	// If we have a complete word in the vocabulary, use it
	if _, ok := t.Vocab[word]; ok {
		return []string{word}
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
