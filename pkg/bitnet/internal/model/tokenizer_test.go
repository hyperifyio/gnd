package model

import (
	"encoding/json"
	"errors"
	"io/fs"
	"strings"
	"testing"
)

func TestNewTokenizer(t *testing.T) {
	// Create test vocabulary
	vocab := map[string]int{
		"hello": 1,
		"world": 2,
		"[UNK]": 3,
		"▁":     4, // Special space token
	}

	// Create test special tokens
	specialTokens := map[string]int{
		"[UNK]": 3,
		"[PAD]": 5,
	}

	// Create test tokenizer files
	testFS := &testFS{
		files: map[string][]byte{
			"tokenizer/vocab.json": func() []byte {
				data, _ := json.Marshal(vocab)
				return data
			}(),
			// Merges as an ordered list (simulate merges.txt as in real BPE)
			"tokenizer/merges.txt": []byte("h e he\nl l ll\nhe l hello\nw o wo\nwo r wor\nwor l worl\nworl d world\n"),
			"tokenizer/special_tokens.json": func() []byte {
				data, _ := json.Marshal(specialTokens)
				return data
			}(),
		},
	}

	tokenizer, err := NewTokenizer(testFS, "tokenizer")
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	if tokenizer == nil {
		t.Fatal("NewTokenizer returned nil")
	}

	if tokenizer.modelPath != "tokenizer" {
		t.Errorf("expected modelPath to be 'tokenizer', got %q", tokenizer.modelPath)
	}

	if len(tokenizer.Vocab) != 4 {
		t.Errorf("expected 4 vocabulary items, got %d", len(tokenizer.Vocab))
	}

	if tokenizer.Vocab["hello"] != 1 {
		t.Errorf("expected 'hello' to have ID 1, got %d", tokenizer.Vocab["hello"])
	}

	if len(tokenizer.Merges) != 2 {
		t.Errorf("expected 2 merges, got %d", len(tokenizer.Merges))
	}

	if len(tokenizer.SpecialTokens) != 2 {
		t.Errorf("expected 2 special tokens, got %d", len(tokenizer.SpecialTokens))
	}

	if tokenizer.SpecialTokens["[UNK]"] != 3 {
		t.Errorf("expected '[UNK]' to have ID 3, got %d", tokenizer.SpecialTokens["[UNK]"])
	}

	if tokenizer.MaxTokens != 4096 {
		t.Errorf("expected MaxTokens to be 4096, got %d", tokenizer.MaxTokens)
	}
}

func TestNewTokenizerErrors(t *testing.T) {
	tests := []struct {
		name      string
		fs        fs.FS
		modelPath string
		wantErr   error
	}{
		{
			name:      "nil filesystem",
			fs:        nil,
			modelPath: "tokenizer",
			wantErr:   ErrFSNotSet,
		},
		{
			name:      "empty model path",
			fs:        &testFS{},
			modelPath: "",
			wantErr:   ErrPathEmpty,
		},
		{
			name:      "vocab file not found",
			fs:        &testFS{},
			modelPath: "nonexistent",
			wantErr:   ErrTokenizerNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewTokenizer(tt.fs, tt.modelPath)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("expected error %q, got %q", tt.wantErr, err)
			}
		})
	}
}

func TestTokenize(t *testing.T) {
	// Create test vocabulary
	vocab := map[string]int{
		"hello": 1,
		"world": 2,
		"[UNK]": 3,
		"▁":     4,
	}

	// Create test special tokens
	specialTokens := map[string]int{
		"[UNK]": 3,
		"[PAD]": 5,
	}

	// Create test tokenizer files
	testFS := &testFS{
		files: map[string][]byte{
			"tokenizer/vocab.json": func() []byte {
				data, _ := json.Marshal(vocab)
				return data
			}(),
			// Merges as an ordered list (simulate merges.txt as in real BPE)
			"tokenizer/merges.txt": []byte("h e he\nl l ll\nhe l hello\nw o wo\nwo r wor\nwor l worl\nworl d world\n"),
			"tokenizer/special_tokens.json": func() []byte {
				data, _ := json.Marshal(specialTokens)
				return data
			}(),
		},
	}

	tokenizer, err := NewTokenizer(testFS, "tokenizer")
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	tests := []struct {
		name    string
		text    string
		want    []int
		wantErr error
	}{
		{
			name:    "known words",
			text:    "hello world",
			want:    []int{1, 4, 2},
			wantErr: nil,
		},
		{
			name:    "unknown word",
			text:    "hello unknown",
			want:    []int{1, 4, 3},
			wantErr: nil,
		},
		{
			name:    "empty text",
			text:    "",
			want:    []int{},
			wantErr: nil,
		},
		{
			name:    "special token",
			text:    "hello [PAD] world",
			want:    []int{1, 4, 5, 4, 2},
			wantErr: nil,
		},
		{
			name:    "BPE merge",
			text:    "he wo",
			want:    []int{1, 4, 2},
			wantErr: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tokenizer.Tokenize(tt.text)
			if err != tt.wantErr {
				t.Errorf("Tokenize() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if len(got) != len(tt.want) {
				t.Errorf("Tokenize() got %v, want %v", got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("Tokenize() got[%d] = %v, want[%d] = %v", i, got[i], i, tt.want[i])
				}
			}
		})
	}
}

func TestTokenizeErrors(t *testing.T) {
	tokenizer := &Tokenizer{} // No vocabulary loaded

	_, err := tokenizer.Tokenize("test")
	if err != ErrVocabNotLoaded {
		t.Errorf("expected ErrVocabNotLoaded, got %v", err)
	}

	// Test sequence length limit
	tokenizer = &Tokenizer{
		Vocab:     map[string]int{"test": 1},
		MaxTokens: 2,
	}

	_, err = tokenizer.Tokenize("test test test")
	if err != ErrSequenceTooLong {
		t.Errorf("expected ErrSequenceTooLong, got %v", err)
	}
}

func TestDetokenize(t *testing.T) {
	// Create test vocabulary
	vocab := map[string]int{
		"hello": 1,
		"world": 2,
		"[UNK]": 3,
		"▁":     4,
	}

	// Create test special tokens
	specialTokens := map[string]int{
		"[UNK]": 3,
		"[PAD]": 5,
	}

	tokenizer := &Tokenizer{
		Vocab:         vocab,
		SpecialTokens: specialTokens,
	}

	tests := []struct {
		name    string
		ids     []int
		want    string
		wantErr error
	}{
		{
			name:    "known tokens",
			ids:     []int{1, 4, 2},
			want:    "hello world",
			wantErr: nil,
		},
		{
			name:    "unknown token ID",
			ids:     []int{1, 999},
			want:    "",
			wantErr: ErrUnknownTokenID,
		},
		{
			name:    "empty tokens",
			ids:     []int{},
			want:    "",
			wantErr: nil,
		},
		{
			name:    "special token",
			ids:     []int{1, 4, 5, 4, 2},
			want:    "hello [PAD] world",
			wantErr: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tokenizer.Detokenize(tt.ids)
			if err != tt.wantErr {
				t.Errorf("Detokenize() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Detokenize() got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDetokenizeErrors(t *testing.T) {
	tokenizer := &Tokenizer{} // No vocabulary loaded

	_, err := tokenizer.Detokenize([]int{1})
	if err != ErrVocabNotLoaded {
		t.Errorf("expected ErrVocabNotLoaded, got %v", err)
	}
}

func TestSplitText(t *testing.T) {
	tokenizer := &Tokenizer{
		SpecialTokens: map[string]int{
			"[UNK]": 1,
			"[PAD]": 2,
		},
	}

	tests := []struct {
		name string
		text string
		want []string
	}{
		{
			name: "simple text",
			text: "hello world",
			want: []string{"hello", "world"},
		},
		{
			name: "special tokens",
			text: "hello [PAD] world",
			want: []string{"hello", "[PAD]", "world"},
		},
		{
			name: "multiple spaces",
			text: "hello   world",
			want: []string{"hello", "world"},
		},
		{
			name: "newlines",
			text: "hello\nworld",
			want: []string{"hello", "world"},
		},
		{
			name: "tabs",
			text: "hello\tworld",
			want: []string{"hello", "world"},
		},
		{
			name: "empty text",
			text: "",
			want: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tokenizer.splitText(tt.text)
			if len(got) != len(tt.want) {
				t.Errorf("splitText() got %v, want %v", got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("splitText() got[%d] = %q, want[%d] = %q", i, got[i], i, tt.want[i])
				}
			}
		})
	}
}

func TestApplyBPE(t *testing.T) {
	tokenizer := &Tokenizer{}

	tests := []struct {
		name string
		word string
		want []string
	}{
		{
			name: "simple word",
			word: "hello",
			want: []string{"h", "e", "l", "l", "o"},
		},
		{
			name: "word with merge",
			word: "he",
			want: []string{"hello"},
		},
		{
			name: "empty word",
			word: "",
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tokenizer.applyBPE(tt.word)
			if len(got) != len(tt.want) {
				t.Errorf("applyBPE() got %v, want %v", got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("applyBPE() got[%d] = %q, want[%d] = %q", i, got[i], i, tt.want[i])
				}
			}
		})
	}
}

func TestBitNetTokenization(t *testing.T) {
	// Create test vocabulary with LLaMA 3 tokens
	vocab := map[string]int{
		"<s>":    1, // Start of sequence
		"</s>":   2, // End of sequence
		"<unk>":  3, // Unknown token
		"▁":      4, // Special space token
		"hello":  5,
		"world":  6,
		"how":    7,
		"are":    8,
		"you":    9,
		"today":  10,
		"doing":  11,
		"fine":   12,
		"thanks": 13,
		"for":    14,
		"asking": 15,
	}

	// Create test special tokens
	specialTokens := map[string]int{
		"<s>":   1,
		"</s>":  2,
		"<unk>": 3,
	}

	// Create test tokenizer files
	testFS := &testFS{
		files: map[string][]byte{
			"tokenizer/vocab.json": func() []byte {
				data, _ := json.Marshal(vocab)
				return data
			}(),
			// Merges as an ordered list (simulate merges.txt as in real BPE)
			"tokenizer/merges.txt": []byte("h e he\nl l ll\nhe l hello\nw o wo\nwo r wor\nwor l worl\nworl d world\n"),
			"tokenizer/special_tokens.json": func() []byte {
				data, _ := json.Marshal(specialTokens)
				return data
			}(),
		},
	}

	tokenizer, err := NewTokenizer(testFS, "tokenizer")
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	// Test cases with known prompts
	tests := []struct {
		name     string
		input    string
		expected []int
	}{
		{
			name:     "simple greeting",
			input:    "<s>hello world</s>",
			expected: []int{1, 5, 4, 6, 2},
		},
		{
			name:     "conversation",
			input:    "<s>how are you today</s>",
			expected: []int{1, 7, 4, 8, 4, 9, 4, 10, 2},
		},
		{
			name:     "response",
			input:    "<s>doing fine thanks for asking</s>",
			expected: []int{1, 11, 4, 12, 4, 13, 4, 14, 4, 15, 2},
		},
		{
			name:     "unknown token",
			input:    "<s>hello unknown world</s>",
			expected: []int{1, 5, 4, 3, 4, 6, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens, err := tokenizer.Tokenize(tt.input)
			if err != nil {
				t.Errorf("Tokenize() error = %v", err)
				return
			}

			if len(tokens) != len(tt.expected) {
				t.Errorf("Tokenize() got %v tokens, want %v tokens", len(tokens), len(tt.expected))
				return
			}

			for i := range tokens {
				if tokens[i] != tt.expected[i] {
					t.Errorf("Tokenize() got[%d] = %v, want[%d] = %v", i, tokens[i], i, tt.expected[i])
				}
			}

			// Test detokenization
			text, err := tokenizer.Detokenize(tokens)
			if err != nil {
				t.Errorf("Detokenize() error = %v", err)
				return
			}

			// Remove special tokens for comparison
			expectedText := strings.ReplaceAll(tt.input, "<s>", "")
			expectedText = strings.ReplaceAll(expectedText, "</s>", "")
			expectedText = strings.TrimSpace(expectedText)

			if text != expectedText {
				t.Errorf("Detokenize() got %q, want %q", text, expectedText)
			}
		})
	}
}
