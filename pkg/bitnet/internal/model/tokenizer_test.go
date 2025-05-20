package model

import (
	"encoding/json"
	"errors"
	"io/fs"
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

	// Create test merges
	merges := map[string]string{
		"he": "hello",
		"wo": "world",
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
			"tokenizer/merges.txt": []byte("he hello\nwo world\n"),
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

	if tokenizer.Merges["he"] != "hello" {
		t.Errorf("expected 'he' to merge to 'hello', got %q", tokenizer.Merges["he"])
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

	// Create test merges
	merges := map[string]string{
		"he": "hello",
		"wo": "world",
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
			"tokenizer/merges.txt": []byte("he hello\nwo world\n"),
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
		Merges:        map[string]string{},
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
	tokenizer := &Tokenizer{
		Merges: map[string]string{
			"he": "hello",
			"wo": "world",
		},
	}

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
