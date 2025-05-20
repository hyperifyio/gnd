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
	}

	// Create test tokenizer file
	tokenizerData, err := json.Marshal(map[string]interface{}{
		"vocab":          vocab,
		"merges":         map[string]string{},
		"special_tokens": map[string]int{"[UNK]": 3},
	})
	if err != nil {
		t.Fatal(err)
	}

	testFS := &testFS{
		files: map[string][]byte{
			"tokenizer.json": tokenizerData,
		},
	}

	tokenizer, err := NewTokenizer(testFS, "tokenizer.json")
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	if tokenizer == nil {
		t.Fatal("NewTokenizer returned nil")
	}

	if tokenizer.modelPath != "tokenizer.json" {
		t.Errorf("expected modelPath to be 'tokenizer.json', got %q", tokenizer.modelPath)
	}

	if len(tokenizer.Vocab) != 3 {
		t.Errorf("expected 3 vocabulary items, got %d", len(tokenizer.Vocab))
	}

	if tokenizer.Vocab["hello"] != 1 {
		t.Errorf("expected 'hello' to have ID 1, got %d", tokenizer.Vocab["hello"])
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
			modelPath: "tokenizer.json",
			wantErr:   errors.New("filesystem cannot be nil"),
		},
		{
			name:      "empty model path",
			fs:        &testFS{},
			modelPath: "",
			wantErr:   errors.New("model path cannot be empty"),
		},
		{
			name:      "file not found",
			fs:        &testFS{},
			modelPath: "nonexistent.json",
			wantErr:   ErrTokenizerNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewTokenizer(tt.fs, tt.modelPath)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if err.Error() != tt.wantErr.Error() {
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
	}

	tokenizer := &Tokenizer{
		Vocab:         vocab,
		Merges:        map[string]string{},
		SpecialTokens: map[string]int{"[UNK]": 3},
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
			want:    []int{1, 2},
			wantErr: nil,
		},
		{
			name:    "unknown word",
			text:    "hello unknown",
			want:    []int{1, 3},
			wantErr: nil,
		},
		{
			name:    "empty text",
			text:    "",
			want:    []int{},
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
}

func TestDetokenize(t *testing.T) {
	// Create test vocabulary
	vocab := map[string]int{
		"hello": 1,
		"world": 2,
		"[UNK]": 3,
	}

	tokenizer := &Tokenizer{
		Vocab:         vocab,
		Merges:        map[string]string{},
		SpecialTokens: map[string]int{"[UNK]": 3},
	}

	tests := []struct {
		name    string
		ids     []int
		want    string
		wantErr error
	}{
		{
			name:    "known tokens",
			ids:     []int{1, 2},
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
