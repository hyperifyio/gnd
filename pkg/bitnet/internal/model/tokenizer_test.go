package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTokenizer(t *testing.T) {
	// Create a temporary directory for test files
	tmpDir, err := os.MkdirTemp("", "bitnet-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create the directory structure
	tokenizerDir := filepath.Join(tmpDir, "pkg", "bitnet", "internal", "assets", "models", "BitNet-b1.58-2B-4T")
	if err := os.MkdirAll(tokenizerDir, 0755); err != nil {
		t.Fatalf("Failed to create tokenizer directory: %v", err)
	}

	// Create a valid tokenizer JSON file
	tokenizerPath := filepath.Join(tokenizerDir, "tokenizer.json")
	tokenizerJSON := `{
		"vocab": {
			"hello": 1,
			"world": 2,
			"##ing": 3,
			"##ed": 4
		},
		"merges": {
			"h e": "he",
			"he l": "hel",
			"hel l": "hell",
			"hell o": "hello"
		},
		"special_tokens": {
			"[UNK]": 0,
			"[PAD]": 5,
			"[CLS]": 6,
			"[SEP]": 7
		}
	}`
	if err := os.WriteFile(tokenizerPath, []byte(tokenizerJSON), 0644); err != nil {
		t.Fatalf("Failed to write tokenizer file: %v", err)
	}

	// Change to the temp directory for the test
	originalDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Failed to get current directory: %v", err)
	}
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to change to temp directory: %v", err)
	}
	defer os.Chdir(originalDir)

	// Create a new tokenizer
	tokenizer, err := NewTokenizer()
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	// Test cases for tokenization
	tests := []struct {
		name     string
		input    string
		expected []int
		wantErr  bool
	}{
		{
			name:     "simple words",
			input:    "hello world",
			expected: []int{1, 2},
			wantErr:  false,
		},
		{
			name:     "unknown word",
			input:    "unknown",
			expected: []int{0},
			wantErr:  false,
		},
		{
			name:     "empty string",
			input:    "",
			expected: []int{},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens, err := tokenizer.Tokenize(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Tokenize() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if len(tokens) != len(tt.expected) {
					t.Errorf("Tokenize() got %v tokens, want %v", len(tokens), len(tt.expected))
					return
				}
				for i, token := range tokens {
					if token != tt.expected[i] {
						t.Errorf("Tokenize()[%d] = %v, want %v", i, token, tt.expected[i])
					}
				}
			}
		})
	}

	// Test decoding
	decodeTests := []struct {
		name     string
		input    []int
		expected string
		wantErr  bool
	}{
		{
			name:     "simple tokens",
			input:    []int{1, 2},
			expected: "helloworld",
			wantErr:  false,
		},
		{
			name:     "unknown token",
			input:    []int{0},
			expected: "",
			wantErr:  true,
		},
		{
			name:     "empty tokens",
			input:    []int{},
			expected: "",
			wantErr:  false,
		},
	}

	for _, tt := range decodeTests {
		t.Run(tt.name, func(t *testing.T) {
			text, err := tokenizer.Decode(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Decode() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && text != tt.expected {
				t.Errorf("Decode() = %v, want %v", text, tt.expected)
			}
		})
	}

	// Verify tokenizer properties
	if tokenizer.Vocab == nil {
		t.Fatal("Expected non-nil vocabulary")
	}
	if len(tokenizer.Vocab) != 4 {
		t.Errorf("Expected vocabulary size 4, got %d", len(tokenizer.Vocab))
	}
	if tokenizer.Vocab["hello"] != 1 {
		t.Errorf("Expected 'hello' token ID 1, got %d", tokenizer.Vocab["hello"])
	}
	if tokenizer.Vocab["world"] != 2 {
		t.Errorf("Expected 'world' token ID 2, got %d", tokenizer.Vocab["world"])
	}
	if tokenizer.SpecialTokens["[UNK]"] != 0 {
		t.Errorf("Expected '[UNK]' token ID 0, got %d", tokenizer.SpecialTokens["[UNK]"])
	}
}
