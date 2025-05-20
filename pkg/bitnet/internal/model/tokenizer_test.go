package model

import (
	"encoding/json"
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

	// Create test tokenizer file
	testVocab := map[string]int{
		"hello": 1,
		"world": 2,
	}
	testData := struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}{
		Model: struct {
			Vocab map[string]int `json:"vocab"`
		}{
			Vocab: testVocab,
		},
	}

	// Create the directory structure
	modelDir := filepath.Join(tmpDir, "pkg", "bitnet", "internal", "assets", "models", "BitNet-b1.58-2B-4T")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("Failed to create model directory: %v", err)
	}

	// Create and write the test tokenizer file
	tokenizerPath := filepath.Join(modelDir, "tokenizer.json")
	file, err := os.Create(tokenizerPath)
	if err != nil {
		t.Fatalf("Failed to create test tokenizer file: %v", err)
	}
	defer file.Close()

	if err := json.NewEncoder(file).Encode(testData); err != nil {
		t.Fatalf("Failed to write test tokenizer data: %v", err)
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

	// Test tokenizer creation and loading
	tokenizer, err := NewTokenizer()
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	if err := tokenizer.Load(); err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Verify vocabulary
	vocab := tokenizer.GetVocab()
	if len(vocab) != len(testVocab) {
		t.Errorf("Expected vocabulary size %d, got %d", len(testVocab), len(vocab))
	}

	for k, v := range testVocab {
		if vocab[k] != v {
			t.Errorf("Expected token '%s' to have value %d, got %d", k, v, vocab[k])
		}
	}

	// Test model path
	if tokenizer.GetModelPath() != tokenizerPath {
		t.Errorf("Expected model path %s, got %s", tokenizerPath, tokenizer.GetModelPath())
	}
}
