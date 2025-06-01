// Package model implements integration tests for the BitNet model implementation.
//
// # BitNet Model Integration Test Suite
//
// This file provides integration tests that verify the BitNet model works correctly
// with embedded model data and real-world usage scenarios.
//
// Key aspects:
//   - Tests model loading and inference with embedded model data.
//   - Verifies tokenization and detokenization with real inputs.
//   - Tests handling of long sequences and special tokens.
//   - Uses a mock filesystem for testing with embedded assets.
//
// Usage:
//   - Used to validate end-to-end model functionality.
//   - Maintainers should run these tests before deploying changes.
//
// Caveats:
//   - Requires embedded model data to be present.
//   - Tests may take longer to run than unit tests.
//   - Any change must pass all integration tests before being merged.
//
// For more details, see BitNet issue #190 and the BitNet project documentation.
package model

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/assets"
	"io"
	"io/fs"
	"os"
	"testing"
	"time"
)

// integrationTestFS implements fs.FS for testing with embedded model data
type integrationTestFS struct {
	modelData []byte
}

func (t *integrationTestFS) Open(name string) (fs.File, error) {
	if name == "model.gguf" {
		return &integrationTestFile{data: t.modelData}, nil
	}
	return nil, os.ErrNotExist
}

// integrationTestFile implements fs.File for testing
type integrationTestFile struct {
	data []byte
	pos  int64
}

func (t *integrationTestFile) Read(p []byte) (n int, err error) {
	if t.pos >= int64(len(t.data)) {
		return 0, io.EOF
	}
	n = copy(p, t.data[t.pos:])
	t.pos += int64(n)
	return n, nil
}

func (t *integrationTestFile) Close() error {
	return nil
}

func (t *integrationTestFile) Stat() (fs.FileInfo, error) {
	return &integrationTestFileInfo{size: int64(len(t.data))}, nil
}

// integrationTestFileInfo implements fs.FileInfo for testing
type integrationTestFileInfo struct {
	size int64
}

func (t *integrationTestFileInfo) Name() string       { return "model.gguf" }
func (t *integrationTestFileInfo) Size() int64        { return t.size }
func (t *integrationTestFileInfo) Mode() fs.FileMode  { return 0 }
func (t *integrationTestFileInfo) ModTime() time.Time { return time.Time{} }
func (t *integrationTestFileInfo) IsDir() bool        { return false }
func (t *integrationTestFileInfo) Sys() interface{}   { return nil }

func TestModelWithEmbeddedData(t *testing.T) {
	// Load embedded model data
	modelData, err := assets.GetModelFile()
	if err != nil {
		t.Fatalf("Failed to load embedded model data: %v", err)
	}

	// Create test filesystem with model data
	fs := &integrationTestFS{modelData: modelData}

	// Create model instance
	config := NewConfig()
	m, err := NewModel(config, fs)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer m.Close()

	// Load model weights
	if err := m.LoadWeights("model.gguf"); err != nil {
		t.Fatalf("Failed to load model weights: %v", err)
	}

	// Initialize tokenizer
	if err := m.InitTokenizer("tokenizer"); err != nil {
		t.Fatalf("Failed to initialize tokenizer: %v", err)
	}

	// Test cases for token decoding
	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "Simple greeting",
			input:    "Hello, how are you?",
			expected: "Hello, how are you?",
		},
		{
			name:     "Question about AI",
			input:    "What is artificial intelligence?",
			expected: "What is artificial intelligence?",
		},
		{
			name:     "Code example",
			input:    "Write a function to calculate fibonacci numbers",
			expected: "Write a function to calculate fibonacci numbers",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Tokenize input
			tokens, err := m.tokenizer.Tokenize(tc.input)
			if err != nil {
				t.Fatalf("Failed to tokenize input: %v", err)
			}

			// Run inference
			outputTokens, err := m.Infer(tokens)
			if err != nil {
				t.Fatalf("Failed to run inference: %v", err)
			}

			// Detokenize output
			output, err := m.tokenizer.Detokenize(outputTokens)
			if err != nil {
				t.Fatalf("Failed to detokenize output: %v", err)
			}

			// Verify output
			if output != tc.expected {
				t.Errorf("Output mismatch:\nExpected: %s\nGot: %s", tc.expected, output)
			}
		})
	}
}

func TestModelWithLongSequence(t *testing.T) {
	// Load embedded model data
	modelData, err := assets.GetModelFile()
	if err != nil {
		t.Fatalf("Failed to load embedded model data: %v", err)
	}

	// Create test filesystem with model data
	fs := &integrationTestFS{modelData: modelData}

	// Create model instance
	config := NewConfig()
	m, err := NewModel(config, fs)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer m.Close()

	// Load model weights
	if err := m.LoadWeights("model.gguf"); err != nil {
		t.Fatalf("Failed to load model weights: %v", err)
	}

	// Initialize tokenizer
	if err := m.InitTokenizer("tokenizer"); err != nil {
		t.Fatalf("Failed to initialize tokenizer: %v", err)
	}

	// Create a long input sequence
	longInput := "This is a test of the model's ability to handle long sequences. " +
		"We want to verify that the model can process inputs up to the maximum context length " +
		"of 4096 tokens. This test will help ensure that the token decoding implementation " +
		"works correctly with longer inputs."

	// Tokenize input
	tokens, err := m.tokenizer.Tokenize(longInput)
	if err != nil {
		t.Fatalf("Failed to tokenize input: %v", err)
	}

	// Verify token count is within limits
	if len(tokens) > m.config.MaxSeqLength {
		t.Fatalf("Input sequence too long: %d tokens (max: %d)", len(tokens), m.config.MaxSeqLength)
	}

	// Run inference
	outputTokens, err := m.Infer(tokens)
	if err != nil {
		t.Fatalf("Failed to run inference: %v", err)
	}

	// Verify output sequence length
	if len(outputTokens) > m.config.MaxSeqLength {
		t.Fatalf("Output sequence too long: %d tokens (max: %d)", len(outputTokens), m.config.MaxSeqLength)
	}

	// Detokenize output
	output, err := m.tokenizer.Detokenize(outputTokens)
	if err != nil {
		t.Fatalf("Failed to detokenize output: %v", err)
	}

	// Basic output validation
	if len(output) == 0 {
		t.Error("Output is empty")
	}
}

func TestModelWithSpecialTokens(t *testing.T) {
	// Load embedded model data
	modelData, err := assets.GetModelFile()
	if err != nil {
		t.Fatalf("Failed to load embedded model data: %v", err)
	}

	// Create test filesystem with model data
	fs := &integrationTestFS{modelData: modelData}

	// Create model instance
	config := NewConfig()
	m, err := NewModel(config, fs)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer m.Close()

	// Load model weights
	if err := m.LoadWeights("model.gguf"); err != nil {
		t.Fatalf("Failed to load model weights: %v", err)
	}

	// Initialize tokenizer
	if err := m.InitTokenizer("tokenizer"); err != nil {
		t.Fatalf("Failed to initialize tokenizer: %v", err)
	}

	// Test cases with special tokens
	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "With unknown token",
			input:    "This is a [UNK] token test",
			expected: "This is a [UNK] token test",
		},
		{
			name:     "With padding token",
			input:    "This is a [PAD] token test",
			expected: "This is a [PAD] token test",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Tokenize input
			tokens, err := m.tokenizer.Tokenize(tc.input)
			if err != nil {
				t.Fatalf("Failed to tokenize input: %v", err)
			}

			// Run inference
			outputTokens, err := m.Infer(tokens)
			if err != nil {
				t.Fatalf("Failed to run inference: %v", err)
			}

			// Detokenize output
			output, err := m.tokenizer.Detokenize(outputTokens)
			if err != nil {
				t.Fatalf("Failed to detokenize output: %v", err)
			}

			// Verify output
			if output != tc.expected {
				t.Errorf("Output mismatch:\nExpected: %s\nGot: %s", tc.expected, output)
			}
		})
	}
}
