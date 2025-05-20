package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestModelPipeline(t *testing.T) {
	// Create temporary test directory
	tmpDir := t.TempDir()

	// Create test model files
	modelDir := filepath.Join(tmpDir, "model")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("Failed to create model directory: %v", err)
	}

	// Create test weights file
	weights := &Weights{
		TokenEmbedding:    []float32{0.1, 0.2, 0.3},
		PositionEmbedding: []float32{0.4, 0.5, 0.6},
		Layers: []LayerWeights{
			{
				SelfAttention: AttentionWeights{
					Query:  []float32{0.1, 0.2},
					Key:    []float32{0.3, 0.4},
					Value:  []float32{0.5, 0.6},
					Output: []float32{0.7, 0.8},
				},
				FFN: FFNWeights{
					Up:   []float32{0.1, 0.2},
					Down: []float32{0.3, 0.4},
				},
			},
		},
	}

	weightsData, err := json.Marshal(weights)
	if err != nil {
		t.Fatalf("Failed to marshal weights: %v", err)
	}

	if err := os.WriteFile(filepath.Join(modelDir, "weights.json"), weightsData, 0644); err != nil {
		t.Fatalf("Failed to write weights file: %v", err)
	}

	// Create test tokenizer file
	tokenizer := &Tokenizer{
		Vocab: map[string]int{
			"hello": 1,
			"world": 2,
			"##he":  3,
			"##ll":  4,
			"##o":   5,
		},
		Merges: map[string]string{
			"he": "##he",
			"ll": "##ll",
		},
		SpecialTokens: map[string]int{
			"[PAD]": 0,
			"[UNK]": 6,
		},
	}

	tokenizerData, err := json.Marshal(tokenizer)
	if err != nil {
		t.Fatalf("Failed to marshal tokenizer: %v", err)
	}

	if err := os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), tokenizerData, 0644); err != nil {
		t.Fatalf("Failed to write tokenizer file: %v", err)
	}

	// Create model config
	config := &Config{
		VocabSize:     7,
		HiddenSize:    3,
		NumLayers:     1,
		NumHeads:      2,
		MaxSeqLength:  512,
		ModelPath:     modelDir,
		TokenizerPath: modelDir,
	}

	// Test complete pipeline
	model, err := NewModel(config)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	// Test tokenization
	text := "hello world"
	tokens, err := model.Tokenizer.Tokenize(text)
	if err != nil {
		t.Fatalf("Failed to tokenize text: %v", err)
	}

	expectedTokens := []int{1, 2}
	if !compareIntSlices(tokens, expectedTokens) {
		t.Errorf("Tokenize() = %v, want %v", tokens, expectedTokens)
	}

	// Test decoding
	decoded, err := model.Tokenizer.Decode(tokens)
	if err != nil {
		t.Fatalf("Failed to decode tokens: %v", err)
	}

	expectedText := "helloworld"
	if decoded != expectedText {
		t.Errorf("Decode() = %v, want %v", decoded, expectedText)
	}

	// Test model weights
	if model.Weights == nil {
		t.Fatal("Model weights not loaded")
	}

	if len(model.Weights.TokenEmbedding) != 3 {
		t.Errorf("TokenEmbedding size = %v, want %v", len(model.Weights.TokenEmbedding), 3)
	}

	if len(model.Weights.Layers) != 1 {
		t.Errorf("Number of layers = %v, want %v", len(model.Weights.Layers), 1)
	}
}
