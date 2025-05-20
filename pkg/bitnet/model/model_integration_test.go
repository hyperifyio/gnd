package model

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/model/embedded"
)

func TestModelPipeline(t *testing.T) {
	// Create model config
	config := &Config{
		VocabSize:     7,
		HiddenSize:    3,
		NumLayers:     1,
		NumHeads:      2,
		MaxSeqLength:  512,
		ModelPath:     "embedded",
		TokenizerPath: "embedded",
	}

	// Test complete pipeline
	model, err := NewModel(config)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	// Test model weights
	if model.Weights == nil {
		t.Fatal("Model weights not loaded")
	}

	// Test tokenizer
	if model.Tokenizer == nil {
		t.Fatal("Tokenizer not loaded")
	}

	// Test model size
	size, err := embedded.GetModelSize()
	if err != nil {
		t.Fatalf("Failed to get model size: %v", err)
	}
	if size <= 0 {
		t.Errorf("Expected model size > 0, got %d", size)
	}
}
