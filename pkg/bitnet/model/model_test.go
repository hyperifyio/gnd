package model

import (
	"embed"
	"testing"
)

//go:embed testdata
var testFS embed.FS

func TestNewConfig(t *testing.T) {
	config := NewConfig()
	if config == nil {
		t.Fatal("NewConfig returned nil")
	}

	// Verify default values
	if config.HiddenSize != 2048 {
		t.Errorf("expected HiddenSize to be 2048, got %d", config.HiddenSize)
	}
	if config.NumHeads != 16 {
		t.Errorf("expected NumHeads to be 16, got %d", config.NumHeads)
	}
	if config.NumLayers != 24 {
		t.Errorf("expected NumLayers to be 24, got %d", config.NumLayers)
	}
	if config.VocabSize != 32000 {
		t.Errorf("expected VocabSize to be 32000, got %d", config.VocabSize)
	}
	if config.MaxSeqLength != 4096 {
		t.Errorf("expected MaxSeqLength to be 4096, got %d", config.MaxSeqLength)
	}
	if config.IntermediateSize != 8192 {
		t.Errorf("expected IntermediateSize to be 8192, got %d", config.IntermediateSize)
	}
}

func TestNewModel(t *testing.T) {
	// Test with nil config
	model := NewModel(nil, testFS)
	if model == nil {
		t.Fatal("NewModel returned nil")
	}
	if model.config == nil {
		t.Fatal("model.config is nil")
	}

	// Test with custom config
	customConfig := &Config{
		HiddenSize:       1024,
		NumHeads:         8,
		NumLayers:        12,
		VocabSize:        16000,
		MaxSeqLength:     2048,
		IntermediateSize: 4096,
	}
	model = NewModel(customConfig, testFS)
	if model == nil {
		t.Fatal("NewModel returned nil")
	}
	if model.config != customConfig {
		t.Error("model.config does not match custom config")
	}
}

func TestLoadWeights(t *testing.T) {
	model := NewModel(nil, testFS)

	// Test with non-existent file
	err := model.LoadWeights("nonexistent.bin")
	if err == nil {
		t.Error("expected error for non-existent file")
	}

	// Test with invalid file format
	err = model.LoadWeights("testdata/invalid.bin")
	if err == nil {
		t.Error("expected error for invalid file format")
	}
}
