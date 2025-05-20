package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestNewModel(t *testing.T) {
	// Create temporary test files
	tmpDir := t.TempDir()

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

	if err := os.WriteFile(filepath.Join(tmpDir, "weights.json"), weightsData, 0644); err != nil {
		t.Fatalf("Failed to write weights file: %v", err)
	}

	// Create test tokenizer file
	tokenizer := &Tokenizer{
		Vocab: map[string]int{
			"hello": 1,
			"world": 2,
		},
		Merges: map[string]string{
			"he": "##he",
			"ll": "##ll",
		},
		SpecialTokens: map[string]int{
			"[PAD]": 0,
			"[UNK]": 3,
		},
	}

	tokenizerData, err := json.Marshal(tokenizer)
	if err != nil {
		t.Fatalf("Failed to marshal tokenizer: %v", err)
	}

	if err := os.WriteFile(filepath.Join(tmpDir, "tokenizer.json"), tokenizerData, 0644); err != nil {
		t.Fatalf("Failed to write tokenizer file: %v", err)
	}

	// Test cases
	tests := []struct {
		name    string
		config  *Config
		wantErr bool
	}{
		{
			name: "valid config",
			config: &Config{
				VocabSize:     4,
				HiddenSize:    3,
				NumLayers:     1,
				NumHeads:      2,
				MaxSeqLength:  512,
				ModelPath:     tmpDir,
				TokenizerPath: tmpDir,
			},
			wantErr: false,
		},
		{
			name:    "nil config",
			config:  nil,
			wantErr: true,
		},
		{
			name: "invalid model path",
			config: &Config{
				VocabSize:     4,
				HiddenSize:    3,
				NumLayers:     1,
				NumHeads:      2,
				MaxSeqLength:  512,
				ModelPath:     "/nonexistent",
				TokenizerPath: tmpDir,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := NewModel(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewModel() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && model == nil {
				t.Error("NewModel() returned nil model when no error expected")
			}
		})
	}
}
