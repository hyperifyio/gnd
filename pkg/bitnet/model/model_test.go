package model

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/model/embedded"
)

func TestNewModel(t *testing.T) {
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
				ModelPath:     "embedded",
				TokenizerPath: "embedded",
			},
			wantErr: false,
		},
		{
			name:    "nil config",
			config:  nil,
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

func TestModelSize(t *testing.T) {
	size, err := embedded.GetModelSize()
	if err != nil {
		t.Fatalf("Failed to get model size: %v", err)
	}
	if size <= 0 {
		t.Errorf("Expected model size > 0, got %d", size)
	}
}
