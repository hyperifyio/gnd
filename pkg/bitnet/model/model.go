package model

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/hyperifyio/gnd/pkg/bitnet/model/embedded"
)

// Model represents the BitNet model with its weights and tokenizer
type Model struct {
	Config     *Config
	Weights    *Weights
	Tokenizer  *Tokenizer
	Parameters *Parameters
}

// Config holds the model configuration
type Config struct {
	VocabSize     int    `json:"vocab_size"`
	HiddenSize    int    `json:"hidden_size"`
	NumLayers     int    `json:"num_layers"`
	NumHeads      int    `json:"num_heads"`
	MaxSeqLength  int    `json:"max_seq_length"`
	ModelPath     string `json:"model_path"`
	TokenizerPath string `json:"tokenizer_path"`
}

// Weights holds the model weights
type Weights struct {
	// Embedding weights
	TokenEmbedding    []float32 `json:"token_embedding"`
	PositionEmbedding []float32 `json:"position_embedding"`

	// Layer weights
	Layers []LayerWeights `json:"layers"`
}

// LayerWeights represents weights for a single transformer layer
type LayerWeights struct {
	SelfAttention AttentionWeights `json:"self_attention"`
	FFN           FFNWeights       `json:"ffn"`
}

// AttentionWeights holds attention layer weights
type AttentionWeights struct {
	Query  []float32 `json:"query"`
	Key    []float32 `json:"key"`
	Value  []float32 `json:"value"`
	Output []float32 `json:"output"`
}

// FFNWeights holds feed-forward network weights
type FFNWeights struct {
	Up   []float32 `json:"up"`
	Down []float32 `json:"down"`
}

// Parameters holds model hyperparameters
type Parameters struct {
	VocabSize    int     `json:"vocab_size"`
	HiddenSize   int     `json:"hidden_size"`
	NumLayers    int     `json:"num_layers"`
	NumHeads     int     `json:"num_heads"`
	MaxSeqLength int     `json:"max_seq_length"`
	Dropout      float32 `json:"dropout"`
	LayerNormEps float32 `json:"layer_norm_eps"`
}

// NewModel creates a new model instance
func NewModel(config *Config) (*Model, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	model := &Model{
		Config: config,
	}

	// Load weights
	if err := model.loadWeights(); err != nil {
		return nil, fmt.Errorf("failed to load weights: %w", err)
	}

	// Load tokenizer
	if err := model.loadTokenizer(); err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	return model, nil
}

// loadWeights loads the model weights from the embedded file
func (m *Model) loadWeights() error {
	file, err := embedded.GetModelFile()
	if err != nil {
		return fmt.Errorf("failed to open embedded weights file: %w", err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return fmt.Errorf("failed to read weights file: %w", err)
	}

	if err := json.Unmarshal(data, &m.Weights); err != nil {
		return fmt.Errorf("failed to unmarshal weights: %w", err)
	}

	return nil
}

// loadTokenizer loads the tokenizer from the embedded file
func (m *Model) loadTokenizer() error {
	file, err := embedded.GetTokenizerFile()
	if err != nil {
		return fmt.Errorf("failed to open embedded tokenizer file: %w", err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return fmt.Errorf("failed to read tokenizer file: %w", err)
	}

	if err := json.Unmarshal(data, &m.Tokenizer); err != nil {
		return fmt.Errorf("failed to unmarshal tokenizer: %w", err)
	}

	return nil
}
