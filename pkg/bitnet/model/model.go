package model

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
)

var (
	ErrConfigNil         = errors.New("config cannot be nil")
	ErrModelNotFound     = errors.New("model file not found")
	ErrTokenizerNotFound = errors.New("tokenizer file not found")
	ErrInvalidJSON       = errors.New("invalid JSON data")
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

	// Raw binary data
	RawData []byte `json:"-"`
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
		return nil, ErrConfigNil
	}

	model := &Model{
		Config: config,
	}

	// Load weights
	if err := model.loadWeights(); err != nil {
		return nil, err
	}

	// Load tokenizer
	if err := model.loadTokenizer(); err != nil {
		return nil, err
	}

	return model, nil
}

// loadWeights loads the model weights from the file
func (m *Model) loadWeights() error {
	modelPath := filepath.Join("pkg", "bitnet", "internal", "assets", "models", "BitNet-b1.58-2B-4T", "model.bin")

	file, err := os.Open(modelPath)
	if err != nil {
		return ErrModelNotFound
	}
	defer file.Close()

	// Read the entire file into memory
	data, err := io.ReadAll(file)
	if err != nil {
		return err
	}

	// Store the raw data
	m.Weights = &Weights{
		RawData: data,
	}

	return nil
}

// loadTokenizer loads the tokenizer from the file
func (m *Model) loadTokenizer() error {
	tokenizerPath := filepath.Join("pkg", "bitnet", "internal", "assets", "models", "BitNet-b1.58-2B-4T", "tokenizer.json")

	file, err := os.Open(tokenizerPath)
	if err != nil {
		return ErrTokenizerNotFound
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return err
	}

	if err := json.Unmarshal(data, &m.Tokenizer); err != nil {
		return ErrInvalidJSON
	}

	return nil
}
