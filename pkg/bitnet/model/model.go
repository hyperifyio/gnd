package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/fs"

	"github.com/hyperifyio/gnd/pkg/bitnet/internal/model"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

// Static errors
var (
	ErrInvalidWeightsFile      = errors.New("bitnet: invalid weights file format")
	ErrUnsupportedVersion      = errors.New("bitnet: unsupported weights file version")
	ErrInferenceNotImplemented = errors.New("bitnet: inference not implemented yet")
	ErrWeightsFileOpen         = errors.New("bitnet: failed to open weights file")
	ErrWeightsFileRead         = errors.New("bitnet: failed to read weights file")
	ErrWeightsNotLoaded        = errors.New("bitnet: weights not loaded")
	ErrInvalidToken            = errors.New("bitnet: invalid token")
	ErrTokenizerNotLoaded      = errors.New("bitnet: tokenizer not loaded")
	ErrTokenizerInit           = errors.New("bitnet: failed to initialize tokenizer")
	ErrTokenization            = errors.New("bitnet: tokenization error")
	ErrInvalidWeightValue      = errors.New("bitnet: invalid weight value")
	ErrSequenceTooLong         = errors.New("bitnet: sequence length exceeds maximum")
)

// Model represents a BitNet model
type Model struct {
	config    *Config
	fs        fs.FS
	weights   *ModelWeights
	tokenizer *model.Tokenizer
	done      chan struct{}
	readBuf   []byte // Buffer for reading ternary weights
}

// Config holds the model configuration
type Config struct {
	// Model dimensions
	HiddenSize       int
	NumHeads         int
	NumLayers        int
	VocabSize        int
	MaxSeqLength     int
	IntermediateSize int
}

// NewConfig creates a new default configuration for BitNet b1.58-2B-4T
func NewConfig() *Config {
	return &Config{
		HiddenSize:       2048,
		NumHeads:         16,
		NumLayers:        24,
		VocabSize:        32000,
		MaxSeqLength:     4096,
		IntermediateSize: 8192,
	}
}

// NewModel creates a new Model instance
func NewModel(config *Config, fs fs.FS) *Model {
	if config == nil {
		config = NewConfig()
	}
	return &Model{
		config: config,
		fs:     fs,
		done:   make(chan struct{}),
	}
}

// LoadWeights loads the model weights from a file
func (m *Model) LoadWeights(path string) error {
	// Open the weights file
	file, err := m.fs.Open(path)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to open weights file: %v", err)
		return ErrWeightsFileOpen
	}
	defer file.Close()

	// Read the header
	header := make([]byte, 8)
	if _, err := io.ReadFull(file, header); err != nil {
		loggers.Printf(loggers.Debug, "failed to read weights file header: %v", err)
		return ErrWeightsFileRead
	}

	// Verify magic number
	if binary.LittleEndian.Uint32(header[0:4]) != 0x424E4554 { // "BNET"
		return ErrInvalidWeightsFile
	}

	// Verify version
	if binary.LittleEndian.Uint32(header[4:8]) != 1 {
		return ErrUnsupportedVersion
	}

	// Initialize tokenizer
	tokenizer, err := model.NewTokenizer(m.fs, "tokenizer")
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to initialize tokenizer: %v", err)
		return ErrTokenizerInit
	}
	m.tokenizer = tokenizer

	// Pre-calculate sizes for all allocations
	embeddingSize := m.config.VocabSize * m.config.HiddenSize
	qkvSize := m.config.HiddenSize * 3 * m.config.HiddenSize
	outSize := m.config.HiddenSize * m.config.HiddenSize
	ffnUpSize := m.config.HiddenSize * m.config.IntermediateSize
	ffnDownSize := m.config.IntermediateSize * m.config.HiddenSize

	// Initialize weights structure with pre-allocated slices
	m.weights = &ModelWeights{
		TokenEmbedding: make([]int8, embeddingSize),
		Blocks:         make([]*TransformerBlock, m.config.NumLayers),
		FinalNorm:      make([]int8, m.config.HiddenSize),
	}

	// Pre-allocate all transformer blocks
	for i := 0; i < m.config.NumLayers; i++ {
		m.weights.Blocks[i] = &TransformerBlock{
			QKVProj:  make([]int8, qkvSize),
			OutProj:  make([]int8, outSize),
			FFNUp:    make([]int8, ffnUpSize),
			FFNDown:  make([]int8, ffnDownSize),
			AttnNorm: make([]int8, m.config.HiddenSize),
			FFNNorm:  make([]int8, m.config.HiddenSize),
		}
	}

	// Read token embeddings
	if err := m.readTernaryWeights(file, m.weights.TokenEmbedding); err != nil {
		return err
	}

	// Read transformer blocks
	for i := 0; i < m.config.NumLayers; i++ {
		block := m.weights.Blocks[i]

		// Read all weights for this block
		if err := m.readTernaryWeights(file, block.QKVProj); err != nil {
			return err
		}
		if err := m.readTernaryWeights(file, block.OutProj); err != nil {
			return err
		}
		if err := m.readTernaryWeights(file, block.FFNUp); err != nil {
			return err
		}
		if err := m.readTernaryWeights(file, block.FFNDown); err != nil {
			return err
		}

		// Read normalization weights
		if err := m.readTernaryWeights(file, block.AttnNorm); err != nil {
			return err
		}
		if err := m.readTernaryWeights(file, block.FFNNorm); err != nil {
			return err
		}
	}

	// Read final normalization
	if err := m.readTernaryWeights(file, m.weights.FinalNorm); err != nil {
		return err
	}

	return nil
}

// Infer performs inference on the input text
func (m *Model) Infer(input string) (string, error) {
	if m.tokenizer == nil {
		return "", ErrTokenizerNotLoaded
	}

	// Tokenize input
	tokens, err := m.tokenizer.Tokenize(input)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrTokenization, err)
	}

	// Check sequence length
	if len(tokens) > m.config.MaxSeqLength {
		return "", fmt.Errorf("%w: sequence length %d exceeds maximum %d", ErrSequenceTooLong, len(tokens), m.config.MaxSeqLength)
	}

	// TODO: Implement actual inference
	return "", ErrInferenceNotImplemented
}

// infer is the internal implementation of Infer
func (m *Model) infer(input string) (string, error) {
	if m.tokenizer == nil {
		loggers.Printf(loggers.Debug, "tokenizer not loaded")
		return "", ErrTokenizerNotLoaded
	}

	// Tokenize input
	tokens, err := m.tokenizer.Tokenize(input)
	if err != nil {
		loggers.Printf(loggers.Debug, "tokenization error: %v", err)
		return "", ErrTokenization
	}

	// Check sequence length
	if len(tokens) > m.config.MaxSeqLength {
		loggers.Printf(loggers.Debug, "sequence length %d exceeds maximum %d", len(tokens), m.config.MaxSeqLength)
		return "", ErrSequenceTooLong
	}

	// TODO: Implement actual inference
	return "", ErrInferenceNotImplemented
}

// Close releases any resources held by the model
func (m *Model) Close() {
	select {
	case <-m.done:
		// Already closed
	default:
		close(m.done)
	}
}

// readTernaryWeights reads and unpacks ternary weights from the file
// Each byte contains 4 ternary values (-1, 0, +1) packed as 2 bits each
func (m *Model) readTernaryWeights(file io.Reader, weights []int8) error {
	if file == nil {
		loggers.Printf(loggers.Debug, "nil reader")
		return ErrWeightsFileRead
	}
	if weights == nil {
		loggers.Printf(loggers.Debug, "nil weights slice")
		return ErrWeightsFileRead
	}

	// Calculate number of bytes needed
	numBytes := (len(weights) + 3) / 4 // Round up to nearest byte
	if cap(m.readBuf) < numBytes {
		m.readBuf = make([]byte, numBytes)
	} else {
		m.readBuf = m.readBuf[:numBytes]
	}

	// Read packed weights
	if _, err := io.ReadFull(file, m.readBuf); err != nil {
		loggers.Printf(loggers.Debug, "failed to read weights: %v", err)
		return ErrWeightsFileRead
	}

	// Unpack weights
	for i := 0; i < len(weights); i++ {
		byteIdx := i / 4
		bitOffset := (i % 4) * 2
		packed := m.readBuf[byteIdx] >> bitOffset & 0x03
		switch packed {
		case 0:
			weights[i] = -1
		case 1:
			weights[i] = 0
		case 2:
			weights[i] = 1
		default:
			loggers.Printf(loggers.Debug, "invalid weight value: %d", packed)
			return ErrInvalidWeightValue
		}
	}

	return nil
}

// Add new structures for model parameters:

// TransformerBlock represents a single transformer block's parameters
type TransformerBlock struct {
	// Attention parameters
	QKVProj []int8 // QKV projection weights (ternary)
	OutProj []int8 // Output projection weights (ternary)

	// Feed-forward parameters
	FFNUp   []int8 // First FFN layer weights (ternary)
	FFNDown []int8 // Second FFN layer weights (ternary)

	// Normalization parameters
	AttnNorm []int8 // Attention normalization weights (ternary)
	FFNNorm  []int8 // FFN normalization weights (ternary)
}

// ModelWeights represents all model parameters
type ModelWeights struct {
	// Token embeddings (shared with output layer)
	TokenEmbedding []int8 // Token embedding weights (ternary)
	Blocks         []*TransformerBlock
	FinalNorm      []int8 // Final normalization weights (ternary)
}
