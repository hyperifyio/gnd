// Package model implements the BitNet neural network model architecture.
// It provides functionality for loading model weights, performing inference,
// and managing the model's lifecycle. The package supports ternary quantization
// for efficient model storage and computation.
package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"runtime"
	"sync"

	"github.com/hyperifyio/gnd/pkg/bitnet/internal/math"
	"github.com/hyperifyio/gnd/pkg/bitnet/internal/model"
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

// Common errors returned by model operations
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
	ErrDetokenization          = errors.New("bitnet: detokenization error")
	ErrInvalidInputShape       = errors.New("bitnet: invalid input shape")
	ErrAttentionSublayer       = errors.New("bitnet: failed to create attention sublayer")
	ErrAttentionWeights        = errors.New("bitnet: failed to set attention weights")
	ErrAttentionForward        = errors.New("bitnet: attention forward pass failed")
	ErrUnexpectedTensorShape   = errors.New("bitnet: unexpected tensor shape")
	ErrInvalidTokenID          = errors.New("model: invalid token ID")
)

// Model represents a BitNet model instance. It manages the model's configuration,
// weights, tokenizer, and provides methods for inference.
type Model struct {
	config    *Config
	fs        fs.FS
	weights   *ModelWeights
	tokenizer *model.Tokenizer
	done      chan struct{}
	readBuf   []byte     // Buffer for reading ternary weights
	closeMu   sync.Mutex // Mutex to protect Close() operations
}

// Config represents the model configuration parameters.
// These parameters define the architecture and capacity of the model.
type Config struct {
	// Vocabulary size defines the number of unique tokens the model can process
	VocabSize int
	// HiddenSize defines the dimension of the model's hidden states
	HiddenSize int
	// NumHeads defines the number of attention heads in each layer
	NumHeads int
	// NumKVHeads defines the number of key/value heads for grouped-query attention
	NumKVHeads int
	// NumLayers defines the number of transformer layers in the model
	NumLayers int
	// IntermediateSize defines the dimension of the feed-forward network's hidden layer
	IntermediateSize int
	// MaxSeqLength defines the maximum sequence length the model can process
	MaxSeqLength int
}

// NewConfig creates a new default configuration for BitNet b1.58-2B-4T.
// The configuration is optimized for the 2B parameter model with 4-bit quantization.
func NewConfig() *Config {
	return &Config{
		HiddenSize:       2048,
		NumHeads:         16,
		NumKVHeads:       16,
		NumLayers:        24,
		VocabSize:        32000,
		MaxSeqLength:     4096,
		IntermediateSize: 8192,
	}
}

// NewModel creates a new Model instance with the given configuration and filesystem.
// If config is nil, a default configuration is used.
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

// LoadWeights loads the model weights from a file.
// The weights file must be in the correct format with a valid magic number and version.
// The function reads and initializes all model parameters including embeddings,
// transformer blocks, and normalization layers.
func (m *Model) LoadWeights(path string) error {
	if m == nil {
		return ErrWeightsNotLoaded
	}
	if m.fs == nil {
		return ErrWeightsFileOpen
	}

	// Open the weights file
	file, err := m.fs.Open(path)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to open weights file: %v", err)
		return ErrWeightsFileOpen
	}
	defer file.Close()

	// Read the header
	header := make([]byte, 8)
	n, err := io.ReadFull(file, header)
	if err != nil {
		loggers.Printf(loggers.Debug, "[DEBUG] failed to read weights file header: %v", err)
		return ErrWeightsFileRead
	}
	if n < 8 {
		loggers.Printf(loggers.Debug, "[DEBUG] header too short: got %d bytes", n)
		return ErrWeightsFileRead
	}

	// Verify version first
	if binary.LittleEndian.Uint32(header[4:8]) != 1 {
		loggers.Printf(loggers.Debug, "[DEBUG] unsupported version: %d", binary.LittleEndian.Uint32(header[4:8]))
		return ErrUnsupportedVersion
	}
	// Verify magic number
	if binary.LittleEndian.Uint32(header[0:4]) != 0x424E4554 { // "BNET"
		loggers.Printf(loggers.Debug, "[DEBUG] invalid magic number: %x", header[0:4])
		return ErrInvalidWeightsFile
	}

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
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return ErrWeightsFileRead
		}
		return err
	}

	// Read transformer blocks
	for i := 0; i < m.config.NumLayers; i++ {
		if m.weights == nil || m.weights.Blocks == nil || i >= len(m.weights.Blocks) {
			return ErrWeightsNotLoaded
		}

		block := m.weights.Blocks[i]
		if block == nil {
			return ErrWeightsNotLoaded
		}

		// Read all weights for this block
		if err := m.readTernaryWeights(file, block.QKVProj); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return ErrWeightsFileRead
			}
			return err
		}
		if err := m.readTernaryWeights(file, block.OutProj); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return ErrWeightsFileRead
			}
			return err
		}
		if err := m.readTernaryWeights(file, block.FFNUp); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return ErrWeightsFileRead
			}
			return err
		}
		if err := m.readTernaryWeights(file, block.FFNDown); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return ErrWeightsFileRead
			}
			return err
		}
		if err := m.readTernaryWeights(file, block.AttnNorm); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return ErrWeightsFileRead
			}
			return err
		}
		if err := m.readTernaryWeights(file, block.FFNNorm); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return ErrWeightsFileRead
			}
			return err
		}
	}

	// Read final normalization weights
	if err := m.readTernaryWeights(file, m.weights.FinalNorm); err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return ErrWeightsFileRead
		}
		return err
	}

	// Initialize tokenizer (after all weights are loaded)
	tokenizer, err := model.NewTokenizer(m.fs, "tokenizer")
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to initialize tokenizer: %v", err)
		return ErrTokenizerInit
	}
	m.tokenizer = tokenizer

	return nil
}

// Infer performs inference on the input tokens
// input: slice of token IDs
// Returns: slice of output token IDs
func (m *Model) Infer(tokens []int) ([]int, error) {
	if len(tokens) == 0 {
		return nil, ErrInvalidToken
	}

	if len(tokens) > m.config.MaxSeqLength {
		return nil, ErrSequenceTooLong
	}

	if m.weights == nil {
		return nil, ErrWeightsNotLoaded
	}

	// Convert tokens to hidden states using embedding layer
	hiddenStates, err := m.embedTokens(tokens)
	if err != nil {
		return nil, err
	}

	// Convert hidden states to tensor with shape [batch, seq, hidden]
	hiddenStatesTensor := tensor.NewTensor(1, len(tokens), m.config.HiddenSize)
	defer hiddenStatesTensor.Close()
	for i := 0; i < len(tokens); i++ {
		for j := 0; j < m.config.HiddenSize; j++ {
			hiddenStatesTensor.Set(int8(hiddenStates[i][j]), 0, i, j)
		}
	}

	// Process through transformer blocks (stacking logic)
	for _, block := range m.weights.Blocks {
		// Create attention sublayer
		attn, err := math.NewAttentionSublayer(m.config.HiddenSize, m.config.NumHeads, m.config.NumKVHeads)
		if err != nil {
			return nil, fmt.Errorf("failed to create attention sublayer: %w", err)
		}
		defer attn.Close()

		// Convert weights to tensors
		h := m.config.HiddenSize
		qTensor := tensor.NewTensor(h, h)
		kTensor := tensor.NewTensor(h, h)
		vTensor := tensor.NewTensor(h, h)
		outTensor := tensor.NewTensor(h, h)

		// Copy weights into projection matrices
		for i := 0; i < h; i++ {
			for j := 0; j < h; j++ {
				// Q projection
				qTensor.Set(block.QKVProj[i*h+j], i, j)
				// K projection
				kTensor.Set(block.QKVProj[h*h+i*h+j], i, j)
				// V projection
				vTensor.Set(block.QKVProj[2*h*h+i*h+j], i, j)
				// Output projection
				outTensor.Set(block.OutProj[i*h+j], i, j)
			}
		}

		// Set attention weights
		if err := attn.SetWeights(qTensor, kTensor, vTensor, outTensor); err != nil {
			return nil, fmt.Errorf("failed to set attention weights: %w", err)
		}

		// Convert attention norm to float32 and create tensor
		attnGammaTensor := tensor.NewTensor(h)
		for i := 0; i < h; i++ {
			attnGammaTensor.Set(int8(block.AttnNorm[i]), i)
		}
		if err := attn.SetGamma(attnGammaTensor); err != nil {
			return nil, fmt.Errorf("failed to set attention gamma: %w", err)
		}

		// Create FFN sublayer
		ffn := math.NewFFNSublayer(m.config.HiddenSize, m.config.IntermediateSize)
		defer ffn.Close()

		// Convert FFN weights to tensors
		ffnUpTensor := tensor.NewTensor(m.config.IntermediateSize, m.config.HiddenSize)
		ffnDownTensor := tensor.NewTensor(m.config.HiddenSize, m.config.IntermediateSize)

		// Copy FFN weights
		for i := 0; i < m.config.IntermediateSize; i++ {
			for j := 0; j < m.config.HiddenSize; j++ {
				ffnUpTensor.Set(block.FFNUp[i*m.config.HiddenSize+j], i, j)
			}
		}
		for i := 0; i < m.config.HiddenSize; i++ {
			for j := 0; j < m.config.IntermediateSize; j++ {
				ffnDownTensor.Set(block.FFNDown[i*m.config.IntermediateSize+j], i, j)
			}
		}

		// Set FFN weights
		ffn.SetWeights(ffnUpTensor, ffnDownTensor)

		// Convert FFN norm to float32
		ffnGamma := make([]float32, m.config.HiddenSize)
		for i := 0; i < m.config.HiddenSize; i++ {
			ffnGamma[i] = float32(block.FFNNorm[i])
		}
		ffn.SetGamma(ffnGamma)

		// Apply attention
		hiddenStatesTensor, err = attn.Forward(hiddenStatesTensor)
		if err != nil {
			return nil, fmt.Errorf("attention forward pass failed: %w", err)
		}

		// Apply FFN
		hiddenStatesTensor, err = ffn.Forward(hiddenStatesTensor)
		if err != nil {
			return nil, fmt.Errorf("FFN forward pass failed: %w", err)
		}
	}

	// Apply final normalization
	finalNorm := math.NewLayerNorm(m.config.HiddenSize)
	defer finalNorm.Close()

	// Convert final norm weights to tensor
	finalNormTensor := tensor.NewTensor(m.config.HiddenSize)
	for i := 0; i < m.config.HiddenSize; i++ {
		finalNormTensor.Set(m.weights.FinalNorm[i], i)
	}

	// Set final norm gamma
	finalNormGammaTensor := tensor.NewTensor(m.config.HiddenSize)
	finalNormGammaData := convertInt8ToFloat32(finalNormTensor.Data())
	for i := 0; i < m.config.HiddenSize; i++ {
		finalNormGammaTensor.Set(int8(finalNormGammaData[i]), i)
	}
	if err := finalNorm.SetGamma(finalNormGammaTensor); err != nil {
		return nil, fmt.Errorf("failed to set final norm gamma: %w", err)
	}

	// Apply final normalization
	hiddenStatesTensor, err = finalNorm.Forward(hiddenStatesTensor)
	if err != nil {
		return nil, fmt.Errorf("final norm forward pass failed: %w", err)
	}

	// For now, just return input tokens as output
	// TODO: Implement proper output projection and token prediction
	outputTokens := make([]int, len(tokens))
	for i := 0; i < len(tokens); i++ {
		outputTokens[i] = tokens[i]
	}
	return outputTokens, nil
}

// embedTokens converts token IDs to embeddings using the model's token embedding layer.
func (m *Model) embedTokens(tokens []int) ([][]float32, error) {
	if len(tokens) == 0 {
		return nil, ErrInvalidToken
	}
	if m.weights == nil || m.weights.TokenEmbedding == nil {
		return nil, ErrWeightsNotLoaded
	}

	// Pre-allocate embeddings slice
	embeddings := make([][]float32, len(tokens))
	for i := range embeddings {
		embeddings[i] = make([]float32, m.config.HiddenSize)
	}

	// Process each token
	for i, tokenID := range tokens {
		if tokenID < 0 || tokenID >= m.config.VocabSize {
			return nil, ErrInvalidToken
		}

		// Get embedding vector for this token
		embeddingStart := tokenID * m.config.HiddenSize
		for j := 0; j < m.config.HiddenSize; j++ {
			weight := m.weights.TokenEmbedding[embeddingStart+j]
			// Convert ternary value (-1, 0, +1) to float32
			switch weight {
			case -1:
				embeddings[i][j] = -1.0
			case 0:
				embeddings[i][j] = 0.0
			case 1:
				embeddings[i][j] = 1.0
			default:
				return nil, ErrInvalidWeightValue
			}
		}
	}

	return embeddings, nil
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

	// Perform inference
	outputTokens, err := m.Infer(tokens)
	if err != nil {
		loggers.Printf(loggers.Debug, "inference error: %v", err)
		return "", err
	}

	// Detokenize output
	output, err := m.tokenizer.Detokenize(outputTokens)
	if err != nil {
		loggers.Printf(loggers.Debug, "detokenization error: %v", err)
		return "", ErrDetokenization
	}

	return output, nil
}

// Close releases all resources associated with the model.
// After calling Close, the model cannot be used anymore.
func (m *Model) Close() {
	if m == nil {
		return
	}

	// Acquire mutex to prevent concurrent Close() calls
	m.closeMu.Lock()
	defer m.closeMu.Unlock()

	// Signal all goroutines to stop
	if m.done != nil {
		select {
		case <-m.done:
			// Channel already closed
		default:
			close(m.done)
		}
	}

	// Clear weights
	if m.weights != nil {
		// Clear token embeddings
		m.weights.TokenEmbedding = nil

		// Clear transformer blocks
		for _, block := range m.weights.Blocks {
			if block != nil {
				block.QKVProj = nil
				block.OutProj = nil
				block.FFNUp = nil
				block.FFNDown = nil
				block.AttnNorm = nil
				block.FFNNorm = nil
			}
		}
		m.weights.Blocks = nil
		m.weights.FinalNorm = nil
		m.weights = nil
	}

	// Clear read buffer
	m.readBuf = nil

	// Clear tokenizer
	m.tokenizer = nil

	// Force GC
	runtime.GC()
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

// TransformerBlock represents a single transformer layer in the model.
// It contains all the parameters needed for attention and feed-forward operations.
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

// ModelWeights contains all the model's learnable parameters.
// All weights are stored in ternary format (-1, 0, 1) for efficiency.
type ModelWeights struct {
	// Token embeddings (shared with output layer)
	TokenEmbedding []int8 // Token embedding weights (ternary)
	Blocks         []*TransformerBlock
	FinalNorm      []int8 // Final normalization weights (ternary)
}

// convertInt8ToFloat32 converts a slice of int8 values to float32.
// This is used internally for converting ternary weights to floating point
// during computation.
func convertInt8ToFloat32(values []int8) []float32 {
	result := make([]float32, len(values))
	for i, v := range values {
		result[i] = float32(v)
	}
	return result
}
