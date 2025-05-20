package model

import (
	"encoding/binary"
	"errors"
	"io"
	"io/fs"
	"sync"
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
)

// Model represents the BitNet b1.58-2B-4T model structure
type Model struct {
	config  *Config
	fs      fs.FS
	done    chan struct{}
	weights *ModelWeights

	// Reusable buffers
	readBuf    []byte
	resultChan chan string
	errChan    chan error
	mu         sync.Mutex
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

// NewModel creates a new BitNet model instance
func NewModel(config *Config, fs fs.FS) *Model {
	if config == nil {
		config = NewConfig()
	}
	return &Model{
		config:     config,
		fs:         fs,
		done:       make(chan struct{}),
		resultChan: make(chan string, 1),
		errChan:    make(chan error, 1),
	}
}

// LoadWeights loads the model weights from the embedded filesystem
func (m *Model) LoadWeights(path string) error {
	file, err := m.fs.Open(path)
	if err != nil {
		return ErrWeightsFileOpen
	}
	defer file.Close()

	// Read and validate magic number
	var magic uint32
	if err := binary.Read(file, binary.LittleEndian, &magic); err != nil {
		return ErrWeightsFileRead
	}
	if magic != 0x424E4554 { // "BNET" in hex
		return ErrInvalidWeightsFile
	}

	// Read version
	var version uint32
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return ErrWeightsFileRead
	}
	if version != 1 {
		return ErrUnsupportedVersion
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
		FinalNorm:      make([]float32, m.config.HiddenSize),
	}

	// Pre-allocate all transformer blocks
	for i := 0; i < m.config.NumLayers; i++ {
		m.weights.Blocks[i] = &TransformerBlock{
			QKVProj:  make([]int8, qkvSize),
			OutProj:  make([]int8, outSize),
			FFNUp:    make([]int8, ffnUpSize),
			FFNDown:  make([]int8, ffnDownSize),
			AttnNorm: make([]float32, m.config.HiddenSize),
			FFNNorm:  make([]float32, m.config.HiddenSize),
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
		if err := binary.Read(file, binary.LittleEndian, block.AttnNorm); err != nil {
			return ErrWeightsFileRead
		}
		if err := binary.Read(file, binary.LittleEndian, block.FFNNorm); err != nil {
			return ErrWeightsFileRead
		}
	}

	// Read final normalization
	if err := binary.Read(file, binary.LittleEndian, m.weights.FinalNorm); err != nil {
		return ErrWeightsFileRead
	}

	return nil
}

// readTernaryWeights reads and unpacks ternary weights from the file
// Each byte contains 4 ternary values (-1, 0, +1) packed as 2 bits each
func (m *Model) readTernaryWeights(file io.Reader, weights []int8) error {
	if len(weights) == 0 {
		return nil
	}
	// Calculate number of bytes needed (4 values per byte)
	numBytes := (len(weights) + 3) / 4

	// Get or create read buffer
	m.mu.Lock()
	if m.readBuf == nil || cap(m.readBuf) < numBytes {
		m.readBuf = make([]byte, numBytes)
	} else {
		m.readBuf = m.readBuf[:numBytes]
	}
	m.mu.Unlock()

	// Read packed weights
	n, err := file.Read(m.readBuf)
	if err != nil && err != io.EOF {
		return ErrWeightsFileRead
	}
	if n == 0 && numBytes > 0 {
		return ErrWeightsFileRead
	}
	if n < numBytes {
		// If we have enough bytes for the weights, allow partial read
		for i := n * 4; i < len(weights); i++ {
			weights[i] = 0 // fill remaining with 0
		}
	}

	// Unpack ternary values
	for i := 0; i < len(weights); i++ {
		byteIndex := i / 4
		if byteIndex >= n {
			weights[i] = 0
			continue
		}
		bitOffset := (i % 4) * 2
		packed := (m.readBuf[byteIndex] >> bitOffset) & 0x03

		// Convert 2-bit value to ternary
		switch packed {
		case 0, 3:
			weights[i] = -1
		case 1:
			weights[i] = 0
		case 2:
			weights[i] = 1
		}
	}

	return nil
}

// Infer performs inference on the input text
func (m *Model) Infer(input string) (string, error) {
	if m.weights == nil {
		return "", ErrWeightsNotLoaded
	}

	// Create a channel to receive the result
	resultChan := make(chan string, 1)
	errChan := make(chan error, 1)

	// Run inference in a goroutine
	go func() {
		select {
		case <-m.done:
			return
		default:
			// Tokenize input
			tokens, err := m.tokenize(input)
			if err != nil {
				errChan <- err
				return
			}

			// Run transformer blocks
			hidden := make([]float32, m.config.HiddenSize)
			for i := 0; i < len(tokens); i++ {
				// Get token embedding
				tokenIdx := tokens[i]
				if tokenIdx >= m.config.VocabSize {
					errChan <- ErrInvalidToken
					return
				}
				embeddingStart := tokenIdx * m.config.HiddenSize
				for j := 0; j < m.config.HiddenSize; j++ {
					hidden[j] = float32(m.weights.TokenEmbedding[embeddingStart+j])
				}

				// Run through transformer blocks
				for _, block := range m.weights.Blocks {
					// Self-attention
					attnOut := m.selfAttention(hidden, block)
					// Add & norm
					for j := 0; j < m.config.HiddenSize; j++ {
						hidden[j] = (hidden[j] + attnOut[j]) * block.AttnNorm[j]
					}

					// FFN
					ffnOut := m.feedForward(hidden, block)
					// Add & norm
					for j := 0; j < m.config.HiddenSize; j++ {
						hidden[j] = (hidden[j] + ffnOut[j]) * block.FFNNorm[j]
					}
				}

				// Final normalization
				for j := 0; j < m.config.HiddenSize; j++ {
					hidden[j] *= m.weights.FinalNorm[j]
				}
			}

			// Generate output tokens
			output := m.generateOutput(hidden)
			resultChan <- output
		}
	}()

	// Wait for result or error
	select {
	case result := <-resultChan:
		return result, nil
	case err := <-errChan:
		return "", err
	}
}

// tokenize converts input text to token IDs
func (m *Model) tokenize(input string) ([]int, error) {
	// TODO: Implement proper tokenization
	// For now, return a simple character-based tokenization
	tokens := make([]int, len(input))
	for i, c := range input {
		if int(c) >= m.config.VocabSize {
			return nil, ErrInvalidToken
		}
		tokens[i] = int(c)
	}
	return tokens, nil
}

// selfAttention performs self-attention computation
func (m *Model) selfAttention(hidden []float32, block *TransformerBlock) []float32 {
	// TODO: Implement proper self-attention
	// For now, return a simple projection
	output := make([]float32, m.config.HiddenSize)
	for i := 0; i < m.config.HiddenSize; i++ {
		for j := 0; j < m.config.HiddenSize; j++ {
			output[i] += float32(block.QKVProj[i*m.config.HiddenSize+j]) * hidden[j]
		}
	}
	return output
}

// feedForward performs feed-forward network computation
func (m *Model) feedForward(hidden []float32, block *TransformerBlock) []float32 {
	// First projection: hidden_size -> intermediate_size
	intermediate := make([]float32, m.config.IntermediateSize)
	for i := 0; i < m.config.IntermediateSize; i++ {
		for j := 0; j < m.config.HiddenSize; j++ {
			intermediate[i] += float32(block.FFNUp[i*m.config.HiddenSize+j]) * hidden[j]
		}
	}

	// Second projection: intermediate_size -> hidden_size
	output := make([]float32, m.config.HiddenSize)
	for i := 0; i < m.config.HiddenSize; i++ {
		for j := 0; j < m.config.IntermediateSize; j++ {
			output[i] += float32(block.FFNDown[i*m.config.IntermediateSize+j]) * intermediate[j]
		}
	}

	return output
}

// generateOutput converts hidden state to output text
func (m *Model) generateOutput(hidden []float32) string {
	// TODO: Implement proper output generation
	// For now, return a simple character-based output
	var output string
	for i := 0; i < len(hidden); i++ {
		if hidden[i] > 0 {
			output += string(rune(i % m.config.VocabSize))
		}
	}
	return output
}

// Close stops all goroutines and cleans up resources
func (m *Model) Close() {
	select {
	case <-m.done:
		// Channel already closed
		return
	default:
		close(m.done)
	}
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
	AttnNorm []float32 // Attention normalization weights
	FFNNorm  []float32 // FFN normalization weights
}

// ModelWeights holds all the model's parameters
type ModelWeights struct {
	// Token embeddings (shared with output layer)
	TokenEmbedding []int8 // Token embedding weights (ternary)

	// Transformer blocks
	Blocks []*TransformerBlock

	// Final normalization
	FinalNorm []float32
}
