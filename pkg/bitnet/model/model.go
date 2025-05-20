package model

import (
	"encoding/binary"
	"errors"
	"io"
	"io/fs"
)

// Static errors
var (
	ErrInvalidWeightsFile      = errors.New("bitnet: invalid weights file format")
	ErrUnsupportedVersion      = errors.New("bitnet: unsupported weights file version")
	ErrInferenceNotImplemented = errors.New("bitnet: inference not implemented yet")
	ErrWeightsFileOpen         = errors.New("bitnet: failed to open weights file")
	ErrWeightsFileRead         = errors.New("bitnet: failed to read weights file")
)

// Model represents the BitNet b1.58-2B-4T model structure
// Implementation details will be covered in issue #173
type Model struct {
	config  *Config
	fs      fs.FS
	done    chan struct{}
	weights *ModelWeights
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
		config: config,
		fs:     fs,
		done:   make(chan struct{}),
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

	// Initialize weights structure
	m.weights = &ModelWeights{
		Blocks: make([]*TransformerBlock, m.config.NumLayers),
	}

	// Read token embeddings (vocab_size × hidden_size)
	embeddingSize := m.config.VocabSize * m.config.HiddenSize
	m.weights.TokenEmbedding = make([]int8, embeddingSize)
	if err := readTernaryWeights(file, m.weights.TokenEmbedding); err != nil {
		return err
	}

	// Read transformer blocks
	for i := 0; i < m.config.NumLayers; i++ {
		block := &TransformerBlock{}

		// QKV projection (hidden_size × 3*hidden_size)
		qkvSize := m.config.HiddenSize * 3 * m.config.HiddenSize
		block.QKVProj = make([]int8, qkvSize)
		if err := readTernaryWeights(file, block.QKVProj); err != nil {
			return err
		}

		// Output projection (hidden_size × hidden_size)
		outSize := m.config.HiddenSize * m.config.HiddenSize
		block.OutProj = make([]int8, outSize)
		if err := readTernaryWeights(file, block.OutProj); err != nil {
			return err
		}

		// FFN up projection (hidden_size × intermediate_size)
		ffnUpSize := m.config.HiddenSize * m.config.IntermediateSize
		block.FFNUp = make([]int8, ffnUpSize)
		if err := readTernaryWeights(file, block.FFNUp); err != nil {
			return err
		}

		// FFN down projection (intermediate_size × hidden_size)
		ffnDownSize := m.config.IntermediateSize * m.config.HiddenSize
		block.FFNDown = make([]int8, ffnDownSize)
		if err := readTernaryWeights(file, block.FFNDown); err != nil {
			return err
		}

		// Normalization weights
		block.AttnNorm = make([]float32, m.config.HiddenSize)
		if err := binary.Read(file, binary.LittleEndian, block.AttnNorm); err != nil {
			return ErrWeightsFileRead
		}

		block.FFNNorm = make([]float32, m.config.HiddenSize)
		if err := binary.Read(file, binary.LittleEndian, block.FFNNorm); err != nil {
			return ErrWeightsFileRead
		}

		m.weights.Blocks[i] = block
	}

	// Read final normalization
	m.weights.FinalNorm = make([]float32, m.config.HiddenSize)
	if err := binary.Read(file, binary.LittleEndian, m.weights.FinalNorm); err != nil {
		return ErrWeightsFileRead
	}

	return nil
}

// readTernaryWeights reads and unpacks ternary weights from the file
// Each byte contains 4 ternary values (-1, 0, +1) packed as 2 bits each
func readTernaryWeights(file io.Reader, weights []int8) error {
	// Calculate number of bytes needed (4 values per byte)
	numBytes := (len(weights) + 3) / 4
	buf := make([]byte, numBytes)

	// Read packed weights
	if _, err := file.Read(buf); err != nil {
		return ErrWeightsFileRead
	}

	// Unpack ternary values
	for i := 0; i < len(weights); i++ {
		byteIndex := i / 4
		bitOffset := (i % 4) * 2
		packed := (buf[byteIndex] >> bitOffset) & 0x03

		// Convert 2-bit value to ternary
		switch packed {
		case 0:
			weights[i] = -1
		case 1:
			weights[i] = 0
		case 2:
			weights[i] = 1
		default:
			return ErrInvalidWeightsFile
		}
	}

	return nil
}

// Infer performs inference on the input text
// Implementation will be completed in issue #173
func (m *Model) Infer(input string) (string, error) {
	// Create a channel to receive the result
	resultChan := make(chan string, 1)
	errChan := make(chan error, 1)

	// Run inference in a goroutine
	go func() {
		select {
		case <-m.done:
			return
		default:
			// TODO: Implement inference logic
			// This will be implemented in issue #190 (Token Decoding (Inference Loop))
			resultChan <- ""
			errChan <- ErrInferenceNotImplemented
		}
	}()

	// Wait for result or error
	select {
	case result := <-resultChan:
		return result, <-errChan
	case err := <-errChan:
		return "", err
	}
}

// Close stops all goroutines and cleans up resources
func (m *Model) Close() {
	close(m.done)
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
