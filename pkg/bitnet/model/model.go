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
	"math"
	"runtime"
	"sync"

	bitnetmath "github.com/hyperifyio/gnd/pkg/bitnet/internal/math"
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
	ErrAttentionGamma          = errors.New("bitnet: failed to set attention gamma")
	ErrFFNForward              = errors.New("bitnet: FFN forward pass failed")
	ErrFinalNormGamma          = errors.New("bitnet: failed to set final norm gamma")
	ErrFinalNormForward        = errors.New("bitnet: final norm forward pass failed")
	ErrFSNotSet                = errors.New("bitnet: fs not set")
	ErrPathEmpty               = errors.New("bitnet: path is empty")
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
	forwardMu sync.Mutex // Mutex to protect forward() operations

	// Reusable sublayers
	attnSublayers []*bitnetmath.AttentionSublayer
	ffnSublayers  []*bitnetmath.FFNSublayer
	finalNorm     *bitnetmath.LayerNorm

	// Memory pools for frequently allocated objects
	tensorOps        *bitnetmath.TensorOps
	hiddenStatesPool sync.Pool
	logitsPool       sync.Pool
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

// NewModel creates a new BitNet model instance with the given configuration.
func NewModel(config *Config, fs fs.FS) *Model {
	if config == nil {
		config = NewConfig()
	}

	// Initialize memory pools
	tensorOps := bitnetmath.NewTensorOps(config.MaxSeqLength, config.HiddenSize)

	hiddenStatesPool := sync.Pool{
		New: func() interface{} {
			return make([][]float32, config.MaxSeqLength)
		},
	}

	logitsPool := sync.Pool{
		New: func() interface{} {
			return make([]float32, config.VocabSize)
		},
	}

	return &Model{
		config:           config,
		fs:               fs,
		done:             make(chan struct{}),
		tensorOps:        tensorOps,
		hiddenStatesPool: hiddenStatesPool,
		logitsPool:       logitsPool,
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

	// Initialize reusable sublayers
	m.attnSublayers = make([]*bitnetmath.AttentionSublayer, m.config.NumLayers)
	m.ffnSublayers = make([]*bitnetmath.FFNSublayer, m.config.NumLayers)
	m.finalNorm = bitnetmath.NewLayerNorm(m.config.HiddenSize)

	// Create and initialize attention sublayers
	for i := 0; i < m.config.NumLayers; i++ {
		attn, err := bitnetmath.NewAttentionSublayer(m.config.HiddenSize, m.config.NumHeads, m.config.NumKVHeads)
		if err != nil {
			return ErrAttentionSublayer
		}
		m.attnSublayers[i] = attn

		// Set attention weights
		if err := m.setAttentionWeights(attn, m.weights.Blocks[i]); err != nil {
			return err
		}
	}

	// Create and initialize FFN sublayers
	for i := 0; i < m.config.NumLayers; i++ {
		ffn := bitnetmath.NewFFNSublayer(m.config.HiddenSize, m.config.IntermediateSize)
		m.ffnSublayers[i] = ffn

		// Set FFN weights
		if err := m.setFFNWeights(ffn, m.weights.Blocks[i]); err != nil {
			return err
		}
	}

	// Set final norm weights
	if err := m.setFinalNormWeights(m.finalNorm); err != nil {
		return err
	}

	return nil
}

// Infer performs inference on the input tokens and returns the predicted tokens.
// It implements a generation loop that continues until an end-of-sequence token
// is produced or the maximum sequence length is reached.
func (m *Model) Infer(tokens []int) ([]int, error) {
	if m.weights == nil {
		return nil, ErrWeightsNotLoaded
	}

	if m.tokenizer == nil {
		return nil, ErrTokenizerNotLoaded
	}

	// Check sequence length
	if len(tokens) > m.config.MaxSeqLength {
		return nil, ErrSequenceTooLong
	}

	// Initialize output sequence with input tokens
	outputTokens := make([]int, len(tokens))
	copy(outputTokens, tokens)

	// Generation loop
	for len(outputTokens) < m.config.MaxSeqLength {
		// Get logits from model forward pass
		logits, err := m.forward(outputTokens)
		if err != nil {
			return nil, err
		}

		// Apply softmax to get probability distribution
		probs := softmax(logits)

		// Greedy decoding: select token with highest probability
		nextToken := argmax(probs)

		// Check for end-of-sequence token
		if nextToken == m.tokenizer.SpecialTokens["</s>"] {
			break
		}

		// Append predicted token to output sequence
		outputTokens = append(outputTokens, nextToken)
	}

	return outputTokens, nil
}

// forward performs a single forward pass through the model and returns the logits
func (m *Model) forward(tokens []int) ([]float32, error) {
	// Get embeddings for tokens
	hiddenStates, err := m.embedTokens(tokens)
	if err != nil {
		return nil, err
	}

	// Reshape and copy hidden states to tensor
	hiddenStatesTensor := m.tensorOps.ReshapeAndCopy(hiddenStates, 1, len(tokens), m.config.HiddenSize)
	if hiddenStatesTensor == nil {
		return nil, fmt.Errorf("failed to create hidden states tensor")
	}
	fmt.Printf("[DEBUG] hiddenStatesTensor created with shape: %v\n", hiddenStatesTensor.Shape())

	// Keep track of tensors to close
	var tensorsToClose []*tensor.Tensor
	defer func() {
		for _, t := range tensorsToClose {
			if t != nil {
				fmt.Printf("[DEBUG] closing tensor with shape: %v\n", t.Shape())
				t.Close()
			}
		}
	}()

	currentTensor := hiddenStatesTensor

	// Process through transformer blocks
	for i := 0; i < m.config.NumLayers; i++ {
		fmt.Printf("[DEBUG] Processing transformer block %d\n", i)
		nextTensor, err := m.attnSublayers[i].Forward(currentTensor)
		if err != nil {
			return nil, fmt.Errorf("attention forward pass failed: %w", err)
		}
		if currentTensor != hiddenStatesTensor {
			tensorsToClose = append(tensorsToClose, currentTensor)
		}
		currentTensor = nextTensor
		fmt.Printf("[DEBUG] After attention, currentTensor shape: %v\n", currentTensor.Shape())

		nextTensor, err = m.ffnSublayers[i].Forward(currentTensor)
		if err != nil {
			return nil, fmt.Errorf("ffn forward pass failed: %w", err)
		}
		if currentTensor != hiddenStatesTensor {
			tensorsToClose = append(tensorsToClose, currentTensor)
		}
		currentTensor = nextTensor
		fmt.Printf("[DEBUG] After FFN, currentTensor shape: %v\n", currentTensor.Shape())
	}

	nextTensor, err := m.finalNorm.Forward(currentTensor)
	if err != nil {
		return nil, fmt.Errorf("final norm forward pass failed: %w", err)
	}
	if currentTensor != hiddenStatesTensor {
		tensorsToClose = append(tensorsToClose, currentTensor)
	}
	currentTensor = nextTensor
	fmt.Printf("[DEBUG] After final norm, currentTensor shape: %v\n", currentTensor.Shape())

	// Get logits from pool
	logits := m.logitsPool.Get().([]float32)
	defer m.logitsPool.Put(logits)

	// Get last hidden state and project to vocabulary size
	lastHiddenState := m.tensorOps.GetLastHiddenState(currentTensor, len(tokens), m.config.HiddenSize)
	if lastHiddenState == nil {
		return nil, fmt.Errorf("failed to get last hidden state")
	}
	copy(logits, lastHiddenState)

	// Create a copy of logits to return
	result := make([]float32, len(logits))
	copy(result, logits)

	// Close hiddenStatesTensor after all operations are complete
	if hiddenStatesTensor != nil {
		fmt.Printf("[DEBUG] closing hiddenStatesTensor with shape: %v\n", hiddenStatesTensor.Shape())
		hiddenStatesTensor.Close()
	}

	return m.projectToVocab(result), nil
}

// softmax applies the softmax function to the input logits
func softmax(logits []float32) []float32 {
	// Find maximum value for numerical stability
	maxVal := logits[0]
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp and sum
	expSum := float32(0)
	expVals := make([]float32, len(logits))
	for i, v := range logits {
		expVals[i] = float32(math.Exp(float64(v - maxVal)))
		expSum += expVals[i]
	}

	// Normalize to get probabilities
	probs := make([]float32, len(logits))
	for i, v := range expVals {
		probs[i] = v / expSum
	}

	return probs
}

// argmax returns the index of the maximum value in the slice
func argmax(values []float32) int {
	maxIdx := 0
	maxVal := values[0]
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// projectToVocab projects the hidden state to vocabulary size
func (m *Model) projectToVocab(hiddenState []float32) []float32 {
	logits := make([]float32, m.config.VocabSize)
	for i := 0; i < m.config.VocabSize; i++ {
		sum := float32(0)
		for j := 0; j < m.config.HiddenSize; j++ {
			sum += hiddenState[j] * float32(m.weights.TokenEmbedding[i*m.config.HiddenSize+j])
		}
		logits[i] = sum
	}
	return logits
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

	// Close all sublayers
	for i := 0; i < m.config.NumLayers; i++ {
		if m.attnSublayers != nil && i < len(m.attnSublayers) && m.attnSublayers[i] != nil {
			m.attnSublayers[i].Close()
		}
		if m.ffnSublayers != nil && i < len(m.ffnSublayers) && m.ffnSublayers[i] != nil {
			m.ffnSublayers[i].Close()
		}
	}
	m.attnSublayers = nil
	m.ffnSublayers = nil

	if m.finalNorm != nil {
		m.finalNorm.Close()
		m.finalNorm = nil
	}

	// Close tensor operations
	if m.tensorOps != nil {
		m.tensorOps.Close()
		m.tensorOps = nil
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

// setAttentionWeights sets the attention weights for a transformer block
func (m *Model) setAttentionWeights(attn *bitnetmath.AttentionSublayer, block *TransformerBlock) error {
	// Convert weights to tensors
	h := m.config.HiddenSize
	qTensor := tensor.NewTensor(h, h)
	defer qTensor.Close()
	kTensor := tensor.NewTensor(h, h)
	defer kTensor.Close()
	vTensor := tensor.NewTensor(h, h)
	defer vTensor.Close()
	outTensor := tensor.NewTensor(h, h)
	defer outTensor.Close()

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
		return ErrAttentionWeights
	}

	// Convert attention norm to float32 and create tensor
	attnGammaTensor := tensor.NewTensor(h)
	for i := 0; i < h; i++ {
		attnGammaTensor.Set(block.AttnNorm[i], i)
	}
	if err := attn.SetGamma(attnGammaTensor); err != nil {
		return ErrAttentionGamma
	}

	return nil
}

// setFFNWeights sets the FFN weights for a transformer block
func (m *Model) setFFNWeights(ffn *bitnetmath.FFNSublayer, block *TransformerBlock) error {
	// Convert FFN weights to tensors
	ffnUpTensor := tensor.NewTensor(m.config.IntermediateSize, m.config.HiddenSize)
	defer ffnUpTensor.Close()
	ffnDownTensor := tensor.NewTensor(m.config.HiddenSize, m.config.IntermediateSize)
	defer ffnDownTensor.Close()

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

	return nil
}

// setFinalNormWeights sets the final normalization weights
func (m *Model) setFinalNormWeights(norm *bitnetmath.LayerNorm) error {
	// Convert final norm weights to tensor
	finalNormTensor := tensor.NewTensor(m.config.HiddenSize)
	defer finalNormTensor.Close()
	for i := 0; i < m.config.HiddenSize; i++ {
		finalNormTensor.Set(m.weights.FinalNorm[i], i)
	}

	// Set final norm gamma
	finalNormGammaTensor := tensor.NewTensor(m.config.HiddenSize)
	finalNormGammaData := convertInt8ToFloat32(finalNormTensor.Data())
	for i := 0; i < m.config.HiddenSize; i++ {
		finalNormGammaTensor.Set(int8(finalNormGammaData[i]), i)
	}
	if err := norm.SetGamma(finalNormGammaTensor); err != nil {
		return ErrFinalNormGamma
	}

	return nil
}

// InitTokenizer initializes the tokenizer with the given path
func (m *Model) InitTokenizer(path string) error {
	if m.fs == nil {
		loggers.Printf(loggers.Debug, "filesystem not set")
		return ErrFSNotSet
	}
	if path == "" {
		loggers.Printf(loggers.Debug, "path is empty")
		return ErrPathEmpty
	}

	tokenizer, err := model.NewTokenizer(m.fs, path)
	if err != nil {
		loggers.Printf(loggers.Debug, "failed to initialize tokenizer: %v", err)
		return ErrTokenizerInit
	}

	m.tokenizer = tokenizer
	return nil
}
