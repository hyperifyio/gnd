package model

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math/rand"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"

	bitnetmath "github.com/hyperifyio/gnd/pkg/bitnet/internal/math"
	"github.com/hyperifyio/gnd/pkg/bitnet/tensor"
)

// testFS implements fs.FS for testing
type testFS struct {
	files map[string][]byte
}

func (t *testFS) Open(name string) (fs.File, error) {
	if data, ok := t.files[name]; ok {
		return &testFile{data: data}, nil
	}
	return nil, fs.ErrNotExist
}

// testFile implements fs.File for testing
type testFile struct {
	data []byte
	pos  int64
}

func (t *testFile) Read(p []byte) (n int, err error) {
	if t.pos >= int64(len(t.data)) {
		return 0, io.EOF
	}
	n = copy(p, t.data[t.pos:])
	t.pos += int64(n)
	return n, nil
}

func (t *testFile) Close() error {
	return nil
}

func (t *testFile) Stat() (fs.FileInfo, error) {
	return &testFileInfo{size: int64(len(t.data))}, nil
}

// testFileInfo implements fs.FileInfo for testing
type testFileInfo struct {
	size int64
}

func (t *testFileInfo) Name() string       { return "" }
func (t *testFileInfo) Size() int64        { return t.size }
func (t *testFileInfo) Mode() fs.FileMode  { return 0 }
func (t *testFileInfo) ModTime() time.Time { return time.Time{} }
func (t *testFileInfo) IsDir() bool        { return false }
func (t *testFileInfo) Sys() interface{}   { return nil }

var testDataFS = &testFS{
	files: map[string][]byte{
		"tokenizer/vocab.json": []byte(`{
			"hello": 1,
			"world": 2,
			"[UNK]": 3,
			"▁": 4
		}`),
		"tokenizer/merges.txt": []byte("he hello\nwo world\n"),
		"tokenizer/special_tokens.json": []byte(`{
			"[UNK]": 3,
			"[PAD]": 5
		}`),
	},
}

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
	model := NewModel(nil, testDataFS)
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
	model = NewModel(customConfig, testDataFS)
	if model == nil {
		t.Fatal("NewModel returned nil")
	}
	if model.config != customConfig {
		t.Error("model.config does not match custom config")
	}

	// Test tokenizer initialization
	if model.tokenizer != nil {
		t.Error("expected tokenizer to be nil with test filesystem")
	}
}

func TestReadTernaryWeights(t *testing.T) {
	tests := []struct {
		name    string
		input   []byte
		weights []int8
		want    []int8
		wantErr error
	}{
		{
			name:    "empty input",
			input:   []byte{},
			weights: make([]int8, 0),
			want:    []int8{},
			wantErr: nil,
		},
		{
			name:    "single byte with all values",
			input:   []byte{0x1A}, // 00011010
			weights: make([]int8, 4),
			want:    []int8{1, 1, 0, -1},
			wantErr: nil,
		},
		{
			name:    "multiple bytes",
			input:   []byte{0x1A, 0x2A}, // 00011010, 00101010
			weights: make([]int8, 8),
			want:    []int8{1, 1, 0, -1, 1, 1, 1, -1},
			wantErr: nil,
		},
		{
			name:    "incomplete byte",
			input:   []byte{0x1A},
			weights: make([]int8, 5), // Request 5 weights but only 4 available
			want:    nil,
			wantErr: ErrWeightsFileRead,
		},
		{
			name:    "nil reader",
			input:   nil,
			weights: make([]int8, 4),
			want:    nil,
			wantErr: ErrWeightsFileRead,
		},
		{
			name:    "nil weights slice",
			input:   []byte{0x1A},
			weights: nil,
			want:    nil,
			wantErr: ErrWeightsFileRead,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := &Model{
				config: NewConfig(),
			}

			var reader io.Reader
			if tt.input != nil {
				reader = bytes.NewReader(tt.input)
			}

			err := model.readTernaryWeights(reader, tt.weights)
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("readTernaryWeights() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err == nil && !reflect.DeepEqual(tt.weights, tt.want) {
				t.Errorf("readTernaryWeights() = %v, want %v", tt.weights, tt.want)
			}
		})
	}
}

func TestReadTernaryWeightsEdgeCases(t *testing.T) {
	tests := []struct {
		name    string
		input   []byte
		size    int
		want    []int8
		wantErr error
	}{
		{
			name:    "empty input",
			input:   []byte{},
			size:    0,
			want:    []int8{},
			wantErr: nil,
		},
		{
			name:    "single byte with all values",
			input:   []byte{0x1A}, // 00011010 -> [1, 1, 0, -1]
			size:    4,
			want:    []int8{1, 1, 0, -1},
			wantErr: nil,
		},
		{
			name:    "multiple bytes with mixed values",
			input:   []byte{0x1A, 0x2A}, // [1,1,0,-1,1,1,1,-1]
			size:    8,
			want:    []int8{1, 1, 0, -1, 1, 1, 1, -1},
			wantErr: nil,
		},
		{
			name:    "invalid weight value",
			input:   []byte{0x3A}, // 00111010 -> [3,1,0,-1] (3 is invalid)
			size:    4,
			want:    nil,
			wantErr: ErrInvalidWeightValue,
		},
		{
			name:    "incomplete byte",
			input:   []byte{0x1A},
			size:    5, // Request 5 weights but only 4 available
			want:    nil,
			wantErr: ErrWeightsFileRead,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := &Model{
				config: NewConfig(),
			}

			weights := make([]int8, tt.size)
			err := model.readTernaryWeights(bytes.NewReader(tt.input), weights)
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("readTernaryWeights() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err == nil && !reflect.DeepEqual(weights, tt.want) {
				t.Errorf("readTernaryWeights() = %v, want %v", weights, tt.want)
			}
		})
	}
}

// createValidWeights creates a valid weights file for testing
func createValidWeights() []byte {
	// Create header
	header := make([]byte, 8)
	binary.LittleEndian.PutUint32(header[0:4], 0x424E4554) // "BNET"
	binary.LittleEndian.PutUint32(header[4:8], 1)          // Version 1

	// Create token embeddings (vocab_size x hidden_size)
	tokenEmbeddings := make([]byte, 32000*4096) // Example sizes

	// Create transformer blocks
	blocks := make([]byte, 0)
	for i := 0; i < 12; i++ { // Example: 12 transformer blocks
		// QKV projection (hidden_size x 3*hidden_size)
		qkv := make([]byte, 4096*12288)
		// Output projection (hidden_size x hidden_size)
		out := make([]byte, 4096*4096)
		// Feed-forward weights (hidden_size x intermediate_size)
		ff1 := make([]byte, 4096*16384)
		ff2 := make([]byte, 16384*4096)
		// Layer norms
		ln1 := make([]byte, 4096*2) // mean and variance
		ln2 := make([]byte, 4096*2)

		blocks = append(blocks, qkv...)
		blocks = append(blocks, out...)
		blocks = append(blocks, ff1...)
		blocks = append(blocks, ff2...)
		blocks = append(blocks, ln1...)
		blocks = append(blocks, ln2...)
	}

	// Final layer norm
	finalNorm := make([]byte, 4096*2)

	// Combine all parts
	weights := make([]byte, 0)
	weights = append(weights, header...)
	weights = append(weights, tokenEmbeddings...)
	weights = append(weights, blocks...)
	weights = append(weights, finalNorm...)

	return weights
}

func TestLoadWeights(t *testing.T) {
	// Create test filesystem with valid weights
	fs := &testFS{
		files: map[string][]byte{
			"weights.bin": createValidWeights(),
			// Minimal tokenizer files
			"tokenizer/vocab.json":          []byte(`{"<unk>":0,"▁":1}`),
			"tokenizer/merges.txt":          []byte(""),
			"tokenizer/special_tokens.json": []byte(`{"<unk>":0}`),
		},
	}

	tests := []struct {
		name    string
		path    string
		wantErr error
	}{
		{
			name:    "valid weights",
			path:    "weights.bin",
			wantErr: nil,
		},
		{
			name:    "file not found",
			path:    "nonexistent.bin",
			wantErr: ErrWeightsFileOpen,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := NewModel(nil, fs)
			err := model.LoadWeights(tt.path)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Errorf("LoadWeights() error = %v, wantErr %v", err, tt.wantErr)
				}
			} else if err != nil {
				t.Errorf("LoadWeights() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestLoadWeightsInvalidData(t *testing.T) {
	// Create test filesystem with invalid weights
	fs := &testFS{
		files: map[string][]byte{
			"invalid_weights.bin": []byte{
				// Invalid magic number
				0x00, 0x00, 0x00, 0x00,
				// Version 1
				0x01, 0x00, 0x00, 0x00,
			},
			"invalid_version.bin": []byte{
				// Valid magic number "BNET"
				0x42, 0x4E, 0x45, 0x54,
				// Invalid version 2
				0x02, 0x00, 0x00, 0x00,
			},
			"truncated_weights.bin": []byte{
				// Valid magic number "BNET"
				0x42, 0x4E, 0x45, 0x54,
				// Version 1
				0x01, 0x00, 0x00, 0x00,
				// Not enough data for weights, but at least 8 bytes header
			},
		},
	}

	tests := []struct {
		name    string
		path    string
		wantErr error
	}{
		{
			name:    "invalid magic number",
			path:    "invalid_weights.bin",
			wantErr: ErrInvalidWeightsFile,
		},
		{
			name:    "invalid version",
			path:    "invalid_version.bin",
			wantErr: ErrUnsupportedVersion,
		},
		{
			name:    "truncated weights",
			path:    "truncated_weights.bin",
			wantErr: ErrWeightsFileRead,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := NewModel(nil, fs)
			err := model.LoadWeights(tt.path)
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("LoadWeights() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestClose(t *testing.T) {
	model := NewModel(nil, testDataFS)
	if model == nil {
		t.Fatal("NewModel returned nil")
	}

	// Close should not panic
	model.Close()

	// Second close should not panic
	model.Close()
}

func BenchmarkModel_LoadWeights(b *testing.B) {
	// Create test filesystem with valid weights and tokenizer files
	fs := &testFS{
		files: map[string][]byte{
			"weights.bin":                   createValidWeights(),
			"tokenizer/vocab.json":          []byte(`{"<unk>":0,"▁":1}`),
			"tokenizer/merges.txt":          []byte(""),
			"tokenizer/special_tokens.json": []byte(`{"<unk>":0}`),
		},
	}

	model := NewModel(nil, fs)
	if model == nil {
		b.Fatal("NewModel returned nil")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := model.LoadWeights("weights.bin")
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkModel_ReadTernaryWeights(b *testing.B) {
	// Create test data with valid ternary values
	data := make([]byte, 1024)
	for i := range data {
		// Generate valid ternary values (0, 1, 2)
		data[i] = byte(i % 3)
	}

	model := &Model{
		config: NewConfig(),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weights := make([]int8, 4096)
		err := model.readTernaryWeights(bytes.NewReader(data), weights)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkModel_Infer(b *testing.B) {
	model := NewModel(nil, testDataFS)
	defer model.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := model.Infer([]int{0, 1, 2})
		if err != ErrInferenceNotImplemented {
			b.Fatal(err)
		}
	}
}

func TestEmbedTokens(t *testing.T) {
	// Create a test model with minimal configuration
	config := &Config{
		HiddenSize: 4,
		VocabSize:  3,
	}
	model := NewModel(config, nil)

	// Create test weights with known ternary values
	model.weights = &ModelWeights{
		TokenEmbedding: []int8{
			// Token 0 embeddings
			1, -1, 0, 1,
			// Token 1 embeddings
			-1, 1, 0, -1,
			// Token 2 embeddings
			0, 0, 1, 1,
		},
	}

	tests := []struct {
		name    string
		tokens  []int
		want    [][]float32
		wantErr error
	}{
		{
			name:   "valid tokens",
			tokens: []int{0, 1, 2},
			want: [][]float32{
				{1.0, -1.0, 0.0, 1.0},  // Token 0
				{-1.0, 1.0, 0.0, -1.0}, // Token 1
				{0.0, 0.0, 1.0, 1.0},   // Token 2
			},
			wantErr: nil,
		},
		{
			name:    "invalid token",
			tokens:  []int{0, 3, 2},
			want:    nil,
			wantErr: ErrInvalidToken,
		},
		{
			name:    "negative token",
			tokens:  []int{0, -1, 2},
			want:    nil,
			wantErr: ErrInvalidToken,
		},
		{
			name:    "nil weights",
			tokens:  []int{0, 1, 2},
			want:    nil,
			wantErr: ErrWeightsNotLoaded,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// For the nil weights test
			if tt.name == "nil weights" {
				model.weights = nil
			} else {
				model.weights = &ModelWeights{
					TokenEmbedding: []int8{
						// Token 0 embeddings
						1, -1, 0, 1,
						// Token 1 embeddings
						-1, 1, 0, -1,
						// Token 2 embeddings
						0, 0, 1, 1,
					},
				}
			}

			got, err := model.embedTokens(tt.tokens)
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("embedTokens() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("embedTokens() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEmbedTokensMemoryUsage(t *testing.T) {
	// Skip in short mode as this is a memory-intensive test
	if testing.Short() {
		t.Skip("skipping memory usage test in short mode")
	}

	// Create a test model with large vocabulary
	config := &Config{
		HiddenSize: 2048,
		VocabSize:  32000,
	}
	model := NewModel(config, nil)

	// Create test weights with random ternary values
	model.weights = &ModelWeights{
		TokenEmbedding: make([]int8, config.VocabSize*config.HiddenSize),
	}
	for i := range model.weights.TokenEmbedding {
		model.weights.TokenEmbedding[i] = int8(rand.Intn(3) - 1)
	}

	// Test different sequence lengths
	sequenceLengths := []int{16, 256, 1024, 4096}

	for _, seqLen := range sequenceLengths {
		t.Run(fmt.Sprintf("SequenceLength_%d", seqLen), func(t *testing.T) {
			// Generate test tokens
			tokens := make([]int, seqLen)
			for i := range tokens {
				tokens[i] = i % config.VocabSize
			}

			// Measure memory before
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			before := m.TotalAlloc

			// Run embedding
			hiddenStates, err := model.embedTokens(tokens)
			if err != nil {
				t.Fatal(err)
			}

			// Measure memory after
			runtime.ReadMemStats(&m)
			after := m.TotalAlloc

			// Calculate memory usage
			memoryUsed := after - before
			expectedMemory := uint64(seqLen * config.HiddenSize * 4) // float32 = 4 bytes

			// Allow for some overhead (20%)
			maxAllowedMemory := uint64(float64(expectedMemory) * 1.2)

			// Verify memory usage is within expected bounds
			if memoryUsed > maxAllowedMemory {
				t.Errorf("Memory usage too high: got %d bytes, want <= %d bytes",
					memoryUsed, maxAllowedMemory)
			}

			// Verify output dimensions
			if len(hiddenStates) != seqLen {
				t.Errorf("Wrong number of hidden states: got %d, want %d",
					len(hiddenStates), seqLen)
			}
			for i, state := range hiddenStates {
				if len(state) != config.HiddenSize {
					t.Errorf("Wrong hidden state size at index %d: got %d, want %d",
						i, len(state), config.HiddenSize)
				}
			}
		})
	}
}

func BenchmarkEmbedTokens(b *testing.B) {
	// Create a test model with large vocabulary
	config := &Config{
		HiddenSize: 2048,
		VocabSize:  32000,
	}
	model := NewModel(config, nil)

	// Create test weights with random ternary values
	model.weights = &ModelWeights{
		TokenEmbedding: make([]int8, config.VocabSize*config.HiddenSize),
	}
	for i := range model.weights.TokenEmbedding {
		// Generate random ternary values (-1, 0, 1)
		model.weights.TokenEmbedding[i] = int8(rand.Intn(3) - 1)
	}

	// Test cases with different sequence lengths
	benchmarks := []struct {
		name         string
		sequenceLen  int
		randomTokens bool
	}{
		{
			name:         "ShortSeq_FixedTokens",
			sequenceLen:  16,
			randomTokens: false,
		},
		{
			name:         "ShortSeq_RandomTokens",
			sequenceLen:  16,
			randomTokens: true,
		},
		{
			name:         "MediumSeq_FixedTokens",
			sequenceLen:  256,
			randomTokens: false,
		},
		{
			name:         "MediumSeq_RandomTokens",
			sequenceLen:  256,
			randomTokens: true,
		},
		{
			name:         "LongSeq_FixedTokens",
			sequenceLen:  1024,
			randomTokens: false,
		},
		{
			name:         "LongSeq_RandomTokens",
			sequenceLen:  1024,
			randomTokens: true,
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			// Generate test tokens
			tokens := make([]int, bm.sequenceLen)
			if bm.randomTokens {
				for i := range tokens {
					tokens[i] = rand.Intn(config.VocabSize)
				}
			} else {
				// Use fixed tokens for more consistent benchmarking
				for i := range tokens {
					tokens[i] = i % config.VocabSize
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := model.embedTokens(tokens)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestModel_Infer(t *testing.T) {
	// Create a test model with minimal configuration
	config := &Config{
		HiddenSize:       32,
		NumHeads:         4,
		NumKVHeads:       4,
		NumLayers:        1,
		VocabSize:        10,
		MaxSeqLength:     8,
		IntermediateSize: 16,
	}
	model := NewModel(config, testDataFS)

	// Create test weights
	model.weights = &ModelWeights{
		TokenEmbedding: make([]int8, config.VocabSize*config.HiddenSize),
		Blocks:         make([]*TransformerBlock, config.NumLayers),
		FinalNorm:      make([]int8, config.HiddenSize),
	}

	// Initialize token embeddings with test values
	for i := 0; i < config.VocabSize*config.HiddenSize; i++ {
		model.weights.TokenEmbedding[i] = int8(i%3 - 1) // -1, 0, or 1
	}

	// Initialize transformer block
	block := &TransformerBlock{
		// QKV projection: [3 * hidden_size * hidden_size] (Q, K, V concatenated)
		QKVProj: make([]int8, 3*config.HiddenSize*config.HiddenSize),
		// Output projection: [hidden_size, hidden_size]
		OutProj: make([]int8, config.HiddenSize*config.HiddenSize),
		// FFN up projection: [intermediate_size, hidden_size]
		FFNUp: make([]int8, config.IntermediateSize*config.HiddenSize),
		// FFN down projection: [hidden_size, intermediate_size]
		FFNDown: make([]int8, config.HiddenSize*config.IntermediateSize),
		// Layer norms: [hidden_size]
		AttnNorm: make([]int8, config.HiddenSize),
		FFNNorm:  make([]int8, config.HiddenSize),
	}

	// Initialize block weights with test values
	// QKV projection: [3 * hidden_size * hidden_size]
	// Each projection matrix is [hidden_size, hidden_size]
	for i := 0; i < config.HiddenSize*config.HiddenSize; i++ {
		// Q projection
		block.QKVProj[i] = int8(i%3 - 1)
		// K projection
		block.QKVProj[i+config.HiddenSize*config.HiddenSize] = int8(i%3 - 1)
		// V projection
		block.QKVProj[i+2*config.HiddenSize*config.HiddenSize] = int8(i%3 - 1)
	}

	// Output projection: [hidden_size, hidden_size]
	for i := range block.OutProj {
		block.OutProj[i] = int8(i%3 - 1)
	}

	// FFN up projection: [intermediate_size, hidden_size]
	for i := 0; i < config.IntermediateSize; i++ {
		for j := 0; j < config.HiddenSize; j++ {
			block.FFNUp[i*config.HiddenSize+j] = int8((i+j)%3 - 1)
		}
	}

	// FFN down projection: [hidden_size, intermediate_size]
	for i := 0; i < config.HiddenSize; i++ {
		for j := 0; j < config.IntermediateSize; j++ {
			block.FFNDown[i*config.IntermediateSize+j] = int8((i+j)%3 - 1)
		}
	}

	// Layer norms: [hidden_size]
	for i := range block.AttnNorm {
		block.AttnNorm[i] = int8(i%3 - 1)
	}
	for i := range block.FFNNorm {
		block.FFNNorm[i] = int8(i%3 - 1)
	}

	model.weights.Blocks[0] = block

	// Initialize final normalization weights
	for i := range model.weights.FinalNorm {
		model.weights.FinalNorm[i] = int8(i%3 - 1)
	}

	tests := []struct {
		name    string
		tokens  []int
		wantErr error
	}{
		{
			name:    "empty input",
			tokens:  []int{},
			wantErr: nil,
		},
		{
			name:    "single token",
			tokens:  []int{1},
			wantErr: nil,
		},
		{
			name:    "multiple tokens",
			tokens:  []int{1, 2, 3},
			wantErr: nil,
		},
		{
			name:    "invalid token",
			tokens:  []int{config.VocabSize + 1},
			wantErr: ErrInvalidToken,
		},
		{
			name:    "sequence too long",
			tokens:  make([]int, config.MaxSeqLength+1),
			wantErr: ErrSequenceTooLong,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := model.Infer(tt.tokens)
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("Infer() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestModel_TensorOperations(t *testing.T) {
	// Create a test tensor
	tensor := tensor.NewTensor(2, 3)

	// Test tensor creation
	if tensor == nil {
		t.Fatal("NewTensor returned nil")
	}

	// Test tensor shape
	shape := tensor.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Errorf("Tensor.Shape() = %v, want [2 3]", shape)
	}

	// Test tensor operations
	tests := []struct {
		name    string
		value   int8
		indices []int
		want    int8
	}{
		{
			name:    "set and get value",
			value:   1,
			indices: []int{0, 0},
			want:    1,
		},
		{
			name:    "clamp positive value",
			value:   2,
			indices: []int{0, 1},
			want:    1,
		},
		{
			name:    "clamp negative value",
			value:   -2,
			indices: []int{1, 0},
			want:    -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor.Set(tt.value, tt.indices...)
			got := tensor.Get(tt.indices...)
			if got != tt.want {
				t.Errorf("Tensor.Get() = %v, want %v", got, tt.want)
			}
		})
	}

	// Test tensor data
	data := tensor.Data()
	if len(data) != 6 {
		t.Errorf("Tensor.Data() length = %v, want %v", len(data), 6)
	}

	// Test tensor reshape
	reshaped := tensor.Reshape(3, 2)
	if reshaped == nil {
		t.Fatal("Reshape returned nil")
	}

	reshapedShape := reshaped.Shape()
	if len(reshapedShape) != 2 || reshapedShape[0] != 3 || reshapedShape[1] != 2 {
		t.Errorf("Reshaped tensor shape = %v, want [3 2]", reshapedShape)
	}

	// Test parallel operations
	var wg sync.WaitGroup
	wg.Add(6) // Add for each element in the 2x3 tensor
	tensor.ParallelForEach(func(indices []int, value int8) {
		defer wg.Done()
		if len(indices) != 2 {
			t.Errorf("ParallelForEach indices length = %v, want 2", len(indices))
		}
	})
	wg.Wait()

	// Test tensor cleanup
	tensor.Close()

	// Verify operations panic after close
	operations := []struct {
		name string
		fn   func()
	}{
		{
			name: "Get",
			fn:   func() { tensor.Get(0, 0) },
		},
		{
			name: "Set",
			fn:   func() { tensor.Set(1, 0, 0) },
		},
		{
			name: "Shape",
			fn:   func() { tensor.Shape() },
		},
		{
			name: "Data",
			fn:   func() { tensor.Data() },
		},
		{
			name: "ParallelForEach",
			fn:   func() { tensor.ParallelForEach(func(indices []int, value int8) {}) },
		},
		{
			name: "Reshape",
			fn:   func() { tensor.Reshape(3, 2) },
		},
	}

	for _, op := range operations {
		t.Run(op.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("%s did not panic after Close()", op.name)
				}
			}()
			op.fn()
		})
	}
}

func TestModelTensorOperations(t *testing.T) {
	// Create a test model with minimal configuration
	config := &Config{
		HiddenSize:       32,
		NumHeads:         4,
		NumKVHeads:       4,
		NumLayers:        1,
		VocabSize:        10,
		MaxSeqLength:     8,
		IntermediateSize: 16,
	}
	model := NewModel(config, testDataFS)

	// Create test weights with known values
	model.weights = &ModelWeights{
		TokenEmbedding: make([]int8, config.VocabSize*config.HiddenSize),
		Blocks:         make([]*TransformerBlock, config.NumLayers),
		FinalNorm:      make([]int8, config.HiddenSize),
	}

	// Initialize token embeddings with test values
	for i := 0; i < config.VocabSize*config.HiddenSize; i++ {
		model.weights.TokenEmbedding[i] = int8(i%3 - 1) // -1, 0, or 1
	}

	// Initialize transformer block
	block := &TransformerBlock{
		QKVProj:  make([]int8, 3*config.HiddenSize*config.HiddenSize),
		OutProj:  make([]int8, config.HiddenSize*config.HiddenSize),
		FFNUp:    make([]int8, config.IntermediateSize*config.HiddenSize),
		FFNDown:  make([]int8, config.HiddenSize*config.IntermediateSize),
		AttnNorm: make([]int8, config.HiddenSize),
		FFNNorm:  make([]int8, config.HiddenSize),
	}

	// Initialize block weights with test values
	for i := range block.QKVProj {
		block.QKVProj[i] = int8(i%3 - 1)
	}
	for i := range block.OutProj {
		block.OutProj[i] = int8(i%3 - 1)
	}
	for i := range block.FFNUp {
		block.FFNUp[i] = int8(i%3 - 1)
	}
	for i := range block.FFNDown {
		block.FFNDown[i] = int8(i%3 - 1)
	}
	for i := range block.AttnNorm {
		block.AttnNorm[i] = int8(i%3 - 1)
	}
	for i := range block.FFNNorm {
		block.FFNNorm[i] = int8(i%3 - 1)
	}

	model.weights.Blocks[0] = block

	// Initialize final normalization weights
	for i := range model.weights.FinalNorm {
		model.weights.FinalNorm[i] = int8(i%3 - 1)
	}

	tests := []struct {
		name    string
		tokens  []int
		wantErr error
	}{
		{
			name:    "single token inference",
			tokens:  []int{1},
			wantErr: nil,
		},
		{
			name:    "multiple tokens inference",
			tokens:  []int{1, 2, 3},
			wantErr: nil,
		},
		{
			name:    "invalid token dimensions",
			tokens:  []int{config.VocabSize + 1},
			wantErr: ErrInvalidToken,
		},
		{
			name:    "sequence too long",
			tokens:  make([]int, config.MaxSeqLength+1),
			wantErr: ErrSequenceTooLong,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Convert tokens to hidden states
			hiddenStates, err := model.embedTokens(tt.tokens)
			if err != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Errorf("embedTokens() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}

			// Verify hidden states dimensions
			if len(hiddenStates) != len(tt.tokens) {
				t.Errorf("hiddenStates length = %v, want %v", len(hiddenStates), len(tt.tokens))
			}
			for i, state := range hiddenStates {
				if len(state) != config.HiddenSize {
					t.Errorf("hiddenStates[%d] length = %v, want %v", i, len(state), config.HiddenSize)
				}
			}

			// Convert to tensor and verify shape
			hiddenStatesTensor := tensor.NewTensor(len(tt.tokens), config.HiddenSize)
			for i := 0; i < len(tt.tokens); i++ {
				for j := 0; j < config.HiddenSize; j++ {
					hiddenStatesTensor.Set(int8(hiddenStates[i][j]), i, j)
				}
			}

			shape := hiddenStatesTensor.Shape()
			if len(shape) != 2 || shape[0] != len(tt.tokens) || shape[1] != config.HiddenSize {
				t.Errorf("tensor shape = %v, want [%d %d]", shape, len(tt.tokens), config.HiddenSize)
			}
		})
	}
}

func TestModelAttentionMechanism(t *testing.T) {
	// Create a test model with minimal configuration
	config := &Config{
		HiddenSize:       32,
		NumHeads:         4,
		NumKVHeads:       4,
		NumLayers:        1,
		VocabSize:        10,
		MaxSeqLength:     8,
		IntermediateSize: 16,
	}
	model := NewModel(config, testDataFS)

	// Create test weights
	model.weights = &ModelWeights{
		TokenEmbedding: make([]int8, config.VocabSize*config.HiddenSize),
		Blocks:         make([]*TransformerBlock, config.NumLayers),
		FinalNorm:      make([]int8, config.HiddenSize),
	}

	// Initialize transformer block with test values
	block := &TransformerBlock{
		QKVProj:  make([]int8, 3*config.HiddenSize*config.HiddenSize),
		OutProj:  make([]int8, config.HiddenSize*config.HiddenSize),
		FFNUp:    make([]int8, config.IntermediateSize*config.HiddenSize),
		FFNDown:  make([]int8, config.HiddenSize*config.IntermediateSize),
		AttnNorm: make([]int8, config.HiddenSize),
		FFNNorm:  make([]int8, config.HiddenSize),
	}

	// Initialize QKV projection weights with test values
	// Each projection matrix is [hidden_size, hidden_size]
	h := config.HiddenSize
	for i := 0; i < h; i++ {
		for j := 0; j < h; j++ {
			// Q projection
			block.QKVProj[i*h+j] = int8((i+j)%3 - 1)
			// K projection
			block.QKVProj[h*h+i*h+j] = int8((i+j)%3 - 1)
			// V projection
			block.QKVProj[2*h*h+i*h+j] = int8((i+j)%3 - 1)
		}
	}

	// Initialize output projection weights
	for i := 0; i < h; i++ {
		for j := 0; j < h; j++ {
			block.OutProj[i*h+j] = int8((i+j)%3 - 1)
		}
	}

	// Initialize normalization weights
	for i := range block.AttnNorm {
		block.AttnNorm[i] = int8(i%3 - 1)
	}
	for i := range block.FFNNorm {
		block.FFNNorm[i] = int8(i%3 - 1)
	}

	model.weights.Blocks[0] = block

	tests := []struct {
		name    string
		tokens  []int
		wantErr error
	}{
		{
			name:    "single token attention",
			tokens:  []int{1},
			wantErr: nil,
		},
		{
			name:    "multiple tokens attention",
			tokens:  []int{1, 2, 3},
			wantErr: nil,
		},
		{
			name:    "invalid token dimensions",
			tokens:  []int{config.VocabSize + 1},
			wantErr: ErrInvalidToken,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Convert tokens to hidden states
			hiddenStates, err := model.embedTokens(tt.tokens)
			if err != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Errorf("embedTokens() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}

			// Convert hidden states to tensor
			hiddenStatesTensor := tensor.NewTensor(len(tt.tokens), config.HiddenSize)
			for i := 0; i < len(tt.tokens); i++ {
				for j := 0; j < config.HiddenSize; j++ {
					hiddenStatesTensor.Set(int8(hiddenStates[i][j]), i, j)
				}
			}

			// Create attention sublayer
			attn, err := bitnetmath.NewAttentionSublayer(config.HiddenSize, config.NumHeads, config.NumKVHeads)
			if err != nil {
				t.Fatalf("failed to create attention sublayer: %v", err)
			}

			// Convert weights to tensors
			qkvProj := block.QKVProj
			qTensor := tensor.NewTensor(h, h)
			kTensor := tensor.NewTensor(h, h)
			vTensor := tensor.NewTensor(h, h)

			// Copy weights into projection matrices
			for i := 0; i < h; i++ {
				for j := 0; j < h; j++ {
					// Q projection
					qTensor.Set(qkvProj[i*h+j], i, j)
					// K projection
					kTensor.Set(qkvProj[h*h+i*h+j], i, j)
					// V projection
					vTensor.Set(qkvProj[2*h*h+i*h+j], i, j)
				}
			}

			outTensor := tensor.NewTensor(h, h)
			for i := 0; i < h; i++ {
				for j := 0; j < h; j++ {
					outTensor.Set(block.OutProj[i*h+j], i, j)
				}
			}

			attnNormTensor := tensor.NewTensor(h)
			for i := 0; i < h; i++ {
				attnNormTensor.Set(block.AttnNorm[i], i)
			}

			// Set attention weights
			if err := attn.SetWeights(qTensor, kTensor, vTensor, outTensor); err != nil {
				t.Fatalf("failed to set attention weights: %v", err)
			}

			// Set gamma for attention normalization
			gammaTensor := tensor.NewTensor(h)
			gammaData := convertInt8ToFloat32(attnNormTensor.Data())
			for i := 0; i < h; i++ {
				gammaTensor.Set(int8(gammaData[i]), i)
			}
			if err := attn.SetGamma(gammaTensor); err != nil {
				t.Fatalf("failed to set attention gamma: %v", err)
			}

			// Apply attention
			output, err := attn.Forward(hiddenStatesTensor)
			if err != nil {
				t.Fatalf("attention forward pass failed: %v", err)
			}

			// Verify output dimensions
			shape := output.Shape()
			if len(shape) != 2 || shape[0] != len(tt.tokens) || shape[1] != config.HiddenSize {
				t.Errorf("attention output shape = %v, want [%d %d]", shape, len(tt.tokens), config.HiddenSize)
			}
		})
	}
}

func TestModelFFNSublayer(t *testing.T) {
	// Create a test model with minimal configuration
	config := &Config{
		HiddenSize:       32,
		NumHeads:         4,
		NumKVHeads:       4,
		NumLayers:        1,
		VocabSize:        10,
		MaxSeqLength:     8,
		IntermediateSize: 16,
	}
	model := NewModel(config, testDataFS)

	// Create test weights
	model.weights = &ModelWeights{
		TokenEmbedding: make([]int8, config.VocabSize*config.HiddenSize),
		Blocks:         make([]*TransformerBlock, config.NumLayers),
		FinalNorm:      make([]int8, config.HiddenSize),
	}

	// Initialize transformer block with test values
	block := &TransformerBlock{
		QKVProj:  make([]int8, 3*config.HiddenSize*config.HiddenSize),
		OutProj:  make([]int8, config.HiddenSize*config.HiddenSize),
		FFNUp:    make([]int8, config.IntermediateSize*config.HiddenSize),
		FFNDown:  make([]int8, config.HiddenSize*config.IntermediateSize),
		AttnNorm: make([]int8, config.HiddenSize),
		FFNNorm:  make([]int8, config.HiddenSize),
	}

	// Initialize FFN weights with test values
	for i := range block.FFNUp {
		block.FFNUp[i] = int8(i%3 - 1)
	}
	for i := range block.FFNDown {
		block.FFNDown[i] = int8(i%3 - 1)
	}
	for i := range block.FFNNorm {
		block.FFNNorm[i] = int8(i%3 - 1)
	}

	model.weights.Blocks[0] = block

	tests := []struct {
		name    string
		input   [][]float32
		wantErr error
	}{
		{
			name: "valid input",
			input: [][]float32{
				{1.0, -1.0, 0.0, 1.0},
			},
			wantErr: nil,
		},
		{
			name: "invalid input length",
			input: [][]float32{
				{1.0, -1.0}, // Too short
			},
			wantErr: ErrInvalidInputShape,
		},
		{
			name:    "empty input",
			input:   [][]float32{},
			wantErr: ErrInvalidInputShape,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input tensor with shape [batchSize=1, seqLen=1, hiddenDim]
			inputTensor := tensor.NewTensor(1, 1, config.HiddenSize)
			if len(tt.input) > 0 {
				for i, v := range tt.input[0] {
					inputTensor.Set(int8(v), 0, 0, i)
				}
			}

			// Create FFN sublayer
			ffn := bitnetmath.NewFFNSublayer(config.HiddenSize, config.IntermediateSize)

			// Convert weights to tensors
			ffnUpTensor := tensor.NewTensor(config.IntermediateSize, config.HiddenSize)
			ffnDownTensor := tensor.NewTensor(config.HiddenSize, config.IntermediateSize)
			ffnNormTensor := tensor.NewTensor(config.HiddenSize)

			// Copy weights into tensors
			for i := 0; i < config.IntermediateSize; i++ {
				for j := 0; j < config.HiddenSize; j++ {
					ffnUpTensor.Set(block.FFNUp[i*config.HiddenSize+j], i, j)
				}
			}
			for i := 0; i < config.HiddenSize; i++ {
				for j := 0; j < config.IntermediateSize; j++ {
					ffnDownTensor.Set(block.FFNDown[i*config.IntermediateSize+j], i, j)
				}
			}
			for i := 0; i < config.HiddenSize; i++ {
				ffnNormTensor.Set(block.FFNNorm[i], i)
			}

			// Set FFN weights
			ffn.SetWeights(ffnUpTensor, ffnDownTensor)
			ffn.SetGamma(convertInt8ToFloat32(ffnNormTensor.Data()))

			// Apply FFN
			output := ffn.Forward(inputTensor)

			// Verify output dimensions
			shape := output.Shape()
			if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != config.HiddenSize {
				t.Errorf("FFN output shape = %v, want [1 1 %d]", shape, config.HiddenSize)
			}
		})
	}
}
