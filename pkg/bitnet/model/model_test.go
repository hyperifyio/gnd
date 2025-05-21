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
	"testing"
	"time"
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
			input:   []byte{0x1B}, // 00 01 10 11 in binary
			weights: make([]int8, 4),
			want:    []int8{-1, 0, 1, 1},
			wantErr: nil,
		},
		{
			name:    "multiple bytes",
			input:   []byte{0x1B, 0x2D}, // 00 01 10 11, 10 11 01 01
			weights: make([]int8, 8),
			want:    []int8{-1, 0, 1, 1, 1, 1, 0, 0},
			wantErr: nil,
		},
		{
			name:    "incomplete byte",
			input:   []byte{0x1B},
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
			input:   []byte{0x1B},
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
	// Create test filesystem with valid weights
	fs := &testFS{
		files: map[string][]byte{
			"weights.bin": createValidWeights(),
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
	// Create test data
	data := make([]byte, 1024)
	for i := range data {
		data[i] = byte(i % 256)
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
		_, err := model.Infer("test input")
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
