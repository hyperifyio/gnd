package model

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"io/fs"
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
		size    int
		want    []int8
		wantErr error
	}{
		{
			name:    "valid weights",
			input:   []byte{0x24}, // 0b00100100 = [-1, 0, 1, -1]
			size:    4,
			want:    []int8{-1, 0, 1, -1},
			wantErr: nil,
		},
		{
			name:    "invalid packed value",
			input:   []byte{0xFF}, // 0b11111111 = invalid packed value (3)
			size:    4,
			want:    nil,
			wantErr: ErrInvalidWeightValue,
		},
		{
			name:    "partial read",
			input:   []byte{0x1B},
			size:    5,
			want:    nil,
			wantErr: ErrWeightsFileRead,
		},
		{
			name:  "empty input",
			input: []byte{},
			size:  0,
			want:  []int8{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			weights := make([]int8, tt.size)
			model := &Model{
				config: NewConfig(),
			}
			err := model.readTernaryWeights(bytes.NewReader(tt.input), weights)
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("readTernaryWeights() error = %v, wantErr %v", err, tt.wantErr)
				return
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
