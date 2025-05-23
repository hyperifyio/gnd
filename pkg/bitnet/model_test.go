package bitnet

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"io/fs"
	"strings"
	"sync"
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/model"
)

// mockFS implements fs.FS for testing
type mockFS struct {
	files map[string][]byte
	mu    sync.RWMutex
}

func (m *mockFS) Open(name string) (fs.File, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	data, ok := m.files[name]
	if !ok {
		return nil, fs.ErrNotExist
	}
	return &mockFile{data: data}, nil
}

// Add this method to satisfy fs.ReadFileFS
func (m *mockFS) ReadFile(name string) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	data, ok := m.files[name]
	if !ok {
		return nil, fs.ErrNotExist
	}
	return data, nil
}

type mockFile struct {
	data []byte
	pos  int64
	mu   sync.Mutex
}

func (m *mockFile) Read(p []byte) (n int, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.pos >= int64(len(m.data)) {
		return 0, io.EOF
	}
	n = copy(p, m.data[m.pos:])
	m.pos += int64(n)
	return n, nil
}

func (m *mockFile) Close() error {
	return nil
}

func (m *mockFile) Stat() (fs.FileInfo, error) {
	return nil, nil
}

func TestLoadWeights(t *testing.T) {
	tests := []struct {
		name    string
		input   io.Reader
		wantErr error
	}{
		{
			name: "valid weights file",
			input: bytes.NewReader([]byte{
				'B', 'I', 'T', 'N', // Magic number
				1,          // Version 1
				1, 2, 3, 4, // Some weights
			}),
			wantErr: nil,
		},
		{
			name: "invalid magic number",
			input: bytes.NewReader([]byte{
				'X', 'Y', 'Z', 'W', // Wrong magic
				1,          // Version 1
				1, 2, 3, 4, // Some weights
			}),
			wantErr: ErrInvalidWeightsFormat,
		},
		{
			name: "unsupported version",
			input: bytes.NewReader([]byte{
				'B', 'I', 'T', 'N', // Magic number
				2,          // Version 2 (unsupported)
				1, 2, 3, 4, // Some weights
			}),
			wantErr: ErrUnsupportedVersion,
		},
		{
			name:    "empty reader",
			input:   strings.NewReader(""),
			wantErr: ErrInvalidWeightsFormat,
		},
		{
			name:    "nil reader",
			input:   nil,
			wantErr: ErrInvalidWeightsFormat,
		},
		{
			name: "truncated magic",
			input: bytes.NewReader([]byte{
				'B', 'I', 'T', // Incomplete magic
			}),
			wantErr: ErrInvalidWeightsFormat,
		},
		{
			name: "truncated version",
			input: bytes.NewReader([]byte{
				'B', 'I', 'T', 'N', // Magic number
				// Missing version
			}),
			wantErr: ErrWeightsFileRead,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := LoadWeights(tt.input)
			if err != tt.wantErr {
				t.Errorf("LoadWeights() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestLoadWeightsLargeFile(t *testing.T) {
	// Create a large weights file (1MB)
	data := make([]byte, 1024*1024)
	copy(data[0:4], []byte{'B', 'I', 'T', 'N'}) // Magic number
	data[4] = 1                                 // Version 1
	// Fill rest with random weights
	for i := 5; i < len(data); i++ {
		data[i] = byte(i % 256)
	}

	err := LoadWeights(bytes.NewReader(data))
	if err != nil {
		t.Errorf("LoadWeights() error = %v, wantErr nil", err)
	}
}

func BenchmarkLoadWeights(b *testing.B) {
	// Create test data with different sizes
	sizes := []struct {
		name string
		size int
	}{
		{"small", 1 * 1024},    // 1KB
		{"medium", 100 * 1024}, // 100KB
		{"large", 1024 * 1024}, // 1MB
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			// Create test data
			data := make([]byte, size.size)
			copy(data[0:4], []byte{'B', 'I', 'T', 'N'}) // Magic number
			data[4] = 1                                 // Version 1
			// Fill rest with random weights
			for i := 5; i < len(data); i++ {
				data[i] = byte(i % 256)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				err := LoadWeights(bytes.NewReader(data))
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkLoadWeightsParallel(b *testing.B) {
	// Create test data
	data := make([]byte, 1024*1024)             // 1MB
	copy(data[0:4], []byte{'B', 'I', 'T', 'N'}) // Magic number
	data[4] = 1                                 // Version 1
	// Fill rest with random weights
	for i := 5; i < len(data); i++ {
		data[i] = byte(i % 256)
	}

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			err := LoadWeights(bytes.NewReader(data))
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func TestNewModel(t *testing.T) {
	tests := []struct {
		name   string
		config *model.Config
	}{
		{
			name:   "default config",
			config: nil,
		},
		{
			name: "custom config",
			config: &model.Config{
				VocabSize:        1000,
				HiddenSize:       512,
				NumHeads:         8,
				NumKVHeads:       8,
				NumLayers:        6,
				IntermediateSize: 2048,
				MaxSeqLength:     1024,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := model.NewModel(tt.config, nil)
			if got == nil {
				t.Error("NewModel() returned nil")
			}
		})
	}
}

func TestModelEmbedTokens(t *testing.T) {
	config := model.NewConfig()
	config.VocabSize = 10
	config.HiddenSize = 16 // must be >= numHeads * 8 for valid head dim
	config.NumLayers = 2   // keep small for test
	config.IntermediateSize = 8
	config.NumHeads = 2   // Add number of attention heads
	config.NumKVHeads = 2 // Add number of KV heads

	// Calculate sizes
	embeddingSize := config.VocabSize * config.HiddenSize
	qkvSize := config.HiddenSize * 3 * config.HiddenSize
	outSize := config.HiddenSize * config.HiddenSize
	ffnUpSize := config.HiddenSize * config.IntermediateSize
	ffnDownSize := config.IntermediateSize * config.HiddenSize
	blockNormSize := config.HiddenSize
	finalNormSize := config.HiddenSize

	// Build weights file
	buf := &bytes.Buffer{}
	// Header
	binary.Write(buf, binary.LittleEndian, uint32(0x424E4554)) // "BNET"
	binary.Write(buf, binary.LittleEndian, uint32(1))          // Version 1
	// Token embeddings
	buf.Write(bytes.Repeat([]byte{1}, embeddingSize))
	// Transformer blocks
	for i := 0; i < config.NumLayers; i++ {
		buf.Write(bytes.Repeat([]byte{1}, qkvSize))
		buf.Write(bytes.Repeat([]byte{1}, outSize))
		buf.Write(bytes.Repeat([]byte{1}, ffnUpSize))
		buf.Write(bytes.Repeat([]byte{1}, ffnDownSize))
		buf.Write(bytes.Repeat([]byte{1}, blockNormSize)) // AttnNorm
		buf.Write(bytes.Repeat([]byte{1}, blockNormSize)) // FFNNorm
	}
	// FinalNorm
	buf.Write(bytes.Repeat([]byte{1}, finalNormSize))

	// Create test vocabulary
	vocab := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
		"▁":     3, // Special space token
		"a":     4,
		"b":     5,
		"c":     6,
		"d":     7,
		"e":     8,
		"f":     9,
	}

	// Create test special tokens
	specialTokens := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
	}

	// Create mock filesystem with both weights and tokenizer files
	mockFS := &mockFS{
		files: map[string][]byte{
			"test_weights.bin": buf.Bytes(),
			"tokenizer/vocab.json": func() []byte {
				data, _ := json.Marshal(vocab)
				return data
			}(),
			"tokenizer/merges.txt": []byte(""), // Empty merges file for simplicity
			"tokenizer/special_tokens.json": func() []byte {
				data, _ := json.Marshal(specialTokens)
				return data
			}(),
		},
	}

	tests := []struct {
		name    string
		tokens  []int
		wantErr bool
	}{
		{
			name:    "single token",
			tokens:  []int{1},
			wantErr: false,
		},
		{
			name:    "multiple tokens",
			tokens:  []int{0, 1},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		tt := tt // capture range variable
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel() // Run subtests in parallel

			// Create a new model instance for each subtest
			m := model.NewModel(config, mockFS)

			// Load weights
			err := m.LoadWeights("test_weights.bin")
			if err != nil {
				t.Fatalf("LoadWeights() error = %v", err)
			}

			got, err := m.Infer(tt.tokens)
			if (err != nil) != tt.wantErr {
				t.Errorf("Infer() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && len(got) != len(tt.tokens) {
				t.Errorf("Infer() returned %d tokens, want %d", len(got), len(tt.tokens))
			}

			// Clean up
			m.Close()
		})
	}
}

func TestModelClose(t *testing.T) {
	config := model.NewConfig()
	m := model.NewModel(config, nil)

	// Test Close
	m.Close()

	// Try to use the model after closing
	_, err := m.Infer([]int{1})
	if err == nil {
		t.Error("Expected error when using closed model")
	}
}
