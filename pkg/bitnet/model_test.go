package bitnet

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

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
