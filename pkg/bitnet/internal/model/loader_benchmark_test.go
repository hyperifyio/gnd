package model

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func BenchmarkLoadModel(b *testing.B) {
	// Create test GGUF file with a full GGUFHeader
	header := &GGUFHeader{
		Magic:       0x46554747, // GGUF magic number
		Version:     1,
		TensorCount: 10,
		KVCount:     5,
	}
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, header); err != nil {
		b.Fatal(err)
	}

	testFS := &testFS{
		files: map[string][]byte{
			"model.gguf": buf.Bytes(),
		},
	}

	loader, err := NewModelLoader(testFS, "model.gguf")
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := loader.LoadModel()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoadModelStream(b *testing.B) {
	// Create test GGUF file with 1MB of data
	data := make([]byte, 1024*1024)
	binary.LittleEndian.PutUint32(data[0:4], 0x46554747) // "GGUF"
	binary.LittleEndian.PutUint32(data[4:8], 1)          // Version 1

	testFS := &testFS{
		files: map[string][]byte{
			"model.gguf": data,
		},
	}

	loader, err := NewModelLoader(testFS, "model.gguf")
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader, file, err := loader.LoadModelStream()
		if err != nil {
			b.Fatal(err)
		}
		file.Close()
		if reader == nil {
			b.Fatal("reader is nil")
		}
	}
}

func BenchmarkLoadModelChunk(b *testing.B) {
	// Create test GGUF file with 1MB of data
	data := make([]byte, 1024*1024)
	binary.LittleEndian.PutUint32(data[0:4], 0x46554747) // "GGUF"
	binary.LittleEndian.PutUint32(data[4:8], 1)          // Version 1

	testFS := &testFS{
		files: map[string][]byte{
			"model.gguf": data,
		},
	}

	loader, err := NewModelLoader(testFS, "model.gguf")
	if err != nil {
		b.Fatal(err)
	}

	reader, file, err := loader.LoadModelStream()
	if err != nil {
		b.Fatal(err)
	}
	defer file.Close()

	chunkSize := 1024 * 64 // 64KB chunks
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := loader.LoadModelChunk(reader, chunkSize)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGetModelSize(b *testing.B) {
	// Create test GGUF file with 1MB of data
	data := make([]byte, 1024*1024)
	binary.LittleEndian.PutUint32(data[0:4], 0x46554747) // "GGUF"
	binary.LittleEndian.PutUint32(data[4:8], 1)          // Version 1

	testFS := &testFS{
		files: map[string][]byte{
			"model.gguf": data,
		},
	}

	loader, err := NewModelLoader(testFS, "model.gguf")
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := loader.GetModelSize()
		if err != nil {
			b.Fatal(err)
		}
	}
}
