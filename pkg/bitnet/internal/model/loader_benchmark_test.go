package model

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkModelLoader(b *testing.B) {
	// Create a temporary directory for test files
	tmpDir, err := os.MkdirTemp("", "bitnet-bench-*")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create test GGUF model file
	header := &GGUFHeader{
		Magic:       0x46554747, // GGUF magic number
		Version:     1,
		TensorCount: 10,
		KVCount:     5,
	}

	// Create the directory structure
	modelDir := filepath.Join(tmpDir, "pkg", "bitnet", "internal", "assets", "models", "BitNet-b1.58-2B-4T")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		b.Fatalf("Failed to create model directory: %v", err)
	}

	// Create and write the test model file
	modelPath := filepath.Join(modelDir, "model.bin")
	file, err := os.Create(modelPath)
	if err != nil {
		b.Fatalf("Failed to create test model file: %v", err)
	}
	defer file.Close()

	// Write the GGUF header
	if err := binary.Write(file, binary.LittleEndian, header); err != nil {
		b.Fatalf("Failed to write GGUF header: %v", err)
	}

	// Write some dummy tensor data (1MB)
	dummyData := make([]byte, 1024*1024)
	if _, err := file.Write(dummyData); err != nil {
		b.Fatalf("Failed to write dummy tensor data: %v", err)
	}

	// Change to the temp directory for the benchmark
	originalDir, err := os.Getwd()
	if err != nil {
		b.Fatalf("Failed to get current directory: %v", err)
	}
	if err := os.Chdir(tmpDir); err != nil {
		b.Fatalf("Failed to change to temp directory: %v", err)
	}
	defer os.Chdir(originalDir)

	b.Run("NewModelLoader", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			loader, err := NewModelLoader()
			if err != nil {
				b.Fatalf("Failed to create model loader: %v", err)
			}
			if loader == nil {
				b.Fatal("Loader is nil")
			}
		}
	})

	b.Run("LoadModelStream", func(b *testing.B) {
		loader, err := NewModelLoader()
		if err != nil {
			b.Fatalf("Failed to create model loader: %v", err)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			reader, file, err := loader.LoadModelStream()
			if err != nil {
				b.Fatalf("Failed to load model stream: %v", err)
			}
			file.Close()
			if reader == nil {
				b.Fatal("Reader is nil")
			}
		}
	})

	b.Run("LoadModelChunk", func(b *testing.B) {
		loader, err := NewModelLoader()
		if err != nil {
			b.Fatalf("Failed to create model loader: %v", err)
		}

		reader, file, err := loader.LoadModelStream()
		if err != nil {
			b.Fatalf("Failed to load model stream: %v", err)
		}
		defer file.Close()

		// Skip the header
		header := &GGUFHeader{}
		if err := binary.Read(reader, binary.LittleEndian, header); err != nil {
			b.Fatalf("Failed to read header: %v", err)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			chunk, err := loader.LoadModelChunk(reader, 1024*1024) // 1MB chunks
			if err != nil {
				b.Fatalf("Failed to load model chunk: %v", err)
			}
			if len(chunk) == 0 {
				b.Fatal("Chunk is empty")
			}
		}
	})
}

func BenchmarkModelLoader_LoadModelChunk(b *testing.B) {
	// Create a temporary test file with different sizes
	sizes := []struct {
		name string
		size int64
	}{
		{"small", 1024 * 1024},       // 1MB
		{"medium", 10 * 1024 * 1024}, // 10MB
		{"large", 100 * 1024 * 1024}, // 100MB
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			// Create test file
			tmpFile := filepath.Join(b.TempDir(), "test_model.bin")
			testData := make([]byte, size.size)
			if err := os.WriteFile(tmpFile, testData, 0644); err != nil {
				b.Fatalf("Failed to create test file: %v", err)
			}

			// Create loader
			loader := &ModelLoader{
				modelPath:  tmpFile,
				bufferSize: 1024 * 1024,
			}

			// Benchmark chunk reading
			b.Run("chunk_1mb", func(b *testing.B) {
				reader, file, err := loader.LoadModelStream()
				if err != nil {
					b.Fatalf("LoadModelStream() error = %v", err)
				}
				defer file.Close()

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, err := loader.LoadModelChunk(reader, 1024*1024)
					if err != nil {
						b.Fatalf("LoadModelChunk() error = %v", err)
					}
				}
			})

			// Benchmark different chunk sizes
			chunkSizes := []int{
				1024,            // 1KB
				1024 * 1024,     // 1MB
				4 * 1024 * 1024, // 4MB
			}

			for _, chunkSize := range chunkSizes {
				b.Run(fmt.Sprintf("chunk_%d", chunkSize), func(b *testing.B) {
					reader, file, err := loader.LoadModelStream()
					if err != nil {
						b.Fatalf("LoadModelStream() error = %v", err)
					}
					defer file.Close()

					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						_, err := loader.LoadModelChunk(reader, chunkSize)
						if err != nil {
							b.Fatalf("LoadModelChunk() error = %v", err)
						}
					}
				})
			}
		})
	}
}

func BenchmarkModelLoader_Memory(b *testing.B) {
	// Create a temporary test file
	tmpFile := filepath.Join(b.TempDir(), "test_model.bin")
	testData := make([]byte, 1024*1024) // 1MB
	if err := os.WriteFile(tmpFile, testData, 0644); err != nil {
		b.Fatalf("Failed to create test file: %v", err)
	}

	// Create loader
	loader := &ModelLoader{
		modelPath:  tmpFile,
		bufferSize: 1024 * 1024,
	}

	// Benchmark memory allocations
	b.Run("allocations", func(b *testing.B) {
		reader, file, err := loader.LoadModelStream()
		if err != nil {
			b.Fatalf("LoadModelStream() error = %v", err)
		}
		defer file.Close()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := loader.LoadModelChunk(reader, 1024*1024)
			if err != nil {
				b.Fatalf("LoadModelChunk() error = %v", err)
			}
		}
	})
}
