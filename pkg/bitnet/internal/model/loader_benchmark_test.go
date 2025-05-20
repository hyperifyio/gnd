package model

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

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
