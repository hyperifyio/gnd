package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestModelLoader(t *testing.T) {
	loader, err := NewModelLoader()
	if err != nil {
		// Skip the test if the model file is not found
		if os.IsNotExist(err) {
			t.Skip("Model file not found, skipping test")
		}
		t.Fatalf("Failed to create model loader: %v", err)
	}

	// Test model path
	modelPath := loader.GetModelPath()
	if modelPath == "" {
		t.Fatal("Model path should not be empty")
	}

	// Test model size
	size, err := loader.GetModelSize()
	if err != nil {
		// Skip the test if the model file is not found
		if os.IsNotExist(err) {
			t.Skip("Model file not found, skipping test")
		}
		t.Fatalf("Failed to get model size: %v", err)
	}
	if size <= 0 {
		t.Errorf("Expected model size > 0, got %d", size)
	}

	// Test model loading
	file, err := loader.LoadModel()
	if err != nil {
		// Skip the test if the model file is not found
		if os.IsNotExist(err) {
			t.Skip("Model file not found, skipping test")
		}
		t.Fatalf("Failed to load model: %v", err)
	}
	defer file.Close()

	// Verify file is readable
	buf := make([]byte, 1024)
	_, err = file.Read(buf)
	if err != nil {
		t.Fatalf("Failed to read model file: %v", err)
	}
}

func TestModelLoader_Streaming(t *testing.T) {
	// Create a temporary test file
	tmpFile := filepath.Join(t.TempDir(), "test_model.bin")
	testData := []byte("test model data for streaming")
	if err := os.WriteFile(tmpFile, testData, 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Create a loader with the test file
	loader := &ModelLoader{
		modelPath:  tmpFile,
		bufferSize: 1024,
	}

	// Test streaming
	reader, file, err := loader.LoadModelStream()
	if err != nil {
		t.Fatalf("LoadModelStream() error = %v", err)
	}
	defer file.Close()

	// Read chunks
	chunk1, err := loader.LoadModelChunk(reader, 10)
	if err != nil {
		t.Fatalf("LoadModelChunk() error = %v", err)
	}
	if string(chunk1) != "test model" {
		t.Errorf("LoadModelChunk() = %v, want %v", string(chunk1), "test model")
	}

	chunk2, err := loader.LoadModelChunk(reader, 20)
	if err != nil {
		t.Fatalf("LoadModelChunk() error = %v", err)
	}
	if string(chunk2) != " data for streaming" {
		t.Errorf("LoadModelChunk() = %v, want %v", string(chunk2), " data for streaming")
	}

	// Test EOF
	chunk3, err := loader.LoadModelChunk(reader, 10)
	if err != nil {
		t.Fatalf("LoadModelChunk() error = %v", err)
	}
	if len(chunk3) != 0 {
		t.Errorf("LoadModelChunk() returned non-empty chunk at EOF: %v", chunk3)
	}
}

func TestModelLoader_InvalidPath(t *testing.T) {
	loader := &ModelLoader{
		modelPath: "/nonexistent/path/model.bin",
	}

	_, _, err := loader.LoadModelStream()
	if err == nil {
		t.Error("LoadModelStream() expected error for invalid path")
	}
}

func TestModelLoader_NilReader(t *testing.T) {
	loader := &ModelLoader{
		modelPath: "test.bin",
	}

	_, err := loader.LoadModelChunk(nil, 1024)
	if err == nil {
		t.Error("LoadModelChunk() expected error for nil reader")
	}
}
