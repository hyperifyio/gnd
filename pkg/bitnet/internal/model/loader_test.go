package model

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestModelLoader(t *testing.T) {
	// Create a temporary directory for test files
	tmpDir, err := os.MkdirTemp("", "bitnet-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
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
		t.Fatalf("Failed to create model directory: %v", err)
	}

	// Create and write the test model file
	modelPath := filepath.Join(modelDir, "model.bin")
	file, err := os.Create(modelPath)
	if err != nil {
		t.Fatalf("Failed to create test model file: %v", err)
	}
	defer file.Close()

	// Write the GGUF header
	if err := binary.Write(file, binary.LittleEndian, header); err != nil {
		t.Fatalf("Failed to write GGUF header: %v", err)
	}

	// Write some dummy tensor data
	dummyData := make([]byte, 1024)
	if _, err := file.Write(dummyData); err != nil {
		t.Fatalf("Failed to write dummy tensor data: %v", err)
	}

	// Change to the temp directory for the test
	originalDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Failed to get current directory: %v", err)
	}
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to change to temp directory: %v", err)
	}
	defer os.Chdir(originalDir)

	// Test model loader creation
	loader, err := NewModelLoader()
	if err != nil {
		t.Fatalf("Failed to create model loader: %v", err)
	}

	// Test model path
	if loader.GetModelPath() != modelPath {
		t.Errorf("Expected model path %s, got %s", modelPath, loader.GetModelPath())
	}

	// Test header loading
	loadedHeader := loader.GetHeader()
	if loadedHeader == nil {
		t.Fatal("Expected header to be loaded")
	}

	if loadedHeader.Magic != header.Magic {
		t.Errorf("Expected magic number %x, got %x", header.Magic, loadedHeader.Magic)
	}
	if loadedHeader.Version != header.Version {
		t.Errorf("Expected version %d, got %d", header.Version, loadedHeader.Version)
	}
	if loadedHeader.TensorCount != header.TensorCount {
		t.Errorf("Expected tensor count %d, got %d", header.TensorCount, loadedHeader.TensorCount)
	}
	if loadedHeader.KVCount != header.KVCount {
		t.Errorf("Expected KV count %d, got %d", header.KVCount, loadedHeader.KVCount)
	}

	// Test model size
	size, err := loader.GetModelSize()
	if err != nil {
		t.Fatalf("Failed to get model size: %v", err)
	}
	expectedSize := int64(binary.Size(header) + len(dummyData))
	if size != expectedSize {
		t.Errorf("Expected model size %d, got %d", expectedSize, size)
	}

	// Test model streaming
	reader, file, err := loader.LoadModelStream()
	if err != nil {
		t.Fatalf("Failed to load model stream: %v", err)
	}
	defer file.Close()

	// Read and verify the header
	readHeader := &GGUFHeader{}
	if err := binary.Read(reader, binary.LittleEndian, readHeader); err != nil {
		t.Fatalf("Failed to read header from stream: %v", err)
	}

	if readHeader.Magic != header.Magic {
		t.Errorf("Expected magic number %x, got %x", header.Magic, readHeader.Magic)
	}

	// Test chunk loading
	chunk, err := loader.LoadModelChunk(reader, 512)
	if err != nil {
		t.Fatalf("Failed to load model chunk: %v", err)
	}
	if len(chunk) == 0 {
		t.Error("Expected non-empty chunk")
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
