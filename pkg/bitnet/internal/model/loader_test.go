package model

import (
	"os"
	"testing"
)

func TestModelLoader(t *testing.T) {
	loader := NewModelLoader()

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
