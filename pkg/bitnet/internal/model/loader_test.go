package model

import (
	"testing"
)

func TestModelLoader(t *testing.T) {
	loader := NewModelLoader()

	// Test model size
	size, err := loader.GetModelSize()
	if err != nil {
		t.Fatalf("Failed to get model size: %v", err)
	}
	if size <= 0 {
		t.Errorf("Expected model size > 0, got %d", size)
	}

	// Test model loading
	file, err := loader.LoadModel()
	if err != nil {
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
