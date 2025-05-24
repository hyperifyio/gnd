package assets

import (
	"testing"
)

func TestGetModelFile(t *testing.T) {
	data, err := GetModelFile()
	if err != nil {
		t.Fatalf("Failed to get model file: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("Model file is empty")
	}
	// The model file should be quite large (several GB)
	if len(data) < 1024*1024 {
		t.Fatalf("Model file seems too small: %d bytes", len(data))
	}
}
