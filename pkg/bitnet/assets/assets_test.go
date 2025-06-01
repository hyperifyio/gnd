package assets

import (
	"os"
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

func TestEmbeddedModelFileSizeMatchesDisk(t *testing.T) {
	embedded, err := GetModelFile()
	if err != nil {
		t.Fatalf("failed to read embedded model: %v", err)
	}
	diskInfo, err := os.Stat("models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
	if err != nil {
		t.Fatalf("failed to stat model file on disk: %v", err)
	}
	if int64(len(embedded)) != diskInfo.Size() {
		t.Errorf("embedded model size (%d) does not match disk file size (%d)", len(embedded), diskInfo.Size())
	}
}
