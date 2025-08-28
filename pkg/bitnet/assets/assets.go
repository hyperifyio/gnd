package assets

import (
	"embed"
	_ "embed"
)

//go:embed models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
//go:embed models/BitNet-b1.58-2B-4T/tokenizer.json
var modelFS embed.FS

// GetModelFile returns the embedded GGUF model file as a byte slice.
func GetModelFile() ([]byte, error) {
	return modelFS.ReadFile("models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
}

// GetTokenizerFile returns the embedded tokenizer file as a byte slice.
func GetTokenizerFile() ([]byte, error) {
	return modelFS.ReadFile("models/BitNet-b1.58-2B-4T/tokenizer.json")
}
