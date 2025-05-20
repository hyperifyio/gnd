package assets

import (
	"embed"
	_ "embed"
)

//go:embed models/BitNet-b1.58-2B-4T/model.bin
var modelFS embed.FS

// GetModelFile returns the embedded model file as a byte slice.
func GetModelFile() ([]byte, error) {
	return modelFS.ReadFile("models/BitNet-b1.58-2B-4T/model.bin")
}
