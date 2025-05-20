package embedded

import (
	"embed"
	"io/fs"
)

//go:embed model.bin tokenizer.json
var embeddedFiles embed.FS

// GetModelFile returns the embedded model file
func GetModelFile() (fs.File, error) {
	return embeddedFiles.Open("model.bin")
}

// GetTokenizerFile returns the embedded tokenizer file
func GetTokenizerFile() (fs.File, error) {
	return embeddedFiles.Open("tokenizer.json")
}

// GetModelSize returns the size of the embedded model file
func GetModelSize() (int64, error) {
	file, err := GetModelFile()
	if err != nil {
		return 0, err
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return 0, err
	}

	return info.Size(), nil
}
