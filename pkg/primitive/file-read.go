package primitive

import (
	"fmt"
	"unicode/utf8"
)

// FileSystem defines the interface for file operations
type FileSystem interface {
	ReadFile(path string) ([]byte, error)
	ListDir(path string) ([]string, error)
	WriteFile(path string, content string) error
}

// FileRead reads the contents of a file with UTF-8 validation
func FileRead(fs FileSystem, path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("empty file path")
	}

	// Read the file
	content, err := fs.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read file %s: %w", path, err)
	}

	// Check for BOM
	if len(content) >= 3 && content[0] == 0xEF && content[1] == 0xBB && content[2] == 0xBF {
		content = content[3:]
	}

	// Validate UTF-8
	if !utf8.Valid(content) {
		return "", fmt.Errorf("file contains invalid UTF-8")
	}

	return string(content), nil
}
