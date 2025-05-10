package core

import (
	"os"
	"path/filepath"
)

// ParseFile parses a GND file and returns its instructions
func ParseFile(filePath string) ([]*Instruction, error) {
	// Read the file content
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	// Get the directory of the file
	dir := filepath.Dir(filePath)

	// Parse the instructions
	return ParseInstructionsString(string(content), dir)
}
