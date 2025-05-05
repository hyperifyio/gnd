package primitive

import (
	"fmt"
	"os"
	"path/filepath"
)

// EmitFile writes a byte sequence to a path
func EmitFile(worldToken string, path string, content string) (string, error) {
	// Log the operation for debugging
	fmt.Printf("EmitFile: writing to file %s\n", path)

	// Create parent directories if they don't exist
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		fmt.Printf("EmitFile: error creating directory %s: %v\n", dir, err)
		return "", fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	// Write the file
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		fmt.Printf("EmitFile: error writing to file %s: %v\n", path, err)
		return "", fmt.Errorf("failed to write to file %s: %w", path, err)
	}

	// Generate a new world token
	newWorldToken := fmt.Sprintf("%s:emit:%s", worldToken, path)

	// Log success for debugging
	fmt.Printf("EmitFile: successfully wrote %d bytes to %s\n", len(content), path)

	return newWorldToken, nil
} 