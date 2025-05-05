package primitive

import (
	"fmt"
	"os"
)

// FileRead reads the contents of a file and returns it along with a new world token
func FileRead(worldToken string, path string) (string, string, error) {
	// Log the operation for debugging
	fmt.Printf("FileRead: reading file %s\n", path)

	// Read the file
	content, err := os.ReadFile(path)
	if err != nil {
		// Log the error for debugging
		fmt.Printf("FileRead: error reading file %s: %v\n", path, err)
		return "", "", fmt.Errorf("failed to read file %s: %w", path, err)
	}

	// Generate a new world token (in a real implementation, this would be more sophisticated)
	newWorldToken := fmt.Sprintf("%s:%s", worldToken, path)

	// Log success for debugging
	fmt.Printf("FileRead: successfully read %d bytes from %s\n", len(content), path)

	return string(content), newWorldToken, nil
} 