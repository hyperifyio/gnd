package primitive

import (
	"fmt"
	"os"
	"sort"
)

// FileList returns an array of names in a directory
func FileList(worldToken string, path string) ([]string, string, error) {
	// Log the operation for debugging
	fmt.Printf("FileList: listing directory %s\n", path)

	// Read the directory
	entries, err := os.ReadDir(path)
	if err != nil {
		// Log the error for debugging
		fmt.Printf("FileList: error reading directory %s: %v\n", path, err)
		return nil, "", fmt.Errorf("failed to read directory %s: %w", path, err)
	}

	// Extract base names and sort them for determinism
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	sort.Strings(names)

	// Generate a new world token
	newWorldToken := fmt.Sprintf("%s:list:%s", worldToken, path)

	// Log success for debugging
	fmt.Printf("FileList: found %d entries in %s\n", len(names), path)

	return names, newWorldToken, nil
} 