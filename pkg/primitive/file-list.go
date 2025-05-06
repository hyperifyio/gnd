package primitive

import (
	"fmt"
	"sort"
)

// FileList returns an array of names in a directory
func FileList(fs FileSystem, path string) ([]string, error) {
	if path == "" {
		return nil, fmt.Errorf("empty directory path")
	}

	// Read the directory
	entries, err := fs.ListDir(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read directory %s: %w", path, err)
	}

	// Sort names for determinism
	sort.Strings(entries)

	return entries, nil
}
