package primitive

import (
	"fmt"
)

// EmitFile writes a byte sequence to a path
func EmitFile(fs FileSystem, path string, content string) (interface{}, error) {
	if path == "" {
		return nil, fmt.Errorf("empty file path")
	}
	if content == "" {
		return nil, fmt.Errorf("empty content")
	}

	// Write the file
	err := fs.WriteFile(path, content)
	if err != nil {
		return nil, fmt.Errorf("failed to write to file %s: %w", path, err)
	}

	return true, nil
}
