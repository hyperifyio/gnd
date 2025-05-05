package primitive

import (
	"fmt"
)

// Format fills placeholders in a template string
func Format(template string, args ...interface{}) (string, error) {
	// Handle empty template
	if template == "" {
		return "", nil
	}

	// Special case: single % is invalid but should return as is
	if template == "%" {
		return "%", nil
	}

	// Use fmt.Sprintf for the actual formatting
	// It will handle mismatched arguments by adding %!(EXTRA type=value) or %!(MISSING)
	result := fmt.Sprintf(template, args...)
	return result, nil
} 