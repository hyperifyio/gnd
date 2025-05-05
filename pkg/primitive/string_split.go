package primitive

import (
	"fmt"
	"strings"
)

// StringSplit splits a string by a delimiter
func StringSplit(s string, delim string) []string {
	// Log the operation for debugging
	fmt.Printf("StringSplit: splitting %q by %q\n", s, delim)

	// Handle empty delimiter case
	if delim == "" {
		// Split into individual characters
		chars := make([]string, len(s))
		for i, r := range s {
			chars[i] = string(r)
		}
		fmt.Printf("StringSplit: split into %d characters\n", len(chars))
		return chars
	}

	// Split the string
	parts := strings.Split(s, delim)
	fmt.Printf("StringSplit: split into %d parts\n", len(parts))
	return parts
} 