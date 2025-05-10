package parsers

// IsEscape returns true if the character is an escape character.
func IsEscape(c byte) bool {
	return c == '\\'
}
