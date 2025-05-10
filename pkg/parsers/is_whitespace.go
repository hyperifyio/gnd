package parsers

// IsWhitespace returns true if the character is a space or tab.
func IsWhitespace(c byte) bool {
	return c == ' ' || c == '\t'
}
