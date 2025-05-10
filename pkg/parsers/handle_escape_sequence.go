package parsers

import "strings"

// HandleEscapeSequence processes an escape sequence in a quoted string.
// It writes the appropriate character to the current token.
func HandleEscapeSequence(current *strings.Builder, c byte) {
	switch c {
	case 'n':
		current.WriteByte('\n')
	case 't':
		current.WriteByte('\t')
	case 'r':
		current.WriteByte('\r')
	case '\\':
		current.WriteByte('\\')
	case '"':
		current.WriteByte('"')
	default:
		current.WriteByte(c)
	}
}
