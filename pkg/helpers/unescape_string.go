package helpers

import "strings"

// UnescapeString converts escape sequences in a string literal to their actual values
func UnescapeString(s string) string {
	var result strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+1 < len(s) {
			switch s[i+1] {
			case 'n':
				result.WriteByte('\n')
			case 't':
				result.WriteByte('\t')
			case 'r':
				result.WriteByte('\r')
			case '\\':
				result.WriteByte('\\')
			case '"':
				result.WriteByte('"')
			default:
				result.WriteByte(s[i])
				result.WriteByte(s[i+1])
			}
			i++ // Skip the next character
		} else {
			result.WriteByte(s[i])
		}
	}
	return result.String()
}
