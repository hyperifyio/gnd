package parsers

import (
	"fmt"
	"strings"
)

// TokenizeLine breaks down a line into tokens, handling strings, arrays, and escape sequences
func TokenizeLine(line string) ([]interface{}, error) {
	var tokens []interface{}
	var current strings.Builder
	inString := false
	escape := false
	var arrayStack [][]interface{} // stack of arrays

	appendToken := func(tok interface{}) {
		if len(arrayStack) > 0 {
			arrayStack[len(arrayStack)-1] = append(arrayStack[len(arrayStack)-1], tok)
		} else {
			tokens = append(tokens, tok)
		}
	}

	for i := 0; i < len(line); i++ {
		c := line[i]

		if escape {
			if inString {
				// Only process escape sequences inside strings
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
					current.WriteByte('\\')
					current.WriteByte(c)
				}
			} else {
				current.WriteByte('\\')
				current.WriteByte(c)
			}
			escape = false
			continue
		}

		if c == '\\' {
			escape = true
			continue
		}

		if c == '"' {
			if inString {
				// End of string
				appendToken(current.String())
				current.Reset()
				inString = false
			} else {
				// Start of string
				if current.Len() > 0 {
					appendToken(current.String())
					current.Reset()
				}
				inString = true
			}
			continue
		}

		if c == '[' && !inString {
			if current.Len() > 0 {
				appendToken(current.String())
				current.Reset()
			}
			// Start a new array
			arrayStack = append(arrayStack, []interface{}{})
			continue
		}

		if c == ']' && !inString {
			if current.Len() > 0 {
				appendToken(current.String())
				current.Reset()
			}
			if len(arrayStack) == 0 {
				return nil, fmt.Errorf("unexpected closing bracket")
			}
			// Pop the last array
			arr := arrayStack[len(arrayStack)-1]
			arrayStack = arrayStack[:len(arrayStack)-1]
			appendToken(arr)
			continue
		}

		if !inString && (c == ' ' || c == '\t') {
			if current.Len() > 0 {
				appendToken(current.String())
				current.Reset()
			}
			continue
		}

		current.WriteByte(c)
	}

	if current.Len() > 0 {
		appendToken(current.String())
	}

	// Check for unclosed strings or arrays
	if inString {
		return nil, fmt.Errorf("unclosed string")
	}
	if len(arrayStack) > 0 {
		return nil, fmt.Errorf("unclosed array")
	}

	return tokens, nil
}
