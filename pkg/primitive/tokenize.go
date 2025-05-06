package primitive

import (
	"fmt"
	"unicode"
)

type Token struct {
	Type  string `json:"type"`
	Value string `json:"value"`
}

func Tokenize(input string) []Token {
	fmt.Printf("Tokenize: tokenizing %q\n", input)

	if input == "" {
		fmt.Printf("Tokenize: found 0 tokens\n")
		return []Token{}
	}

	var tokens []Token
	var current string
	var currentType string

	emitToken := func() {
		if current != "" {
			tokens = append(tokens, Token{Type: currentType, Value: current})
			current = ""
		}
	}

	for i := 0; i < len(input); i++ {
		c := rune(input[i])
		var newType string

		switch {
		case unicode.IsSpace(c):
			newType = "whitespace"
		case unicode.IsLetter(c) || c == '_':
			newType = "identifier"
		case unicode.IsDigit(c):
			newType = "literal"
		case c == '"' || c == '\'':
			newType = "literal"
		case c == '#' || (c == '/' && i+1 < len(input) && input[i+1] == '/'):
			newType = "comment"
		case c == '+' || c == '-' || c == '*' || c == '/' || c == '=' || c == '<' || c == '>':
			newType = "operator"
		default:
			newType = "delimiter"
		}

		if currentType != "" && newType != currentType {
			emitToken()
		}

		currentType = newType

		switch currentType {
		case "comment":
			for i < len(input) && input[i] != '\n' {
				current += string(input[i])
				i++
			}
			emitToken()
			continue
		case "literal":
			if c == '"' || c == '\'' {
				quote := c
				current += string(c)
				i++
				for i < len(input) && input[i] != byte(quote) {
					if input[i] == '\\' && i+1 < len(input) {
						current += string(input[i : i+2])
						i++
					} else {
						current += string(input[i])
					}
					i++
				}
				if i < len(input) {
					current += string(input[i])
				}
				emitToken()
				continue
			}
		case "whitespace":
			continue
		}

		current += string(c)
	}

	emitToken()

	fmt.Printf("Tokenize: found %d tokens\n", len(tokens))
	return tokens
}
