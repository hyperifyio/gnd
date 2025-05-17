package parsers

import (
	"errors"
)

var EmptyLineError = errors.New("empty line")

// TokenizeLine tokenizes a line at the top level, ensuring the first two tokens are plain strings,
// and subsequent tokens are wrapped as PropertyRef if unquoted.
func TokenizeLine(line string) ([]interface{}, error) {
	p := NewLineParser(line)
	var tokens []interface{}

	// Parse optional destination
	p.ParseWhitespace()
	if p.IsEOF() {
		return nil, EmptyLineError
	}

	if p.IsDollar() {
		token, err := p.ParseDestination()
		if err != nil {
			return nil, err
		}
		tokens = append(tokens, token)
	}

	// Parse operation code
	p.ParseWhitespace()
	if p.IsEOF() {
		return nil, EmptyLineError
	}
	token, err := p.ParseOpCode()
	if err != nil {
		return tokens, nil
	}
	tokens = append(tokens, token)

	// Parse remaining tokens
	p.ParseWhitespace()
	if p.IsEOF() {
		return tokens, nil
	}
	remainingTokens, err := p.ParseRemainingTokens()
	if err != nil {
		return nil, err
	}
	tokens = append(tokens, remainingTokens...)
	return tokens, nil
}
