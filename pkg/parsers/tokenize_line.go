package parsers

import (
	"fmt"
)

// TokenizeLine tokenizes a line at the top level, ensuring the first two tokens are plain strings,
// and subsequent tokens are wrapped as PropertyRef if unquoted.
func TokenizeLine(line string) ([]interface{}, error) {
	p := NewLineParser(line)
	var tokens []interface{}

	// Parse operation code
	p.ParseWhitespace()
	if p.IsEOF() {
		return nil, fmt.Errorf("empty line")
	}

	token, err := p.ParseOpCode()
	if err != nil {
		return nil, err
	}
	tokens = append(tokens, token)

	// Parse destination
	p.ParseWhitespace()
	if p.IsEOF() {
		return tokens, nil
	}

	token, err = p.ParseDestination()
	if err != nil {
		return nil, err
	}
	tokens = append(tokens, token)

	// Parse remaining tokens
	remainingTokens, err := p.ParseRemainingTokens()
	if err != nil {
		return nil, err
	}
	tokens = append(tokens, remainingTokens...)

	return tokens, nil
}
