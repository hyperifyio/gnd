package parsers

import (
	"fmt"
	"strings"
)

// LineParser maintains the state for parsing a single line
type LineParser struct {
	line string
	pos  int
}

// NewLineParser creates a new parser for the given line
func NewLineParser(line string) *LineParser {
	return &LineParser{line: line}
}

// IsEOF returns true if we've reached the end of the line
func (p *LineParser) IsEOF() bool {
	return p.pos >= len(p.line)
}

// ParseWhitespace advances the position past any whitespace characters
func (p *LineParser) ParseWhitespace() {
	for p.pos < len(p.line) && IsWhitespace(p.line[p.pos]) {
		p.pos++
	}
}

// ParseEscapeSequence parses an escape sequence starting at the current position
func (p *LineParser) ParseEscapeSequence() (string, error) {
	if !IsEscape(p.line[p.pos]) {
		return "", fmt.Errorf("expected escape character at position %d", p.pos)
	}
	p.pos++ // Skip escape character

	if p.IsEOF() {
		return "", fmt.Errorf("unterminated escape sequence at position %d", p.pos-1)
	}

	c := p.line[p.pos]
	p.pos++
	var result strings.Builder
	HandleEscapeSequence(&result, c)
	return result.String(), nil
}

// ParseQuotedString parses a quoted string starting at the current position
func (p *LineParser) ParseQuotedString() (string, error) {
	if !IsQuote(p.line[p.pos]) {
		return "", fmt.Errorf("expected quote at position %d", p.pos)
	}
	var result strings.Builder
	p.pos++ // Skip opening quote

	for !p.IsEOF() {
		c := p.line[p.pos]
		if IsEscape(c) {
			escaped, err := p.ParseEscapeSequence()
			if err != nil {
				return "", err
			}
			result.WriteString(escaped)
			continue
		}
		if IsQuote(c) {
			p.pos++
			return result.String(), nil
		}
		result.WriteByte(c)
		p.pos++
	}

	return result.String(), fmt.Errorf("unterminated quoted string")
}

// ParseUnquotedToken parses an unquoted token starting at the current position
func (p *LineParser) ParseUnquotedToken() (string, error) {
	var result strings.Builder
	start := p.pos

	for !p.IsEOF() {
		c := p.line[p.pos]
		if IsWhitespace(c) || IsArrayStart(c) || IsArrayEnd(c) {
			break
		}
		if IsEscape(c) {
			result.WriteByte('\\')
			if p.pos+1 < len(p.line) {
				result.WriteByte(p.line[p.pos+1])
				p.pos++
			}
		} else {
			result.WriteByte(c)
		}
		p.pos++
	}

	if p.pos == start {
		return "", fmt.Errorf("expected token at position %d", p.pos)
	}
	return result.String(), nil
}

// ParseArray parses an array starting at the current position
func (p *LineParser) ParseArray() ([]interface{}, error) {
	if !IsArrayStart(p.line[p.pos]) {
		return nil, fmt.Errorf("expected array start at position %d", p.pos)
	}
	p.pos++ // Skip opening bracket

	var tokens []interface{}
	for !p.IsEOF() {
		p.ParseWhitespace()
		if p.IsEOF() {
			return nil, fmt.Errorf("unterminated array")
		}

		token, err := p.ParseArrayElement()
		if err != nil {
			return nil, err
		}
		if token == nil {
			return tokens, nil
		}
		tokens = append(tokens, token)
	}

	return nil, fmt.Errorf("unterminated array")
}

// ParseArrayElement parses a single element within an array
func (p *LineParser) ParseArrayElement() (interface{}, error) {
	switch {
	case IsQuote(p.line[p.pos]):
		return p.ParseQuotedString()
	case IsArrayStart(p.line[p.pos]):
		return p.ParseArray()
	case IsArrayEnd(p.line[p.pos]):
		p.pos++
		return nil, nil
	default:
		token, err := p.ParseUnquotedToken()
		if err != nil {
			return nil, err
		}
		return PropertyRef{Name: token}, nil
	}
}

// ParseOpCode parses the operation code (first token) of the line
func (p *LineParser) ParseOpCode() (interface{}, error) {
	if p.IsEOF() {
		return nil, fmt.Errorf("unexpected EOF")
	}
	switch {
	case IsQuote(p.line[p.pos]):
		return nil, fmt.Errorf("operation code cannot be a quoted string")
	case IsArrayStart(p.line[p.pos]):
		return nil, fmt.Errorf("operation code cannot be an array")
	case IsArrayEnd(p.line[p.pos]):
		return nil, fmt.Errorf("unexpected array end character ']' at position %d", p.pos)
	default:
		return p.ParseUnquotedToken()
	}
}

// ParseDestination parses the destination (second token) of the line
func (p *LineParser) ParseDestination() (interface{}, error) {
	if p.IsEOF() {
		return nil, fmt.Errorf("unexpected EOF")
	}
	switch {
	case IsQuote(p.line[p.pos]):
		return nil, fmt.Errorf("destination cannot be a quoted string")
	case IsArrayStart(p.line[p.pos]):
		return nil, fmt.Errorf("destination cannot be an array")
	case IsArrayEnd(p.line[p.pos]):
		return nil, fmt.Errorf("unexpected array end character ']' at position %d", p.pos)
	default:
		return p.ParseUnquotedToken()
	}
}

// ParseRemainingTokens parses all remaining tokens in the line
func (p *LineParser) ParseRemainingTokens() ([]interface{}, error) {
	var tokens []interface{}
	for !p.IsEOF() {
		p.ParseWhitespace()
		if p.IsEOF() {
			break
		}

		token, err := p.ParseRemainingToken()
		if err != nil {
			return nil, err
		}
		tokens = append(tokens, token)
	}
	return tokens, nil
}

// ParseRemainingToken parses a single token in the remaining part of the line
func (p *LineParser) ParseRemainingToken() (interface{}, error) {
	switch {
	case IsQuote(p.line[p.pos]):
		return p.ParseQuotedString()
	case IsArrayStart(p.line[p.pos]):
		return p.ParseArray()
	case IsArrayEnd(p.line[p.pos]):
		return nil, fmt.Errorf("unexpected array end character ']' at position %d", p.pos)
	default:
		token, err := p.ParseUnquotedToken()
		if err != nil {
			return nil, err
		}
		return PropertyRef{Name: token}, nil
	}
}
