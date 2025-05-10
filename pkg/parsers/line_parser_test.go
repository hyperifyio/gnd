package parsers

import (
	"reflect"
	"testing"
)

func slicesEqual(a, b []interface{}) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}

func TestNewLineParser(t *testing.T) {
	line := "test line"
	parser := NewLineParser(line)
	if parser.line != line {
		t.Errorf("NewLineParser() line = %v, want %v", parser.line, line)
	}
	if parser.pos != 0 {
		t.Errorf("NewLineParser() pos = %v, want 0", parser.pos)
	}
}

func TestIsEOF(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		pos      int
		expected bool
	}{
		{"empty string", "", 0, true},
		{"at start", "test", 0, false},
		{"at end", "test", 4, true},
		{"beyond end", "test", 5, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LineParser{line: tt.line, pos: tt.pos}
			if got := parser.IsEOF(); got != tt.expected {
				t.Errorf("IsEOF() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestParseWhitespace(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		pos      int
		expected int
	}{
		{"no whitespace", "test", 0, 0},
		{"leading whitespace", "  test", 0, 2},
		{"middle whitespace", "test  test", 4, 6},
		{"all whitespace", "   ", 0, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LineParser{line: tt.line, pos: tt.pos}
			parser.ParseWhitespace()
			if parser.pos != tt.expected {
				t.Errorf("ParseWhitespace() pos = %v, want %v", parser.pos, tt.expected)
			}
		})
	}
}

func TestParseEscapeSequence(t *testing.T) {
	tests := []struct {
		name        string
		line        string
		pos         int
		expected    string
		expectError bool
	}{
		{"valid escape", "\\n", 0, "\n", false},
		{"invalid escape", "a", 0, "", true},
		{"unterminated", "\\", 0, "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LineParser{line: tt.line, pos: tt.pos}
			got, err := parser.ParseEscapeSequence()
			if (err != nil) != tt.expectError {
				t.Errorf("ParseEscapeSequence() error = %v, expectError %v", err, tt.expectError)
				return
			}
			if !tt.expectError && got != tt.expected {
				t.Errorf("ParseEscapeSequence() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestParseQuotedString(t *testing.T) {
	tests := []struct {
		name        string
		line        string
		pos         int
		expected    string
		expectError bool
	}{
		{"simple quoted", "\"test\"", 0, "test", false},
		{"with escape", "\"test\\n\"", 0, "test\n", false},
		{"unterminated", "\"test", 0, "", true},
		{"no quote", "test", 0, "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LineParser{line: tt.line, pos: tt.pos}
			got, err := parser.ParseQuotedString()
			if (err != nil) != tt.expectError {
				t.Errorf("ParseQuotedString() error = %v, expectError %v", err, tt.expectError)
				return
			}
			if !tt.expectError && got != tt.expected {
				t.Errorf("ParseQuotedString() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestParseUnquotedToken(t *testing.T) {
	tests := []struct {
		name        string
		line        string
		pos         int
		expected    string
		expectError bool
	}{
		{"simple token", "test", 0, "test", false},
		{"with escape", "test\\n", 0, "test\\n", false},
		{"with whitespace", "test test", 0, "test", false},
		{"empty", "", 0, "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LineParser{line: tt.line, pos: tt.pos}
			got, err := parser.ParseUnquotedToken()
			if (err != nil) != tt.expectError {
				t.Errorf("ParseUnquotedToken() error = %v, expectError %v", err, tt.expectError)
				return
			}
			if !tt.expectError && got != tt.expected {
				t.Errorf("ParseUnquotedToken() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestParseArray(t *testing.T) {
	tests := []struct {
		name        string
		line        string
		pos         int
		expected    []interface{}
		expectError bool
	}{
		{"empty array", "[]", 0, []interface{}{}, false},
		{"simple array", "[a b c]", 0, []interface{}{PropertyRef{Name: "a"}, PropertyRef{Name: "b"}, PropertyRef{Name: "c"}}, false},
		{"nested array", "[[a] [b]]", 0, []interface{}{
			[]interface{}{PropertyRef{Name: "a"}},
			[]interface{}{PropertyRef{Name: "b"}},
		}, false},
		{"unterminated", "[a", 0, nil, true},
		{"no array", "a", 0, nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LineParser{line: tt.line, pos: tt.pos}
			got, err := parser.ParseArray()
			if (err != nil) != tt.expectError {
				t.Errorf("ParseArray() error = %v, expectError %v", err, tt.expectError)
				return
			}
			if !tt.expectError && !slicesEqual(got, tt.expected) {
				t.Errorf("ParseArray() = %#v, want %#v", got, tt.expected)
			}
		})
	}
}

func TestParseOpCode(t *testing.T) {
	tests := []struct {
		name        string
		line        string
		pos         int
		expected    string
		expectError bool
	}{
		{"valid opcode", "test", 0, "test", false},
		{"quoted string", "\"test\"", 0, "", true},
		{"array", "[test]", 0, "", true},
		{"empty", "", 0, "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LineParser{line: tt.line, pos: tt.pos}
			got, err := parser.ParseOpCode()
			if (err != nil) != tt.expectError {
				t.Errorf("ParseOpCode() error = %v, expectError %v", err, tt.expectError)
				return
			}
			if !tt.expectError && got != tt.expected {
				t.Errorf("ParseOpCode() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestParseDestination(t *testing.T) {
	tests := []struct {
		name        string
		line        string
		pos         int
		expected    string
		expectError bool
	}{
		{"valid destination", "test", 0, "test", false},
		{"quoted string", "\"test\"", 0, "", true},
		{"array", "[test]", 0, "", true},
		{"empty", "", 0, "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LineParser{line: tt.line, pos: tt.pos}
			got, err := parser.ParseDestination()
			if (err != nil) != tt.expectError {
				t.Errorf("ParseDestination() error = %v, expectError %v", err, tt.expectError)
				return
			}
			if !tt.expectError && got != tt.expected {
				t.Errorf("ParseDestination() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestParseRemainingTokens(t *testing.T) {
	tests := []struct {
		name        string
		line        string
		pos         int
		expected    []interface{}
		expectError bool
	}{
		{"simple tokens", "a b c", 0, []interface{}{PropertyRef{Name: "a"}, PropertyRef{Name: "b"}, PropertyRef{Name: "c"}}, false},
		{"mixed tokens", "a \"b\" [c]", 0, []interface{}{PropertyRef{Name: "a"}, "b", []interface{}{PropertyRef{Name: "c"}}}, false},
		{"empty", "", 0, []interface{}{}, false},
		{"whitespace only", "   ", 0, []interface{}{}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LineParser{line: tt.line, pos: tt.pos}
			got, err := parser.ParseRemainingTokens()
			if (err != nil) != tt.expectError {
				t.Errorf("ParseRemainingTokens() error = %v, expectError %v", err, tt.expectError)
				return
			}
			if !tt.expectError && !slicesEqual(got, tt.expected) {
				t.Errorf("ParseRemainingTokens() = %#v, want %#v", got, tt.expected)
			}
		})
	}
}
