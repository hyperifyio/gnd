package primitive

import (
	"reflect"
	"testing"
)

func TestTokenize(t *testing.T) {
	// Test cases
	tests := []struct {
		name   string
		input  string
		expect []Token
	}{
		{
			name:   "Empty string",
			input:  "",
			expect: []Token{},
		},
		{
			name:  "Simple identifier",
			input: "hello",
			expect: []Token{
				{Type: "identifier", Value: "hello"},
			},
		},
		{
			name:  "Identifier with underscore",
			input: "hello_world",
			expect: []Token{
				{Type: "identifier", Value: "hello_world"},
			},
		},
		{
			name:  "Number literal",
			input: "123",
			expect: []Token{
				{Type: "literal", Value: "123"},
			},
		},
		{
			name:  "String literal",
			input: `"hello"`,
			expect: []Token{
				{Type: "literal", Value: `"hello"`},
			},
		},
		{
			name:  "String literal with escape",
			input: `"hello\nworld"`,
			expect: []Token{
				{Type: "literal", Value: `"hello\nworld"`},
			},
		},
		{
			name:  "Comment",
			input: "# This is a comment",
			expect: []Token{
				{Type: "comment", Value: "# This is a comment"},
			},
		},
		{
			name:  "Operator",
			input: "+",
			expect: []Token{
				{Type: "operator", Value: "+"},
			},
		},
		{
			name:  "Delimiter",
			input: "(",
			expect: []Token{
				{Type: "delimiter", Value: "("},
			},
		},
		{
			name:  "Complex expression",
			input: "x = 123 + \"hello\" # comment",
			expect: []Token{
				{Type: "identifier", Value: "x"},
				{Type: "operator", Value: "="},
				{Type: "literal", Value: "123"},
				{Type: "operator", Value: "+"},
				{Type: "literal", Value: `"hello"`},
				{Type: "comment", Value: "# comment"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Tokenize(tt.input)
			if !reflect.DeepEqual(got, tt.expect) {
				t.Errorf("Tokenize(%q) = %v, want %v",
					tt.input, got, tt.expect)
			}
		})
	}
}
