package parsers

import (
	"strings"
	"testing"
)

// TestHandleEscapeSequence tests the HandleEscapeSequence function for all supported escape characters.
func TestHandleEscapeSequence(t *testing.T) {
	tests := []struct {
		name     string
		input    byte
		expected string
	}{
		{"newline", 'n', "\n"},
		{"tab", 't', "\t"},
		{"carriage return", 'r', "\r"},
		{"backslash", '\\', "\\"},
		{"double quote", '"', "\""},
		{"other character", 'a', "a"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var current strings.Builder
			HandleEscapeSequence(&current, tt.input)
			if current.String() != tt.expected {
				t.Errorf("HandleEscapeSequence() = %v, want %v", current.String(), tt.expected)
			}
		})
	}
}
