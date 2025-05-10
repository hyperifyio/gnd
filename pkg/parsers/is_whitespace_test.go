package parsers

import "testing"

// TestIsWhitespace tests the IsWhitespace function for space, tab, and other characters.
func TestIsWhitespace(t *testing.T) {
	tests := []struct {
		input    byte
		expected bool
	}{
		{' ', true},
		{'\t', true},
		{'a', false},
		{'\n', false},
	}

	for _, tt := range tests {
		t.Run(string([]byte{tt.input}), func(t *testing.T) {
			got := IsWhitespace(tt.input)
			if got != tt.expected {
				t.Errorf("IsWhitespace(%q) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}
