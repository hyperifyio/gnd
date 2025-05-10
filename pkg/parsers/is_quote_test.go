package parsers

import "testing"

// TestIsQuote tests the IsQuote function for '"' and other characters.
func TestIsQuote(t *testing.T) {
	tests := []struct {
		input    byte
		expected bool
	}{
		{'"', true},
		{'a', false},
		{'\'', false},
	}

	for _, tt := range tests {
		t.Run(string([]byte{tt.input}), func(t *testing.T) {
			got := IsQuote(tt.input)
			if got != tt.expected {
				t.Errorf("IsQuote(%q) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}
