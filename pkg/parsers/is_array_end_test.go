package parsers

import "testing"

// TestIsArrayEnd tests the IsArrayEnd function for ']' and other characters.
func TestIsArrayEnd(t *testing.T) {
	tests := []struct {
		input    byte
		expected bool
	}{
		{']', true},
		{'a', false},
		{'[', false},
	}

	for _, tt := range tests {
		t.Run(string([]byte{tt.input}), func(t *testing.T) {
			got := IsArrayEnd(tt.input)
			if got != tt.expected {
				t.Errorf("IsArrayEnd(%q) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}
