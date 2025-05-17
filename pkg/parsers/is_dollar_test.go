package parsers

import "testing"

// TestIsDollar tests the IsDollar function for '$' and other characters.
func TestIsDollar(t *testing.T) {
	tests := []struct {
		input    byte
		expected bool
	}{
		{'$', true},
		{'"', false},
		{'a', false},
		{'\'', false},
	}

	for _, tt := range tests {
		t.Run(string([]byte{tt.input}), func(t *testing.T) {
			got := IsDollar(tt.input)
			if got != tt.expected {
				t.Errorf("IsDollar(%q) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}
