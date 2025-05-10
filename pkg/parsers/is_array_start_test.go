package parsers

import "testing"

// TestIsArrayStart tests the IsArrayStart function for '[' and other characters.
func TestIsArrayStart(t *testing.T) {
	tests := []struct {
		input    byte
		expected bool
	}{
		{'[', true},
		{'a', false},
		{']', false},
	}

	for _, tt := range tests {
		t.Run(string([]byte{tt.input}), func(t *testing.T) {
			got := IsArrayStart(tt.input)
			if got != tt.expected {
				t.Errorf("IsArrayStart(%q) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}
