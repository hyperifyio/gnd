package parsers

import "testing"

// TestIsHashtag tests the IsHashtag function for '#' and other characters.
func TestIsHashtag(t *testing.T) {
	tests := []struct {
		input    byte
		expected bool
	}{
		{'#', true},
		{'"', false},
		{'a', false},
		{'\'', false},
	}

	for _, tt := range tests {
		t.Run(string([]byte{tt.input}), func(t *testing.T) {
			got := IsHashtag(tt.input)
			if got != tt.expected {
				t.Errorf("IsHashtag(%q) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}
