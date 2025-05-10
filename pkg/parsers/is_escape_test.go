package parsers

import "testing"

func TestIsEscape(t *testing.T) {
	tests := []struct {
		name     string
		input    byte
		expected bool
	}{
		{
			name:     "escape character",
			input:    '\\',
			expected: true,
		},
		{
			name:     "non-escape character",
			input:    'a',
			expected: false,
		},
		{
			name:     "number",
			input:    '1',
			expected: false,
		},
		{
			name:     "special character",
			input:    '@',
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsEscape(tt.input)
			if result != tt.expected {
				t.Errorf("IsEscape(%q) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}
