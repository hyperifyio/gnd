package core

import (
	"testing"
)

func TestUnescapeString(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "no escape sequences",
			input:    "hello world",
			expected: "hello world",
		},
		{
			name:     "newline escape",
			input:    "hello\\nworld",
			expected: "hello\nworld",
		},
		{
			name:     "tab escape",
			input:    "hello\\tworld",
			expected: "hello\tworld",
		},
		{
			name:     "carriage return escape",
			input:    "hello\\rworld",
			expected: "hello\rworld",
		},
		{
			name:     "backslash escape",
			input:    "hello\\\\world",
			expected: "hello\\world",
		},
		{
			name:     "quote escape",
			input:    "hello\\\"world",
			expected: "hello\"world",
		},
		{
			name:     "multiple escape sequences",
			input:    "line1\\nline2\\tindented\\r\\n",
			expected: "line1\nline2\tindented\r\n",
		},
		{
			name:     "invalid escape sequence",
			input:    "hello\\xworld",
			expected: "hello\\xworld",
		},
		{
			name:     "backslash at end",
			input:    "hello\\",
			expected: "hello\\",
		},
		{
			name:     "mixed content",
			input:    "\\n\\t\\r\\\\\\\"normal text\\n",
			expected: "\n\t\r\\\"normal text\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := UnescapeString(tt.input)
			if result != tt.expected {
				t.Errorf("UnescapeString(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}
