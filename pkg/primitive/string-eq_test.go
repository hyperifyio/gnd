package primitive

import (
	"testing"
)

func TestStringEq(t *testing.T) {
	tests := []struct {
		name     string
		s1       string
		s2       string
		expected bool
	}{
		{
			name:     "equal strings",
			s1:       "hello",
			s2:       "hello",
			expected: true,
		},
		{
			name:     "different strings",
			s1:       "hello",
			s2:       "world",
			expected: false,
		},
		{
			name:     "empty strings",
			s1:       "",
			s2:       "",
			expected: true,
		},
		{
			name:     "case sensitive",
			s1:       "Hello",
			s2:       "hello",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := StringEq(tt.s1, tt.s2)
			if err != nil {
				t.Errorf("StringEq() error = %v", err)
				return
			}
			if result != tt.expected {
				t.Errorf("StringEq() = %v, want %v", result, tt.expected)
			}
		})
	}
}
