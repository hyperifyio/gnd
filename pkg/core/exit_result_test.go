package core

import (
	"testing"
)

func TestExitResult(t *testing.T) {
	tests := []struct {
		name     string
		code     int
		expected string
	}{
		{
			name:     "exit code 0",
			code:     0,
			expected: "exit with code 0",
		},
		{
			name:     "exit code 1",
			code:     1,
			expected: "exit with code 1",
		},
		{
			name:     "exit code -1",
			code:     -1,
			expected: "exit with code -1",
		},
		{
			name:     "exit code 255",
			code:     255,
			expected: "exit with code 255",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := &ExitResult{Code: tt.code}
			if err.Error() != tt.expected {
				t.Errorf("ExitResult.Error() = %q, want %q", err.Error(), tt.expected)
			}
		})
	}
}
