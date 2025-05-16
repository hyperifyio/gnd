package core

import (
	"path/filepath"
	"testing"
)

func TestSubroutinePath(t *testing.T) {
	tests := []struct {
		name     string
		inName   string
		dir      string
		expected string
	}{
		{
			name:     "absolute path, already has .gnd",
			inName:   filepath.Join(string(filepath.Separator), "foo", "bar", "baz.gnd"),
			dir:      "/ignored/dir",
			expected: filepath.Join(string(filepath.Separator), "foo", "bar", "baz.gnd"),
		},
		{
			name:     "absolute path, missing .gnd",
			inName:   filepath.Join(string(filepath.Separator), "foo", "bar", "baz"),
			dir:      "/ignored/dir",
			expected: filepath.Join(string(filepath.Separator), "foo", "bar", "baz.gnd"),
		},
		{
			name:     "relative path, already has .gnd",
			inName:   "baz.gnd",
			dir:      filepath.Join(string(filepath.Separator), "work", "subroutines"),
			expected: filepath.Join(string(filepath.Separator), "work", "subroutines", "baz.gnd"),
		},
		{
			name:     "relative path, missing .gnd",
			inName:   "baz",
			dir:      filepath.Join(string(filepath.Separator), "work", "subroutines"),
			expected: filepath.Join(string(filepath.Separator), "work", "subroutines", "baz.gnd"),
		},
	}

	for _, tt := range tests {
		tt := tt // capture range variable
		t.Run(tt.name, func(t *testing.T) {
			got := SubroutinePath(tt.inName, tt.dir)
			if got != tt.expected {
				t.Errorf("SubroutinePath(%q, %q) = %q; want %q",
					tt.inName, tt.dir, got, tt.expected)
			}
		})
	}
}
