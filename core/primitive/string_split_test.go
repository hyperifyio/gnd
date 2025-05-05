package primitive

import (
	"reflect"
	"testing"
)

func TestStringSplit(t *testing.T) {
	// Test cases
	tests := []struct {
		name   string
		input  string
		delim  string
		expect []string
	}{
		{
			name:   "Split by comma",
			input:  "a,b,c",
			delim:  ",",
			expect: []string{"a", "b", "c"},
		},
		{
			name:   "Split with empty fields",
			input:  "a,,c",
			delim:  ",",
			expect: []string{"a", "", "c"},
		},
		{
			name:   "Split empty string",
			input:  "",
			delim:  ",",
			expect: []string{""},
		},
		{
			name:   "Split by empty delimiter",
			input:  "abc",
			delim:  "",
			expect: []string{"a", "b", "c"},
		},
		{
			name:   "Split with delimiter not found",
			input:  "abc",
			delim:  ",",
			expect: []string{"abc"},
		},
		{
			name:   "Split with multiple delimiters",
			input:  "a,b,c,d",
			delim:  ",",
			expect: []string{"a", "b", "c", "d"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := StringSplit(tt.input, tt.delim)
			if !reflect.DeepEqual(got, tt.expect) {
				t.Errorf("StringSplit(%q, %q) = %v, want %v",
					tt.input, tt.delim, got, tt.expect)
			}
		})
	}
} 