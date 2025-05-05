package primitive

import (
	"reflect"
	"testing"
)

func TestStringMatch(t *testing.T) {
	// Test cases
	tests := []struct {
		name    string
		input   string
		pattern string
		expect  []Match
		wantErr bool
	}{
		{
			name:    "Simple match",
			input:   "Hello, World!",
			pattern: "World",
			expect: []Match{
				{Start: 7, End: 12, Text: "World"},
			},
			wantErr: false,
		},
		{
			name:    "Multiple matches",
			input:   "a1b2c3",
			pattern: "[0-9]",
			expect: []Match{
				{Start: 1, End: 2, Text: "1"},
				{Start: 3, End: 4, Text: "2"},
				{Start: 5, End: 6, Text: "3"},
			},
			wantErr: false,
		},
		{
			name:    "No matches",
			input:   "Hello, World!",
			pattern: "xyz",
			expect:  []Match{},
			wantErr: false,
		},
		{
			name:    "Invalid pattern",
			input:   "Hello, World!",
			pattern: "[",
			expect:  nil,
			wantErr: true,
		},
		{
			name:    "Empty string",
			input:   "",
			pattern: ".*",
			expect: []Match{
				{Start: 0, End: 0, Text: ""},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := StringMatch(tt.input, tt.pattern)
			
			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if !reflect.DeepEqual(got, tt.expect) {
				t.Errorf("StringMatch(%q, %q) = %v, want %v",
					tt.input, tt.pattern, got, tt.expect)
			}
		})
	}
} 