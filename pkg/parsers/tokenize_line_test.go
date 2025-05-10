package parsers

import (
	"testing"
)

func TestTokenizeLine(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []interface{}
		wantErr  bool
	}{
		{
			name:     "simple space-separated tokens",
			input:    "hello world test",
			expected: []interface{}{"hello", "world", "test"},
			wantErr:  false,
		},
		{
			name:     "tokens with tabs",
			input:    "hello\tworld\ttest",
			expected: []interface{}{"hello", "world", "test"},
			wantErr:  false,
		},
		{
			name:     "string with quotes",
			input:    `hello "world test" end`,
			expected: []interface{}{"hello", "world test", "end"},
			wantErr:  false,
		},
		{
			name:     "string with escape sequences",
			input:    `"hello\nworld" "test\tend"`,
			expected: []interface{}{"hello\nworld", "test\tend"},
			wantErr:  false,
		},
		{
			name:     "array notation",
			input:    "command [arg1 arg2]",
			expected: []interface{}{"command", []interface{}{"arg1", "arg2"}},
			wantErr:  false,
		},
		{
			name:     "array notation two times",
			input:    "command [arg1 arg2] [arg3 arg4]",
			expected: []interface{}{"command", []interface{}{"arg1", "arg2"}, []interface{}{"arg3", "arg4"}},
			wantErr:  false,
		},
		{
			name:     "nested arrays",
			input:    "command [arg1 [nested arg]]",
			expected: []interface{}{"command", []interface{}{"arg1", []interface{}{"nested", "arg"}}},
			wantErr:  false,
		},
		{
			name:     "array with quotes",
			input:    `command ["arg1" "hello world"]`,
			expected: []interface{}{"command", []interface{}{"arg1", "hello world"}},
			wantErr:  false,
		},
		{
			name:     "nested arrays with quotes",
			input:    `command [arg1 [nested "hello world"]]`,
			expected: []interface{}{"command", []interface{}{"arg1", []interface{}{"nested", "hello world"}}},
			wantErr:  false,
		},
		{
			name:     "unclosed string",
			input:    `"hello world`,
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "unclosed array",
			input:    "command [arg1",
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "empty input",
			input:    "",
			expected: []interface{}{},
			wantErr:  false,
		},
		{
			name:     "multiple spaces between tokens",
			input:    "hello    world    test",
			expected: []interface{}{"hello", "world", "test"},
			wantErr:  false,
		},
		{
			name:     "string with escaped quotes",
			input:    `"hello \"world\""`,
			expected: []interface{}{`hello "world"`},
			wantErr:  false,
		},
		{
			name:     "string with escaped backslash",
			input:    `"hello\\world"`,
			expected: []interface{}{`hello\world`},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := TokenizeLine(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("%s: TokenizeLine() error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if len(got) != len(tt.expected) {
					t.Errorf("%s: TokenizeLine() got length = %v, want length %v", tt.name, len(got), len(tt.expected))
					return
				}
				for i := range got {
					compareTokens(t, tt.name, i, got[i], tt.expected[i])
				}
			}
		})
	}
}

func compareTokens(t *testing.T, testName string, index int, got, want interface{}) {
	switch want := want.(type) {
	case string:
		if gotStr, ok := got.(string); !ok || gotStr != want {
			t.Errorf("%s: TokenizeLine() got[%d] = %v (%T), want[%d] = %v (string)", testName, index, got, got, index, want)
		}
	case []interface{}:
		if gotArr, ok := got.([]interface{}); !ok {
			t.Errorf("%s: TokenizeLine() got[%d] = %v (%T), want[%d] = %v ([]interface{})", testName, index, got, got, index, want)
		} else {
			if len(gotArr) != len(want) {
				t.Errorf("%s: TokenizeLine() got[%d] length = %v, want[%d] length = %v", testName, index, len(gotArr), index, len(want))
				return
			}
			for j := range gotArr {
				compareTokens(t, testName, j, gotArr[j], want[j])
			}
		}
	default:
		t.Errorf("%s: TokenizeLine() unexpected type in want[%d]: %T", testName, index, want)
	}
}
