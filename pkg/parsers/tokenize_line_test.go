package parsers

import (
	"reflect"
	"testing"
)

func TestTokenizeLine(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		expected []interface{}
		wantErr  bool
	}{
		{
			name:     "simple line",
			line:     "op dest arg1 arg2",
			expected: []interface{}{"op", "dest", NewPropertyRef("arg1"), NewPropertyRef("arg2")},
			wantErr:  false,
		},
		{
			name: "quoted string after first two tokens",
			line: "op dest \"quoted string\" arg1",
			expected: []interface{}{
				"op", "dest", "quoted string", NewPropertyRef("arg1"),
			},
			wantErr: false,
		},
		{
			name:     "multiple quoted strings",
			line:     "op dest \"first quoted\" \"second quoted\"",
			expected: []interface{}{"op", "dest", "first quoted", "second quoted"},
			wantErr:  false,
		},
		{
			name: "array",
			line: "op dest [arg1 arg2] arg3",
			expected: []interface{}{
				"op",
				"dest",
				[]interface{}{
					NewPropertyRef("arg1"),
					NewPropertyRef("arg2"),
				},
				NewPropertyRef("arg3"),
			},
			wantErr: false,
		},
		{
			name:     "unexpected array end",
			line:     "op ] arg1",
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "quoted string as first token",
			line:     "\"quoted\" dest arg1",
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "quoted string as second token",
			line:     "op \"quoted\" arg1",
			expected: nil,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := TokenizeLine(tt.line)
			if (err != nil) != tt.wantErr {
				t.Errorf("TokenizeLine() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !reflect.DeepEqual(got, tt.expected) {
				t.Errorf("TokenizeLine() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestTokenizeLineSimple(t *testing.T) {
	want := []interface{}{"op", "dest", NewPropertyRef("arg1"), NewPropertyRef("arg2")}
	got, err := TokenizeLine("op dest arg1 arg2")
	if err != nil {
		t.Errorf("TokenizeLine() error = %v", err)
		return
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("TokenizeLine() = %v, want %v", got, want)
	}
}

func TestTokenizeLine_TopLevel(t *testing.T) {
	got, err := TokenizeLine("op dest arg1 arg2")
	if err != nil {
		t.Errorf("TokenizeLine() error = %v", err)
		return
	}
	want := []interface{}{"op", "dest", NewPropertyRef("arg1"), NewPropertyRef("arg2")}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("TokenizeLine() = %v, want %v", got, want)
	}
}
