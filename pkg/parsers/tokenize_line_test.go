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
			name: "simple line",
			line: "$dest op arg1 arg2",
			expected: []interface{}{
				NewPropertyRef("dest"),
				"op",
				"arg1",
				"arg2",
			},
			wantErr: false,
		},
		{
			name: "simple line with props",
			line: "$dest op $arg1 $arg2",
			expected: []interface{}{
				NewPropertyRef("dest"),
				"op",
				NewPropertyRef("arg1"),
				NewPropertyRef("arg2"),
			},
			wantErr: false,
		},
		{
			name: "quoted string after first two tokens",
			line: "$dest op \"quoted string\" $arg1",
			expected: []interface{}{
				NewPropertyRef("dest"),
				"op",
				"quoted string",
				NewPropertyRef("arg1"),
			},
			wantErr: false,
		},
		{
			name: "multiple quoted strings",
			line: "$dest op \"first quoted\" \"second quoted\"",
			expected: []interface{}{
				NewPropertyRef("dest"),
				"op",
				"first quoted",
				"second quoted",
			},
			wantErr: false,
		},
		{
			name: "array",
			line: "$dest op [arg1 arg2] arg3",
			expected: []interface{}{
				NewPropertyRef("dest"),
				"op",
				[]interface{}{
					"arg1",
					"arg2",
				},
				"arg3",
			},
			wantErr: false,
		},
		{
			name: "array with props",
			line: "$dest op [$arg1 $arg2] $arg3",
			expected: []interface{}{
				NewPropertyRef("dest"),
				"op",
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
			line:     "\"quoted\" op arg1",
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "quoted string as second token",
			line:     "op \"quoted\" arg1",
			expected: []interface{}{"op", "quoted", "arg1"},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := TokenizeLine(tt.line)
			if err != nil && !tt.wantErr {
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
	want := []interface{}{
		NewPropertyRef("dest"), "op", "arg1", "arg2",
	}
	got, err := TokenizeLine("$dest op arg1 arg2")
	if err != nil {
		t.Errorf("TestTokenizeLineSimple() error = %v", err)
		return
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("TestTokenizeLineSimple() = %v, want %v", got, want)
	}
}

func TestTokenizeLine_TopLevel(t *testing.T) {
	got, err := TokenizeLine("$dest op $arg1 $arg2")
	if err != nil {
		t.Errorf("TestTokenizeLine_TopLevel() error = %v", err)
		return
	}
	want := []interface{}{NewPropertyRef("dest"), "op", NewPropertyRef("arg1"), NewPropertyRef("arg2")}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("TestTokenizeLine_TopLevel() = %v, want %v", got, want)
	}
}
