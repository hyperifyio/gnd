package primitives

import (
	"reflect"
	"testing"
)

func TestConcat(t *testing.T) {
	tests := []struct {
		name     string
		args     []interface{}
		expected interface{}
		wantErr  bool
	}{
		{
			name:     "single string",
			args:     []interface{}{"hello"},
			expected: "hello",
			wantErr:  false,
		},
		{
			name:     "string concatenation",
			args:     []interface{}{"hello", " ", "world"},
			expected: "hello world",
			wantErr:  false,
		},
		{
			name:     "string with newlines",
			args:     []interface{}{"Hello\n", "World\n"},
			expected: "Hello\nWorld\n",
			wantErr:  false,
		},
		{
			name:     "string with escaped quotes",
			args:     []interface{}{"Hello \"World\"", "!"},
			expected: "Hello \"World\"!",
			wantErr:  false,
		},
		{
			name:     "string with mixed escapes",
			args:     []interface{}{"Hello\n\"World\"", "\t!"},
			expected: "Hello\n\"World\"\t!",
			wantErr:  false,
		},
		{
			name:     "array concatenation",
			args:     []interface{}{[]interface{}{"a", "b"}, []interface{}{"c", "d"}},
			expected: []interface{}{"a", "b", "c", "d"},
			wantErr:  false,
		},
		{
			name:     "mixed array and string",
			args:     []interface{}{[]interface{}{"a", "b"}, "x", []interface{}{"c", "d"}},
			expected: []interface{}{"a", "b", "x", "c", "d"},
			wantErr:  false,
		},
		{
			name:     "empty array",
			args:     []interface{}{[]interface{}{}},
			expected: []interface{}{},
			wantErr:  false,
		},
		{
			name:     "no args",
			args:     []interface{}{},
			expected: nil,
			wantErr:  true,
		},
	}

	concatPrim := &Concat{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := concatPrim.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Concat.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if reflect.TypeOf(got) != reflect.TypeOf(tt.expected) {
					t.Errorf("Concat.Execute() type = %T, want %T", got, tt.expected)
					return
				}
				if !reflect.DeepEqual(got, tt.expected) {
					t.Errorf("Concat.Execute() = %v, want %v", got, tt.expected)
				}
			}
		})
	}
}

func TestConcatName(t *testing.T) {
	concatPrim := &Concat{}
	expected := "/gnd/concat"
	if got := concatPrim.Name(); got != expected {
		t.Errorf("Concat.Name() = %v, want %v", got, expected)
	}
}
