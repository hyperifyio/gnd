package primitive

import (
	"reflect"
	"testing"
)

func TestLet(t *testing.T) {
	tests := []struct {
		name     string
		args     []interface{}
		expected interface{}
		wantErr  bool
	}{
		{
			name:     "empty args",
			args:     []interface{}{},
			expected: "",
			wantErr:  false,
		},
		{
			name:     "single value",
			args:     []interface{}{"value"},
			expected: "value",
			wantErr:  false,
		},
		{
			name:     "string with newline",
			args:     []interface{}{"Hello\nWorld"},
			expected: "Hello\nWorld",
			wantErr:  false,
		},
		{
			name:     "string with escaped quotes",
			args:     []interface{}{"Hello \"World\""},
			expected: "Hello \"World\"",
			wantErr:  false,
		},
		{
			name:     "string with mixed escapes",
			args:     []interface{}{"Hello\n\"World\"\t!"},
			expected: "Hello\n\"World\"\t!",
			wantErr:  false,
		},
		{
			name:     "simple array",
			args:     []interface{}{[]interface{}{1, 2, 3}},
			expected: []interface{}{1, 2, 3},
			wantErr:  false,
		},
		{
			name:     "nested array",
			args:     []interface{}{[]interface{}{1, []interface{}{2, 3}, 4}},
			expected: []interface{}{1, []interface{}{2, 3}, 4},
			wantErr:  false,
		},
		{
			name:     "deeply nested array",
			args:     []interface{}{[]interface{}{1, []interface{}{2, []interface{}{3, 4}}, 5}},
			expected: []interface{}{1, []interface{}{2, []interface{}{3, 4}}, 5},
			wantErr:  false,
		},
		{
			name:     "mixed type array",
			args:     []interface{}{[]interface{}{1, "two", true, []interface{}{3, 4}}},
			expected: []interface{}{1, "two", true, []interface{}{3, 4}},
			wantErr:  false,
		},
		{
			name:     "empty array",
			args:     []interface{}{[]interface{}{}},
			expected: []interface{}{},
			wantErr:  false,
		},
	}

	let := &Let{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := let.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Let.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.expected) {
				t.Errorf("Let.Execute() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestLetName(t *testing.T) {
	let := &Let{}
	expected := "/gnd/let"
	if got := let.Name(); got != expected {
		t.Errorf("Let.Name() = %v, want %v", got, expected)
	}
}
