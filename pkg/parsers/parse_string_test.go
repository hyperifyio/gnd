package parsers

import "testing"

func TestParseString(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		want    string
		wantErr bool
	}{
		{
			name:    "nil",
			input:   nil,
			want:    "nil",
			wantErr: false,
		},
		{
			name:    "string",
			input:   "hello",
			want:    "hello",
			wantErr: false,
		},
		{
			name:    "int",
			input:   123,
			want:    "123",
			wantErr: false,
		},
		{
			name:    "bool",
			input:   true,
			want:    "true",
			wantErr: false,
		},
		{
			name:    "float64",
			input:   3.14,
			want:    "3.14",
			wantErr: false,
		},
		{
			name:    "float32",
			input:   float32(2.71),
			want:    "2.71",
			wantErr: false,
		},
		{
			name:    "map",
			input:   map[string]interface{}{"key": "value"},
			want:    `{ key value }`,
			wantErr: false,
		},
		{
			name:    "map with escapable characters",
			input:   map[string]interface{}{"key": "hello world"},
			want:    `{ key "hello world" }`,
			wantErr: false,
		},
		{
			name:    "array of strings",
			input:   []interface{}{"a", "b"},
			want:    "[ a b ]",
			wantErr: false,
		},
		{
			name:    "nested array",
			input:   []interface{}{"hello", []interface{}{"world", 123}},
			want:    "[ hello [ world 123 ] ]",
			wantErr: false,
		},
		{
			name:    "nested array with spaces in string",
			input:   []interface{}{"hi", []interface{}{"hello world", 123}},
			want:    "[ hi [ \"hello world\" 123 ] ]",
			wantErr: false,
		},
		{
			name:    "invalid type",
			input:   struct{}{},
			want:    "",
			wantErr: true,
		},
		{
			name:    "invalid element in array",
			input:   []interface{}{"ok", struct{}{}},
			want:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseString(tt.input)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseString(%v) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("ParseString(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}
