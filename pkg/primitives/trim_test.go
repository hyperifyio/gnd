package primitives

import (
	"testing"
)

func TestTrim(t *testing.T) {
	tests := []struct {
		name     string
		args     []interface{}
		expected interface{}
		wantErr  bool
	}{
		{
			name:     "trim whitespace",
			args:     []interface{}{"   Hello, World!   "},
			expected: "Hello, World!",
			wantErr:  false,
		},
		{
			name:     "trim custom chars",
			args:     []interface{}{"!!!Warning!!!", ".!"},
			expected: "Warning",
			wantErr:  false,
		},
		{
			name:     "trim newlines and tabs",
			args:     []interface{}{"\n\tHello, World!\n\t"},
			expected: "Hello, World!",
			wantErr:  false,
		},
		{
			name:     "trim mixed whitespace",
			args:     []interface{}{" \t\nHello, World!\n\t "},
			expected: "Hello, World!",
			wantErr:  false,
		},
		{
			name:     "trim multiple custom chars",
			args:     []interface{}{"***Hello***World***", "*"},
			expected: "Hello***World",
			wantErr:  false,
		},
		{
			name:     "trim non-existent chars",
			args:     []interface{}{"Hello, World!", "xyz"},
			expected: "Hello, World!",
			wantErr:  false,
		},
		{
			name:     "empty string",
			args:     []interface{}{""},
			expected: "",
			wantErr:  false,
		},
		{
			name:     "no args",
			args:     []interface{}{},
			expected: nil,
			wantErr:  true,
		},
	}

	trimPrim := &Trim{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := trimPrim.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Trim.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.expected {
				t.Errorf("Trim.Execute() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestTrimName(t *testing.T) {
	trimPrim := &Trim{}
	expected := "/gnd/trim"
	if got := trimPrim.Name(); got != expected {
		t.Errorf("Trim.Name() = %v, want %v", got, expected)
	}
}
