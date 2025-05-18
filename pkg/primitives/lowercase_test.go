package primitives

import (
	"testing"
)

func TestLowercase(t *testing.T) {
	tests := []struct {
		name     string
		args     []interface{}
		expected interface{}
		wantErr  bool
	}{
		{
			name:     "simple lowercase",
			args:     []interface{}{"Hello, World!"},
			expected: "hello, world!",
			wantErr:  false,
		},
		{
			name:     "already lowercase",
			args:     []interface{}{"hello, world!"},
			expected: "hello, world!",
			wantErr:  false,
		},
		{
			name:     "mixed case",
			args:     []interface{}{"HeLLo, WoRlD!"},
			expected: "hello, world!",
			wantErr:  false,
		},
		{
			name:     "special characters",
			args:     []interface{}{"HELLO 123 !@#$%^&*()"},
			expected: "hello 123 !@#$%^&*()",
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

	lowercasePrim := &Lowercase{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := lowercasePrim.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Lowercase.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.expected {
				t.Errorf("Lowercase.Execute() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestLowercaseName(t *testing.T) {
	lowercasePrim := &Lowercase{}
	expected := "/gnd/lowercase"
	if got := lowercasePrim.Name(); got != expected {
		t.Errorf("Lowercase.Name() = %v, want %v", got, expected)
	}
}
