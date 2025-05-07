package primitive

import (
	"testing"
)

func TestUppercase(t *testing.T) {
	tests := []struct {
		name     string
		args     []interface{}
		expected interface{}
		wantErr  bool
	}{
		{
			name:     "simple uppercase",
			args:     []interface{}{"Hello, World!"},
			expected: "HELLO, WORLD!",
			wantErr:  false,
		},
		{
			name:     "already uppercase",
			args:     []interface{}{"HELLO, WORLD!"},
			expected: "HELLO, WORLD!",
			wantErr:  false,
		},
		{
			name:     "mixed case",
			args:     []interface{}{"HeLLo, WoRlD!"},
			expected: "HELLO, WORLD!",
			wantErr:  false,
		},
		{
			name:     "special characters",
			args:     []interface{}{"hello 123 !@#$%^&*()"},
			expected: "HELLO 123 !@#$%^&*()",
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

	uppercasePrim := &Uppercase{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := uppercasePrim.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Uppercase.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.expected {
				t.Errorf("Uppercase.Execute() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestUppercaseName(t *testing.T) {
	uppercasePrim := &Uppercase{}
	expected := "/gnd/uppercase"
	if got := uppercasePrim.Name(); got != expected {
		t.Errorf("Uppercase.Name() = %v, want %v", got, expected)
	}
}
