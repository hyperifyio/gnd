package primitive

import (
	"testing"
)

func TestLet(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		expected interface{}
		wantErr  bool
	}{
		{
			name:     "let x y",
			args:     []string{"x", "y"},
			expected: "y",
			wantErr:  false,
		},
		{
			name:     "let x",
			args:     []string{"x"},
			expected: "x",
			wantErr:  false,
		},
		{
			name:     "let",
			args:     []string{},
			expected: "",
			wantErr:  false,
		},
		{
			name:     "too many args",
			args:     []string{"x", "y", "z"},
			expected: nil,
			wantErr:  true,
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
			if got != tt.expected {
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
