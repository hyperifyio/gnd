package primitives

import (
	"testing"
)

func TestFirst(t *testing.T) {
	tests := []struct {
		name     string
		args     []interface{}
		expected interface{}
		wantErr  bool
	}{
		{
			name:     "no arguments",
			args:     []interface{}{},
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "no items in array",
			args:     []interface{}{[]interface{}{}},
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "single string argument",
			args:     []interface{}{"test"},
			expected: "test",
			wantErr:  false,
		},
		{
			name:     "single string array",
			args:     []interface{}{[]interface{}{"test"}},
			expected: "test",
			wantErr:  false,
		},
		{
			name:     "multiple arguments",
			args:     []interface{}{42, "test", true},
			expected: 42,
			wantErr:  false,
		},
		{
			name:     "array",
			args:     []interface{}{[]interface{}{42, "test", true}},
			expected: 42,
			wantErr:  false,
		},
		{
			name:     "nil argument",
			args:     []interface{}{nil, "test"},
			expected: nil,
			wantErr:  false,
		},
		{
			name:     "nil argument in array",
			args:     []interface{}{[]interface{}{nil, "test"}},
			expected: nil,
			wantErr:  false,
		},
	}

	firstPrim := &First{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := firstPrim.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("First.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.expected {
				t.Errorf("First.Execute() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestFirstName(t *testing.T) {
	firstPrim := &First{}
	expected := "/gnd/first"
	if got := firstPrim.Name(); got != expected {
		t.Errorf("First.Name() = %v, want %v", got, expected)
	}
}
