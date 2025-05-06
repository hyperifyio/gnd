package primitive

import (
	"testing"
)

func TestSelect(t *testing.T) {
	tests := []struct {
		name     string
		args     []interface{}
		expected interface{}
		wantErr  bool
	}{
		{
			name:     "true condition",
			args:     []interface{}{"true", "Proceed", "Abort"},
			expected: "Proceed",
			wantErr:  false,
		},
		{
			name:     "false condition",
			args:     []interface{}{"false", "Proceed", "Abort"},
			expected: "Abort",
			wantErr:  false,
		},
		{
			name:     "uppercase true",
			args:     []interface{}{"TRUE", "Proceed", "Abort"},
			expected: "Abort",
			wantErr:  false,
		},
		{
			name:     "uppercase false",
			args:     []interface{}{"FALSE", "Proceed", "Abort"},
			expected: "Abort",
			wantErr:  false,
		},
		{
			name:     "mixed case true",
			args:     []interface{}{"True", "Proceed", "Abort"},
			expected: "Abort",
			wantErr:  false,
		},
		{
			name:     "mixed case false",
			args:     []interface{}{"False", "Proceed", "Abort"},
			expected: "Abort",
			wantErr:  false,
		},
		{
			name:     "too few args",
			args:     []interface{}{"true", "Proceed"},
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "too many args",
			args:     []interface{}{"true", "Proceed", "Abort", "Extra"},
			expected: nil,
			wantErr:  true,
		},
	}

	selectPrim := &Select{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := selectPrim.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Select.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.expected {
				t.Errorf("Select.Execute() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestSelectName(t *testing.T) {
	selectPrim := &Select{}
	expected := "/gnd/select"
	if got := selectPrim.Name(); got != expected {
		t.Errorf("Select.Name() = %v, want %v", got, expected)
	}
}
