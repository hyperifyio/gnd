package parsers

import (
	"strings"
	"testing"
)

func TestParseInt(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		want    int
		wantErr bool
		errMsg  string
	}{
		{
			name:    "integer",
			input:   10,
			want:    10,
			wantErr: false,
			errMsg:  "",
		},
		{
			name:    "valid integer string",
			input:   "15",
			want:    15,
			wantErr: false,
			errMsg:  "",
		},
		{
			name:    "invalid integer string",
			input:   "abc",
			want:    0,
			wantErr: true,
			errMsg:  "value invalid: abc",
		},
		{
			name:    "invalid type",
			input:   true,
			want:    0,
			wantErr: true,
			errMsg:  "value must be an integer: bool",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseInt(tt.input)
			if err != nil && !tt.wantErr {
				t.Errorf("ParseInt(%v) error = %v, wantErr %v", tt.input, err, tt.wantErr)
				return
			}
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("ParseInt(%v) error = %v, want error starting with %v", tt.input, err, tt.errMsg)
				}
				return
			}
			if got != tt.want {
				t.Errorf("ParseInt(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}
