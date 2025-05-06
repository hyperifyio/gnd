package primitive

import (
	"testing"
)

func TestExit(t *testing.T) {
	tests := []struct {
		name    string
		args    []interface{}
		wantErr bool
		errMsg  string
	}{
		{
			name:    "no arguments",
			args:    []interface{}{},
			wantErr: false,
			errMsg:  "",
		},
		{
			name:    "integer exit code",
			args:    []interface{}{2},
			wantErr: false,
			errMsg:  "",
		},
		{
			name:    "string exit code",
			args:    []interface{}{"3"},
			wantErr: false,
			errMsg:  "",
		},
		{
			name:    "destination and integer exit code",
			args:    []interface{}{"code", 4},
			wantErr: false,
			errMsg:  "",
		},
		{
			name:    "destination and string exit code",
			args:    []interface{}{"code", "5"},
			wantErr: false,
			errMsg:  "",
		},
		{
			name:    "invalid exit code type",
			args:    []interface{}{"not a number"},
			wantErr: true,
			errMsg:  "exit code must be an integer",
		},
		{
			name:    "invalid exit code type with destination",
			args:    []interface{}{"code", "not a number"},
			wantErr: true,
			errMsg:  "exit code must be an integer",
		},
	}

	exitPrim := &Exit{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := exitPrim.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Exit.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && err != nil {
				if tt.errMsg != "" && err.Error()[:len(tt.errMsg)] != tt.errMsg {
					t.Errorf("Exit.Execute() error message = %v, want %v", err.Error(), tt.errMsg)
				}
			}
		})
	}
}

func TestExitName(t *testing.T) {
	exitPrim := &Exit{}
	expected := "/gnd/exit"
	if got := exitPrim.Name(); got != expected {
		t.Errorf("Exit.Name() = %v, want %v", got, expected)
	}
}
