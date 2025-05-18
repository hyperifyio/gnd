package primitives

import (
	"strings"
	"testing"
)

func TestExit(t *testing.T) {
	tests := []struct {
		name       string
		args       []interface{}
		wantErr    bool
		wantStatus int
		wantValue  interface{}
		errMsg     string
	}{
		{
			name:       "no arguments",
			args:       []interface{}{},
			wantErr:    false,
			errMsg:     "",
			wantStatus: 1,
			wantValue:  nil,
		},
		{
			name:       "integer exit code",
			args:       []interface{}{2},
			wantErr:    false,
			errMsg:     "",
			wantStatus: 2,
			wantValue:  nil,
		},
		{
			name:       "string exit code",
			args:       []interface{}{"3"},
			wantErr:    false,
			errMsg:     "",
			wantStatus: 3,
			wantValue:  nil,
		},
		{
			name:    "destination and integer exit code",
			args:    []interface{}{"code", 4},
			wantErr: true,
			errMsg:  "ParseInt: value invalid: code",
		},
		{
			name:    "destination and string exit code",
			args:    []interface{}{"code", "5"},
			wantErr: true,
			errMsg:  "ParseInt: value invalid: code",
		},
		{
			name:    "invalid exit code type",
			args:    []interface{}{"not a number"},
			wantErr: true,
			errMsg:  "ParseInt: value invalid: not a number",
		},
		{
			name:    "invalid exit code type with destination",
			args:    []interface{}{"code", "not a number"},
			wantErr: true,
			errMsg:  "ParseInt: value invalid: code",
		},
	}

	exitPrim := &Exit{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := exitPrim.Execute(tt.args)

			if err != nil {
				if tt.wantErr {
					if tt.errMsg != "" {
						str := err.Error()
						if !strings.Contains(str, tt.errMsg) {
							t.Errorf("Exit.Execute(%s): error message = %v, want %v", tt.name, str, tt.errMsg)
						}
					}
				} else {

					if result, ok := GetExitResult(err); ok {
						if got := result.Code; got != tt.wantStatus {
							t.Errorf("Exit.Execute(%s): status: got = %v, want %v", tt.name, got, tt.wantStatus)
						}
						if got := result.Value; got != tt.wantValue {
							t.Errorf("Exit.Execute(%s): value: got = %v, want %v", tt.name, got, tt.wantValue)
						}
					} else {
						t.Errorf("Exit.Execute(%s): error = %v, wantErr %v", tt.name, err, tt.wantErr)
					}

				}
			} else {
				t.Errorf("Exit.Execute(%s): Did not expect success", tt.name)
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
