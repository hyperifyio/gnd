package primitives_test

import (
	"github.com/hyperifyio/gnd/pkg/interpreters"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitives"
	"reflect"
	"testing"
)

func TestHandleExecResult(t *testing.T) {
	tests := []struct {
		name       string
		source     string
		execResult *primitives.ExecResult
		want       interface{}
		wantErr    bool
	}{
		{
			name:   "empty routine returns empty slice",
			source: "test",
			execResult: primitives.NewExecResult(
				[]*parsers.Instruction{},
				[]interface{}{},
			),
			want:    []interface{}{},
			wantErr: false,
		},
		{
			name:   "routine with return value",
			source: "test",
			execResult: primitives.NewExecResult(
				[]*parsers.Instruction{
					{
						Opcode:      "/gnd/return",
						Destination: parsers.NewPropertyRef("_"),
						Arguments:   []interface{}{"hello"},
					},
				},
				[]interface{}{},
			),
			want:    "hello",
			wantErr: false,
		},
		{
			name:   "routine with arguments",
			source: "test",
			execResult: primitives.NewExecResult(
				[]*parsers.Instruction{
					{
						Opcode:      "/gnd/return",
						Destination: parsers.NewPropertyRef("_"),
						Arguments:   []interface{}{"arg1"},
					},
				},
				[]interface{}{"arg1"},
			),
			want:    "arg1",
			wantErr: false,
		},
		{
			name:   "routine with error",
			source: "test",
			execResult: primitives.NewExecResult(
				[]*parsers.Instruction{
					{
						Opcode:      "/gnd/throw",
						Destination: parsers.NewPropertyRef("_"),
						Arguments:   []interface{}{"test error"},
					},
				},
				[]interface{}{},
			),
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			i := interpreters.NewInterpreter("test", map[string]string{
				"return": "/gnd/return",
				"error":  "/gnd/error",
			})
			got, err := primitives.HandleExecResult(i, tt.execResult)
			if (err != nil) != tt.wantErr {
				t.Errorf("HandleExecResult(%s) error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("HandleExecResult(%s) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
